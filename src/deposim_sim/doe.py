"""DOE execution helpers for AIB workflows with case-dimension outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_report.plot_catalog import DOE_KPI_MAPS, DOE_ZREF_PLOT, to_plot_record
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .common.overrides import normalize_overrides, normalize_sweep
from .common.run_artifacts import (
    build_provenance_metadata,
    build_manifest_and_summary,
    create_run_layout,
    finalize_run_outputs,
    standard_artifact_rows,
)
from .metrics import compute_kpi_metrics
from .output_manifest import artifact_links, artifact_paths, build_manifest, load_manifest, write_manifest
from .pipeline import run_aib_from_spec
from .results_index import update_project_files
from .validation import validate_run_spec
from .zarr_output import load_array_store, save_array_store

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DoeRunResult:
    run_dir: Path
    case_count: int
    summary: dict[str, Any]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for DOE workflows.")


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/+-")
    if all(ch in safe for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _grid_cases(sweep: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = [key for key in sorted(sweep) if sweep[key]]
    if not keys:
        raise ValueError("sweep must define at least one non-empty parameter list")
    values = [list(sweep[key]) for key in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def _random_cases(
    sweep: Mapping[str, Sequence[Any]],
    *,
    n_cases: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    _require_numpy()
    keys = [key for key in sorted(sweep) if sweep[key]]
    if not keys:
        raise ValueError("sweep must define at least one non-empty parameter list")
    rng = np.random.default_rng(int(random_seed))
    cases: list[dict[str, Any]] = []
    for _ in range(int(n_cases)):
        case = {key: list(sweep[key])[int(rng.integers(0, len(sweep[key])))] for key in keys}
        cases.append(case)
    return cases


def _plot_metric(
    values: np.ndarray,
    output_path: Path,
    *,
    ylabel: str,
    xlabel: str = "Case Index",
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(values.shape[0]), values, marker="o", lw=1.5)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_doe(
    *,
    config_name: str,
    sweep: Mapping[str, Sequence[Any]],
    sampling: str = "grid",
    random_cases: int = 0,
    random_seed: int = 0,
    base_overrides: Sequence[str] | None = None,
    run_dir_name: str | None = None,
) -> DoeRunResult:
    """Run DOE with case-dimension output storage."""
    _require_numpy()
    sampling_mode = str(sampling).strip().lower()
    if sampling_mode not in {"grid", "random"}:
        raise ValueError("sampling must be 'grid' or 'random'")

    sweep_norm = normalize_sweep(sweep)
    base_norm = normalize_overrides(base_overrides, prefix_sim=True)

    if sampling_mode == "grid":
        cases = _grid_cases(sweep_norm)
    else:
        if random_cases < 1:
            raise ValueError("random_cases must be >= 1 when sampling='random'")
        cases = _random_cases(sweep_norm, n_cases=random_cases, random_seed=random_seed)

    base_spec = compose_sim_config(config_name, overrides=base_norm)
    sim = getattr(base_spec, "sim", base_spec)
    root_name = run_dir_name or f"{sim.output.run_name}_doe"
    layout = create_run_layout(
        root_dir=Path(sim.output.root_dir),
        project=str(sim.output.project),
        run_name=str(root_name),
        with_inputs_dir=False,
    )
    run_id = layout.run_id
    run_dir = layout.run_dir
    outputs_dir = layout.outputs_dir
    plots_dir = layout.plots_dir

    compose_and_save_sim_config(
        run_dir / "config_resolved.yaml",
        config_name=config_name,
        overrides=base_norm,
    )
    (run_dir / "doe_sweep.json").write_text(
        json.dumps({"sampling": sampling_mode, "sweep": sweep_norm}, indent=2),
        encoding="utf-8",
    )

    case_payload: list[dict[str, Any]] = []
    thickness_cases: list[np.ndarray] = []
    deposition_rate_cases: list[np.ndarray] = []
    nu_values: list[float] = []
    center_edge_values: list[float | None] = []
    z_refs: list[float] = []
    input_paths: set[str] = set()
    grid_shape: tuple[int, ...] | None = None

    for case_index, case in enumerate(cases):
        case_overrides = [*base_norm, *[f"{key}={_literal(value)}" for key, value in case.items()]]
        spec = compose_sim_config(config_name, overrides=case_overrides)
        validate_run_spec(spec)
        result = run_aib_from_spec(spec)

        thickness = np.asarray(result.thickness, dtype=float)
        dep_rate = np.asarray(result.deposition_rate, dtype=float)

        if grid_shape is None:
            grid_shape = thickness.shape
        elif thickness.shape != grid_shape:
            raise ValueError(
                f"DOE case shape mismatch: expected {grid_shape}, got {thickness.shape} at case {case_index}"
            )

        kpi = compute_kpi_metrics(thickness, result.grid)
        thickness_cases.append(thickness)
        deposition_rate_cases.append(dep_rate)
        nu_values.append(float(kpi["nu_percent"]))
        center_edge_values.append(kpi["center_edge_delta"])

        sim_case = getattr(spec, "sim", spec)
        input_paths.add(str(sim_case.inputs.fluent.file))
        if bool(getattr(sim_case.measurement, "enabled", False)) and str(getattr(sim_case.measurement, "file", "")).strip():
            input_paths.add(str(sim_case.measurement.file))
        z_refs.append(float(sim_case.reference_plane.z_ref_mm))
        case_payload.append(
            {
                "case_index": case_index,
                "parameters": dict(case),
                "overrides": case_overrides,
                "kpi": kpi,
            }
        )

    thick = np.stack(thickness_cases, axis=0)
    dep_rate = np.stack(deposition_rate_cases, axis=0)
    nu_arr = np.asarray(nu_values, dtype=float)
    center_edge_arr = np.asarray(
        [float(v) if v is not None else np.nan for v in center_edge_values],
        dtype=float,
    )
    z_ref_arr = np.asarray(z_refs, dtype=float)

    store_fmt = str(getattr(sim.output, "store", {}).get("format", "npz"))
    doe_store = save_array_store(
        base_path=outputs_dir / "doe_cases",
        arrays={
            "thickness": thick,
            "deposition_rate": dep_rate,
            "nu_percent": nu_arr,
            "center_edge_delta": center_edge_arr,
            "z_ref_mm": z_ref_arr,
        },
        store=store_fmt,
    )
    (outputs_dir / "doe_cases.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")

    rank_idx = np.argsort(nu_arr)
    ranking = [
        {"rank": int(rank + 1), "case_index": int(idx), "nu_percent": float(nu_arr[idx])}
        for rank, idx in enumerate(rank_idx[: min(10, len(rank_idx))])
    ]
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    doe_store_rel = Path(doe_store["path"]).relative_to(run_dir).as_posix()
    plot_records: list[dict[str, Any]] = []
    _plot_metric(nu_arr, plots_dir / DOE_KPI_MAPS[0].filename, ylabel=DOE_KPI_MAPS[0].title)
    plot_records.append(to_plot_record(DOE_KPI_MAPS[0], rel_path=f"plots/{DOE_KPI_MAPS[0].filename}"))
    if np.isfinite(center_edge_arr).any():
        _plot_metric(center_edge_arr, plots_dir / DOE_KPI_MAPS[1].filename, ylabel=DOE_KPI_MAPS[1].title)
        plot_records.append(to_plot_record(DOE_KPI_MAPS[1], rel_path=f"plots/{DOE_KPI_MAPS[1].filename}"))

    artifact_rows = standard_artifact_rows(
        include_report=True,
        extra_rows=[
            {"id": "doe_cases_store", "path": doe_store_rel, "kind": str(doe_store["store_used"]), "required": True},
            {"id": "doe_cases_json", "path": "outputs/doe_cases.json", "kind": "json", "required": True},
            {"id": "doe_sweep", "path": "doe_sweep.json", "kind": "json", "required": True},
        ],
    )
    provenance = build_provenance_metadata(
        workflow_name="doe",
        config_payload={
            "base_spec": base_spec,
            "sweep": sweep_norm,
            "sampling": sampling_mode,
            "random_cases": int(random_cases),
            "random_seed": int(random_seed),
        },
        input_paths=sorted(input_paths),
    )
    manifest, summary = build_manifest_and_summary(
        run_id=run_id,
        mode="doe",
        artifacts=artifact_rows,
        plots=plot_records,
        metadata={**provenance, "sampling": sampling_mode, "case_count": int(len(cases))},
        timestamp_utc=timestamp_utc,
        summary_fields={
        "sampling": sampling_mode,
        "case_count": int(len(cases)),
        "grid_shape": list(grid_shape or ()),
        "best_case_index": int(rank_idx[0]),
        "best_nu_percent": float(nu_arr[rank_idx[0]]),
        "mean_nu_percent": float(np.mean(nu_arr)),
        "sweep_keys": sorted(sweep_norm),
        "ranking_top_nu": ranking,
        "doe_cases_store_used": doe_store["store_used"],
        "doe_cases_store_path": doe_store_rel,
        **provenance,
        },
    )
    ranking_rows = "".join(
        f"<tr><td>{row['rank']}</td><td>{row['case_index']}</td><td>{row['nu_percent']:.8g}</td></tr>"
        for row in ranking
    )
    output_links = artifact_links(manifest)
    output_items = "".join(f"<li><a href='{path}'>{path}</a></li>" for path in output_links)
    report_html = render_report_page(
        title=f"DOE Report: {run_id}",
        heading=f"DOE Report: {run_id}",
        sections=[
            "<h2>Summary</h2>"
            "<ul>"
            f"<li>sampling: {sampling_mode}</li>"
            f"<li>case_count: {len(cases)}</li>"
            f"<li>best_case_index (min NU): {summary['best_case_index']}</li>"
            f"<li>best_nu_percent: {summary['best_nu_percent']:.8g}</li>"
            "</ul>",
            "<h2>Top Ranking (NU%)</h2>"
            "<table border='1' cellspacing='0' cellpadding='4'><tr><th>Rank</th><th>Case</th><th>NU%</th></tr>"
            f"{ranking_rows}</table>",
            f"<h2>Artifacts</h2><ul>{output_items}</ul>",
        ],
    )
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    finalize_run_outputs(layout=layout, summary=summary, manifest=manifest)
    return DoeRunResult(run_dir=run_dir, case_count=len(cases), summary=summary)


def run_zref_sensitivity(
    *,
    config_name: str,
    z_ref_values_mm: Sequence[float],
    base_overrides: Sequence[str] | None = None,
) -> DoeRunResult:
    """Run z_ref sensitivity workflow as a DOE factor and emit dedicated artifact."""

    if not z_ref_values_mm:
        raise ValueError("z_ref_values_mm must be non-empty")
    result = run_doe(
        config_name=config_name,
        sweep={"sim.reference_plane.z_ref_mm": list(z_ref_values_mm)},
        sampling="grid",
        base_overrides=base_overrides,
        run_dir_name="zref_sensitivity",
    )
    if np is None:
        return result

    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    case_store_path = result.run_dir / str(summary["doe_cases_store_path"])
    outputs = load_array_store(case_store_path)
    z_ref = np.asarray(outputs["z_ref_mm"], dtype=float)
    nu_percent = np.asarray(outputs["nu_percent"], dtype=float)
    order = np.argsort(z_ref)
    z_sorted = z_ref[order]
    nu_sorted = nu_percent[order]
    store_requested = str(summary.get("doe_cases_store_used", "npz")).strip().lower()
    if store_requested not in {"npz", "zarr", "hdf5"}:
        store_requested = "npz"
    zref_store = save_array_store(
        base_path=result.run_dir / "outputs" / "zref_sensitivity",
        arrays={"z_ref_mm": z_sorted, "nu_percent": nu_sorted},
        store=store_requested,
    )

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(z_sorted, nu_sorted, marker="o", lw=1.5)
        ax.set(xlabel="z_ref [mm]", ylabel="NU Percent", title="z_ref Sensitivity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(result.run_dir / "plots" / "zref_sensitivity.png", dpi=140)
        plt.close(fig)

    manifest_path = result.run_dir / "outputs" / "manifest.json"
    manifest = load_manifest(manifest_path)
    artifacts = list(manifest.get("artifacts", []))
    plots = list(manifest.get("plots", []))
    zref_rel = f"outputs/{Path(zref_store['path']).name}"
    artifacts.append({"id": "zref_sensitivity_store", "path": zref_rel, "kind": store_requested, "required": True})
    if plt is not None:
        plots.append(to_plot_record(DOE_ZREF_PLOT, rel_path=f"plots/{DOE_ZREF_PLOT.filename}"))
    manifest_updated = build_manifest(
        run_id=str(manifest["run_id"]),
        mode=str(manifest["mode"]),
        created_at_utc=str(manifest["created_at_utc"]),
        artifacts=artifacts,
        plots=plots,
        metadata=dict(manifest.get("metadata", {})),
    )
    write_manifest(result.run_dir, manifest_updated)
    artifact_map = artifact_paths(manifest_updated)
    summary["manifest_path"] = "outputs/manifest.json"
    summary["artifact_paths"] = artifact_map
    (result.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = result.run_dir / "report.html"
    report_text = report_path.read_text(encoding="utf-8")
    extra = (
        "<h2>z_ref Sensitivity</h2>"
        "<ul>"
        f"<li><a href='{zref_rel}'>{zref_rel}</a></li>"
        "<li><a href='plots/zref_sensitivity.png'>plots/zref_sensitivity.png</a></li>"
        "</ul>"
    )
    report_path.write_text(report_text.replace("</body></html>", extra + "</body></html>"), encoding="utf-8")
    project_dir = result.run_dir.parent.parent
    update_project_files(project_dir, summary)
    return result


__all__ = ["DoeRunResult", "run_doe", "run_zref_sensitivity"]
