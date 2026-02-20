"""DOE execution helpers with case-dimension outputs and z_ref sensitivity workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .domain import build_domain_grid
from .input_builder import build_field_bundle
from .metrics import compute_kpi_metrics
from .physics.cvd_steady import run_cvd_steady
from .results_index import next_run_dir, update_project_files
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

    if sampling_mode == "grid":
        cases = _grid_cases(sweep)
    else:
        if random_cases < 1:
            raise ValueError("random_cases must be >= 1 when sampling='random'")
        cases = _random_cases(sweep, n_cases=random_cases, random_seed=random_seed)

    base_spec = compose_sim_config(config_name, overrides=base_overrides)
    project_dir = Path(base_spec.output.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    root_name = run_dir_name or f"{base_spec.output.run_dir_name}_doe"
    run_id, run_dir = next_run_dir(project_dir, root_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    compose_and_save_sim_config(
        run_dir / base_spec.output.resolved_config_filename,
        config_name=config_name,
        overrides=base_overrides,
    )
    (run_dir / "doe_sweep.json").write_text(json.dumps({"sampling": sampling_mode, "sweep": sweep}, indent=2), encoding="utf-8")

    case_payload: list[dict[str, Any]] = []
    thickness_cases: list[np.ndarray] = []
    deposition_rate_cases: list[np.ndarray] = []
    nu_values: list[float] = []
    center_edge_values: list[float | None] = []
    z_refs: list[float] = []
    grid_shape: tuple[int, ...] | None = None

    for case_index, case in enumerate(cases):
        case_overrides = list(base_overrides or [])
        case_overrides.extend(f"{key}={_literal(value)}" for key, value in case.items())
        spec = compose_sim_config(config_name, overrides=case_overrides)
        validate_run_spec(spec)
        grid = build_domain_grid(spec.domain)
        fields = build_field_bundle(spec, grid)
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=spec.model,
            process_time_s=spec.time.process_time_s,
            solver_config=spec.solver,
        )
        if grid_shape is None:
            grid_shape = result.thickness.shape
        elif result.thickness.shape != grid_shape:
            raise ValueError(
                f"DOE case shape mismatch: expected {grid_shape}, got {result.thickness.shape} at case {case_index}"
            )

        kpi = compute_kpi_metrics(
            np.asarray(result.thickness, dtype=float),
            grid,
            spec_min=spec.kpi.spec_min,
            spec_max=spec.kpi.spec_max,
            ring_count=spec.kpi.ring_count,
        )
        thickness_cases.append(np.asarray(result.thickness, dtype=float))
        deposition_rate_cases.append(np.asarray(result.deposition_rate, dtype=float))
        nu_values.append(float(kpi["nu_percent"]))
        center_edge_values.append(kpi["center_edge_delta"])
        z_refs.append(float(spec.reference_plane.z_ref_mm))
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

    doe_store = save_array_store(
        base_path=outputs_dir / "doe_cases",
        arrays={
            "thickness": thick,
            "deposition_rate": dep_rate,
            "nu_percent": nu_arr,
            "center_edge_delta": center_edge_arr,
            "z_ref_mm": z_ref_arr,
        },
        store=str(base_spec.output.array_store),
    )
    (outputs_dir / "doe_cases.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")

    rank_idx = np.argsort(nu_arr)
    ranking = [
        {"rank": int(rank + 1), "case_index": int(idx), "nu_percent": float(nu_arr[idx])}
        for rank, idx in enumerate(rank_idx[: min(10, len(rank_idx))])
    ]
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "doe",
        "sampling": sampling_mode,
        "case_count": int(len(cases)),
        "grid_shape": list(grid_shape or ()),
        "best_case_index": int(rank_idx[0]),
        "best_nu_percent": float(nu_arr[rank_idx[0]]),
        "mean_nu_percent": float(np.mean(nu_arr)),
        "sweep_keys": sorted(sweep),
        "ranking_top_nu": ranking,
        "doe_cases_store_used": doe_store["store_used"],
        "doe_cases_store_path": Path(doe_store["path"]).relative_to(run_dir).as_posix(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_metric(nu_arr, plots_dir / "kpi_nu_percent.png", ylabel="NU Percent")
    if np.isfinite(center_edge_arr).any():
        _plot_metric(center_edge_arr, plots_dir / "kpi_center_edge_delta.png", ylabel="Center-Edge Delta")

    ranking_rows = "".join(
        f"<tr><td>{row['rank']}</td><td>{row['case_index']}</td><td>{row['nu_percent']:.8g}</td></tr>"
        for row in ranking
    )
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
            "<h2>Artifacts</h2>"
            "<ul>"
            f"<li><a href='{summary['doe_cases_store_path']}'>{summary['doe_cases_store_path']}</a></li>"
            "<li><a href='outputs/doe_cases.json'>outputs/doe_cases.json</a></li>"
            "<li><a href='doe_sweep.json'>doe_sweep.json</a></li>"
            "<li><a href='summary.json'>summary.json</a></li>"
            "<li><a href='plots/kpi_nu_percent.png'>plots/kpi_nu_percent.png</a></li>"
            "</ul>",
        ],
    )
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")

    update_project_files(project_dir, summary)
    return DoeRunResult(run_dir=run_dir, case_count=len(cases), summary=summary)


def run_zref_sensitivity(
    *,
    config_name: str,
    z_ref_values_mm: Sequence[float],
    base_overrides: Sequence[str] | None = None,
) -> DoeRunResult:
    """Run z_ref sensitivity workflow as a DOE factor and emit a dedicated artifact."""
    if not z_ref_values_mm:
        raise ValueError("z_ref_values_mm must be non-empty")
    result = run_doe(
        config_name=config_name,
        sweep={"reference_plane.z_ref_mm": list(z_ref_values_mm)},
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

    report_path = result.run_dir / "report.html"
    report_text = report_path.read_text(encoding="utf-8")
    extra = (
        "<h2>z_ref Sensitivity</h2>"
        "<ul>"
        f"<li><a href='outputs/{Path(zref_store['path']).name}'>outputs/{Path(zref_store['path']).name}</a></li>"
        "<li><a href='plots/zref_sensitivity.png'>plots/zref_sensitivity.png</a></li>"
        "</ul>"
    )
    report_path.write_text(report_text.replace("</body></html>", extra + "</body></html>"), encoding="utf-8")
    return result


__all__ = ["DoeRunResult", "run_doe", "run_zref_sensitivity"]
