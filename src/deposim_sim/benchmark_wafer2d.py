"""AIB wafer-2D benchmark runner with class-based role coverage."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_report.plot_catalog import benchmark_physviz_specs, to_plot_record
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .common.overrides import as_bool, normalize_overrides
from .common.render_tri import render_unstructured_map
from .output_manifest import artifact_links, artifact_paths, build_manifest, write_manifest
from .pipeline import run_aib_from_spec
from .results_index import next_run_dir, update_project_files
from .validation import validate_run_spec

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


_EPS = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for wafer2d benchmark execution.")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    class_id: str
    description: str
    role_a: str
    role_i: str | None
    role_b: str | None
    overrides: tuple[str, ...]


def build_wafer2d_cases() -> list[BenchmarkCase]:
    """Deterministic AIB benchmark set (A/AI/AB/AIB)."""

    return [
        BenchmarkCase(
            case_id="CASE-A",
            class_id="A",
            description="Base adsorption/reaction without inhibitor/byproduct role.",
            role_a="s0",
            role_i=None,
            role_b=None,
            overrides=(
                "sim.model.params.kinetics.k_rxn=0.012",
                "sim.model.params.inhibitor.K_I=0.0",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-AI",
            class_id="AI",
            description="Inhibitor role enabled with stronger inhibition coupling.",
            role_a="s0",
            role_i="s1",
            role_b=None,
            overrides=(
                "sim.model.params.kinetics.k_rxn=0.012",
                "sim.model.params.inhibitor.K_I=2.5",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-AB",
            class_id="AB",
            description="Byproduct role enabled with transport feedback.",
            role_a="s0",
            role_i=None,
            role_b="s2",
            overrides=(
                "sim.model.params.kinetics.k_rxn=0.014",
                "sim.model.params.scaling.C_B_scale=1.0",
                "sim.model.params.inhibitor.K_I=0.0",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-AIB",
            class_id="AIB",
            description="Inhibitor + byproduct roles jointly enabled.",
            role_a="s0",
            role_i="s1",
            role_b="s2",
            overrides=(
                "sim.model.params.kinetics.k_rxn=0.014",
                "sim.model.params.scaling.C_B_scale=1.0",
                "sim.model.params.inhibitor.K_I=3.0",
            ),
        ),
    ]


def _base_xy_points() -> np.ndarray:
    # Non-collinear deterministic point cloud for from_fluent_xy.
    return np.asarray(
        [
            [-45.0, -45.0],
            [-45.0, 0.0],
            [-45.0, 45.0],
            [0.0, -45.0],
            [0.0, 0.0],
            [0.0, 45.0],
            [45.0, -45.0],
            [45.0, 0.0],
            [45.0, 45.0],
            [-20.0, 20.0],
            [20.0, -20.0],
            [30.0, 20.0],
        ],
        dtype=float,
    )


def _build_cref(case: BenchmarkCase, xy_mm: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(np.square(xy_mm), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    theta = np.arctan2(xy_mm[:, 1], xy_mm[:, 0])

    scale_i = 1.4 if case.role_i is not None else 1.0
    scale_b = 1.2 if case.role_b is not None else 1.0

    s0 = np.clip(1.2 - 0.4 * r_norm + 0.10 * np.cos(theta), 0.02, np.inf)
    s1 = np.clip((0.55 + 0.20 * np.sin(2.0 * theta)) * scale_i, 0.0, np.inf)
    s2 = np.clip((0.65 + 0.35 * (r_norm**2)) * scale_b, 0.02, np.inf)
    s3 = np.full_like(s0, 0.08)
    return np.stack([s0, s1, s2, s3], axis=1)


def _build_measurement_h(xy_mm: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(np.square(xy_mm), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    return 0.008 + 0.0015 * (1.0 - r_norm)


def write_case_input_npz(
    *,
    case: BenchmarkCase,
    output_dir: Path,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Create deterministic Fluent-like NPZ payload for one benchmark case."""

    _require_numpy()
    xy_mm = _base_xy_points()
    cref = _build_cref(case, xy_mm)
    payload_path = output_dir / f"{case.case_id.lower()}_fluent.npz"
    np.savez(payload_path, xy=xy_mm, cref=cref)
    return payload_path, xy_mm, cref


def _masked_nanmean(values: Any, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(arr[valid]))


def _masked_nanmean_abs(values: Any, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(arr[valid])))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _save_physviz_map(path: Path, xy_mm: np.ndarray, values: np.ndarray, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = render_unstructured_map(
        ax,
        xy_mm=np.asarray(xy_mm, dtype=float),
        values=np.asarray(values, dtype=float),
        valid_mask=np.ones(np.asarray(values).shape, dtype=bool),
        cmap="viridis",
        discrete=False,
    )
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_trend_assertions(case_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate AIB benchmark trend assertions."""

    required = ("CASE-A", "CASE-AI", "CASE-AB", "CASE-AIB")
    for case_id in required:
        if case_id not in case_metrics:
            raise ValueError(f"missing benchmark metrics for case {case_id}")

    a = case_metrics["CASE-A"]
    ai = case_metrics["CASE-AI"]
    ab = case_metrics["CASE-AB"]
    aib = case_metrics["CASE-AIB"]

    checks: dict[str, bool] = {
        "assert_class_coverage": True,
        "assert_ai_inhibition": float(ai["mean_f_I"]) < float(a["mean_f_I"]),
        "assert_aib_inhibition_vs_ab": float(aib["mean_h_nm"]) < float(ab["mean_h_nm"]),
        "assert_ab_phi_b_finite": bool(np.isfinite(float(ab["mean_phi_B"]))),
        "assert_aib_phi_b_finite": bool(np.isfinite(float(aib["mean_phi_B"]))),
        "assert_a_phi_b_nan": bool(not np.isfinite(float(a["mean_phi_B"]))),
        "assert_ai_phi_b_nan": bool(not np.isfinite(float(ai["mean_phi_B"]))),
    }
    overall = all(checks.values())
    return {
        "delta_f_I_A_minus_AI": float(a["mean_f_I"]) - float(ai["mean_f_I"]),
        "delta_h_AB_minus_AIB": float(ab["mean_h_nm"]) - float(aib["mean_h_nm"]),
        **checks,
        "overall_passed": overall,
    }


def write_benchmark_report(
    *,
    run_dir: Path,
    run_id: str,
    summary: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    trend_assertions: Mapping[str, Any],
    output_links: Sequence[str],
    physviz_plots: Sequence[str] | None,
) -> Path:
    """Write benchmark HTML report to run directory."""

    assertion_rows = "".join(
        (
            f"<tr><th>{escape(str(name))}</th><td>{'PASS' if bool(value) else 'FAIL'}</td></tr>"
            if isinstance(value, bool)
            else f"<tr><th>{escape(str(name))}</th><td>{escape(str(value))}</td></tr>"
        )
        for name, value in trend_assertions.items()
    )

    case_table_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(str(row['case_id']))}</td>"
            f"<td>{escape(str(row['class_id']))}</td>"
            f"<td>{float(row['mean_h_nm']):.8g}</td>"
            f"<td>{float(row['mean_phi_B']):.8g}</td>"
            f"<td>{float(row['mean_f_I']):.8g}</td>"
            f"<td>{float(row['mean_CsA_over_CrefA']):.8g}</td>"
            f"<td>{float(row['mean_CsB_over_CrefB']):.8g}</td>"
            f"<td>{float(row['mean_abs_residual_nm']):.8g}</td>"
            f"<td>{float(row['score']):.8g}</td>"
            "</tr>"
        )
        for row in case_rows
    )

    artifact_links = "".join(f'<li><a href="{escape(str(path))}">{escape(str(path))}</a></li>' for path in output_links)
    if not artifact_links:
        artifact_links = "<li>None</li>"

    sections: list[str] = [
        "<h2>Summary</h2>"
        "<ul>"
        f"<li>case_count: {summary.get('case_count')}</li>"
        f"<li>overall_passed: {summary.get('trend_assertions', {}).get('overall_passed')}</li>"
        "</ul>",
        "<h2>Trend Assertions</h2>"
        f"<table>{assertion_rows}</table>",
        "<h2>Case Metrics</h2>"
        "<table>"
        "<tr><th>case_id</th><th>class</th><th>mean_h_nm</th><th>mean_phi_B</th><th>mean_f_I</th>"
        "<th>mean_CsA_over_CrefA</th><th>mean_CsB_over_CrefB</th><th>mean_abs_residual_nm</th><th>score</th></tr>"
        f"{case_table_rows}</table>",
        f"<h2>Artifacts</h2><ul>{artifact_links}</ul>",
    ]

    if physviz_plots:
        plot_items = "".join(f'<li><a href="{escape(p)}">{escape(p)}</a></li>' for p in physviz_plots)
        sections.append(f"<h2>Physviz Maps</h2><ul>{plot_items}</ul>")

    style = (
        "body { font-family: sans-serif; margin: 1.2rem 1.8rem; }"
        "table { border-collapse: collapse; margin-bottom: 1rem; }"
        "th, td { border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; }"
    )

    page = render_report_page(
        title=f"Wafer2D AIB Benchmark: {run_id}",
        heading=f"Wafer2D AIB Benchmark: {run_id}",
        style=style,
        sections=sections,
    )
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def run_wafer2d_benchmark(
    config_name: str = "cvd_steady_min",
    overrides: Sequence[str] | None = None,
    *,
    with_physviz: bool = False,
    physviz_fast: bool = False,
) -> dict[str, Any]:
    """Run AIB wafer benchmark and persist artifacts."""

    _require_numpy()

    normalized_overrides = normalize_overrides(overrides, prefix_sim=False)
    base_overrides = [
        "sim.model.name=aib_ode",
        "sim.output.run_name=benchmark_wafer2d",
    ]

    selected_config = config_name
    try:
        base_spec = compose_sim_config(selected_config, overrides=[*base_overrides, *normalized_overrides])
    except Exception:
        # Legacy default compatibility (old command used --config-name smoke).
        if selected_config != "cvd_steady_min":
            selected_config = "cvd_steady_min"
            base_spec = compose_sim_config(selected_config, overrides=[*base_overrides, *normalized_overrides])
        else:
            raise

    sim = getattr(base_spec, "sim", base_spec)
    if not hasattr(sim, "model") or str(getattr(sim.model, "name", "")) != "aib_ode":
        raise ValueError("wafer2d benchmark requires sim.model.name='aib_ode'")

    project_dir = Path(str(sim.output.root_dir)) / str(sim.output.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = next_run_dir(project_dir, str(sim.output.run_name))
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    compose_and_save_sim_config(
        run_dir / "config_resolved.yaml",
        config_name=selected_config,
        overrides=[*base_overrides, *normalized_overrides],
    )

    cases = build_wafer2d_cases()
    case_rows: list[dict[str, Any]] = []
    case_metrics_by_id: dict[str, dict[str, Any]] = {}

    h_stack: list[np.ndarray] = []
    phi_stack: list[np.ndarray] = []
    fi_stack: list[np.ndarray] = []
    csa_stack: list[np.ndarray] = []
    csb_stack: list[np.ndarray] = []
    residual_stack: list[np.ndarray] = []

    representative: dict[str, Any] | None = None

    for case in cases:
        payload_path, xy_mm, _cref = write_case_input_npz(case=case, output_dir=inputs_dir)

        meas_h = _build_measurement_h(xy_mm)
        meas_path = inputs_dir / f"{case.case_id.lower()}_meas.npz"
        np.savez(meas_path, h_nm=meas_h, xy=xy_mm)

        case_overrides = [
            *base_overrides,
            *normalized_overrides,
            f"sim.inputs.fluent.file={payload_path.as_posix()}",
            "sim.inputs.fluent.mode=steady",
            "sim.roles.A=s0",
            "sim.measurement.enabled=true",
            f"sim.measurement.file={meas_path.as_posix()}",
            *list(case.overrides),
        ]
        if case.role_i is not None:
            case_overrides.append(f"sim.roles.I={case.role_i}")
        if case.role_b is not None:
            case_overrides.append(f"sim.roles.B={case.role_b}")

        case_spec = compose_sim_config(selected_config, overrides=case_overrides)
        validate_run_spec(case_spec)
        result = run_aib_from_spec(case_spec)

        edge_mask = np.asarray(result.grid.edge_mask, dtype=bool)
        mean_h = _masked_nanmean(result.fields["h_nm"], edge_mask)
        mean_phi = _masked_nanmean(result.fields["phi_B"], edge_mask)
        mean_fi = _masked_nanmean(result.fields["f_I"], edge_mask)
        mean_csa = _masked_nanmean(result.fields["CsA_over_CrefA"], edge_mask)
        mean_csb = _masked_nanmean(result.fields["CsB_over_CrefB"], edge_mask)
        mean_resid_abs = _masked_nanmean_abs(result.fields["residual_nm"], edge_mask)

        complexity = int(case.role_i is not None) + int(case.role_b is not None)
        score = mean_resid_abs + 0.05 * complexity

        row = {
            "case_id": case.case_id,
            "class_id": case.class_id,
            "description": case.description,
            "roles": {"A": case.role_a, "I": case.role_i, "B": case.role_b},
            "mean_h_nm": mean_h,
            "mean_phi_B": mean_phi,
            "mean_f_I": mean_fi,
            "mean_CsA_over_CrefA": mean_csa,
            "mean_CsB_over_CrefB": mean_csb,
            "mean_abs_residual_nm": mean_resid_abs,
            "score": score,
        }
        case_rows.append(row)
        case_metrics_by_id[case.case_id] = row

        h_stack.append(np.asarray(result.fields["h_nm"], dtype=float))
        phi_stack.append(np.asarray(result.fields["phi_B"], dtype=float))
        fi_stack.append(np.asarray(result.fields["f_I"], dtype=float))
        csa_stack.append(np.asarray(result.fields["CsA_over_CrefA"], dtype=float))
        csb_stack.append(np.asarray(result.fields["CsB_over_CrefB"], dtype=float))
        residual_stack.append(np.asarray(result.fields["residual_nm"], dtype=float))

        if case.case_id == "CASE-AIB":
            representative = {
                "xy_mm": np.asarray(result.diagnostics.get("xy_mm"), dtype=float),
                "h_nm": np.asarray(result.fields["h_nm"], dtype=float),
                "phi_B": np.asarray(result.fields["phi_B"], dtype=float),
                "f_I": np.asarray(result.fields["f_I"], dtype=float),
                "residual_nm": np.asarray(result.fields["residual_nm"], dtype=float),
            }

    np.savez(
        outputs_dir / "benchmark_cases.npz",
        h_nm=np.stack(h_stack, axis=0),
        phi_B=np.stack(phi_stack, axis=0),
        f_I=np.stack(fi_stack, axis=0),
        CsA_over_CrefA=np.stack(csa_stack, axis=0),
        CsB_over_CrefB=np.stack(csb_stack, axis=0),
        residual_nm=np.stack(residual_stack, axis=0),
    )

    metrics_path = outputs_dir / "benchmark_case_metrics.json"
    metrics_path.write_text(json.dumps(case_rows, indent=2), encoding="utf-8")

    ranking_rows = sorted(case_rows, key=lambda row: float(row["score"]))
    ranking_csv = outputs_dir / "ranking.csv"
    _write_csv(
        ranking_csv,
        ranking_rows,
        fieldnames=(
            "case_id",
            "class_id",
            "mean_h_nm",
            "mean_phi_B",
            "mean_f_I",
            "mean_CsA_over_CrefA",
            "mean_CsB_over_CrefB",
            "mean_abs_residual_nm",
            "score",
        ),
    )

    class_compare_rows: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in ranking_rows:
        by_class.setdefault(str(row["class_id"]), []).append(row)
    for class_id in sorted(by_class):
        best = by_class[class_id][0]
        class_compare_rows.append(
            {
                "class_id": class_id,
                "best_case_id": best["case_id"],
                "best_score": best["score"],
                "mean_h_nm": best["mean_h_nm"],
                "mean_phi_B": best["mean_phi_B"],
                "mean_f_I": best["mean_f_I"],
            }
        )

    class_compare_csv = outputs_dir / "class_compare.csv"
    _write_csv(
        class_compare_csv,
        class_compare_rows,
        fieldnames=("class_id", "best_case_id", "best_score", "mean_h_nm", "mean_phi_B", "mean_f_I"),
    )

    trend_assertions = evaluate_trend_assertions(case_metrics_by_id)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    physviz_rel_paths: list[str] = []
    plot_records: list[dict[str, Any]] = []
    if with_physviz and representative is not None:
        physviz_npz = outputs_dir / "physviz_maps.npz"
        np.savez(
            physviz_npz,
            xy_mm=representative["xy_mm"],
            h_nm=representative["h_nm"],
            phi_B=representative["phi_B"],
            f_I=representative["f_I"],
            residual_nm=representative["residual_nm"],
        )

        if plt is not None:
            for spec in benchmark_physviz_specs(fast=as_bool(physviz_fast)):
                values = representative.get(spec.source_key)
                if values is None:
                    continue
                _save_physviz_map(plots_dir / spec.filename, representative["xy_mm"], values, spec.title)
                rel = f"plots/{spec.filename}"
                physviz_rel_paths.append(rel)
                plot_records.append(to_plot_record(spec, rel_path=rel))

    artifact_rows: list[dict[str, Any]] = [
        {"id": "config", "path": "config_resolved.yaml", "kind": "yaml", "required": True},
        {"id": "summary", "path": "summary.json", "kind": "json", "required": True},
        {"id": "report", "path": "report.html", "kind": "html", "required": True},
        {"id": "manifest", "path": "outputs/manifest.json", "kind": "json", "required": True},
        {"id": "benchmark_cases", "path": "outputs/benchmark_cases.npz", "kind": "npz", "required": True},
        {"id": "benchmark_case_metrics", "path": "outputs/benchmark_case_metrics.json", "kind": "json", "required": True},
        {"id": "class_compare", "path": "outputs/class_compare.csv", "kind": "csv", "required": True},
        {"id": "ranking", "path": "outputs/ranking.csv", "kind": "csv", "required": True},
    ]
    if with_physviz and representative is not None:
        artifact_rows.append({"id": "physviz_maps", "path": "outputs/physviz_maps.npz", "kind": "npz", "required": True})

    manifest = build_manifest(
        run_id=run_id,
        mode="benchmark_wafer2d",
        created_at_utc=timestamp_utc,
        artifacts=artifact_rows,
        plots=plot_records,
        metadata={"sim_model": "aib_ode"},
    )
    artifact_map = artifact_paths(manifest)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "mode": "benchmark_wafer2d",
        "sim_model": "aib_ode",
        "case_count": len(case_rows),
        "case_ids": [row["case_id"] for row in case_rows],
        "trend_assertions": trend_assertions,
        "manifest_path": "outputs/manifest.json",
        "artifact_paths": artifact_map,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_links = artifact_links(manifest)
    write_benchmark_report(
        run_dir=run_dir,
        run_id=run_id,
        summary=summary,
        case_rows=case_rows,
        trend_assertions=trend_assertions,
        output_links=output_links,
        physviz_plots=physviz_rel_paths,
    )
    write_manifest(run_dir, manifest)
    update_project_files(project_dir, summary)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "case_metrics": case_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AIB wafer2d benchmark.")
    parser.add_argument("--config-name", default="cvd_steady_min")
    parser.add_argument("--with-physviz", action="store_true")
    parser.add_argument("--physviz-fast", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_wafer2d_benchmark(
        config_name=args.config_name,
        overrides=args.overrides,
        with_physviz=bool(args.with_physviz),
        physviz_fast=bool(args.physviz_fast),
    )
    run_dir = result["run_dir"]
    overall = bool(result["summary"]["trend_assertions"]["overall_passed"])
    print(f"[benchmark_wafer2d] wrote artifacts to: {run_dir}")
    print(f"[benchmark_wafer2d] overall_passed={overall}")
    return 0 if overall else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
