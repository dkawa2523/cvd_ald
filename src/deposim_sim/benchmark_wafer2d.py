"""AIB wafer-2D benchmark runner with class-based role coverage."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_report.plot_catalog import benchmark_physviz_specs, to_plot_record
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .common.csv_io import write_rows_csv
from .common.overrides import as_bool, normalize_overrides
from .common.run_artifacts import create_run_layout, finalize_run_outputs
from .common.render_tri import render_unstructured_map
from .output_manifest import artifact_links, artifact_paths, build_manifest
from .pipeline import run_aib_from_spec
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

_RANKING_FIELDNAMES: tuple[str, ...] = (
    "case_id",
    "class_id",
    "mean_h_nm",
    "mean_phi_B",
    "mean_f_I",
    "mean_CsA_over_CrefA",
    "mean_CsB_over_CrefB",
    "mean_abs_residual_nm",
    "mean_km_A",
    "mean_tau_A",
    "mean_abs_residual_nm_flux_km",
    "mean_km_A_flux_km",
    "delta_score_flux_minus_free",
    "relative_delta_flux_minus_free",
    "score",
)

_CASE_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("case_id", "case_id"),
    ("class_id", "class"),
    ("mean_h_nm", "mean_h_nm"),
    ("mean_phi_B", "mean_phi_B"),
    ("mean_f_I", "mean_f_I"),
    ("mean_CsA_over_CrefA", "mean_CsA_over_CrefA"),
    ("mean_CsB_over_CrefB", "mean_CsB_over_CrefB"),
    ("mean_abs_residual_nm", "mean_abs_residual_nm"),
    ("mean_km_A", "mean_km_A"),
    ("mean_tau_A", "mean_tau_A"),
    ("mean_abs_residual_nm_flux_km", "mean_abs_residual_nm_flux_km"),
    ("mean_km_A_flux_km", "mean_km_A_flux_km"),
    ("delta_score_flux_minus_free", "delta_score_flux_minus_free"),
    ("relative_delta_flux_minus_free", "relative_delta_flux_minus_free"),
    ("score", "score"),
)


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


def _build_flux_sink(case: BenchmarkCase, xy_mm: np.ndarray, cref: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(np.square(xy_mm), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    theta = np.arctan2(xy_mm[:, 1], xy_mm[:, 0])
    vel = np.clip(0.05 - 0.02 * r_norm + 0.01 * np.cos(2.0 * theta), 0.005, np.inf)
    class_boost = 1.1 if case.class_id in {"AB", "AIB"} else 1.0
    vel = vel * class_boost
    return np.clip(cref * vel[:, None], 0.0, np.inf)


def _build_measurement_h(xy_mm: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(np.square(xy_mm), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    return 0.008 + 0.0015 * (1.0 - r_norm)


def write_case_input_npz(
    *,
    case: BenchmarkCase,
    output_dir: Path,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic Fluent-like NPZ payload for one benchmark case."""

    _require_numpy()
    xy_mm = _base_xy_points()
    cref = _build_cref(case, xy_mm)
    flux_sink = _build_flux_sink(case, xy_mm, cref)
    payload_path = output_dir / f"{case.case_id.lower()}_fluent.npz"
    np.savez(payload_path, xy=xy_mm, cref=cref, flux_sink=flux_sink)
    return payload_path, xy_mm, cref, flux_sink


def _masked_mean(
    values: Any,
    mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    take_abs: bool = False,
) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        valid &= np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float("nan")
    sample = np.abs(arr[valid]) if take_abs else arr[valid]
    if w is None:
        return float(np.mean(sample))
    wv = np.asarray(w[valid], dtype=float)
    denom = float(np.sum(wv))
    if denom <= _EPS:
        return float("nan")
    return float(np.sum(sample * wv) / denom)


def _build_case_overrides(
    *,
    base_overrides: Sequence[str],
    normalized_overrides: Sequence[str],
    payload_path: Path,
    meas_path: Path,
    case: BenchmarkCase,
    extra_overrides: Sequence[str] | None = None,
) -> list[str]:
    out = [
        *list(base_overrides),
        *list(normalized_overrides),
        f"sim.inputs.fluent.file={payload_path.as_posix()}",
        "sim.inputs.fluent.mode=steady",
        "sim.roles.A=s0",
        "sim.measurement.enabled=true",
        f"sim.measurement.file={meas_path.as_posix()}",
        *list(case.overrides),
    ]
    if case.role_i is not None:
        out.append(f"sim.roles.I={case.role_i}")
    if case.role_b is not None:
        out.append(f"sim.roles.B={case.role_b}")
    if extra_overrides:
        out.extend(list(extra_overrides))
    return out


def _run_case_spec(*, config_name: str, overrides: Sequence[str]) -> Any:
    spec = compose_sim_config(config_name, overrides=list(overrides))
    validate_run_spec(spec)
    return run_aib_from_spec(spec)


def _summarize_result(result: Any) -> dict[str, float]:
    edge_mask = np.asarray(result.grid.edge_mask, dtype=bool)
    area_w = np.asarray(result.grid.area_weights_mm2, dtype=float)
    return {
        "mean_h_nm": _masked_mean(result.fields["h_nm"], edge_mask, weights=area_w),
        "mean_phi_B": _masked_mean(result.fields["phi_B"], edge_mask, weights=area_w),
        "mean_f_I": _masked_mean(result.fields["f_I"], edge_mask, weights=area_w),
        "mean_CsA_over_CrefA": _masked_mean(result.fields["CsA_over_CrefA"], edge_mask, weights=area_w),
        "mean_CsB_over_CrefB": _masked_mean(result.fields["CsB_over_CrefB"], edge_mask, weights=area_w),
        "mean_abs_residual_nm": _masked_mean(result.fields["residual_nm"], edge_mask, weights=area_w, take_abs=True),
        "mean_km_A": _masked_mean(result.diagnostics.get("km_A_map"), edge_mask, weights=area_w),
        "mean_tau_A": _masked_mean(result.diagnostics.get("tau_A_map"), edge_mask, weights=area_w),
    }


def _format_case_cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if key in {"case_id", "class_id"}:
        return escape(str(value))
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return "nan"


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


def _add_extra_physviz_plot(
    *,
    plots_dir: Path,
    xy_mm: np.ndarray,
    values: np.ndarray,
    filename: str,
    title: str,
    cmap: str,
    rel_paths: list[str],
    records: list[dict[str, Any]],
) -> None:
    _save_physviz_map(plots_dir / filename, xy_mm, values, title)
    rel = f"plots/{filename}"
    rel_paths.append(rel)
    records.append(
        {
            "plot_id": filename[:-4] if filename.endswith(".png") else filename,
            "path": rel,
            "source_key": (filename[8:-4] if filename.startswith("physviz_") and filename.endswith(".png") else filename),
            "cmap": cmap,
            "discrete": False,
        }
    )


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
    flux_deltas = np.asarray(
        [
            float(case_metrics[cid].get("delta_score_flux_minus_free", float("nan")))
            for cid in required
        ],
        dtype=float,
    )
    flux_relative = np.asarray(
        [
            float(case_metrics[cid].get("relative_delta_flux_minus_free", float("nan")))
            for cid in required
        ],
        dtype=float,
    )
    if np.any(np.isfinite(flux_deltas)):
        checks["assert_flux_km_mean_not_worse"] = bool(float(np.nanmean(flux_deltas)) <= 0.0)
    if np.any(np.isfinite(flux_relative)):
        checks["assert_flux_km_relative_not_worse"] = bool(float(np.nanmean(flux_relative)) <= 0.0)
    mandatory = {k: v for k, v in checks.items() if not k.startswith("assert_flux_km_")}
    overall = all(mandatory.values())
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
        "<tr>"
        + "".join(f"<td>{_format_case_cell(row, key)}</td>" for key, _title in _CASE_TABLE_COLUMNS)
        + "</tr>"
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
        f"<li>km_spread_ratio: {summary.get('km_spread_ratio')}</li>"
        f"<li>flux_delta_mean: {summary.get('flux_delta_mean')}</li>"
        f"<li>flux_relative_delta_mean: {summary.get('flux_relative_delta_mean')}</li>"
        f"<li>p1_recommendation: {summary.get('p1_recommendation')}</li>"
        "</ul>",
        "<h2>Trend Assertions</h2>"
        f"<table>{assertion_rows}</table>",
        "<h2>Case Metrics</h2>"
        "<table>"
        f"<tr>{''.join(f'<th>{escape(title)}</th>' for _key, title in _CASE_TABLE_COLUMNS)}</tr>"
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
    compare_flux_km: bool = False,
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

    layout = create_run_layout(
        root_dir=Path(str(sim.output.root_dir)),
        project=str(sim.output.project),
        run_name=str(sim.output.run_name),
        with_inputs_dir=True,
    )
    run_id = layout.run_id
    run_dir = layout.run_dir
    outputs_dir = layout.outputs_dir
    inputs_dir = layout.inputs_dir
    plots_dir = layout.plots_dir
    if inputs_dir is None:
        raise RuntimeError("benchmark layout requires inputs_dir")

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
        payload_path, xy_mm, _cref, _flux = write_case_input_npz(case=case, output_dir=inputs_dir)

        meas_h = _build_measurement_h(xy_mm)
        meas_path = inputs_dir / f"{case.case_id.lower()}_meas.npz"
        np.savez(meas_path, h_nm=meas_h, xy=xy_mm)

        case_overrides = _build_case_overrides(
            base_overrides=base_overrides,
            normalized_overrides=normalized_overrides,
            payload_path=payload_path,
            meas_path=meas_path,
            case=case,
        )
        result = _run_case_spec(config_name=selected_config, overrides=case_overrides)
        summary_free = _summarize_result(result)

        complexity = int(case.role_i is not None) + int(case.role_b is not None)
        score = float(summary_free["mean_abs_residual_nm"]) + 0.05 * complexity
        row_flux: dict[str, float] = {}
        if compare_flux_km:
            flux_overrides = _build_case_overrides(
                base_overrides=base_overrides,
                normalized_overrides=normalized_overrides,
                payload_path=payload_path,
                meas_path=meas_path,
                case=case,
                extra_overrides=(
                    "sim.model.params.transport.km_source=from_cfd_flux_sink",
                    "sim.model.params.transport.gamma_km_A=1.0",
                    "sim.model.params.transport.gamma_km_B=1.0",
                    "sim.model.params.transport.from_cfd_flux_sink.flux_negative_policy=error",
                ),
            )
            flux_result = _run_case_spec(config_name=selected_config, overrides=flux_overrides)
            summary_flux = _summarize_result(flux_result)
            mean_resid_flux = float(summary_flux["mean_abs_residual_nm"])
            mean_km_flux = float(summary_flux["mean_km_A"])
            mean_resid_abs = float(summary_free["mean_abs_residual_nm"])
            rel_delta = (mean_resid_flux - mean_resid_abs) / max(abs(mean_resid_abs), _EPS)
            row_flux = {
                "mean_abs_residual_nm_flux_km": float(mean_resid_flux),
                "mean_km_A_flux_km": float(mean_km_flux),
                "delta_score_flux_minus_free": float(mean_resid_flux - mean_resid_abs),
                "relative_delta_flux_minus_free": float(rel_delta),
            }

        row = {
            "case_id": case.case_id,
            "class_id": case.class_id,
            "description": case.description,
            "roles": {"A": case.role_a, "I": case.role_i, "B": case.role_b},
            **summary_free,
            "score": score,
            **row_flux,
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
                "km_A": np.asarray(result.diagnostics.get("km_A_map"), dtype=float),
                "tau_A": np.asarray(result.diagnostics.get("tau_A_map"), dtype=float),
                "input_cref_A": np.asarray(_cref[:, 0], dtype=float),
                "input_flux_A": np.asarray(_flux[:, 0], dtype=float),
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

    (outputs_dir / "benchmark_case_metrics.json").write_text(json.dumps(case_rows, indent=2), encoding="utf-8")

    ranking_rows = sorted(case_rows, key=lambda row: float(row["score"]))
    ranking_csv = outputs_dir / "ranking.csv"
    write_rows_csv(
        ranking_csv,
        [dict(row) for row in ranking_rows],
        fieldnames=list(_RANKING_FIELDNAMES),
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

    write_rows_csv(
        outputs_dir / "class_compare.csv",
        class_compare_rows,
        fieldnames=["class_id", "best_case_id", "best_score", "mean_h_nm", "mean_phi_B", "mean_f_I"],
    )

    trend_assertions = evaluate_trend_assertions(case_metrics_by_id)
    km_values = np.asarray(
        [
            float(
                row.get(
                    "mean_km_A_flux_km",
                    row.get("mean_km_A", float("nan")),
                )
            )
            for row in case_rows
        ],
        dtype=float,
    )
    km_valid = km_values[np.isfinite(km_values) & (km_values > _EPS)]
    km_spread_ratio = float(np.max(km_valid) / np.min(km_valid)) if km_valid.size > 0 else float("nan")
    flux_delta = np.asarray([float(row.get("delta_score_flux_minus_free", float("nan"))) for row in case_rows], dtype=float)
    flux_delta_mean = float(np.nanmean(flux_delta)) if np.any(np.isfinite(flux_delta)) else float("nan")
    flux_rel_delta = np.asarray(
        [float(row.get("relative_delta_flux_minus_free", float("nan"))) for row in case_rows],
        dtype=float,
    )
    flux_relative_delta_mean = float(np.nanmean(flux_rel_delta)) if np.any(np.isfinite(flux_rel_delta)) else float("nan")
    p1_recommendation = bool(np.isfinite(km_spread_ratio) and km_spread_ratio >= 10.0)
    if np.isfinite(flux_delta_mean):
        p1_recommendation = p1_recommendation or bool(flux_delta_mean < 0.0)
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
            km_A=representative["km_A"],
            tau_A=representative["tau_A"],
            input_cref_A=representative["input_cref_A"],
            input_flux_A=representative["input_flux_A"],
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
            _add_extra_physviz_plot(
                plots_dir=plots_dir,
                xy_mm=representative["xy_mm"],
                values=representative["input_cref_A"],
                filename="physviz_input_cref_A.png",
                title="Input Cref(A) [a.u.]",
                cmap="viridis",
                rel_paths=physviz_rel_paths,
                records=plot_records,
            )
            _add_extra_physviz_plot(
                plots_dir=plots_dir,
                xy_mm=representative["xy_mm"],
                values=representative["input_flux_A"],
                filename="physviz_input_flux_A.png",
                title="Input flux_sink(A) [a.u.]",
                cmap="magma",
                rel_paths=physviz_rel_paths,
                records=plot_records,
            )

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
        "km_spread_ratio": km_spread_ratio,
        "flux_delta_mean": flux_delta_mean,
        "flux_relative_delta_mean": flux_relative_delta_mean,
        "p1_recommendation": p1_recommendation,
        "manifest_path": "outputs/manifest.json",
        "artifact_paths": artifact_map,
    }
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
    finalize_run_outputs(layout=layout, summary=summary, manifest=manifest)

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
    parser.add_argument("--compare-flux-km", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_wafer2d_benchmark(
        config_name=args.config_name,
        overrides=args.overrides,
        with_physviz=bool(args.with_physviz),
        physviz_fast=bool(args.physviz_fast),
        compare_flux_km=bool(args.compare_flux_km),
    )
    run_dir = result["run_dir"]
    overall = bool(result["summary"]["trend_assertions"]["overall_passed"])
    print(f"[benchmark_wafer2d] wrote artifacts to: {run_dir}")
    print(f"[benchmark_wafer2d] overall_passed={overall}")
    return 0 if overall else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
