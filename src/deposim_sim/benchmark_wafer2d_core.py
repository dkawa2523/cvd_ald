"""Core evaluation/report helpers for wafer2d benchmark orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_report.map_plot import save_map
from deposim_report.plot_catalog import PlotSpec, to_plot_record
from deposim_schema import compose_sim_config

from .pipeline import run_aib_from_spec
from .validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-12
_DEFAULT_FLUX_GAMMA_GRID: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

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
    ("mean_abs_residual_nm_flux_km_calibrated", "mean_abs_residual_nm_flux_km_calibrated"),
    ("mean_km_A_flux_km_calibrated", "mean_km_A_flux_km_calibrated"),
    ("delta_score_flux_calibrated_minus_free", "delta_score_flux_calibrated_minus_free"),
    ("relative_delta_flux_calibrated_minus_free", "relative_delta_flux_calibrated_minus_free"),
    ("score", "score"),
)


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
    case: Any,
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


def _normalize_flux_gamma_grid(values: Sequence[float] | None) -> tuple[float, ...]:
    raw = _DEFAULT_FLUX_GAMMA_GRID if values is None else tuple(values)
    out: list[float] = []
    seen: set[float] = set()
    for entry in raw:
        gamma = float(entry)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(f"flux gamma must be finite and > 0, got {entry!r}")
        if gamma in seen:
            continue
        seen.add(gamma)
        out.append(gamma)
    if not out:
        raise ValueError("flux gamma grid must contain at least one positive value")
    return tuple(out)


def _flux_compare_extra_overrides(*, gamma_km_a: float, gamma_km_b: float | None = None) -> tuple[str, ...]:
    gamma_b = gamma_km_a if gamma_km_b is None else gamma_km_b
    return (
        "sim.model.params.transport.km_source=from_cfd_flux_sink",
        f"sim.model.params.transport.gamma_km_A={float(gamma_km_a)}",
        f"sim.model.params.transport.gamma_km_B={float(gamma_b)}",
        "sim.model.params.transport.from_cfd_flux_sink.flux_negative_policy=error",
    )


def _parse_flux_gamma_grid_arg(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        return None
    return tuple(float(tok) for tok in tokens)


def _is_gamma_grid_edge(best: float, gamma_grid: Sequence[float]) -> bool:
    if not gamma_grid:
        return False
    grid = [float(g) for g in gamma_grid]
    lo = min(grid)
    hi = max(grid)
    tol = 1.0e-12
    return abs(float(best) - lo) <= tol or abs(float(best) - hi) <= tol


def _build_flux_km_judge(
    *,
    compare_flux_km: bool,
    flux_delta_mean: float,
    flux_delta_calibrated_mean: float,
    flux_eval_basis: str,
    flux_gamma_best: float | None,
    flux_gamma_grid: Sequence[float],
) -> dict[str, Any]:
    if not compare_flux_km:
        return {
            "status": "SKIP",
            "basis": "disabled",
            "delta_mean": float("nan"),
            "reason_codes": ["compare_flux_km_disabled"],
        }

    basis = "calibrated" if str(flux_eval_basis) == "calibrated" else "default"
    delta_mean = flux_delta_calibrated_mean if basis == "calibrated" else flux_delta_mean
    reasons: list[str] = []
    status = "PASS"

    if not np.isfinite(delta_mean):
        status = "WARN"
        reasons.append("delta_not_available")
    elif float(delta_mean) > 0.0:
        status = "FAIL"
        reasons.append("mean_residual_worse_than_free")
    else:
        reasons.append("mean_residual_not_worse_than_free")

    if basis == "calibrated" and np.isfinite(flux_delta_mean) and float(flux_delta_mean) > 0.0:
        reasons.append("default_worse_but_calibrated_improves")

    if basis == "calibrated" and flux_gamma_best is not None and _is_gamma_grid_edge(float(flux_gamma_best), flux_gamma_grid):
        if status == "PASS":
            status = "WARN"
        reasons.append("gamma_best_at_grid_edge")

    if flux_gamma_best is None:
        reasons.append("gamma_best_unavailable")

    return {
        "status": status,
        "basis": basis,
        "delta_mean": float(delta_mean),
        "default_delta_mean": float(flux_delta_mean),
        "calibrated_delta_mean": float(flux_delta_calibrated_mean),
        "gamma_best": None if flux_gamma_best is None else float(flux_gamma_best),
        "reason_codes": reasons,
    }


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


def _run_benchmark_case(
    *,
    config_name: str,
    base_overrides: Sequence[str],
    normalized_overrides: Sequence[str],
    payload_path: Path,
    meas_path: Path,
    case: Any,
    compare_flux_km: bool,
    flux_gamma_scan: Sequence[float],
) -> tuple[Any, dict[str, float], dict[str, float], dict[float, dict[str, float]]]:
    case_overrides = _build_case_overrides(
        base_overrides=base_overrides,
        normalized_overrides=normalized_overrides,
        payload_path=payload_path,
        meas_path=meas_path,
        case=case,
    )
    result = _run_case_spec(config_name=config_name, overrides=case_overrides)
    summary_free = _summarize_result(result)

    row_flux: dict[str, float] = {}
    case_gamma_rows: dict[float, dict[str, float]] = {}
    if not compare_flux_km:
        return result, summary_free, row_flux, case_gamma_rows

    flux_overrides = _build_case_overrides(
        base_overrides=base_overrides,
        normalized_overrides=normalized_overrides,
        payload_path=payload_path,
        meas_path=meas_path,
        case=case,
        extra_overrides=_flux_compare_extra_overrides(gamma_km_a=1.0, gamma_km_b=1.0),
    )
    flux_result = _run_case_spec(config_name=config_name, overrides=flux_overrides)
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

    for gamma in flux_gamma_scan:
        if abs(float(gamma) - 1.0) <= _EPS:
            summary_gamma = summary_flux
        else:
            gamma_overrides = _build_case_overrides(
                base_overrides=base_overrides,
                normalized_overrides=normalized_overrides,
                payload_path=payload_path,
                meas_path=meas_path,
                case=case,
                extra_overrides=_flux_compare_extra_overrides(gamma_km_a=float(gamma), gamma_km_b=float(gamma)),
            )
            gamma_result = _run_case_spec(config_name=config_name, overrides=gamma_overrides)
            summary_gamma = _summarize_result(gamma_result)

        case_gamma_rows[float(gamma)] = {
            "mean_abs_residual_nm": float(summary_gamma["mean_abs_residual_nm"]),
            "mean_km_A": float(summary_gamma["mean_km_A"]),
        }

    return result, summary_free, row_flux, case_gamma_rows


def _apply_flux_calibration(
    *,
    case_rows: list[dict[str, Any]],
    flux_gamma_best: float | None,
    flux_gamma_case_summaries: Mapping[str, Mapping[float, Mapping[str, float]]],
) -> None:
    if flux_gamma_best is None:
        return
    for row in case_rows:
        case_id = str(row["case_id"])
        gamma_rows = flux_gamma_case_summaries.get(case_id, {})
        selected = gamma_rows.get(float(flux_gamma_best))
        if selected is None:
            continue
        mean_resid_cal = float(selected["mean_abs_residual_nm"])
        mean_km_cal = float(selected["mean_km_A"])
        mean_resid_free = float(row["mean_abs_residual_nm"])
        rel_delta_cal = (mean_resid_cal - mean_resid_free) / max(abs(mean_resid_free), _EPS)
        row["mean_abs_residual_nm_flux_km_calibrated"] = mean_resid_cal
        row["mean_km_A_flux_km_calibrated"] = mean_km_cal
        row["delta_score_flux_calibrated_minus_free"] = float(mean_resid_cal - mean_resid_free)
        row["relative_delta_flux_calibrated_minus_free"] = float(rel_delta_cal)


def _format_case_cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if key in {"case_id", "class_id"}:
        return escape(str(value))
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return "nan"


def _add_physviz_plot(
    *,
    plots_dir: Path,
    grid: Any,
    xy_mm: np.ndarray,
    values: np.ndarray | Sequence[float],
    spec: PlotSpec,
    rel_paths: list[str],
    records: list[dict[str, Any]],
) -> None:
    save_map(
        plots_dir / spec.filename,
        grid=grid,
        value=np.asarray(values, dtype=float),
        title=spec.title,
        cmap=spec.cmap,
        xy_mm=np.asarray(xy_mm, dtype=float),
        valid_mask=np.asarray(grid.edge_mask, dtype=bool),
        discrete=bool(spec.discrete),
    )
    rel = f"plots/{spec.filename}"
    rel_paths.append(rel)
    records.append(to_plot_record(spec, rel_path=rel))


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
    flux_cal_deltas = np.asarray(
        [
            float(case_metrics[cid].get("delta_score_flux_calibrated_minus_free", float("nan")))
            for cid in required
        ],
        dtype=float,
    )
    flux_cal_relative = np.asarray(
        [
            float(case_metrics[cid].get("relative_delta_flux_calibrated_minus_free", float("nan")))
            for cid in required
        ],
        dtype=float,
    )
    if np.any(np.isfinite(flux_deltas)):
        checks["assert_flux_km_mean_not_worse"] = bool(float(np.nanmean(flux_deltas)) <= 0.0)
    if np.any(np.isfinite(flux_relative)):
        checks["assert_flux_km_relative_not_worse"] = bool(float(np.nanmean(flux_relative)) <= 0.0)
    if np.any(np.isfinite(flux_cal_deltas)):
        checks["assert_flux_km_calibrated_mean_not_worse"] = bool(float(np.nanmean(flux_cal_deltas)) <= 0.0)
    if np.any(np.isfinite(flux_cal_relative)):
        checks["assert_flux_km_calibrated_relative_not_worse"] = bool(float(np.nanmean(flux_cal_relative)) <= 0.0)
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

    flux_judge = dict(summary.get("flux_km_judge", {}) or {})
    flux_judge_reasons = flux_judge.get("reason_codes", [])
    if isinstance(flux_judge_reasons, (list, tuple)):
        reason_text = ",".join(str(item) for item in flux_judge_reasons)
    else:
        reason_text = str(flux_judge_reasons)

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
    flux_judge_rows = ""
    for key in ("status", "basis", "delta_mean", "default_delta_mean", "calibrated_delta_mean", "gamma_best", "reason_codes"):
        if key not in flux_judge:
            continue
        value = flux_judge.get(key)
        if isinstance(value, (list, tuple)):
            text = ", ".join(str(item) for item in value)
        else:
            text = str(value)
        flux_judge_rows += f"<tr><th>{escape(key)}</th><td>{escape(text)}</td></tr>"

    artifact_list_html = "".join(f'<li><a href="{escape(str(path))}">{escape(str(path))}</a></li>' for path in output_links)
    if not artifact_list_html:
        artifact_list_html = "<li>None</li>"

    sections: list[str] = [
        "<h2>Summary</h2>"
        "<ul>"
        f"<li>case_count: {summary.get('case_count')}</li>"
        f"<li>overall_passed: {summary.get('trend_assertions', {}).get('overall_passed')}</li>"
        f"<li>km_spread_ratio: {summary.get('km_spread_ratio')}</li>"
        f"<li>flux_delta_mean: {summary.get('flux_delta_mean')}</li>"
        f"<li>flux_relative_delta_mean: {summary.get('flux_relative_delta_mean')}</li>"
        f"<li>flux_gamma_best: {summary.get('flux_gamma_best')}</li>"
        f"<li>flux_delta_calibrated_mean: {summary.get('flux_delta_calibrated_mean')}</li>"
        f"<li>flux_relative_delta_calibrated_mean: {summary.get('flux_relative_delta_calibrated_mean')}</li>"
        f"<li>flux_eval_basis: {summary.get('flux_eval_basis')}</li>"
        f"<li>flux_judge_status: {flux_judge.get('status')}</li>"
        f"<li>flux_judge_reasons: {reason_text}</li>"
        f"<li>p1_recommendation: {summary.get('p1_recommendation')}</li>"
        "</ul>",
        "<h2>Trend Assertions</h2>"
        f"<table>{assertion_rows}</table>",
        "<h2>Flux-KM Judge</h2>"
        f"<table>{flux_judge_rows or '<tr><td>None</td></tr>'}</table>",
        "<h2>Case Metrics</h2>"
        "<table>"
        f"<tr>{''.join(f'<th>{escape(title)}</th>' for _key, title in _CASE_TABLE_COLUMNS)}</tr>"
        f"{case_table_rows}</table>",
        f"<h2>Artifacts</h2><ul>{artifact_list_html}</ul>",
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


__all__ = [
    "_add_physviz_plot",
    "_apply_flux_calibration",
    "_build_flux_km_judge",
    "_normalize_flux_gamma_grid",
    "_parse_flux_gamma_grid_arg",
    "_run_benchmark_case",
    "evaluate_trend_assertions",
    "write_benchmark_report",
]
