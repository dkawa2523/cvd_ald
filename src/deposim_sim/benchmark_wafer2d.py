"""AIB wafer-2D benchmark runner with class-based role coverage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deposim_report.plot_catalog import PlotSpec, benchmark_physviz_specs
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .benchmark_wafer2d_core import (
    _add_physviz_plot,
    _apply_flux_calibration,
    _build_flux_km_judge,
    _normalize_flux_gamma_grid,
    _parse_flux_gamma_grid_arg,
    _run_benchmark_case,
    evaluate_trend_assertions,
    write_benchmark_report,
)
from .common.csv_io import write_rows_csv
from .common.overrides import as_bool, normalize_overrides
from .common.run_artifacts import build_manifest_and_summary, create_run_layout, finalize_run_outputs, standard_artifact_rows
from .output_manifest import artifact_links

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
    "mean_abs_residual_nm_flux_km_calibrated",
    "mean_km_A_flux_km_calibrated",
    "delta_score_flux_calibrated_minus_free",
    "relative_delta_flux_calibrated_minus_free",
    "score",
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


def run_wafer2d_benchmark(
    config_name: str = "cvd_steady_min",
    overrides: Sequence[str] | None = None,
    *,
    with_physviz: bool = False,
    physviz_fast: bool = False,
    compare_flux_km: bool = False,
    calibrate_flux_km: bool = True,
    flux_gamma_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run AIB wafer benchmark and persist artifacts.

    When compare_flux_km is enabled, this runner always evaluates flux-km with
    gamma=1.0 and can optionally run a gamma mini-search for calibrated
    free-vs-flux comparison.
    """

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
    flux_gamma_scan: tuple[float, ...] = ()
    flux_gamma_scores: dict[float, list[float]] = {}
    flux_gamma_case_summaries: dict[str, dict[float, dict[str, float]]] = {}
    if compare_flux_km and calibrate_flux_km:
        flux_gamma_scan = _normalize_flux_gamma_grid(flux_gamma_grid)
        flux_gamma_scores = {float(g): [] for g in flux_gamma_scan}

    for case in cases:
        payload_path, xy_mm, _cref, _flux = write_case_input_npz(case=case, output_dir=inputs_dir)

        meas_h = _build_measurement_h(xy_mm)
        meas_path = inputs_dir / f"{case.case_id.lower()}_meas.npz"
        np.savez(meas_path, h_nm=meas_h, xy=xy_mm)

        result, summary_free, row_flux, case_gamma_rows = _run_benchmark_case(
            config_name=selected_config,
            base_overrides=base_overrides,
            normalized_overrides=normalized_overrides,
            payload_path=payload_path,
            meas_path=meas_path,
            case=case,
            compare_flux_km=bool(compare_flux_km),
            flux_gamma_scan=flux_gamma_scan,
        )

        complexity = int(case.role_i is not None) + int(case.role_b is not None)
        score = float(summary_free["mean_abs_residual_nm"]) + 0.05 * complexity
        if flux_gamma_scan:
            flux_gamma_case_summaries[case.case_id] = case_gamma_rows
            for gamma, gamma_row in case_gamma_rows.items():
                flux_gamma_scores[float(gamma)].append(float(gamma_row["mean_abs_residual_nm"]))

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
                "grid": result.grid,
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

    flux_gamma_scan_rows: list[dict[str, Any]] = []
    flux_gamma_best: float | None = None
    if flux_gamma_scores:
        finite_rows: list[tuple[float, float]] = []
        for gamma in flux_gamma_scan:
            values = np.asarray(flux_gamma_scores.get(float(gamma), []), dtype=float)
            mean_residual = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
            flux_gamma_scan_rows.append(
                {
                    "gamma_km": float(gamma),
                    "mean_abs_residual_nm": mean_residual,
                    "case_count": int(values.size),
                }
            )
            if np.isfinite(mean_residual):
                finite_rows.append((float(gamma), mean_residual))
        if finite_rows:
            flux_gamma_best = min(finite_rows, key=lambda row: row[1])[0]

    _apply_flux_calibration(
        case_rows=case_rows,
        flux_gamma_best=flux_gamma_best,
        flux_gamma_case_summaries=flux_gamma_case_summaries,
    )

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
    if flux_gamma_scan_rows:
        write_rows_csv(
            outputs_dir / "flux_gamma_scan.csv",
            [dict(row) for row in flux_gamma_scan_rows],
            fieldnames=["gamma_km", "mean_abs_residual_nm", "case_count"],
        )

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
                    "mean_km_A_flux_km_calibrated",
                    row.get(
                        "mean_km_A_flux_km",
                        row.get("mean_km_A", float("nan")),
                    ),
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
    flux_cal_delta = np.asarray(
        [float(row.get("delta_score_flux_calibrated_minus_free", float("nan"))) for row in case_rows],
        dtype=float,
    )
    flux_delta_calibrated_mean = (
        float(np.nanmean(flux_cal_delta)) if np.any(np.isfinite(flux_cal_delta)) else float("nan")
    )
    flux_cal_rel_delta = np.asarray(
        [float(row.get("relative_delta_flux_calibrated_minus_free", float("nan"))) for row in case_rows],
        dtype=float,
    )
    flux_relative_delta_calibrated_mean = (
        float(np.nanmean(flux_cal_rel_delta)) if np.any(np.isfinite(flux_cal_rel_delta)) else float("nan")
    )
    flux_eval_basis = "default"
    flux_eval_delta_mean = flux_delta_mean
    if np.isfinite(flux_delta_calibrated_mean):
        flux_eval_basis = "calibrated"
        flux_eval_delta_mean = flux_delta_calibrated_mean
    flux_km_judge = _build_flux_km_judge(
        compare_flux_km=bool(compare_flux_km),
        flux_delta_mean=float(flux_delta_mean),
        flux_delta_calibrated_mean=float(flux_delta_calibrated_mean),
        flux_eval_basis=flux_eval_basis,
        flux_gamma_best=flux_gamma_best,
        flux_gamma_grid=flux_gamma_scan,
    )
    p1_recommendation = bool(np.isfinite(km_spread_ratio) and km_spread_ratio >= 10.0)
    if np.isfinite(flux_eval_delta_mean):
        p1_recommendation = p1_recommendation or bool(flux_eval_delta_mean < 0.0)
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
                _add_physviz_plot(
                    plots_dir=plots_dir,
                    grid=representative["grid"],
                    xy_mm=representative["xy_mm"],
                    values=values,
                    spec=spec,
                    rel_paths=physviz_rel_paths,
                    records=plot_records,
                )
            for spec in (
                PlotSpec(
                    plot_id="physviz_input_cref_A",
                    filename="physviz_input_cref_A.png",
                    source_key="input_cref_A",
                    title="Input Cref(A) [a.u.]",
                    cmap="viridis",
                ),
                PlotSpec(
                    plot_id="physviz_input_flux_A",
                    filename="physviz_input_flux_A.png",
                    source_key="input_flux_A",
                    title="Input flux_sink(A) [a.u.]",
                    cmap="magma",
                ),
            ):
                _add_physviz_plot(
                    plots_dir=plots_dir,
                    grid=representative["grid"],
                    xy_mm=representative["xy_mm"],
                    values=representative[spec.source_key],
                    spec=spec,
                    rel_paths=physviz_rel_paths,
                    records=plot_records,
                )

    artifact_rows = standard_artifact_rows(
        include_report=True,
        extra_rows=[
            {"id": "benchmark_cases", "path": "outputs/benchmark_cases.npz", "kind": "npz", "required": True},
            {"id": "benchmark_case_metrics", "path": "outputs/benchmark_case_metrics.json", "kind": "json", "required": True},
            {"id": "class_compare", "path": "outputs/class_compare.csv", "kind": "csv", "required": True},
            {"id": "ranking", "path": "outputs/ranking.csv", "kind": "csv", "required": True},
        ],
    )
    if flux_gamma_scan_rows:
        artifact_rows.append({"id": "flux_gamma_scan", "path": "outputs/flux_gamma_scan.csv", "kind": "csv", "required": True})
    if with_physviz and representative is not None:
        artifact_rows.append({"id": "physviz_maps", "path": "outputs/physviz_maps.npz", "kind": "npz", "required": True})

    manifest, summary = build_manifest_and_summary(
        run_id=run_id,
        mode="benchmark_wafer2d",
        artifacts=artifact_rows,
        plots=plot_records,
        metadata={"sim_model": "aib_ode"},
        timestamp_utc=timestamp_utc,
        summary_fields={
        "sim_model": "aib_ode",
        "case_count": len(case_rows),
        "case_ids": [row["case_id"] for row in case_rows],
        "trend_assertions": trend_assertions,
        "km_spread_ratio": km_spread_ratio,
        "flux_delta_mean": flux_delta_mean,
        "flux_relative_delta_mean": flux_relative_delta_mean,
        "flux_delta_calibrated_mean": flux_delta_calibrated_mean,
        "flux_relative_delta_calibrated_mean": flux_relative_delta_calibrated_mean,
        "flux_eval_basis": flux_eval_basis,
        "flux_gamma_best": flux_gamma_best,
        "flux_gamma_grid": list(flux_gamma_scan) if flux_gamma_scan else [],
        "flux_km_judge": flux_km_judge,
        "p1_recommendation": p1_recommendation,
        },
    )
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
    parser.add_argument("--flux-calibrate", dest="flux_calibrate", action="store_true", default=True)
    parser.add_argument("--no-flux-calibrate", dest="flux_calibrate", action="store_false")
    parser.add_argument("--flux-gamma-grid", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    flux_gamma_grid = _parse_flux_gamma_grid_arg(args.flux_gamma_grid)

    result = run_wafer2d_benchmark(
        config_name=args.config_name,
        overrides=args.overrides,
        with_physviz=bool(args.with_physviz),
        physviz_fast=bool(args.physviz_fast),
        compare_flux_km=bool(args.compare_flux_km),
        calibrate_flux_km=bool(args.flux_calibrate),
        flux_gamma_grid=flux_gamma_grid,
    )
    run_dir = result["run_dir"]
    overall = bool(result["summary"]["trend_assertions"]["overall_passed"])
    print(f"[benchmark_wafer2d] wrote artifacts to: {run_dir}")
    print(f"[benchmark_wafer2d] overall_passed={overall}")
    return 0 if overall else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
