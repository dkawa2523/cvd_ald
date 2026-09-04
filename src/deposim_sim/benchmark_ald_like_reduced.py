"""ALD-like reduced transient benchmark for ALD role-model readiness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .common.csv_io import write_rows_csv
from .common.run_artifacts import (
    build_manifest_and_summary,
    build_provenance_metadata,
    create_run_layout,
    finalize_run_outputs,
    standard_artifact_rows,
)
from .input_builder import apply_roles
from .models.ald_role_state import run_ald_role_state_transient
from .models.aib_ode import step_theta_implicit
from .pipeline import run_sim_from_spec

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


SCENARIOS: tuple[str, ...] = ("low_dose", "nominal", "high_dose", "short_purge", "long_purge")
DOSE_ORDER: tuple[str, ...] = ("low_dose", "nominal", "high_dose")
PURGE_PHASE_CODES: tuple[int, ...] = (2, 4)


class _StaticKmProvider:
    def __init__(self, km_a: np.ndarray, km_b: np.ndarray) -> None:
        self._km_a = np.asarray(km_a, dtype=float)
        self._km_b = np.asarray(km_b, dtype=float)

    def get_km(self, role: str, *, t_index: int | None = None) -> np.ndarray:
        del t_index
        return self._km_b if str(role).upper() == "B" else self._km_a


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for ALD-like reduced benchmark execution.")


def _wiwnu_percent(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    if not np.isfinite(mean) or abs(mean) <= 1.0e-30:
        return float("nan")
    return float(100.0 * (np.nanmax(arr) - np.nanmin(arr)) / (2.0 * mean))


def _center_edge_delta(values: Any, xy_mm: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    xy = np.asarray(xy_mm, dtype=float)
    r = np.sqrt(np.sum(np.square(xy), axis=1))
    center_mask = r <= np.nanpercentile(r, 20.0)
    edge_mask = r >= np.nanpercentile(r, 80.0)
    return float(np.nanmean(arr[center_mask]) - np.nanmean(arr[edge_mask]))


def _resolve_param(value: Any, shape: tuple[int, ...], default: float) -> np.ndarray:
    if value is None:
        raw = default
    elif isinstance(value, dict):
        raw = float(value.get("value", default))
    else:
        raw = value
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    return np.broadcast_to(arr, shape).astype(float, copy=True)


def _scenario_paths(data_dir: Path, scenario: str) -> tuple[Path, Path]:
    fluent = data_dir / f"ald_like_{scenario}.npz"
    measurement = data_dir / f"ald_like_{scenario}_meas.npz"
    if not fluent.exists():
        raise FileNotFoundError(f"missing generated ALD-like input: {fluent}")
    if not measurement.exists():
        raise FileNotFoundError(f"missing generated ALD-like measurement: {measurement}")
    return fluent, measurement


def _cycles_from_phase_code(fluent_path: Path) -> int:
    with np.load(fluent_path, allow_pickle=False) as data:
        codes = np.asarray(data["phase_code"], dtype=int)
    return max(int(np.sum(codes == 1)), 1)


def _simulate_interval_snapshots(*, spec: Any, fluent_path: Path) -> dict[str, Any]:
    sim = getattr(spec, "sim", spec)
    with np.load(fluent_path, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        cref = np.asarray(data["cref"], dtype=float)
        xy = np.asarray(data["xy"], dtype=float)
        phase_code = np.asarray(data["phase_code"], dtype=int)

    c_a, c_i, c_b = apply_roles(
        cref=cref,
        species=list(sim.inputs.fluent.species),
        role_a=str(sim.roles.A),
        role_i=None if sim.roles.I is None else str(sim.roles.I),
        role_b=None if sim.roles.B is None else str(sim.roles.B),
    )
    if str(sim.model.name) == "role_ald_state":
        return _simulate_role_state_interval_snapshots(
            sim=sim,
            xy=xy,
            time=time,
            c_a=c_a,
            c_i=c_i,
            c_b=c_b,
            phase_code=phase_code,
        )
    n_pts = int(xy.shape[0])
    shape = (n_pts,)
    theta = np.full(shape, float(sim.initial_conditions.theta_A.value), dtype=float)
    h = np.full(shape, float(sim.initial_conditions.h_nm.value), dtype=float)

    params = sim.model.params
    orders = sim.model.orders
    transport = dict(params.transport)
    kinetics = dict(params.kinetics)
    inhibitor = dict(params.inhibitor)
    thickness = dict(params.thickness)
    scaling = dict(params.scaling)

    km_a = _resolve_param(transport.get("km_A"), shape, 0.02)
    km_b = _resolve_param(transport.get("km_B"), shape, 0.02)
    gamma_s = _resolve_param(transport.get("Gamma_s"), shape, 1.0)
    nu_a = _resolve_param(transport.get("nu_A"), shape, 1.0)
    k_ads = _resolve_param(kinetics.get("k_ads"), shape, 1.0)
    k_des = _resolve_param(kinetics.get("k_des"), shape, 0.1)
    k_rxn = _resolve_param(kinetics.get("k_rxn"), shape, 0.01)
    k_i = _resolve_param(inhibitor.get("K_I"), shape, 0.0)
    alpha_h = _resolve_param(thickness.get("alpha_h"), shape, 1.0)
    c_b_scale = _resolve_param(scaling.get("C_B_scale"), shape, 1.0)

    interval_h: list[np.ndarray] = []
    interval_theta: list[np.ndarray] = []
    interval_delta_mean: list[float] = []
    interval_codes: list[int] = []
    interval_end_time: list[float] = []
    non_bracketed_total = 0
    dt_max = float(sim.time.dt_s)

    for idx in range(time.shape[0] - 1):
        h_start = h.copy()
        seg = float(time[idx + 1] - time[idx])
        n_sub = max(int(np.ceil(seg / dt_max)), 1)
        dt_s = seg / float(n_sub)
        for _ in range(n_sub):
            step = step_theta_implicit(
                theta_n=theta,
                h_n=h,
                dt_s=dt_s,
                cref_a=np.asarray(c_a[idx], dtype=float),
                cref_i=np.asarray(c_i[idx], dtype=float),
                cref_b=np.asarray(c_b[idx], dtype=float),
                km_a=km_a,
                km_b=km_b,
                k_ads=k_ads,
                k_des=k_des,
                k_rxn=k_rxn,
                K_I=k_i,
                gamma_s=gamma_s,
                nu_a=nu_a,
                alpha_h=alpha_h,
                c_b_scale=c_b_scale,
                m_ads=int(orders.adsorption_site_order),
                p_a=int(orders.reaction_site_order_A),
                p_star=int(orders.reaction_site_order_star),
                has_b=sim.roles.B is not None,
                max_iter=int(sim.time.solver.max_iter),
                theta_tol=float(sim.time.solver.theta_tol),
            )
            theta = np.asarray(step.theta_next, dtype=float)
            h = np.asarray(step.h_next, dtype=float)
            non_bracketed_total += int(step.diagnostics.get("non_bracketed_count", 0))
        interval_h.append(h.copy())
        interval_theta.append(theta.copy())
        interval_delta_mean.append(float(np.nanmean(h - h_start)))
        interval_codes.append(int(phase_code[idx]))
        interval_end_time.append(float(time[idx + 1]))

    codes = np.asarray(interval_codes, dtype=int)
    h_stack = np.stack(interval_h, axis=0)
    theta_stack = np.stack(interval_theta, axis=0)
    cycle_indices = np.where(codes == 4)[0]
    cycle_h = h_stack[cycle_indices] if cycle_indices.size else h_stack[-1:][None, ...]
    cycle_theta = theta_stack[cycle_indices] if cycle_indices.size else theta_stack[-1:][None, ...]
    prev = np.zeros_like(cycle_h[:1])
    cycle_delta = np.diff(np.concatenate([prev, cycle_h], axis=0), axis=0)
    purge_growth = float(np.sum([delta for delta, code in zip(interval_delta_mean, interval_codes) if code in PURGE_PHASE_CODES]))
    final_mean = float(np.nanmean(h_stack[-1]))
    purge_fraction = float(purge_growth / max(final_mean, 1.0e-30))

    return {
        "xy_mm": xy,
        "interval_end_time_s": np.asarray(interval_end_time, dtype=float),
        "interval_phase_code": codes,
        "interval_h_nm": h_stack,
        "interval_theta_A": theta_stack,
        "interval_delta_mean_nm": np.asarray(interval_delta_mean, dtype=float),
        "cycle_end_h_nm": cycle_h,
        "cycle_end_theta_A": cycle_theta,
        "cycle_gpc_nm": cycle_delta,
        "cycle_gpc_mean_nm": np.asarray([float(np.nanmean(v)) for v in cycle_delta], dtype=float),
        "purge_growth_mean_nm": purge_growth,
        "purge_growth_fraction": purge_fraction,
        "snapshot_non_bracketed_total": int(non_bracketed_total),
    }


def _simulate_role_state_interval_snapshots(
    *,
    sim: Any,
    xy: np.ndarray,
    time: np.ndarray,
    c_a: np.ndarray,
    c_i: np.ndarray,
    c_b: np.ndarray,
    phase_code: np.ndarray,
) -> dict[str, Any]:
    n_pts = int(xy.shape[0])
    shape = (n_pts,)
    params = sim.model.params
    transport = dict(params.transport)
    kinetics = dict(params.kinetics)
    inhibitor = dict(params.inhibitor)
    thickness = dict(params.thickness)
    if str(transport.get("km_source", "fit_scalar")).strip().lower() != "fit_scalar":
        raise ValueError("ALD role-state benchmark snapshots currently require km_source=fit_scalar")

    km_provider = _StaticKmProvider(
        km_a=_resolve_param(transport.get("km_A"), shape, 0.02),
        km_b=_resolve_param(transport.get("km_B"), shape, 0.02),
    )
    k_store_a = _resolve_param(kinetics.get("k_store_A", kinetics.get("k_ads")), shape, 1.0)
    k_release_a = _resolve_param(kinetics.get("k_release_A", kinetics.get("k_des")), shape, 0.1)
    k_convert_a = _resolve_param(kinetics.get("k_convert_A", kinetics.get("k_rxn")), shape, 0.01)
    k_convert_ab = _resolve_param(kinetics.get("k_convert_AB", kinetics.get("k_rxn")), shape, 0.01)
    k_store_i = _resolve_param(inhibitor.get("k_store_I", inhibitor.get("K_I")), shape, 0.0)
    k_release_i = _resolve_param(inhibitor.get("k_release_I"), shape, 0.1)
    alpha_h = _resolve_param(thickness.get("alpha_h"), shape, 1.0)
    theta0 = np.full(shape, float(sim.initial_conditions.theta_A.value), dtype=float)
    h0 = np.full(shape, float(sim.initial_conditions.h_nm.value), dtype=float)

    interval_h: list[np.ndarray] = []
    interval_theta: list[np.ndarray] = []
    interval_delta_mean: list[float] = []
    interval_codes: list[int] = []
    interval_end_time: list[float] = []

    prev_h = h0.copy()
    for idx in range(time.shape[0] - 1):
        result = run_ald_role_state_transient(
            c_a=np.asarray(c_a[: idx + 2], dtype=float),
            c_i=np.asarray(c_i[: idx + 2], dtype=float),
            c_b=np.asarray(c_b[: idx + 2], dtype=float),
            km_provider=km_provider,
            time=np.asarray(time[: idx + 2], dtype=float),
            dt_max_s=float(sim.time.dt_s),
            theta_a0=theta0,
            h0=h0,
            k_store_a=k_store_a,
            k_release_a=k_release_a,
            k_convert_a=k_convert_a,
            k_convert_ab=k_convert_ab,
            k_store_i=k_store_i,
            k_release_i=k_release_i,
            alpha_h=alpha_h,
            has_b=sim.roles.B is not None,
            has_i=sim.roles.I is not None,
        )
        h_now = np.asarray(result.h_nm, dtype=float)
        interval_h.append(h_now.copy())
        interval_theta.append(np.asarray(result.theta_a, dtype=float).copy())
        interval_delta_mean.append(float(np.nanmean(h_now - prev_h)))
        interval_codes.append(int(phase_code[idx]))
        interval_end_time.append(float(time[idx + 1]))
        prev_h = h_now

    codes = np.asarray(interval_codes, dtype=int)
    h_stack = np.stack(interval_h, axis=0)
    theta_stack = np.stack(interval_theta, axis=0)
    cycle_indices = np.where(codes == 4)[0]
    cycle_h = h_stack[cycle_indices] if cycle_indices.size else h_stack[-1:][None, ...]
    cycle_theta = theta_stack[cycle_indices] if cycle_indices.size else theta_stack[-1:][None, ...]
    prev = np.zeros_like(cycle_h[:1])
    cycle_delta = np.diff(np.concatenate([prev, cycle_h], axis=0), axis=0)
    purge_growth = float(np.sum([delta for delta, code in zip(interval_delta_mean, interval_codes) if code in PURGE_PHASE_CODES]))
    final_mean = float(np.nanmean(h_stack[-1]))
    purge_fraction = float(purge_growth / max(final_mean, 1.0e-30))

    return {
        "xy_mm": xy,
        "interval_end_time_s": np.asarray(interval_end_time, dtype=float),
        "interval_phase_code": codes,
        "interval_h_nm": h_stack,
        "interval_theta_A": theta_stack,
        "interval_delta_mean_nm": np.asarray(interval_delta_mean, dtype=float),
        "cycle_end_h_nm": cycle_h,
        "cycle_end_theta_A": cycle_theta,
        "cycle_gpc_nm": cycle_delta,
        "cycle_gpc_mean_nm": np.asarray([float(np.nanmean(v)) for v in cycle_delta], dtype=float),
        "purge_growth_mean_nm": purge_growth,
        "purge_growth_fraction": purge_fraction,
        "snapshot_non_bracketed_total": 0,
    }


def _load_measurement_mean(measurement_path: Path) -> float:
    with np.load(measurement_path, allow_pickle=False) as data:
        return float(np.nanmean(np.asarray(data["h_nm"], dtype=float)))


def _evaluate_scenario(*, config_name: str, data_dir: Path, scenario: str) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    fluent_path, measurement_path = _scenario_paths(data_dir, scenario)
    cycles = _cycles_from_phase_code(fluent_path)
    spec = compose_sim_config(
        config_name,
        overrides=[
            f"sim.inputs.fluent.file={fluent_path.as_posix()}",
            "sim.measurement.enabled=true",
            f"sim.measurement.file={measurement_path.as_posix()}",
        ],
    )
    result = run_sim_from_spec(spec)
    snapshots = _simulate_interval_snapshots(spec=spec, fluent_path=fluent_path)

    h = np.asarray(result.fields["h_nm"], dtype=float)
    theta = np.asarray(result.fields["theta_A"], dtype=float)
    residual = np.asarray(result.fields["residual_nm"], dtype=float)
    xy = np.asarray(result.diagnostics["xy_mm"], dtype=float)

    row = {
        "scenario": scenario,
        "cycles": cycles,
        "h_mean_nm": float(np.nanmean(h)),
        "h_min_nm": float(np.nanmin(h)),
        "h_max_nm": float(np.nanmax(h)),
        "gpc_mean_nm_per_cycle": float(np.nanmean(h) / float(cycles)),
        "wiwnu_percent": _wiwnu_percent(h),
        "center_edge_delta_nm": _center_edge_delta(h, xy),
        "theta_mean": float(np.nanmean(theta)),
        "theta_min": float(np.nanmin(theta)),
        "theta_max": float(np.nanmax(theta)),
        "bounded_violation_count": int(result.diagnostics.get("bounded_violation_count", 0)),
        "residual_mae_nm": float(np.nanmean(np.abs(residual))),
        "measurement_mean_nm": _load_measurement_mean(measurement_path),
        "purge_growth_mean_nm": float(snapshots["purge_growth_mean_nm"]),
        "purge_growth_fraction": float(snapshots["purge_growth_fraction"]),
        "cycle_gpc_mean_nm": [float(v) for v in np.asarray(snapshots["cycle_gpc_mean_nm"], dtype=float)],
        "cycle_gpc_std_nm": float(np.nanstd(np.asarray(snapshots["cycle_gpc_mean_nm"], dtype=float))),
        "non_bracketed_total": int(result.diagnostics.get("non_bracketed_total", 0)),
    }
    return row, snapshots, h


def _build_assertions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["scenario"]): row for row in rows}
    dose_values = [float(by_name[name]["gpc_mean_nm_per_cycle"]) for name in DOSE_ORDER]
    dose_monotonic = dose_values[0] <= dose_values[1] <= dose_values[2]
    wiwnu_improves = float(by_name["high_dose"]["wiwnu_percent"]) <= float(by_name["low_dose"]["wiwnu_percent"])
    short_purge_more_growth = float(by_name["short_purge"]["h_mean_nm"]) >= float(by_name["long_purge"]["h_mean_nm"])
    first_gain = float(by_name["nominal"]["gpc_mean_nm_per_cycle"]) - float(by_name["low_dose"]["gpc_mean_nm_per_cycle"])
    high_gain = float(by_name["high_dose"]["gpc_mean_nm_per_cycle"]) - float(by_name["nominal"]["gpc_mean_nm_per_cycle"])
    plateau_gain_ratio = float(high_gain / max(first_gain, 1.0e-30))
    plateau_reached = plateau_gain_ratio <= 0.5
    solver_healthy = all(int(row["non_bracketed_total"]) == 0 for row in rows)
    bounded_healthy = all(int(row.get("bounded_violation_count", 0)) == 0 for row in rows)
    theta_bounded = all(0.0 <= float(row["theta_min"]) and float(row["theta_max"]) <= 1.0 for row in rows)
    return {
        "assert_dose_gpc_monotonic": bool(dose_monotonic),
        "assert_high_dose_wiwnu_not_worse_than_low_dose": bool(wiwnu_improves),
        "assert_short_purge_growth_not_less_than_long_purge": bool(short_purge_more_growth),
        "assert_solver_non_bracketed_zero": bool(solver_healthy),
        "assert_theta_bounded": bool(theta_bounded),
        "plateau_gain_ratio": plateau_gain_ratio,
        "assert_saturation_plateau_reached": bool(plateau_reached),
        "assert_bounded_violation_zero": bool(bounded_healthy),
        "operational_passed": bool(dose_monotonic and wiwnu_improves and short_purge_more_growth and solver_healthy and bounded_healthy and theta_bounded),
        "model_readiness_passed": bool(plateau_reached and dose_monotonic and wiwnu_improves and short_purge_more_growth and solver_healthy and bounded_healthy and theta_bounded),
        "overall_passed": bool(dose_monotonic and wiwnu_improves and short_purge_more_growth and solver_healthy and bounded_healthy and theta_bounded),
    }


def _write_plots(
    *,
    plots_dir: Path,
    rows: Sequence[dict[str, Any]],
    xy_mm: np.ndarray | None,
    final_h_by_scenario: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if plt is None:
        return []
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_name = {str(row["scenario"]): row for row in rows}
    records: list[dict[str, Any]] = []

    dose_x = ["low", "nominal", "high"]
    dose_y = [float(by_name[name]["gpc_mean_nm_per_cycle"]) for name in DOSE_ORDER]
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    ax.plot(dose_x, dose_y, marker="o")
    ax.set_xlabel("Dose scenario")
    ax.set_ylabel("GPC mean [nm/cycle]")
    ax.set_title("ALD-like dose response")
    out = plots_dir / "ald_like_gpc_vs_dose.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    records.append({"plot_id": "ald_like_gpc_vs_dose", "path": f"plots/{out.name}", "source_key": "gpc_mean_nm_per_cycle"})

    scenario_names = list(SCENARIOS)
    wiwnu = [float(by_name[name]["wiwnu_percent"]) for name in scenario_names]
    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    ax.bar(scenario_names, wiwnu)
    ax.set_ylabel("WIWNU [%]")
    ax.set_title("ALD-like spatial non-uniformity")
    ax.tick_params(axis="x", rotation=25)
    out = plots_dir / "ald_like_wiwnu_by_scenario.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    records.append({"plot_id": "ald_like_wiwnu_by_scenario", "path": f"plots/{out.name}", "source_key": "wiwnu_percent"})

    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    for name in scenario_names:
        gpc_by_cycle = [float(value) for value in by_name[name]["cycle_gpc_mean_nm"]]
        cycle_index = np.arange(1, len(gpc_by_cycle) + 1)
        ax.plot(cycle_index, gpc_by_cycle, marker="o", label=name)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cycle GPC mean [nm]")
    ax.set_title("ALD-like GPC by cycle")
    ax.legend(fontsize=7)
    out = plots_dir / "ald_like_cycle_gpc_by_scenario.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    records.append({"plot_id": "ald_like_cycle_gpc_by_scenario", "path": f"plots/{out.name}", "source_key": "cycle_gpc_mean_nm"})

    if xy_mm is not None and {"low_dose", "high_dose"}.issubset(final_h_by_scenario):
        xy = np.asarray(xy_mm, dtype=float)
        low = np.asarray(final_h_by_scenario["low_dose"], dtype=float)
        high = np.asarray(final_h_by_scenario["high_dose"], dtype=float)
        delta = high - low
        fields = (
            ("low dose h_nm", low),
            ("high dose h_nm", high),
            ("high - low h_nm", delta),
        )
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
        for ax, (title, values) in zip(axes, fields):
            artist = ax.tricontourf(xy[:, 0], xy[:, 1], values, levels=18)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            fig.colorbar(artist, ax=ax, shrink=0.82)
        out = plots_dir / "ald_like_low_high_thickness_compare.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        records.append(
            {
                "plot_id": "ald_like_low_high_thickness_compare",
                "path": f"plots/{out.name}",
                "source_key": "h_nm.low_high_delta",
            }
        )
    return records


def _write_calibration(
    *,
    outputs_dir: Path,
    final_h_by_scenario: dict[str, np.ndarray],
    data_dir: Path,
    alpha_h: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pred_all: list[np.ndarray] = []
    meas_all: list[np.ndarray] = []
    for scenario, pred in final_h_by_scenario.items():
        _fluent, measurement_path = _scenario_paths(data_dir, scenario)
        with np.load(measurement_path, allow_pickle=False) as data:
            meas = np.asarray(data["h_nm"], dtype=float)
        pred_arr = np.asarray(pred, dtype=float)
        pred_all.append(pred_arr.reshape(-1))
        meas_all.append(meas.reshape(-1))
        scale = float(np.sum(pred_arr * meas) / max(float(np.sum(pred_arr * pred_arr)), 1.0e-30))
        calibrated = pred_arr * scale
        rows.append(
            {
                "scenario": scenario,
                "alpha_h_current": float(alpha_h),
                "alpha_h_fit": float(alpha_h * scale),
                "scale_factor": scale,
                "mae_before_nm": float(np.nanmean(np.abs(pred_arr - meas))),
                "mae_after_nm": float(np.nanmean(np.abs(calibrated - meas))),
                "measurement_mean_nm": float(np.nanmean(meas)),
            }
        )
    pred_cat = np.concatenate(pred_all)
    meas_cat = np.concatenate(meas_all)
    global_scale = float(np.sum(pred_cat * meas_cat) / max(float(np.sum(pred_cat * pred_cat)), 1.0e-30))
    global_summary = {
        "alpha_h_current": float(alpha_h),
        "alpha_h_global_fit": float(alpha_h * global_scale),
        "global_scale_factor": global_scale,
        "global_mae_before_nm": float(np.nanmean(np.abs(pred_cat - meas_cat))),
        "global_mae_after_nm": float(np.nanmean(np.abs(pred_cat * global_scale - meas_cat))),
        "calibration_data_kind": "synthetic measurement targets unless replaced by real measurement files",
    }
    payload = {"global": global_summary, "by_scenario": rows}
    (outputs_dir / "ald_like_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_rows_csv(
        outputs_dir / "ald_like_calibration.csv",
        rows,
        fieldnames=[
            "scenario",
            "alpha_h_current",
            "alpha_h_fit",
            "scale_factor",
            "mae_before_nm",
            "mae_after_nm",
            "measurement_mean_nm",
        ],
    )
    return payload


def run_ald_like_reduced_benchmark(
    *,
    config_name: str = "ald_like_reduced",
    data_dir: Path = Path("runs/generated_inputs/ald_like_reduced"),
) -> dict[str, Any]:
    _require_numpy()
    base_spec = compose_sim_config(config_name)
    sim = getattr(base_spec, "sim", base_spec)
    layout = create_run_layout(
        root_dir=Path(str(sim.output.root_dir)),
        project=str(sim.output.project),
        run_name="benchmark_ald_like_reduced",
        with_inputs_dir=False,
    )
    run_dir = layout.run_dir
    outputs_dir = layout.outputs_dir
    plots_dir = layout.plots_dir

    config_overrides = [
        f"sim.inputs.fluent.file={data_dir / 'ald_like_nominal.npz'}",
        f"sim.measurement.file={data_dir / 'ald_like_nominal_meas.npz'}",
        "sim.measurement.enabled=true",
    ]
    compose_and_save_sim_config(
        run_dir / "config_resolved.yaml",
        config_name=config_name,
        overrides=config_overrides,
    )
    rows: list[dict[str, Any]] = []
    snapshots_by_scenario: dict[str, dict[str, Any]] = {}
    final_h_by_scenario: dict[str, np.ndarray] = {}
    xy_mm: np.ndarray | None = None
    for scenario in SCENARIOS:
        row, snapshots, h = _evaluate_scenario(config_name=config_name, data_dir=data_dir, scenario=scenario)
        rows.append(row)
        snapshots_by_scenario[scenario] = snapshots
        final_h_by_scenario[scenario] = np.asarray(h, dtype=float)
        if xy_mm is None:
            xy_mm = np.asarray(snapshots["xy_mm"], dtype=float)
    assertions = _build_assertions(rows)

    metrics_path = outputs_dir / "ald_like_case_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_rows_csv(
        outputs_dir / "ald_like_case_metrics.csv",
        [dict(row) for row in rows],
        fieldnames=[
            "scenario",
            "cycles",
            "h_mean_nm",
            "gpc_mean_nm_per_cycle",
            "wiwnu_percent",
            "center_edge_delta_nm",
            "theta_mean",
            "theta_min",
            "theta_max",
            "bounded_violation_count",
            "residual_mae_nm",
            "measurement_mean_nm",
            "purge_growth_mean_nm",
            "purge_growth_fraction",
            "cycle_gpc_std_nm",
            "non_bracketed_total",
        ],
    )
    np.savez(
        outputs_dir / "ald_like_cycle_snapshots.npz",
        xy_mm=xy_mm,
        **{
            f"{scenario}__cycle_end_h_nm": np.asarray(snapshots["cycle_end_h_nm"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__cycle_end_theta_A": np.asarray(snapshots["cycle_end_theta_A"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__cycle_gpc_nm": np.asarray(snapshots["cycle_gpc_nm"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__cycle_gpc_mean_nm": np.asarray(snapshots["cycle_gpc_mean_nm"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__interval_phase_code": np.asarray(snapshots["interval_phase_code"], dtype=int)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__interval_end_time_s": np.asarray(snapshots["interval_end_time_s"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
        **{
            f"{scenario}__interval_delta_mean_nm": np.asarray(snapshots["interval_delta_mean_nm"], dtype=float)
            for scenario, snapshots in snapshots_by_scenario.items()
        },
    )
    alpha_h = float(getattr(base_spec.model.params, "thickness", {}).get("alpha_h", 1.0))
    calibration = _write_calibration(
        outputs_dir=outputs_dir,
        final_h_by_scenario=final_h_by_scenario,
        data_dir=data_dir,
        alpha_h=alpha_h,
    )
    plot_records = _write_plots(
        plots_dir=plots_dir,
        rows=rows,
        xy_mm=xy_mm,
        final_h_by_scenario=final_h_by_scenario,
    )

    provenance = build_provenance_metadata(
        workflow_name="benchmark_ald_like_reduced",
        config_payload=base_spec,
        input_paths=[str(path) for path in sorted(data_dir.glob("ald_like_*.npz"))],
        extra_metadata={
            "benchmark_scope": "ALD-like reduced transient role-model readiness",
            "sim_model": str(sim.model.name),
        },
    )
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    artifact_rows = standard_artifact_rows(
        include_report=False,
        extra_rows=[
            {"id": "ald_like_case_metrics_json", "path": "outputs/ald_like_case_metrics.json", "kind": "json", "required": True},
            {"id": "ald_like_case_metrics_csv", "path": "outputs/ald_like_case_metrics.csv", "kind": "csv", "required": True},
            {"id": "ald_like_cycle_snapshots", "path": "outputs/ald_like_cycle_snapshots.npz", "kind": "npz", "required": True},
            {"id": "ald_like_calibration_json", "path": "outputs/ald_like_calibration.json", "kind": "json", "required": True},
            {"id": "ald_like_calibration_csv", "path": "outputs/ald_like_calibration.csv", "kind": "csv", "required": True},
        ],
    )
    manifest, summary = build_manifest_and_summary(
        run_id=layout.run_id,
        mode="benchmark_ald_like_reduced",
        artifacts=artifact_rows,
        plots=plot_records,
        metadata=provenance,
        timestamp_utc=timestamp_utc,
        summary_fields={
            "case_count": len(rows),
            "scenarios": list(SCENARIOS),
            "assertions": assertions,
            "nominal_gpc_nm_per_cycle": next(row["gpc_mean_nm_per_cycle"] for row in rows if row["scenario"] == "nominal"),
            "nominal_residual_mae_nm": next(row["residual_mae_nm"] for row in rows if row["scenario"] == "nominal"),
            "calibration": calibration["global"],
            **provenance,
        },
    )
    finalize_run_outputs(layout=layout, summary=summary, manifest=manifest)
    return {"run_dir": str(run_dir), "rows": rows, "assertions": assertions, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ALD-like reduced transient benchmark.")
    parser.add_argument("--config-name", default="ald_like_reduced")
    parser.add_argument("--data-dir", default="runs/generated_inputs/ald_like_reduced")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_ald_like_reduced_benchmark(config_name=args.config_name, data_dir=Path(args.data_dir))
    print(f"[benchmark_ald_like_reduced] wrote artifacts to: {result['run_dir']}")
    print(f"[benchmark_ald_like_reduced] operational_passed={result['assertions']['operational_passed']}")
    print(f"[benchmark_ald_like_reduced] model_readiness_passed={result['assertions']['model_readiness_passed']}")
    print(f"[benchmark_ald_like_reduced] plateau_gain_ratio={result['assertions']['plateau_gain_ratio']:.6g}")
    for row in result["rows"]:
        print(
            "[benchmark_ald_like_reduced] "
            f"{row['scenario']}: gpc={row['gpc_mean_nm_per_cycle']:.6g} nm/cycle, "
            f"wiwnu={row['wiwnu_percent']:.4g} %, "
            f"mae={row['residual_mae_nm']:.6g} nm"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
