"""Unified AIB-ODE simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from deposim_schema import compose_sim_config

from .common.overrides import as_bool, normalize_overrides
from .domain import DomainGrid, build_domain_grid
from .input_builder import apply_roles, build_domain_from_fluent_xy, load_fluent_npz_v2, normalize_xy_mm
from .measurement_adapter import align_point_measurement_to_points
from .models.aib_ode import compute_diagnostics, step_theta_implicit
from .transport_provider import CfdFluxSinkKmProvider, FitScalarKmProvider, TransportProvider
from .validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for AIB pipeline execution")


@dataclass(frozen=True)
class SimRunResult:
    thickness: np.ndarray
    deposition_rate: np.ndarray
    R: np.ndarray
    Cs: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    fields: dict[str, np.ndarray]
    grid: DomainGrid


def _resolve_param(value: Any, shape: tuple[int, ...], default: float) -> np.ndarray:
    if value is None:
        raw = default
    elif isinstance(value, dict):
        if str(value.get("mode", "constant")) != "constant":
            raise ValueError("Only constant parameter mode is supported in v1")
        raw = float(value.get("value", default))
    else:
        raw = value

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        out = np.full(shape, float(arr), dtype=float)
    else:
        out = np.broadcast_to(arr, shape).astype(float, copy=True)
    return out


def _transport_dict(sim: Any) -> dict[str, Any]:
    return dict(getattr(sim.model.params, "transport", {}) or {})


def _build_transport_provider(
    *,
    sim: Any,
    c_a: np.ndarray,
    c_b: np.ndarray,
    flux_a: np.ndarray | None,
    flux_b: np.ndarray | None,
) -> tuple[str, TransportProvider]:
    transport = _transport_dict(sim)
    km_source = str(transport.get("km_source", "fit_scalar")).strip().lower()
    reference_shape = tuple(np.asarray(c_a, dtype=float).shape)
    time_dependent = str(sim.time_mode) == "transient"

    if km_source == "fit_scalar":
        return km_source, FitScalarKmProvider.from_transport_params(
            transport=transport,
            reference_shape=reference_shape,
            time_dependent=time_dependent,
        )
    if km_source == "from_cfd_flux_sink":
        if flux_a is None:
            raise ValueError("km_source=from_cfd_flux_sink requires flux_sink input for role A")
        provider = CfdFluxSinkKmProvider.from_arrays(
            cref_a=c_a,
            cref_b=c_b,
            flux_a=flux_a,
            flux_b=flux_b,
            transport=transport,
            time_dependent=time_dependent,
        )
        return km_source, provider
    raise ValueError(f"unsupported km_source: {km_source}")


def _as_xy_pair(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            items = [tok.strip() for tok in text[1:-1].split(",") if tok.strip()]
            if len(items) >= 2:
                return float(items[0]), float(items[1])
    return float(default[0]), float(default[1])


def _load_measurement(sim: Any, *, xy_mm: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    meas = sim.measurement
    if not bool(getattr(meas, "enabled", False)):
        return None, None
    path = Path(str(meas.file))
    if not path.exists():
        raise FileNotFoundError(f"measurement file not found: {path}")
    keys = dict(meas.keys)
    align = dict(getattr(meas, "align", {}) or {})
    align_enabled = as_bool(align.get("enable", False))
    with np.load(path, allow_pickle=False) as data:
        h_raw = np.asarray(data[keys.get("h", "h_nm")], dtype=float)
        xy_key = keys.get("xy", "xy")
        has_xy = xy_key in data.files
        xy_raw = np.asarray(data[xy_key], dtype=float) if has_xy else None

    h = np.asarray(h_raw, dtype=float).reshape(-1)
    n_pts = int(xy_mm.shape[0])

    if align_enabled:
        if xy_raw is None:
            raise ValueError("measurement.align.enable=true requires measurement xy key in measurement file")
        shift = _as_xy_pair(align.get("shift_mm", [0.0, 0.0]))
        rotate_deg = float(align.get("rotate_deg", 0.0))
        scale = float(align.get("scale", 1.0))
        mask_radius_mm = align.get("mask_radius_mm")
        mask_radius = None if mask_radius_mm is None else float(mask_radius_mm)
        aligned, valid = align_point_measurement_to_points(
            values=h,
            source_xy_mm=np.asarray(xy_raw, dtype=float),
            target_xy_mm=np.asarray(xy_mm, dtype=float),
            shift_mm=shift,
            rotation_deg=rotate_deg,
            scale=scale,
            mask_radius_mm=mask_radius,
        )
        return aligned, valid

    if h.ndim != 1 or h.shape[0] != n_pts:
        raise ValueError("measurement h must be shape [n_pts] when alignment is disabled")
    valid = np.isfinite(h)
    return h, valid


def _grid_xy_points(grid: DomainGrid) -> np.ndarray:
    if grid.x_grid_mm is not None and grid.y_grid_mm is not None:
        x = np.asarray(grid.x_grid_mm, dtype=float).reshape(-1)
        y = np.asarray(grid.y_grid_mm, dtype=float).reshape(-1)
        return np.stack([x, y], axis=1)
    if grid.kind == "wafer_2d_polar" and grid.theta_grid_rad is not None:
        x = np.asarray(grid.r_grid_mm * np.cos(grid.theta_grid_rad), dtype=float).reshape(-1)
        y = np.asarray(grid.r_grid_mm * np.sin(grid.theta_grid_rad), dtype=float).reshape(-1)
        return np.stack([x, y], axis=1)
    if grid.kind == "wafer_1d_radial":
        x = np.asarray(grid.r_grid_mm, dtype=float).reshape(-1)
        y = np.zeros_like(x, dtype=float)
        return np.stack([x, y], axis=1)
    raise ValueError(f"Unsupported grid kind for XY projection: {grid.kind}")


def _project_values_to_grid(
    values: np.ndarray,
    *,
    source_xy_mm: np.ndarray,
    target_xy_mm: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        aligned, _valid = align_point_measurement_to_points(
            values=arr,
            source_xy_mm=source_xy_mm,
            target_xy_mm=target_xy_mm,
        )
        return np.asarray(aligned, dtype=float).reshape(target_shape)
    if arr.ndim == 2:
        frames: list[np.ndarray] = []
        for idx in range(arr.shape[0]):
            aligned, _valid = align_point_measurement_to_points(
                values=arr[idx],
                source_xy_mm=source_xy_mm,
                target_xy_mm=target_xy_mm,
            )
            frames.append(np.asarray(aligned, dtype=float).reshape(target_shape))
        return np.stack(frames, axis=0)
    raise ValueError(f"Expected [n_pts] or [n_t,n_pts] values for projection, got shape {arr.shape}")


def _prepare_domain_inputs(
    *,
    sim: Any,
    fluent: Any,
) -> tuple[DomainGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    source_xy_mm = normalize_xy_mm(np.asarray(fluent.xy, dtype=float), sim.domain.xy_unit)
    c_a_raw, c_i_raw, c_b_raw = apply_roles(
        cref=fluent.cref,
        species=fluent.species,
        role_a=sim.roles.A,
        role_i=sim.roles.I,
        role_b=sim.roles.B,
    )

    flux_a_raw = None
    flux_b_raw = None
    if fluent.flux_sink is not None:
        flux_a_raw, _flux_i_raw, flux_b_raw = apply_roles(
            cref=fluent.flux_sink,
            species=fluent.species,
            role_a=sim.roles.A,
            role_i=sim.roles.I,
            role_b=sim.roles.B,
        )

    if str(sim.domain.kind) == "from_fluent_xy":
        grid = build_domain_from_fluent_xy(
            xy=fluent.xy,
            xy_unit=sim.domain.xy_unit,
            wafer_radius_mm=float(sim.domain.wafer_radius_mm),
        )
        return (
            grid,
            np.asarray(source_xy_mm, dtype=float),
            np.asarray(c_a_raw, dtype=float),
            np.asarray(c_i_raw, dtype=float),
            np.asarray(c_b_raw, dtype=float),
            None if flux_a_raw is None else np.asarray(flux_a_raw, dtype=float),
            None if flux_b_raw is None else np.asarray(flux_b_raw, dtype=float),
        )

    grid = build_domain_grid(sim.domain)
    target_xy_mm = _grid_xy_points(grid)
    target_shape = tuple(grid.shape)

    c_a = _project_values_to_grid(
        np.asarray(c_a_raw, dtype=float),
        source_xy_mm=source_xy_mm,
        target_xy_mm=target_xy_mm,
        target_shape=target_shape,
    )
    c_i = _project_values_to_grid(
        np.asarray(c_i_raw, dtype=float),
        source_xy_mm=source_xy_mm,
        target_xy_mm=target_xy_mm,
        target_shape=target_shape,
    )
    c_b = _project_values_to_grid(
        np.asarray(c_b_raw, dtype=float),
        source_xy_mm=source_xy_mm,
        target_xy_mm=target_xy_mm,
        target_shape=target_shape,
    )
    flux_a = (
        None
        if flux_a_raw is None
        else _project_values_to_grid(
            np.asarray(flux_a_raw, dtype=float),
            source_xy_mm=source_xy_mm,
            target_xy_mm=target_xy_mm,
            target_shape=target_shape,
        )
    )
    flux_b = (
        None
        if flux_b_raw is None
        else _project_values_to_grid(
            np.asarray(flux_b_raw, dtype=float),
            source_xy_mm=source_xy_mm,
            target_xy_mm=target_xy_mm,
            target_shape=target_shape,
        )
    )
    return grid, target_xy_mm, c_a, c_i, c_b, flux_a, flux_b


def _simulate_steady(
    *,
    sim: Any,
    c_a: np.ndarray,
    c_i: np.ndarray,
    c_b: np.ndarray,
    theta: np.ndarray,
    h: np.ndarray,
    has_b: bool,
    m_ads: int,
    p_a: int,
    p_star: int,
    km_a: np.ndarray,
    km_b: np.ndarray,
    k_ads: np.ndarray,
    k_des: np.ndarray,
    k_rxn: np.ndarray,
    K_I: np.ndarray,
    gamma_s: np.ndarray,
    nu_a: np.ndarray,
    alpha_h: np.ndarray,
    c_b_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, np.ndarray, np.ndarray]:
    dt = float(sim.time.dt_s)
    n_steps = max(int(np.ceil(float(sim.time.t_proc_s) / dt)), 1)
    non_bracketed_total = 0
    final_diag: dict[str, np.ndarray] = {}
    iter_total = np.zeros(theta.shape, dtype=int)
    fallback_count_map = np.zeros(theta.shape, dtype=int)

    for _ in range(n_steps):
        step = step_theta_implicit(
            theta_n=theta,
            h_n=h,
            dt_s=dt,
            cref_a=c_a,
            cref_i=c_i,
            cref_b=c_b,
            km_a=km_a,
            km_b=km_b,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            K_I=K_I,
            gamma_s=gamma_s,
            nu_a=nu_a,
            alpha_h=alpha_h,
            c_b_scale=c_b_scale,
            m_ads=m_ads,
            p_a=p_a,
            p_star=p_star,
            has_b=has_b,
            max_iter=int(sim.time.solver.max_iter),
            theta_tol=float(sim.time.solver.theta_tol),
        )
        theta = step.theta_next
        h = step.h_next
        non_bracketed_total += int(step.diagnostics["non_bracketed_count"])
        iter_total += np.asarray(step.diagnostics.get("iteration_count", np.zeros(theta.shape, dtype=int)), dtype=int)
        fallback_count_map += np.asarray(
            step.diagnostics.get("fallback_mask", np.zeros(theta.shape, dtype=bool)),
            dtype=bool,
        ).astype(int)
        final_diag = dict(step.diagnostics)

    return theta, h, final_diag, non_bracketed_total, iter_total, fallback_count_map


def _simulate_transient(
    *,
    sim: Any,
    c_a: np.ndarray,
    c_i: np.ndarray,
    c_b: np.ndarray,
    theta: np.ndarray,
    h: np.ndarray,
    has_b: bool,
    m_ads: int,
    p_a: int,
    p_star: int,
    km_provider: TransportProvider,
    k_ads: np.ndarray,
    k_des: np.ndarray,
    k_rxn: np.ndarray,
    K_I: np.ndarray,
    gamma_s: np.ndarray,
    nu_a: np.ndarray,
    alpha_h: np.ndarray,
    c_b_scale: np.ndarray,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, np.ndarray, np.ndarray]:
    dt_max = float(sim.time.dt_s)
    non_bracketed_total = 0
    final_diag: dict[str, np.ndarray] = {}
    iter_total = np.zeros(theta.shape, dtype=int)
    fallback_count_map = np.zeros(theta.shape, dtype=int)

    for i in range(time.shape[0] - 1):
        seg = float(time[i + 1] - time[i])
        if seg <= 0.0:
            raise ValueError("transient time array must be strictly increasing")
        n_sub = max(int(np.ceil(seg / dt_max)), 1)
        dt = seg / float(n_sub)

        c_a_i = c_a[i]
        c_i_i = c_i[i]
        c_b_i = c_b[i]
        km_a_i = np.asarray(km_provider.get_km("A", t_index=i), dtype=float)
        km_b_i = (
            np.asarray(km_provider.get_km("B", t_index=i), dtype=float)
            if has_b
            else np.zeros_like(km_a_i, dtype=float)
        )

        for _ in range(n_sub):
            step = step_theta_implicit(
                theta_n=theta,
                h_n=h,
                dt_s=dt,
                cref_a=c_a_i,
                cref_i=c_i_i,
                cref_b=c_b_i,
                km_a=km_a_i,
                km_b=km_b_i,
                k_ads=k_ads,
                k_des=k_des,
                k_rxn=k_rxn,
                K_I=K_I,
                gamma_s=gamma_s,
                nu_a=nu_a,
                alpha_h=alpha_h,
                c_b_scale=c_b_scale,
                m_ads=m_ads,
                p_a=p_a,
                p_star=p_star,
                has_b=has_b,
                max_iter=int(sim.time.solver.max_iter),
                theta_tol=float(sim.time.solver.theta_tol),
            )
            theta = step.theta_next
            h = step.h_next
            non_bracketed_total += int(step.diagnostics["non_bracketed_count"])
            iter_total += np.asarray(
                step.diagnostics.get("iteration_count", np.zeros(theta.shape, dtype=int)),
                dtype=int,
            )
            fallback_count_map += np.asarray(
                step.diagnostics.get("fallback_mask", np.zeros(theta.shape, dtype=bool)),
                dtype=bool,
            ).astype(int)
            final_diag = dict(step.diagnostics)

    return theta, h, final_diag, non_bracketed_total, iter_total, fallback_count_map


def compose_aib_spec(
    config_name: str = "cvd_steady_min",
    *,
    overrides: Sequence[str] | None = None,
) -> Any:
    """Compose a SimSpecV2-compatible object for AIB execution."""

    return compose_sim_config(
        config_name,
        overrides=normalize_overrides(overrides, prefix_sim=True),
    )


def run_aib_from_config(
    config_name: str = "cvd_steady_min",
    *,
    overrides: Sequence[str] | None = None,
) -> tuple[Any, SimRunResult]:
    """Compose config then execute AIB pipeline. Returns (spec, result)."""

    spec = compose_aib_spec(config_name, overrides=overrides)
    return spec, run_aib_from_spec(spec)


def run_aib_from_spec(run_spec: Any) -> SimRunResult:
    """Execute AIB simulation from a SimSpecV2-compatible object."""

    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)

    fluent = load_fluent_npz_v2(
        path=sim.inputs.fluent.file,
        mode=sim.inputs.fluent.mode,
        keys=sim.inputs.fluent.keys,
        species=sim.inputs.fluent.species,
    )
    grid, xy_mm, c_a, c_i, c_b, flux_a, flux_b = _prepare_domain_inputs(sim=sim, fluent=fluent)
    has_b = sim.roles.B is not None
    km_source, km_provider = _build_transport_provider(
        sim=sim,
        c_a=c_a,
        c_b=c_b,
        flux_a=flux_a,
        flux_b=flux_b,
    )

    shape = tuple(grid.shape)
    theta = np.full(shape, float(sim.initial_conditions.theta_A.value), dtype=float)
    h = np.full(shape, float(sim.initial_conditions.h_nm.value), dtype=float)

    params = sim.model.params
    orders = sim.model.orders

    gamma_s = _resolve_param(params.transport.get("Gamma_s"), shape, 1.0)
    nu_a = _resolve_param(params.transport.get("nu_A"), shape, 1.0)

    k_ads = _resolve_param(params.kinetics.get("k_ads"), shape, 1.0)
    k_des = _resolve_param(params.kinetics.get("k_des"), shape, 0.1)
    k_rxn = _resolve_param(params.kinetics.get("k_rxn"), shape, 0.01)
    K_I = _resolve_param(params.inhibitor.get("K_I"), shape, 0.0)
    alpha_h = _resolve_param(params.thickness.get("alpha_h"), shape, 1.0)
    c_b_scale = _resolve_param(params.scaling.get("C_B_scale"), shape, 1.0)

    if sim.time_mode == "steady":
        km_a = np.asarray(km_provider.get_km("A", t_index=0), dtype=float)
        km_b = np.asarray(km_provider.get_km("B", t_index=0), dtype=float) if has_b else np.zeros_like(km_a, dtype=float)
        theta, h, step_diag, non_bracketed_total, root_iteration_count, root_non_bracket_count_map = _simulate_steady(
            sim=sim,
            c_a=c_a,
            c_i=c_i,
            c_b=c_b,
            theta=theta,
            h=h,
            has_b=has_b,
            m_ads=int(orders.adsorption_site_order),
            p_a=int(orders.reaction_site_order_A),
            p_star=int(orders.reaction_site_order_star),
            km_a=km_a,
            km_b=km_b,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            K_I=K_I,
            gamma_s=gamma_s,
            nu_a=nu_a,
            alpha_h=alpha_h,
            c_b_scale=c_b_scale,
        )
        total_time = float(sim.time.t_proc_s)
    else:
        if fluent.time is None:
            raise ValueError("transient mode requires Fluent time array")
        theta, h, step_diag, non_bracketed_total, root_iteration_count, root_non_bracket_count_map = _simulate_transient(
            sim=sim,
            c_a=c_a,
            c_i=c_i,
            c_b=c_b,
            theta=theta,
            h=h,
            has_b=has_b,
            m_ads=int(orders.adsorption_site_order),
            p_a=int(orders.reaction_site_order_A),
            p_star=int(orders.reaction_site_order_star),
            km_provider=km_provider,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            K_I=K_I,
            gamma_s=gamma_s,
            nu_a=nu_a,
            alpha_h=alpha_h,
            c_b_scale=c_b_scale,
            time=fluent.time,
        )
        total_time = float(fluent.time[-1] - fluent.time[0])

    final_t_index = 0 if sim.time_mode == "steady" else int((fluent.time.shape[0] - 1) if fluent.time is not None else 0)
    km_a_diag = dict(km_provider.get_diagnostics("A", t_index=final_t_index))
    km_b_diag = (
        dict(km_provider.get_diagnostics("B", t_index=final_t_index))
        if has_b
        else {"km_used": np.full(shape, np.nan, dtype=float), "km_cfd": np.full(shape, np.nan, dtype=float)}
    )
    km_a_final = np.asarray(km_a_diag.get("km_used"), dtype=float)
    km_b_final = np.asarray(km_b_diag.get("km_used"), dtype=float)
    km_a_cfd = np.asarray(km_a_diag.get("km_cfd", km_a_final), dtype=float)
    km_b_cfd = np.asarray(km_b_diag.get("km_cfd", km_b_final), dtype=float)
    z_ref_mm = float(sim.reference_plane.z_ref_mm)
    tau_a = z_ref_mm / np.maximum(km_a_final, 1.0e-12)
    tau_b = z_ref_mm / np.maximum(km_b_final, 1.0e-12)

    theta_star = np.asarray(step_diag.get("theta_star", np.zeros_like(theta)), dtype=float)
    cs_a = np.asarray(step_diag.get("CsA", np.zeros_like(theta)), dtype=float)
    cs_b = np.asarray(step_diag.get("CsB", np.full(theta.shape, np.nan, dtype=float)), dtype=float)
    r_event = np.asarray(step_diag.get("r_event", np.zeros_like(theta)), dtype=float)

    if sim.time_mode == "steady":
        cref_a_final = c_a
        cref_i_final = c_i
        cref_b_final = c_b
    else:
        cref_a_final = c_a[-1]
        cref_i_final = c_i[-1]
        cref_b_final = c_b[-1]

    diag_fields = compute_diagnostics(
        theta_a=theta,
        theta_star=theta_star,
        cs_a=cs_a,
        cs_b=cs_b,
        cref_a=cref_a_final,
        cref_b=cref_b_final,
        cref_i=cref_i_final,
        gamma_s=gamma_s,
        k_rxn=k_rxn,
        km_b=km_b_final,
        c_b_scale=c_b_scale,
        p_a=int(orders.reaction_site_order_A),
        p_star=int(orders.reaction_site_order_star),
        K_I=K_I,
        has_b=has_b,
    )

    measurement, meas_valid = _load_measurement(sim, xy_mm=xy_mm)
    if measurement is not None:
        measurement = np.asarray(measurement, dtype=float).reshape(shape)
    if meas_valid is not None:
        meas_valid = np.asarray(meas_valid, dtype=bool).reshape(shape)
    if measurement is None:
        residual = np.full(theta.shape, np.nan, dtype=float)
    else:
        residual = h - measurement
        if meas_valid is not None:
            residual = np.asarray(residual, dtype=float)
            residual[~np.asarray(meas_valid, dtype=bool)] = np.nan

    dep_rate = h / max(total_time, 1.0e-12)

    fields = {
        "h_nm": h,
        "theta_A": theta,
        "theta_star": theta_star,
        "CsA_over_CrefA": diag_fields["CsA_over_CrefA"],
        "CsB_over_CrefB": diag_fields["CsB_over_CrefB"],
        "phi_B": diag_fields["phi_B"],
        "f_I": diag_fields["f_I"],
        "residual_nm": residual,
        "km_A": km_a_final,
        "km_B": km_b_final,
        "tau_A": tau_a,
        "tau_B": tau_b,
    }

    diagnostics = {
        "non_bracketed_total": non_bracketed_total,
        "dispatch_mode": sim.time_mode,
        "km_source": km_source,
        "z_ref_mm": z_ref_mm,
        "xy_mm": xy_mm,
        "species": list(fluent.species),
        "roles": {"A": sim.roles.A, "I": sim.roles.I, "B": sim.roles.B},
        "measurement_thickness": measurement,
        "measurement_valid_mask": meas_valid,
        "Cs_over_Cref": {
            "A": diag_fields["CsA_over_CrefA"],
            "B": diag_fields["CsB_over_CrefB"],
        },
        "phi_B": diag_fields["phi_B"],
        "f_I": diag_fields["f_I"],
        "Da_proxy": np.nan_to_num(diag_fields["phi_B"], nan=0.0),
        "R_event": r_event,
        "km_A_map": km_a_final,
        "km_B_map": km_b_final,
        "km_A_cfd_map": km_a_cfd,
        "km_B_cfd_map": km_b_cfd,
        "tau_A_map": tau_a,
        "tau_B_map": tau_b,
        "transport_units_hint": str(km_a_diag.get("units_hint", "") or km_b_diag.get("units_hint", "")),
        "root_iteration_count": np.asarray(root_iteration_count, dtype=float),
        "root_non_bracket_count_map": np.asarray(root_non_bracket_count_map, dtype=float),
        "root_status_map": np.asarray(np.asarray(root_non_bracket_count_map, dtype=int) > 0, dtype=int),
    }

    return SimRunResult(
        thickness=h,
        deposition_rate=dep_rate,
        R=r_event,
        Cs={"A": cs_a, "B": cs_b},
        diagnostics=diagnostics,
        fields=fields,
        grid=grid,
    )


def run_from_run_spec(run_spec: Any) -> SimRunResult:
    """Backward-compatible dispatch shim."""
    return run_aib_from_spec(run_spec)


__all__ = [
    "SimRunResult",
    "compose_aib_spec",
    "run_aib_from_config",
    "run_aib_from_spec",
    "run_from_run_spec",
]
