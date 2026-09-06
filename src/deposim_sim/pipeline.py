"""Role-based CVD and ALD simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from deposim_schema import compose_sim_config

from .common.overrides import as_bool, normalize_overrides
from .domain import DomainGrid, build_domain_grid
from .input_builder import apply_roles, build_domain_from_fluent_xy, normalize_xy_mm
from .io_plugins import MeasurementData, load_fluent_from_run_spec, load_measurement_from_run_spec
from .measurement_adapter import align_point_measurement_to_points, point_alignment_distance_stats, compare_point_observations
from .models.ald_role_state import run_ald_role_state_transient
from .models.aib_ode import compute_diagnostics, step_theta_implicit
from .models.mvk_state import run_mvk_state
from .models.process_models import canonical_process_implementation, validate_process_model_choice
from .transport_provider import (
    CfdFluxSinkKmProvider,
    DirectSurfaceConcentrationProvider,
    FitScalarKmProvider,
    TransportProvider,
)
from .validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for role-based pipeline execution")


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
    if km_source == "direct_surface":
        return km_source, DirectSurfaceConcentrationProvider.from_reference_shape(
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


def _load_measurement(
    sim: Any,
    *,
    xy_mm: np.ndarray,
    prediction_nm: np.ndarray,
    duration_s: float,
    initial_nm: Any = 0.0,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
    dict[str, Any] | None,
    MeasurementData | None,
]:
    meas = sim.measurement
    if not bool(getattr(meas, "enabled", False)):
        return None, None, {"enabled": False}, None, None
    align = dict(getattr(meas, "align", {}) or {})
    align_enabled = as_bool(align.get("enable", False))
    align["enable"] = align_enabled
    align["shift_mm"] = _as_xy_pair(align.get("shift_mm", [0.0, 0.0]))
    loaded = load_measurement_from_run_spec(sim)
    h_raw = loaded.h
    xy_raw = normalize_xy_mm(loaded.xy, getattr(meas, "xy_unit", "mm"))

    h = np.asarray(h_raw, dtype=float).reshape(-1)
    n_pts = int(xy_mm.shape[0])
    quantity = str(getattr(meas, "quantity", "thickness"))
    observation = compare_point_observations(
        prediction_nm=prediction_nm, model_xy_mm=xy_mm,
        measured=h, measurement_xy_mm=xy_raw, align=align,
        quantity=quantity, duration_s=duration_s, initial_nm=initial_nm,
        sigma=loaded.sigma if loaded.sigma is not None else getattr(meas, "sigma", None),
    )
    if quantity == "mean_rate":
        h = h * duration_s

    if align_enabled:
        shift = _as_xy_pair(align.get("shift_mm", [0.0, 0.0]))
        rotate_deg = float(align.get("rotate_deg", 0.0))
        scale = float(align.get("scale", 1.0))
        mask_radius_mm = align.get("mask_radius_mm")
        mask_radius = None if mask_radius_mm is None else float(mask_radius_mm)
        max_distance_raw = align.get("max_nearest_distance_mm")
        max_distance = None if max_distance_raw is None else float(max_distance_raw)
        aligned, valid = align_point_measurement_to_points(
            values=h,
            source_xy_mm=np.asarray(xy_raw, dtype=float),
            target_xy_mm=np.asarray(xy_mm, dtype=float),
            shift_mm=shift,
            rotation_deg=rotate_deg,
            scale=scale,
            mask_radius_mm=mask_radius,
            max_nearest_distance_mm=max_distance,
        )
        alignment_diag = point_alignment_distance_stats(
            source_xy_mm=np.asarray(xy_raw, dtype=float),
            target_xy_mm=np.asarray(xy_mm, dtype=float),
            shift_mm=shift,
            rotation_deg=rotate_deg,
            scale=scale,
            max_nearest_distance_mm=max_distance,
        )
        alignment_diag.update({"enabled": True, "valid_count": int(np.sum(valid))})
        if quantity == "mean_rate":
            aligned += np.broadcast_to(np.asarray(initial_nm), prediction_nm.shape).ravel()
        return aligned, valid, alignment_diag, observation, loaded

    if h.ndim != 1 or h.shape[0] != n_pts:
        raise ValueError("measurement h must be shape [n_pts] when alignment is disabled")
    valid = np.isfinite(h)
    if quantity == "mean_rate":
        h += np.broadcast_to(np.asarray(initial_nm), prediction_nm.shape).ravel()
    return (
        h,
        valid,
        {"enabled": False, "valid_count": int(np.sum(valid)), "target_count": n_pts},
        observation,
        loaded,
    )


_MVK_HISTORY_OBSERVATIONS = (
    "h_nm_history",
    "oxidized_fraction_history",
    "reduction_rate_history_s-1",
    "regeneration_rate_history_s-1",
    "Cs_A_history_kmol_m3",
    "Cs_B_history_kmol_m3",
    "J_A_surface_history",
    "J_B_surface_history",
)


def _mvk_multi_observations(
    measurement_data: MeasurementData | None,
    point_observation: dict[str, Any] | None,
    *,
    measurement_xy_mm: np.ndarray,
    model_xy_mm: np.ndarray,
    align: dict[str, Any],
    time_s: np.ndarray,
    fields: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Adapt configured MvK histories to the unit-standardized objective."""

    if measurement_data is None:
        return {}
    extra = measurement_data.extra
    configured = [name for name in _MVK_HISTORY_OBSERVATIONS if name in extra]
    if not configured:
        return {}
    if "time" not in extra:
        raise ValueError("MvK history observations require measurement.keys.time")
    measured_time = np.asarray(extra["time"], dtype=float).reshape(-1)
    model_time = np.asarray(time_s, dtype=float).reshape(-1)
    if measured_time.shape != model_time.shape or not np.allclose(
        measured_time, model_time, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("MvK history observation times must match Fluent input times")
    if point_observation is None or point_observation.get("sigma_nm") is None:
        raise ValueError(
            "MvK multi-observation fitting requires film measurement uncertainty"
        )

    observations: dict[str, dict[str, Any]] = {
        "film": {
            "target": np.asarray(point_observation["target_nm"], dtype=float),
            "prediction": np.asarray(point_observation["prediction_nm"], dtype=float),
            "sigma": np.asarray(point_observation["sigma_nm"], dtype=float),
        }
    }
    for name in configured:
        sigma_name = f"{name}_sigma"
        if sigma_name not in extra:
            raise ValueError(
                f"MvK history observation {name!r} requires measurement.keys.{sigma_name}"
            )
        prediction = np.asarray(fields[name], dtype=float)
        target = np.asarray(extra[name], dtype=float)
        if target.ndim < 2 or target.shape[0] != model_time.size:
            raise ValueError(
                f"MvK history observation {name!r} must have shape [time, *space]"
            )
        target = target.reshape(model_time.size, -1)
        if target.shape[1] != measurement_xy_mm.shape[0]:
            raise ValueError(
                f"MvK history observation {name!r} has {target.shape[1]} spatial values; "
                f"expected {measurement_xy_mm.shape[0]} measurement points"
            )
        sigma = np.asarray(extra[sigma_name], dtype=float)
        try:
            sigma = np.broadcast_to(sigma, target.shape)
        except ValueError as exc:
            raise ValueError(
                f"MvK history uncertainty {sigma_name!r} cannot broadcast to "
                f"{target.shape}"
            ) from exc
        prediction = prediction.reshape(model_time.size, -1)

        # Film is already an observation at the final time.  When a complete
        # thickness history is supplied, omit its final row so that the same
        # measurement is not counted twice in the objective.
        stop = model_time.size - 1 if name == "h_nm_history" else model_time.size
        compared = [
            compare_point_observations(
                prediction_nm=prediction[index],
                model_xy_mm=model_xy_mm,
                measured=target[index],
                measurement_xy_mm=measurement_xy_mm,
                align=align,
                sigma=sigma[index],
            )
            for index in range(stop)
        ]
        if not compared:
            continue
        observations[name] = {
            "target": np.concatenate([item["target_nm"] for item in compared]),
            "prediction": np.concatenate([item["prediction_nm"] for item in compared]),
            "sigma": np.concatenate([item["sigma_nm"] for item in compared]),
        }
    return observations


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
    nu_b: np.ndarray,
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
            nu_b=nu_b,
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
    nu_b: np.ndarray,
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
                nu_b=nu_b,
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
    """Compose config then execute the current compatibility pipeline."""

    spec = compose_aib_spec(config_name, overrides=overrides)
    return spec, run_aib_from_spec(spec)


def run_aib_from_spec(run_spec: Any) -> SimRunResult:
    """Execute the AIB-compatible implementation from a SimSpecV2 object."""

    sim = getattr(run_spec, "sim", run_spec)
    implementation = canonical_process_implementation(str(sim.model.name))
    if implementation != "aib_ode":
        raise ValueError(f"run_aib_from_spec cannot execute process model implementation {implementation!r}")
    return _run_cvd_aib_from_spec(run_spec)


def run_sim_from_config(
    config_name: str = "cvd_steady_min",
    *,
    overrides: Sequence[str] | None = None,
) -> tuple[Any, SimRunResult]:
    """Compose config then execute through the process-model dispatcher."""

    spec = compose_aib_spec(config_name, overrides=overrides)
    return spec, run_sim_from_spec(spec)


def run_sim_from_spec(run_spec: Any) -> SimRunResult:
    """Execute a simulation through the small process-model dispatcher."""

    sim = getattr(run_spec, "sim", run_spec)
    info = validate_process_model_choice(
        name=str(sim.model.name),
        process=str(sim.process),
        time_mode=str(sim.time_mode),
    )
    if info.implementation == "aib_ode":
        return _run_cvd_aib_from_spec(run_spec)
    if info.implementation == "ald_role_state":
        return _run_ald_role_state_from_spec(run_spec)
    if info.implementation == "mvk_state":
        return _run_cvd_mvk_from_spec(run_spec)
    raise ValueError(f"process model implementation is registered but not executable: {info.implementation!r}")


def _param_with_fallback(params: dict[str, Any], name: str, fallback_name: str, shape: tuple[int, ...], default: float) -> np.ndarray:
    value = params.get(name, params.get(fallback_name, default))
    return _resolve_param(value, shape, default)


def _run_ald_role_state_from_spec(run_spec: Any) -> SimRunResult:
    """Execute the minimal ALD role-state assimilation model."""

    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)
    if str(sim.time_mode) != "transient":
        raise ValueError("role_ald_state requires sim.time_mode=transient")

    fluent = load_fluent_from_run_spec(sim)
    if fluent.time is None:
        raise ValueError("role_ald_state requires Fluent time array")

    grid, xy_mm, c_a, c_i, c_b, flux_a, flux_b = _prepare_domain_inputs(sim=sim, fluent=fluent)
    has_b = sim.roles.B is not None
    has_i = sim.roles.I is not None
    km_source, km_provider = _build_transport_provider(
        sim=sim,
        c_a=c_a,
        c_b=c_b,
        flux_a=flux_a,
        flux_b=flux_b,
    )

    shape = tuple(grid.shape)
    theta_a0 = np.full(shape, float(sim.initial_conditions.theta_A.value), dtype=float)
    h0 = np.full(shape, float(sim.initial_conditions.h_nm.value), dtype=float)

    params = sim.model.params
    kinetics = dict(getattr(params, "kinetics", {}) or {})
    inhibitor = dict(getattr(params, "inhibitor", {}) or {})
    thickness = dict(getattr(params, "thickness", {}) or {})

    k_store_a = _param_with_fallback(kinetics, "k_store_A", "k_ads", shape, 1.0)
    k_release_a = _param_with_fallback(kinetics, "k_release_A", "k_des", shape, 0.1)
    k_convert_a = _param_with_fallback(kinetics, "k_convert_A", "k_rxn", shape, 0.01)
    k_convert_ab = _param_with_fallback(kinetics, "k_convert_AB", "k_rxn", shape, 0.01)
    k_store_i = _param_with_fallback(inhibitor, "k_store_I", "K_I", shape, 0.0)
    k_release_i = _resolve_param(inhibitor.get("k_release_I"), shape, 0.1)
    alpha_h = _resolve_param(thickness.get("alpha_h"), shape, 1.0)
    gamma_s = _resolve_param(params.transport.get("Gamma_s"), shape, 1.0)
    nu_b = _resolve_param(params.transport.get("nu_B"), shape, 1.0)

    result = run_ald_role_state_transient(
        c_a=c_a,
        c_i=c_i,
        c_b=c_b,
        km_provider=km_provider,
        time=np.asarray(fluent.time, dtype=float),
        dt_max_s=float(sim.time.dt_s),
        theta_a0=theta_a0,
        h0=h0,
        k_store_a=k_store_a,
        k_release_a=k_release_a,
        k_convert_a=k_convert_a,
        k_convert_ab=k_convert_ab,
        k_store_i=k_store_i,
        k_release_i=k_release_i,
        alpha_h=alpha_h,
        gamma_s=gamma_s,
        nu_b=nu_b,
        has_b=has_b,
        has_i=has_i,
    )

    final_t_index = max(int(fluent.time.shape[0] - 2), 0)
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
    tau_a = (z_ref_mm * 1.0e-3) / np.maximum(km_a_final, 1.0e-12)
    tau_b = (z_ref_mm * 1.0e-3) / np.maximum(km_b_final, 1.0e-12)

    cref_a_final = np.asarray(c_a[final_t_index], dtype=float)
    cref_i_final = np.asarray(c_i[final_t_index], dtype=float)
    cref_b_final = np.asarray(c_b[final_t_index], dtype=float)
    cs_a_ratio = result.cs_a / np.where(cref_a_final > 1.0e-30, cref_a_final, np.nan)
    if has_b:
        cs_b_ratio = result.cs_b / np.where(cref_b_final > 1.0e-30, cref_b_final, np.nan)
        phi_b = (
            gamma_s
            * nu_b
            * k_convert_ab
            * np.clip(result.theta_a, 0.0, 1.0)
            / np.maximum(km_b_final, 1.0e-30)
        )
    else:
        cs_b_ratio = np.full(shape, np.nan, dtype=float)
        phi_b = np.full(shape, np.nan, dtype=float)
    f_i = np.clip(1.0 - result.theta_i, 0.0, 1.0) if has_i else np.ones(shape, dtype=float)
    with np.errstate(invalid="ignore"):
        j_a_transport = np.where(
            np.isfinite(km_a_final), km_a_final * (cref_a_final - result.cs_a), np.nan
        )
        j_b_transport = (
            np.where(
                np.isfinite(km_b_final),
                km_b_final * (cref_b_final - result.cs_b),
                np.nan,
            )
            if has_b
            else np.full(shape, np.nan, dtype=float)
        )

    total_time = float(fluent.time[-1] - fluent.time[0])
    measurement, meas_valid, measurement_alignment, observation, _measurement_data = _load_measurement(
        sim, xy_mm=xy_mm, prediction_nm=result.h_nm, duration_s=total_time, initial_nm=h0,
    )
    if measurement is not None:
        measurement = np.asarray(measurement, dtype=float).reshape(shape)
    if meas_valid is not None:
        meas_valid = np.asarray(meas_valid, dtype=bool).reshape(shape)
    if measurement is None:
        residual = np.full(shape, np.nan, dtype=float)
    else:
        residual = result.h_nm - measurement
        if meas_valid is not None:
            residual = np.asarray(residual, dtype=float)
            residual[~np.asarray(meas_valid, dtype=bool)] = np.nan

    dep_rate = (result.h_nm - h0) / max(total_time, 1.0e-12)
    not_applicable = np.full(shape, np.nan, dtype=float)

    fields = {
        "h_nm": result.h_nm,
        "theta_A": result.theta_a,
        "theta_I": result.theta_i,
        "theta_free": result.theta_free,
        "theta_star": result.theta_free,
        "CsA_over_CrefA": cs_a_ratio,
        "CsB_over_CrefB": cs_b_ratio,
        "phi_B": phi_b,
        "f_I": f_i,
        "J_A_surface": result.j_a_surface,
        "J_B_surface": result.j_b_surface,
        "J_A_transport": j_a_transport,
        "J_B_transport": j_b_transport,
        "residual_nm": residual,
        "km_A": km_a_final,
        "km_B": km_b_final,
        "tau_A_s": tau_a,
        "tau_B_s": tau_b,
    }

    diagnostics = {
        "non_bracketed_total": 0,
        "bounded_violation_count": int(result.diagnostics["bounded_violation_count"]),
        "state_projection_count": int(result.diagnostics["state_projection_count"]),
        "dispatch_mode": sim.time_mode,
        "process_model_implementation": "ald_role_state",
        "solver_kind": str(sim.time.solver.name),
        "root_metrics_applicable": False,
        "km_source": km_source,
        "concentration_location": str(
            km_a_diag.get("concentration_location", "reference_plane")
        ),
        "flux_semantics": str(
            km_a_diag.get("flux_semantics", "not_used")
        ),
        "z_ref_mm": z_ref_mm,
        "xy_mm": xy_mm,
        "species": list(fluent.species),
        "roles": {"A": sim.roles.A, "I": sim.roles.I, "B": sim.roles.B},
        "measurement_thickness": measurement,
        "measurement_valid_mask": meas_valid,
        "measurement_alignment": measurement_alignment,
        "observation": observation,
        "Cs_over_Cref": {"A": cs_a_ratio, "B": cs_b_ratio},
        "phi_B": phi_b,
        "f_I": f_i,
        "Da_proxy": np.nan_to_num(phi_b, nan=0.0),
        "R_event": result.r_event,
        "surface_flux": {"A": result.j_a_surface, "B": result.j_b_surface},
        "transport_flux": {"A": j_a_transport, "B": j_b_transport},
        "km_A_map": km_a_final,
        "km_B_map": km_b_final,
        "km_A_cfd_map": km_a_cfd,
        "km_B_cfd_map": km_b_cfd,
        "boundary_concentration_A_map": km_a_diag.get("boundary_concentration"),
        "boundary_concentration_B_map": km_b_diag.get("boundary_concentration"),
        "transport_driving_A_map": km_a_diag.get("driving_concentration"),
        "transport_driving_B_map": km_b_diag.get("driving_concentration"),
        "tau_A_s_map": tau_a,
        "tau_B_s_map": tau_b,
        "units": {
            "Gamma_s": "kmol/m^2",
            "surface_flux": "kmol/(m^2 s)",
            "km": "m/s",
            "transport_time": "s",
            "alpha_h": "nm per unit coverage converted",
        },
        "transport_units_hint": str(km_a_diag.get("units_hint", "") or km_b_diag.get("units_hint", "")),
        "root_iteration_count": not_applicable,
        "root_non_bracket_count_map": not_applicable,
        "root_status_map": np.full(shape, -1, dtype=int),
        "ald_role_state": dict(result.diagnostics),
    }

    return SimRunResult(
        thickness=result.h_nm,
        deposition_rate=dep_rate,
        R=result.r_event,
        Cs={"A": result.cs_a, "B": result.cs_b},
        diagnostics=diagnostics,
        fields=fields,
        grid=grid,
    )


def _run_cvd_mvk_from_spec(run_spec: Any) -> SimRunResult:
    """Execute the CVD Mars-van Krevelen redox-reservoir model."""

    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)
    fluent = load_fluent_from_run_spec(sim)
    grid, xy_mm, c_a, _c_i, c_b, flux_a, flux_b = _prepare_domain_inputs(
        sim=sim, fluent=fluent
    )
    km_source, km_provider = _build_transport_provider(
        sim=sim,
        c_a=c_a,
        c_b=c_b,
        flux_a=flux_a,
        flux_b=flux_b,
    )

    shape = tuple(grid.shape)
    params = sim.model.params
    kinetics = dict(getattr(params, "kinetics", {}) or {})
    transport = dict(getattr(params, "transport", {}) or {})
    thickness_params = dict(getattr(params, "thickness", {}) or {})
    k_reduce = _resolve_param(kinetics.get("k_reduce"), shape, 1.0)
    k_regenerate = _resolve_param(kinetics.get("k_regenerate"), shape, 1.0)
    gamma_s = _resolve_param(transport.get("Gamma_s"), shape, 1.0)
    nu_b = _resolve_param(transport.get("nu_B"), shape, 1.0)
    alpha_h = _resolve_param(thickness_params.get("alpha_h"), shape, 1.0)
    chi0 = np.full(
        shape,
        float(sim.initial_conditions.redox_fraction.value),
        dtype=float,
    )
    h0 = np.full(shape, float(sim.initial_conditions.h_nm.value), dtype=float)

    if str(sim.time_mode) == "transient":
        if fluent.time is None:
            raise ValueError("role_cvd_mvk transient execution requires Fluent time array")
        time_s = np.asarray(fluent.time, dtype=float)
        c_a_history = np.asarray(c_a, dtype=float)
        c_b_history = np.asarray(c_b, dtype=float)
        final_t_index = max(int(time_s.size - 2), 0)
    else:
        time_s = np.asarray([0.0, float(sim.time.t_proc_s)], dtype=float)
        c_a_history = np.stack([np.asarray(c_a, dtype=float)] * 2, axis=0)
        c_b_history = np.stack([np.asarray(c_b, dtype=float)] * 2, axis=0)
        final_t_index = 0

    result = run_mvk_state(
        c_a=c_a_history,
        c_b=c_b_history,
        km_provider=km_provider,
        time_s=time_s,
        dt_max_s=float(sim.time.dt_s),
        oxidized_fraction0=chi0,
        h0_nm=h0,
        k_reduce=k_reduce,
        k_regenerate=k_regenerate,
        gamma_s=gamma_s,
        nu_b=nu_b,
        alpha_h=alpha_h,
        max_iter=int(sim.time.solver.max_iter),
        state_tol=float(sim.time.solver.theta_tol),
    )

    cref_a_final = np.asarray(c_a_history[final_t_index], dtype=float)
    cref_b_final = np.asarray(c_b_history[final_t_index], dtype=float)
    km_a_diag = dict(km_provider.get_diagnostics("A", t_index=final_t_index))
    km_b_diag = dict(km_provider.get_diagnostics("B", t_index=final_t_index))
    km_a = np.asarray(km_a_diag["km_used"], dtype=float)
    km_b = np.asarray(km_b_diag["km_used"], dtype=float)
    with np.errstate(invalid="ignore"):
        j_a_transport = np.where(
            np.isfinite(km_a), km_a * (cref_a_final - result.cs_a), np.nan
        )
        j_b_transport = np.where(
            np.isfinite(km_b), km_b * (cref_b_final - result.cs_b), np.nan
        )
    cs_a_ratio = result.cs_a / np.where(cref_a_final > 1.0e-30, cref_a_final, np.nan)
    cs_b_ratio = result.cs_b / np.where(cref_b_final > 1.0e-30, cref_b_final, np.nan)
    total_time = float(time_s[-1] - time_s[0])

    measurement, meas_valid, measurement_alignment, observation, measurement_data = _load_measurement(
        sim,
        xy_mm=xy_mm,
        prediction_nm=result.h_nm,
        duration_s=total_time,
        initial_nm=h0,
    )
    if measurement is not None:
        measurement = np.asarray(measurement, dtype=float).reshape(shape)
    if meas_valid is not None:
        meas_valid = np.asarray(meas_valid, dtype=bool).reshape(shape)
    residual = (
        np.full(shape, np.nan, dtype=float)
        if measurement is None
        else np.asarray(result.h_nm - measurement, dtype=float)
    )
    if measurement is not None and meas_valid is not None:
        residual[~meas_valid] = np.nan

    dep_rate = (result.h_nm - h0) / max(total_time, 1.0e-12)
    redox_balance = np.asarray(result.diagnostics["redox_balance_rate"], dtype=float)
    fields = {
        "h_nm": result.h_nm,
        "time_s": result.time_s,
        "oxidized_fraction": result.oxidized_fraction,
        "reduced_fraction": 1.0 - result.oxidized_fraction,
        "reduction_rate_s-1": result.reduction_rate,
        "regeneration_rate_s-1": result.regeneration_rate,
        "redox_balance_rate_s-1": redox_balance,
        "redox_relaxation_time_s": np.asarray(
            result.diagnostics["relaxation_time_s"], dtype=float
        ),
        "CsA_over_CrefA": cs_a_ratio,
        "CsB_over_CrefB": cs_b_ratio,
        "J_A_surface": result.j_a_surface,
        "J_B_surface": result.j_b_surface,
        "J_A_transport": j_a_transport,
        "J_B_transport": j_b_transport,
        "h_nm_history": result.h_nm_history,
        "oxidized_fraction_history": result.oxidized_fraction_history,
        "reduction_rate_history_s-1": result.reduction_rate_history,
        "regeneration_rate_history_s-1": result.regeneration_rate_history,
        "Cs_A_history_kmol_m3": result.cs_a_history,
        "Cs_B_history_kmol_m3": result.cs_b_history,
        "J_A_surface_history": result.j_a_surface_history,
        "J_B_surface_history": result.j_b_surface_history,
        "km_A": km_a,
        "km_B": km_b,
        "residual_nm": residual,
    }
    units = {
        "concentration": "kmol/m^3",
        "k_reduce": "m^3/(kmol s)",
        "k_regenerate": "m^3/(kmol s)",
        "oxidized_fraction": "1",
        "reduction_rate": "1/s",
        "regeneration_rate": "1/s",
        "Gamma_s": "kmol/m^2",
        "surface_flux": "kmol/(m^2 s)",
        "alpha_h": "nm m^2/kmol",
        "thickness": "nm",
        "deposition_rate": "nm/s",
        "km": "m/s",
    }
    multi_observations = _mvk_multi_observations(
        measurement_data,
        observation,
        measurement_xy_mm=normalize_xy_mm(
            measurement_data.xy, getattr(sim.measurement, "xy_unit", "mm")
        ) if measurement_data is not None else xy_mm,
        model_xy_mm=xy_mm,
        align=dict(getattr(sim.measurement, "align", {}) or {}),
        time_s=result.time_s,
        fields=fields,
    )
    diagnostics = {
        "dispatch_mode": str(sim.time_mode),
        "process_model_implementation": "mvk_state",
        "mechanism": "Mars-van Krevelen surface redox reservoir",
        "pathways": ["A_reduction_growth", "B_regeneration"],
        "state_variable": "oxidized_fraction",
        "state_history_time_s": result.time_s,
        "history_sampling_convention": (
            "states at supplied times; endpoint rates, surface concentrations, and "
            "fluxes use the preceding piecewise-constant Fluent frame"
        ),
        "steady_observable_equivalence": "aib_qss:AB:no_desorption",
        "solver_kind": str(sim.time.solver.name),
        "root_metrics_applicable": True,
        "km_source": km_source,
        "concentration_location": str(
            km_a_diag.get("concentration_location", "reference_plane")
        ),
        "flux_semantics": str(km_a_diag.get("flux_semantics", "not_used")),
        "xy_mm": xy_mm,
        "species": list(fluent.species),
        "roles": {"A": sim.roles.A, "I": None, "B": sim.roles.B},
        "measurement_thickness": measurement,
        "measurement_valid_mask": meas_valid,
        "measurement_alignment": measurement_alignment,
        "observation": observation,
        "observations": multi_observations,
        "R_event": result.reduction_rate,
        "surface_flux": {"A": result.j_a_surface, "B": result.j_b_surface},
        "transport_flux": {"A": j_a_transport, "B": j_b_transport},
        "km_A_map": km_a,
        "km_B_map": km_b,
        "km_A_cfd_map": km_a_diag.get("km_cfd", km_a),
        "km_B_cfd_map": km_b_diag.get("km_cfd", km_b),
        "root_iteration_count": np.asarray(
            result.diagnostics["iteration_count"], dtype=float
        ),
        "root_non_bracket_count_map": np.asarray(
            result.diagnostics["fallback_count_map"], dtype=float
        ),
        "root_status_map": np.asarray(
            np.asarray(result.diagnostics["fallback_count_map"], dtype=int) > 0,
            dtype=int,
        ),
        "redox_state": dict(result.diagnostics),
        "units": units,
    }
    return SimRunResult(
        thickness=result.h_nm,
        deposition_rate=dep_rate,
        R=result.reduction_rate,
        Cs={"A": result.cs_a, "B": result.cs_b},
        diagnostics=diagnostics,
        fields=fields,
        grid=grid,
    )


def _run_cvd_aib_from_spec(run_spec: Any) -> SimRunResult:
    """Execute the continuous CVD A/I/B surface and transport balances."""

    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)

    fluent = load_fluent_from_run_spec(sim)
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
    nu_b = _resolve_param(params.transport.get("nu_B"), shape, 1.0)

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
            nu_b=nu_b,
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
            nu_b=nu_b,
            alpha_h=alpha_h,
            c_b_scale=c_b_scale,
            time=fluent.time,
        )
        total_time = float(fluent.time[-1] - fluent.time[0])

    final_t_index = (
        0
        if sim.time_mode == "steady"
        else max(int((fluent.time.shape[0] - 2) if fluent.time is not None else 0), 0)
    )
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
    tau_a = (z_ref_mm * 1.0e-3) / np.maximum(km_a_final, 1.0e-12)
    tau_b = (z_ref_mm * 1.0e-3) / np.maximum(km_b_final, 1.0e-12)

    theta_star = np.asarray(step_diag.get("theta_star", np.zeros_like(theta)), dtype=float)
    cs_a = np.asarray(step_diag.get("CsA", np.zeros_like(theta)), dtype=float)
    cs_b = np.asarray(step_diag.get("CsB", np.full(theta.shape, np.nan, dtype=float)), dtype=float)
    r_event = np.asarray(step_diag.get("r_event", np.zeros_like(theta)), dtype=float)

    if sim.time_mode == "steady":
        cref_a_final = c_a
        cref_i_final = c_i
        cref_b_final = c_b
    else:
        cref_a_final = c_a[final_t_index]
        cref_i_final = c_i[final_t_index]
        cref_b_final = c_b[final_t_index]

    diag_fields = compute_diagnostics(
        theta_a=theta,
        theta_star=theta_star,
        cs_a=cs_a,
        cs_b=cs_b,
        cref_a=cref_a_final,
        cref_b=cref_b_final,
        cref_i=cref_i_final,
        gamma_s=gamma_s,
        k_ads=k_ads,
        k_des=k_des,
        k_rxn=k_rxn,
        km_a=km_a_final,
        km_b=km_b_final,
        nu_b=nu_b,
        c_b_scale=c_b_scale,
        m_ads=int(orders.adsorption_site_order),
        p_a=int(orders.reaction_site_order_A),
        p_star=int(orders.reaction_site_order_star),
        K_I=K_I,
        has_b=has_b,
    )

    measurement, meas_valid, measurement_alignment, observation, _measurement_data = _load_measurement(
        sim, xy_mm=xy_mm, prediction_nm=h, duration_s=total_time,
        initial_nm=float(sim.initial_conditions.h_nm.value),
    )
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

    dep_rate = (h - float(sim.initial_conditions.h_nm.value)) / max(total_time, 1.0e-12)

    fields = {
        "h_nm": h,
        "theta_A": theta,
        "theta_star": theta_star,
        "CsA_over_CrefA": diag_fields["CsA_over_CrefA"],
        "CsB_over_CrefB": diag_fields["CsB_over_CrefB"],
        "phi_B": diag_fields["phi_B"],
        "f_I": diag_fields["f_I"],
        "J_A_surface": diag_fields["J_A_surface"],
        "J_B_surface": diag_fields["J_B_surface"],
        "J_A_transport": diag_fields["J_A_transport"],
        "J_B_transport": diag_fields["J_B_transport"],
        "residual_nm": residual,
        "km_A": km_a_final,
        "km_B": km_b_final,
        "tau_A_s": tau_a,
        "tau_B_s": tau_b,
    }

    diagnostics = {
        "non_bracketed_total": non_bracketed_total,
        "dispatch_mode": sim.time_mode,
        "solver_kind": str(sim.time.solver.name),
        "root_metrics_applicable": True,
        "km_source": km_source,
        "concentration_location": str(
            km_a_diag.get("concentration_location", "reference_plane")
        ),
        "flux_semantics": str(km_a_diag.get("flux_semantics", "not_used")),
        "z_ref_mm": z_ref_mm,
        "xy_mm": xy_mm,
        "species": list(fluent.species),
        "roles": {"A": sim.roles.A, "I": sim.roles.I, "B": sim.roles.B},
        "measurement_thickness": measurement,
        "measurement_valid_mask": meas_valid,
        "measurement_alignment": measurement_alignment,
        "observation": observation,
        "Cs_over_Cref": {
            "A": diag_fields["CsA_over_CrefA"],
            "B": diag_fields["CsB_over_CrefB"],
        },
        "phi_B": diag_fields["phi_B"],
        "f_I": diag_fields["f_I"],
        "Da_proxy": np.nan_to_num(diag_fields["phi_B"], nan=0.0),
        "R_event": r_event,
        "surface_flux": {
            "A": diag_fields["J_A_surface"],
            "B": diag_fields["J_B_surface"],
        },
        "transport_flux": {
            "A": diag_fields["J_A_transport"],
            "B": diag_fields["J_B_transport"],
        },
        "km_A_map": km_a_final,
        "km_B_map": km_b_final,
        "km_A_cfd_map": km_a_cfd,
        "km_B_cfd_map": km_b_cfd,
        "boundary_concentration_A_map": km_a_diag.get("boundary_concentration"),
        "boundary_concentration_B_map": km_b_diag.get("boundary_concentration"),
        "transport_driving_A_map": km_a_diag.get("driving_concentration"),
        "transport_driving_B_map": km_b_diag.get("driving_concentration"),
        "tau_A_s_map": tau_a,
        "tau_B_s_map": tau_b,
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
    return run_sim_from_spec(run_spec)


__all__ = [
    "SimRunResult",
    "compose_aib_spec",
    "run_aib_from_config",
    "run_aib_from_spec",
    "run_from_run_spec",
    "run_sim_from_config",
    "run_sim_from_spec",
]
