"""Physical-interpretability visualization helpers for benchmark workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .domain import DomainGrid, build_domain_grid
from .input_builder import build_field_bundle
from .physics.ald import run_ald_synthetic
from .physics.cvd_steady import FieldBundle, run_cvd_steady

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-12
_CREF_VALID_THRESHOLD = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for physviz utilities.")


def _as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    try:
        return np.broadcast_to(arr, shape).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to {shape}") from exc


def _as_species_map(value: Mapping[str, Any] | Any, species: Sequence[str], shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        return {
            name: _as_array(value[name], shape, f"species_map[{name}]")
            for name in species
            if name in value
        }
    return {name: _as_array(value, shape, f"species_map[{name}]") for name in species}


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    w = np.asarray(weights[valid], dtype=float)
    v = np.asarray(values[valid], dtype=float)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return float("nan")
    return float(np.sum(v * w) / wsum)


def _edge_weights(grid: DomainGrid) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(grid.edge_mask, dtype=bool)
    weights = np.asarray(grid.area_weights_mm2, dtype=float)
    return weights, mask


def _path_get(root: Any, path: str) -> Any:
    current = root
    for token in path.split("."):
        if hasattr(current, token):
            current = getattr(current, token)
        elif isinstance(current, Mapping):
            current = current[token]
        else:
            raise KeyError(path)
    return current


def _path_set(root: Any, path: str, value: Any) -> None:
    tokens = path.split(".")
    current = root
    for token in tokens[:-1]:
        if hasattr(current, token):
            current = getattr(current, token)
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(path)
    leaf = tokens[-1]
    if hasattr(current, leaf):
        setattr(current, leaf, value)
    elif isinstance(current, dict):
        current[leaf] = value
    else:
        raise KeyError(path)


def _hotspot_radius_mm(effect_map: np.ndarray, grid: DomainGrid, weights: np.ndarray, edge_mask: np.ndarray) -> float:
    score = np.asarray(np.abs(effect_map), dtype=float)
    valid = np.isfinite(score) & edge_mask
    if not np.any(valid):
        return float("nan")
    peak = float(np.max(score[valid]))
    if peak <= 0.0:
        return float(_weighted_mean(np.asarray(grid.r_grid_mm, dtype=float), weights, valid))
    hot = valid & (score >= 0.9 * peak)
    return float(_weighted_mean(np.asarray(grid.r_grid_mm, dtype=float), weights, hot))


def _phase_durations(run_spec: Any) -> tuple[list[str], np.ndarray]:
    phases = list(getattr(run_spec.time, "phases", []) or [])
    if not phases:
        return ["default"], np.asarray([float(run_spec.time.process_time_s)], dtype=float)
    names = [str(phase.get("name", f"phase_{idx+1:02d}")) for idx, phase in enumerate(phases)]
    durations = np.asarray([float(phase.get("duration_s", 0.0)) for phase in phases], dtype=float)
    return names, durations


def _build_input_snapshots(
    fields: Any,
    grid: DomainGrid,
    fractions: np.ndarray,
    *,
    enable_time_variation: bool,
    variation_amplitude: float,
) -> dict[str, np.ndarray]:
    species = [str(name) for name in fields.C_ref]
    base = {name: _as_array(fields.C_ref[name], grid.shape, f"C_ref[{name}]") for name in species}
    out: dict[str, np.ndarray] = {}
    radius = max(float(grid.wafer_radius_mm), _EPS)
    r_norm = np.asarray(grid.r_grid_mm, dtype=float) / radius
    theta = np.asarray(grid.theta_grid_rad, dtype=float) if grid.theta_grid_rad is not None else None
    for name in species:
        snapshots: list[np.ndarray] = []
        base_map = base[name]
        for frac in fractions:
            if enable_time_variation:
                if theta is not None:
                    wave = np.cos(3.0 * theta - 2.0 * np.pi * float(frac)) * (0.35 + 0.65 * r_norm)
                    wave = wave + 0.25 * (1.0 - r_norm) * (2.0 * float(frac) - 1.0)
                else:
                    wave = (1.0 - r_norm) * (2.0 * float(frac) - 1.0)
                mod = np.clip(1.0 + float(variation_amplitude) * wave, 0.05, np.inf)
                snapshots.append(np.clip(base_map * mod, _EPS, np.inf))
            else:
                snapshots.append(base_map.copy())
        out[name] = np.stack(snapshots, axis=0)
    return out


def build_cvd_pseudo_time_snapshots(
    run_spec: Any,
    time_points: Sequence[float],
    *,
    enable_input_time_variation: bool = False,
    input_variation_amplitude: float = 0.2,
) -> dict[str, Any]:
    """Build pseudo-time thickness maps by re-running steady solve at multiple process times."""
    _require_numpy()
    if not time_points:
        raise ValueError("time_points must be non-empty")
    fractions = np.asarray(sorted({float(t) for t in time_points}), dtype=float)
    if np.any(fractions <= 0.0):
        raise ValueError("time_points must be > 0")

    grid = build_domain_grid(run_spec.domain)
    fields = build_field_bundle(run_spec, grid)
    t_final = float(run_spec.time.process_time_s)
    times = fractions * t_final

    input_snapshots = _build_input_snapshots(
        fields,
        grid,
        fractions,
        enable_time_variation=bool(enable_input_time_variation),
        variation_amplitude=float(input_variation_amplitude),
    )

    snapshots: list[np.ndarray] = []
    cumulative = np.zeros(grid.shape, dtype=float)
    t_prev = 0.0
    for idx, t in enumerate(times):
        c_ref_t = {name: np.asarray(series[idx], dtype=float) for name, series in input_snapshots.items()}
        fields_t = FieldBundle(C_ref=c_ref_t, U=fields.U, T=fields.T, scalars=fields.scalars)
        out = run_cvd_steady(
            grid=grid,
            fields=fields_t,
            model_config=run_spec.model,
            process_time_s=1.0,
            solver_config=run_spec.solver,
        )
        dt = max(float(t) - float(t_prev), 0.0)
        cumulative = cumulative + np.asarray(out.deposition_rate, dtype=float) * dt
        snapshots.append(cumulative.copy())
        t_prev = float(t)
    thickness = np.stack(snapshots, axis=0)

    scale = fractions.reshape((-1,) + (1,) * len(grid.shape))
    linear_pred = thickness[-1][None, ...] * scale
    residual = thickness - linear_pred
    delta = np.diff(thickness, axis=0)
    return {
        "time_fractions": fractions,
        "time_seconds": times,
        "thickness_snapshots": thickness,
        "delta_thickness_snapshots": delta,
        "linearity_residual_snapshots": residual,
        "linearity_residual_max": np.max(np.abs(residual), axis=0),
        "input_snapshots": input_snapshots,
        "input_time_variation_enabled": bool(enable_input_time_variation),
        "input_variation_amplitude": float(input_variation_amplitude),
    }


def build_ald_phase_snapshots(run_spec: Any) -> dict[str, Any]:
    """Build ALD phase snapshots (thickness increment, coverage, cumulative thickness)."""
    _require_numpy()
    grid = build_domain_grid(run_spec.domain)
    fields = build_field_bundle(run_spec, grid)
    result = run_ald_synthetic(run_spec, grid=grid, fields=fields)

    coverage_history = np.asarray(result.diagnostics.get("coverage_history"), dtype=float)
    if coverage_history.ndim < 2:
        raise ValueError("ALD diagnostics.coverage_history must be at least 2D (phase, spatial...).")
    phase_names, durations = _phase_durations(run_spec)
    if coverage_history.shape[0] != len(durations):
        raise ValueError("coverage_history phase count does not match run_spec.time.phases")
    growth_nm_s = float((getattr(run_spec.model, "state_params", {}) or {}).get("growth_rate_nm_s", 0.8))
    duration_scale = durations.reshape((-1,) + (1,) * (coverage_history.ndim - 1))
    phase_thickness = growth_nm_s * coverage_history * duration_scale
    cumulative = np.cumsum(phase_thickness, axis=0)
    return {
        "phase_names": phase_names,
        "phase_durations_s": durations,
        "phase_thickness_snapshots": phase_thickness,
        "phase_coverage_snapshots": coverage_history,
        "cumulative_thickness_snapshots": cumulative,
        "final_thickness": np.asarray(result.thickness, dtype=float),
    }


def compute_transport_term_maps(
    result: Any,
    fields: Any,
    km: Mapping[str, Any] | Any,
    nu: Mapping[str, Any] | Any,
) -> dict[str, np.ndarray]:
    """Compute transport-side term maps for physical interpretation."""
    _require_numpy()
    species = [str(name) for name in fields.C_ref]
    shape = np.asarray(result.thickness, dtype=float).shape
    c_ref = {name: _as_array(fields.C_ref[name], shape, f"C_ref[{name}]") for name in species}
    cs = {name: _as_array(result.Cs[name], shape, f"Cs[{name}]") for name in species}
    km_map = _as_species_map(km, species, shape)
    nu_map = _as_species_map(nu, species, shape)
    R = _as_array(result.R, shape, "R")

    out: dict[str, np.ndarray] = {}
    util_terms: list[np.ndarray] = []
    for name in species:
        cref = c_ref[name]
        cs_i = cs[name]
        km_i = np.clip(km_map[name], 0.0, np.inf)
        nu_i = np.clip(nu_map[name], 0.0, np.inf)
        transport_capacity = km_i * cref
        reaction_demand = nu_i * R
        depletion_ratio = np.where(cref > _CREF_VALID_THRESHOLD, 1.0 - (cs_i / (cref + _EPS)), 0.0)
        depletion_ratio = np.clip(depletion_ratio, 0.0, np.inf)
        utilization = np.divide(
            reaction_demand,
            transport_capacity + _EPS,
            out=np.zeros(shape, dtype=float),
            where=cref > _CREF_VALID_THRESHOLD,
        )
        utilization = np.clip(utilization, 0.0, np.inf)
        out[f"transport_capacity__{name}"] = transport_capacity
        out[f"reaction_demand__{name}"] = reaction_demand
        out[f"depletion_ratio__{name}"] = depletion_ratio
        out[f"utilization__{name}"] = utilization
        util_terms.append(utilization)
    if util_terms:
        out["utilization_max"] = np.maximum.reduce(util_terms)
    return out


def compute_net_term_maps(result: Any) -> dict[str, np.ndarray]:
    """Compute net-equation term maps and contribution ratios."""
    _require_numpy()
    net_rate = np.asarray(result.deposition_rate, dtype=float)
    shape = net_rate.shape
    comps = result.diagnostics.get("net_rate_components", {})
    if not isinstance(comps, Mapping):
        comps = {}
    dep = _as_array(comps.get("dep_rate", result.diagnostics.get("gross_deposition_rate", net_rate)), shape, "dep_rate")
    etch = np.clip(_as_array(comps.get("etch_rate", 0.0), shape, "etch_rate"), 0.0, np.inf)
    loss = np.clip(_as_array(comps.get("loss_rate", 0.0), shape, "loss_rate"), 0.0, np.inf)
    denom = np.maximum(dep, _EPS)
    etch_frac = np.divide(etch, denom, out=np.zeros(shape, dtype=float), where=denom > _EPS)
    loss_frac = np.divide(loss, denom, out=np.zeros(shape, dtype=float), where=denom > _EPS)
    return {
        "dep_rate": dep,
        "etch_rate": etch,
        "loss_rate": loss,
        "net_rate": net_rate,
        "etch_fraction_of_dep": np.clip(etch_frac, 0.0, np.inf),
        "loss_fraction_of_dep": np.clip(loss_frac, 0.0, np.inf),
    }


def _reaction_term_specs(run_spec: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str, Any]]]:
    params = getattr(run_spec.model, "kinetics_params", {}) or {}
    if not isinstance(params, Mapping):
        return [], []
    sensitivity: list[tuple[str, str]] = []
    ablation: list[tuple[str, str, Any]] = []

    if isinstance(params.get("k0"), (int, float)):
        sensitivity.append(("k0", "model.kinetics_params.k0"))

    if isinstance(params.get("ea_j_mol"), (int, float)):
        sensitivity.append(("ea_j_mol", "model.kinetics_params.ea_j_mol"))
        ablation.append(("ea_j_mol", "model.kinetics_params.ea_j_mol", 0.0))
    elif isinstance(params.get("ea"), (int, float)):
        sensitivity.append(("ea", "model.kinetics_params.ea"))
        ablation.append(("ea", "model.kinetics_params.ea", 0.0))

    if isinstance(params.get("order"), (int, float)):
        sensitivity.append(("order", "model.kinetics_params.order"))
        ablation.append(("order", "model.kinetics_params.order", 0.0))

    orders = params.get("orders")
    if isinstance(orders, Mapping):
        for species in sorted(str(k) for k in orders):
            path = f"model.kinetics_params.orders.{species}"
            sensitivity.append((f"orders.{species}", path))
            ablation.append((f"orders.{species}", path, 0.0))

    numerator_orders = params.get("numerator_orders")
    if isinstance(numerator_orders, Mapping):
        for species in sorted(str(k) for k in numerator_orders):
            path = f"model.kinetics_params.numerator_orders.{species}"
            sensitivity.append((f"numerator_orders.{species}", path))
            ablation.append((f"numerator_orders.{species}", path, 0.0))

    kinetics = str(getattr(run_spec.model, "kinetics_name", "")).strip().lower()
    den_coeffs = params.get("denominator_coeffs")
    if kinetics in {"saturation_inhibition", "lhhw_competition", "competition_lhhw"} and isinstance(den_coeffs, Mapping):
        for species in sorted(str(k) for k in den_coeffs):
            path = f"model.kinetics_params.denominator_coeffs.{species}"
            sensitivity.append((f"denominator_coeffs.{species}", path))
            ablation.append((f"denominator_coeffs.{species}", path, 0.0))
    den_orders = params.get("denominator_orders")
    if kinetics in {"saturation_inhibition", "lhhw_competition", "competition_lhhw"} and isinstance(den_orders, Mapping):
        for species in sorted(str(k) for k in den_orders):
            path = f"model.kinetics_params.denominator_orders.{species}"
            sensitivity.append((f"denominator_orders.{species}", path))
            ablation.append((f"denominator_orders.{species}", path, 0.0))
    if kinetics in {"saturation_inhibition", "lhhw_competition", "competition_lhhw"} and isinstance(
        params.get("denominator_power"), (int, float)
    ):
        sensitivity.append(("denominator_power", "model.kinetics_params.denominator_power"))
        ablation.append(("denominator_power", "model.kinetics_params.denominator_power", 0.0))
    if kinetics in {"saturation_inhibition", "lhhw_competition", "competition_lhhw"} and isinstance(
        params.get("denominator_base"), (int, float)
    ):
        sensitivity.append(("denominator_base", "model.kinetics_params.denominator_base"))
        ablation.append(("denominator_base", "model.kinetics_params.denominator_base", 1.0))

    if isinstance(params.get("pattern_loading"), (int, float)):
        sensitivity.append(("pattern_loading", "model.kinetics_params.pattern_loading"))
        ablation.append(("pattern_loading", "model.kinetics_params.pattern_loading", 1.0))

    dedup_sens: list[tuple[str, str]] = []
    seen_sens: set[str] = set()
    for term_name, path in sensitivity:
        if term_name in seen_sens:
            continue
        seen_sens.add(term_name)
        dedup_sens.append((term_name, path))

    dedup_abla: list[tuple[str, str, Any]] = []
    seen_abla: set[str] = set()
    for term_name, path, off_value in ablation:
        if term_name in seen_abla:
            continue
        seen_abla.add(term_name)
        dedup_abla.append((term_name, path, off_value))

    return dedup_sens, dedup_abla


def compute_reaction_term_importance(
    run_spec: Any,
    mode: str = "sensitivity+ablation",
    *,
    relative_step: float = 1.0e-2,
) -> dict[str, Any]:
    """Compute reaction-term importance using local sensitivity and ablation deltas."""
    _require_numpy()
    mode_key = str(mode).strip().lower()
    do_sens = "sensitivity" in mode_key
    do_ablation = "ablation" in mode_key
    if not (do_sens or do_ablation):
        raise ValueError("mode must include 'sensitivity' and/or 'ablation'")

    grid = build_domain_grid(run_spec.domain)
    fields = build_field_bundle(run_spec, grid)
    base_result = run_cvd_steady(
        grid=grid,
        fields=fields,
        model_config=run_spec.model,
        process_time_s=float(run_spec.time.process_time_s),
        solver_config=run_spec.solver,
    )
    base_thickness = np.asarray(base_result.thickness, dtype=float)
    weights, edge_mask = _edge_weights(grid)
    base_scale = max(abs(_weighted_mean(base_thickness, weights, edge_mask)), _EPS)

    sens_specs, abla_specs = _reaction_term_specs(run_spec)
    sensitivity_maps: dict[str, np.ndarray] = {}
    ablation_maps: dict[str, np.ndarray] = {}
    scores: dict[str, dict[str, Any]] = {}
    failed_terms: list[dict[str, str]] = []

    def ensure_row(name: str) -> dict[str, Any]:
        if name not in scores:
            scores[name] = {
                "term_name": name,
                "score_sens": 0.0,
                "score_ablation": 0.0,
                "importance_score": 0.0,
                "sign": 0.0,
                "spatial_hotspot_radius_mm": float("nan"),
                "notes": "",
                "status": "ok",
            }
        return scores[name]

    if do_sens:
        for term_name, path in sens_specs:
            row = ensure_row(term_name)
            try:
                base_param = float(_path_get(run_spec, path))
                step = relative_step * max(abs(base_param), 1.0)
                plus = deepcopy(run_spec)
                minus = deepcopy(run_spec)
                _path_set(plus, path, base_param + step)
                _path_set(minus, path, base_param - step)

                plus_fields = build_field_bundle(plus, grid)
                minus_fields = build_field_bundle(minus, grid)
                out_plus = run_cvd_steady(
                    grid=grid,
                    fields=plus_fields,
                    model_config=plus.model,
                    process_time_s=float(plus.time.process_time_s),
                    solver_config=plus.solver,
                )
                out_minus = run_cvd_steady(
                    grid=grid,
                    fields=minus_fields,
                    model_config=minus.model,
                    process_time_s=float(minus.time.process_time_s),
                    solver_config=minus.solver,
                )
                t_plus = np.asarray(out_plus.thickness, dtype=float)
                t_minus = np.asarray(out_minus.thickness, dtype=float)
                denom = np.log((abs(base_param + step) + _EPS) / (abs(base_param - step) + _EPS))
                if abs(denom) <= _EPS:
                    denom = (2.0 * step) / max(abs(base_param), 1.0)
                sens_map = (np.log(np.clip(t_plus, _EPS, np.inf)) - np.log(np.clip(t_minus, _EPS, np.inf))) / denom
                sensitivity_maps[term_name] = sens_map
                row["score_sens"] = float(_weighted_mean(np.abs(sens_map), weights, edge_mask))
                row["sign"] = float(np.sign(_weighted_mean(sens_map, weights, edge_mask)))
                row["spatial_hotspot_radius_mm"] = _hotspot_radius_mm(sens_map, grid, weights, edge_mask)
            except Exception as exc:
                row["status"] = "failed"
                row["notes"] = f"sensitivity failed: {exc}"
                failed_terms.append({"term_name": term_name, "stage": "sensitivity", "reason": str(exc)})

    if do_ablation:
        for term_name, path, off_value in abla_specs:
            row = ensure_row(term_name)
            try:
                ablated = deepcopy(run_spec)
                _path_set(ablated, path, off_value)
                ablated_fields = build_field_bundle(ablated, grid)
                out_abl = run_cvd_steady(
                    grid=grid,
                    fields=ablated_fields,
                    model_config=ablated.model,
                    process_time_s=float(ablated.time.process_time_s),
                    solver_config=ablated.solver,
                )
                delta_map = base_thickness - np.asarray(out_abl.thickness, dtype=float)
                ablation_maps[term_name] = delta_map
                score = _weighted_mean(np.abs(delta_map), weights, edge_mask) / base_scale
                row["score_ablation"] = float(score)
                row["sign"] = float(np.sign(_weighted_mean(delta_map, weights, edge_mask)))
                row["spatial_hotspot_radius_mm"] = _hotspot_radius_mm(delta_map, grid, weights, edge_mask)
            except Exception as exc:
                row["status"] = "failed"
                note = f"ablation failed: {exc}"
                row["notes"] = note if not row["notes"] else f"{row['notes']}; {note}"
                failed_terms.append({"term_name": term_name, "stage": "ablation", "reason": str(exc)})

    ranking: list[dict[str, Any]] = []
    for term_name in sorted(scores):
        row = scores[term_name]
        row["importance_score"] = 0.5 * float(row["score_sens"]) + 0.5 * float(row["score_ablation"])
        ranking.append(row)
    ranking.sort(key=lambda item: float(item["importance_score"]), reverse=True)
    return {
        "mode": mode_key,
        "relative_step": float(relative_step),
        "base_thickness": base_thickness,
        "sensitivity_maps": sensitivity_maps,
        "ablation_maps": ablation_maps,
        "scores": ranking,
        "failed_terms": failed_terms,
    }


__all__ = [
    "build_ald_phase_snapshots",
    "build_cvd_pseudo_time_snapshots",
    "compute_net_term_maps",
    "compute_reaction_term_importance",
    "compute_transport_term_maps",
]
