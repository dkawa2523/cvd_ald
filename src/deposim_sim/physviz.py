"""Physical-interpretability helpers on top of the AIB execution path."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .common.path_tools import get_attr_path, set_attr_path
from .pipeline import run_aib_from_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for physviz utilities.")


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    w = np.asarray(weights[valid], dtype=float)
    v = np.asarray(values[valid], dtype=float)
    return float(np.sum(v * w) / max(np.sum(w), _EPS))


def build_cvd_pseudo_time_snapshots(
    run_spec: Any,
    time_points: Sequence[float],
    *,
    enable_input_time_variation: bool = False,
    input_variation_amplitude: float = 0.2,
) -> dict[str, Any]:
    """Build pseudo-time thickness maps by re-running AIB with scaled process times."""

    _require_numpy()
    if not time_points:
        raise ValueError("time_points must be non-empty")
    fractions = np.asarray(sorted({float(t) for t in time_points}), dtype=float)
    if np.any(fractions <= 0.0):
        raise ValueError("time_points must be > 0")

    base_t = float(getattr(run_spec.time, "t_proc_s", 1.0))
    times = fractions * base_t
    snapshots: list[np.ndarray] = []
    input_snapshots: dict[str, np.ndarray] = {}

    for frac in fractions:
        spec_i = deepcopy(run_spec)
        spec_i.time.t_proc_s = float(max(base_t * float(frac), _EPS))
        if enable_input_time_variation:
            scale = 1.0 + float(input_variation_amplitude) * (2.0 * float(frac) - 1.0)
            k_rxn = float(spec_i.model.params.kinetics.get("k_rxn", 0.01))
            spec_i.model.params.kinetics["k_rxn"] = max(k_rxn * scale, _EPS)
        out = run_aib_from_spec(spec_i)
        snapshots.append(np.asarray(out.thickness, dtype=float))

    thickness = np.stack(snapshots, axis=0)
    linear = thickness[-1][None, ...] * fractions.reshape((-1,) + (1,) * (thickness.ndim - 1))
    residual = thickness - linear
    return {
        "time_fractions": fractions,
        "time_seconds": times,
        "thickness_snapshots": thickness,
        "delta_thickness_snapshots": np.diff(thickness, axis=0),
        "linearity_residual_snapshots": residual,
        "linearity_residual_max": np.max(np.abs(residual), axis=0),
        "input_snapshots": input_snapshots,
        "input_time_variation_enabled": bool(enable_input_time_variation),
        "input_variation_amplitude": float(input_variation_amplitude),
    }


def build_ald_phase_snapshots(run_spec: Any) -> dict[str, Any]:
    """Build ALD-like phase snapshots using transient AIB execution."""

    _require_numpy()
    out = run_aib_from_spec(run_spec)
    phase_names = ["phase_01"]
    phase_thickness = np.asarray(out.thickness, dtype=float)[None, ...]
    coverage = np.asarray(out.fields.get("theta_A", out.thickness), dtype=float)[None, ...]
    return {
        "phase_names": phase_names,
        "phase_durations_s": np.asarray([float(run_spec.time.t_proc_s)], dtype=float),
        "phase_thickness_snapshots": phase_thickness,
        "phase_coverage_snapshots": coverage,
        "cumulative_thickness_snapshots": np.cumsum(phase_thickness, axis=0),
        "final_thickness": np.asarray(out.thickness, dtype=float),
    }


def compute_transport_term_maps(
    result: Any,
    fields: Any | None = None,
    km: Any | None = None,
    nu: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute AIB transport-side term maps from runtime diagnostics."""

    _require_numpy()
    csa = np.asarray(result.fields.get("CsA_over_CrefA"), dtype=float)
    phi_b = np.asarray(result.fields.get("phi_B"), dtype=float)
    fi = np.asarray(result.fields.get("f_I"), dtype=float)
    dep = np.asarray(result.deposition_rate, dtype=float)
    km_a = np.asarray(result.diagnostics.get("km_A_map", np.ones_like(dep)), dtype=float)
    km_b = np.asarray(result.diagnostics.get("km_B_map", np.full(dep.shape, np.nan, dtype=float)), dtype=float)
    tau_a = np.asarray(result.diagnostics.get("tau_A_map", np.full(dep.shape, np.nan, dtype=float)), dtype=float)
    tau_b = np.asarray(result.diagnostics.get("tau_B_map", np.full(dep.shape, np.nan, dtype=float)), dtype=float)

    cs_a = np.asarray(result.Cs.get("A", np.zeros_like(dep)), dtype=float)
    cs_b = np.asarray(result.Cs.get("B", np.full(dep.shape, np.nan, dtype=float)), dtype=float)
    cref_a = np.where(np.abs(csa) > _EPS, cs_a / np.maximum(csa, _EPS), 0.0)
    cref_b = np.where(np.abs(phi_b) > _EPS, cs_b, 0.0)
    transport_capacity_a = np.clip(km_a * np.maximum(cref_a, 0.0), 0.0, np.inf)
    reaction_demand_a = np.clip(dep, 0.0, np.inf)
    utilization_a = reaction_demand_a / np.maximum(transport_capacity_a, _EPS)

    transport_capacity_b = np.clip(km_b * np.maximum(cref_b, 0.0), 0.0, np.inf)
    reaction_demand_b = np.nan_to_num(phi_b, nan=0.0)
    utilization_b = reaction_demand_b / np.maximum(transport_capacity_b, _EPS)
    return {
        "transport_capacity__A": transport_capacity_a,
        "reaction_demand__A": reaction_demand_a,
        "depletion_ratio__A": np.clip(1.0 - csa, 0.0, np.inf),
        "utilization__A": np.clip(utilization_a, 0.0, np.inf),
        "transport_capacity__B": np.nan_to_num(transport_capacity_b, nan=0.0),
        "reaction_demand__B": np.nan_to_num(reaction_demand_b, nan=0.0),
        "depletion_ratio__B": np.nan_to_num(phi_b, nan=0.0),
        "utilization__B": np.nan_to_num(utilization_b, nan=0.0),
        "km_A": km_a,
        "km_B": km_b,
        "tau_A": tau_a,
        "tau_B": tau_b,
        "inhibition_proxy": np.clip(fi, 0.0, np.inf),
    }


def compute_net_term_maps(result: Any) -> dict[str, np.ndarray]:
    """Compute net equation term maps (deposition/etch/loss proxies)."""

    _require_numpy()
    dep = np.asarray(result.deposition_rate, dtype=float)
    etch = np.zeros_like(dep, dtype=float)
    loss = np.zeros_like(dep, dtype=float)
    frac_etch = np.zeros_like(dep, dtype=float)
    frac_loss = np.zeros_like(dep, dtype=float)
    return {
        "dep_rate": dep,
        "etch_rate": etch,
        "loss_rate": loss,
        "etch_fraction_of_dep": frac_etch,
        "loss_fraction_of_dep": frac_loss,
    }


def compute_reaction_term_importance(
    run_spec: Any,
    *,
    do_sens: bool = True,
    do_ablation: bool = True,
    relative_step: float = 0.05,
) -> dict[str, Any]:
    """Approximate reaction-term importance with finite differences on AIB params."""

    _require_numpy()
    base = run_aib_from_spec(run_spec)
    base_h = np.asarray(base.thickness, dtype=float)
    weights = np.asarray(base.grid.area_weights_mm2, dtype=float)
    mask = np.asarray(base.grid.edge_mask, dtype=bool)
    base_scale = max(abs(_weighted_mean(base_h, weights, mask)), _EPS)

    terms = {
        "k_ads": "model.params.kinetics.k_ads",
        "k_des": "model.params.kinetics.k_des",
        "k_rxn": "model.params.kinetics.k_rxn",
        "K_I": "model.params.inhibitor.K_I",
    }

    sensitivity_maps: dict[str, np.ndarray] = {}
    ablation_maps: dict[str, np.ndarray] = {}
    scores: list[dict[str, Any]] = []
    failed_terms: list[dict[str, str]] = []

    for term_name, path in terms.items():
        row = {
            "term_name": term_name,
            "score_sens": 0.0,
            "score_ablation": 0.0,
            "importance_score": 0.0,
            "sign": 0.0,
            "spatial_hotspot_radius_mm": float("nan"),
            "notes": "",
            "status": "ok",
        }
        try:
            base_param = float(get_attr_path(run_spec, path))
            step = max(abs(base_param), 1.0) * float(relative_step)
            if do_sens:
                plus = deepcopy(run_spec)
                minus = deepcopy(run_spec)
                set_attr_path(plus, path, base_param + step, create_missing_mappings=False)
                set_attr_path(minus, path, max(base_param - step, _EPS), create_missing_mappings=False)
                out_plus = run_aib_from_spec(plus)
                out_minus = run_aib_from_spec(minus)
                sens_map = (
                    np.asarray(out_plus.thickness, dtype=float) - np.asarray(out_minus.thickness, dtype=float)
                ) / (2.0 * step)
                sensitivity_maps[term_name] = sens_map
                row["score_sens"] = abs(_weighted_mean(sens_map, weights, mask))
                row["sign"] = float(np.sign(_weighted_mean(sens_map, weights, mask)))

            if do_ablation:
                ablated = deepcopy(run_spec)
                off_value = 0.0 if term_name != "k_des" else base_param
                set_attr_path(ablated, path, off_value, create_missing_mappings=False)
                out_abl = run_aib_from_spec(ablated)
                delta = base_h - np.asarray(out_abl.thickness, dtype=float)
                ablation_maps[term_name] = delta
                row["score_ablation"] = abs(_weighted_mean(delta, weights, mask)) / base_scale
                row["sign"] = float(np.sign(_weighted_mean(delta, weights, mask)))
        except Exception as exc:  # pragma: no cover - defensive path
            row["status"] = "failed"
            row["notes"] = str(exc)
            failed_terms.append({"term_name": term_name, "reason": str(exc)})

        row["importance_score"] = 0.5 * float(row["score_sens"]) + 0.5 * float(row["score_ablation"])
        scores.append(row)

    scores.sort(key=lambda item: float(item["importance_score"]), reverse=True)
    return {
        "mode": "aib",
        "relative_step": float(relative_step),
        "base_thickness": base_h,
        "sensitivity_maps": sensitivity_maps,
        "ablation_maps": ablation_maps,
        "scores": scores,
        "failed_terms": failed_terms,
    }


__all__ = [
    "build_ald_phase_snapshots",
    "build_cvd_pseudo_time_snapshots",
    "compute_net_term_maps",
    "compute_reaction_term_importance",
    "compute_transport_term_maps",
]
