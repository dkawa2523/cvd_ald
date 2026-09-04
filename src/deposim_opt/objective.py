"""Score decomposition utilities for AIB optimization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

_EPS = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for deposim_opt.objective")


def _mean_finite(values: Any, *, fallback: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(fallback)
    return float(np.mean(arr[finite]))


def huber_loss(residual_nm: np.ndarray, *, delta_nm: float) -> float:
    abs_r = np.abs(residual_nm)
    quad = np.minimum(abs_r, delta_nm)
    lin = abs_r - quad
    return float(np.mean(0.5 * quad**2 + delta_nm * lin))


def l1_loss(residual_nm: np.ndarray) -> float:
    return float(np.mean(np.abs(residual_nm)))


def residual_metrics(residual_nm: Any) -> dict[str, float]:
    """Return interpretable error metrics without changing the fit objective."""

    _require_numpy()
    residual = np.asarray(residual_nm, dtype=float)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {
            "mse_nm2": float("nan"),
            "rmse_nm": float("nan"),
            "mae_nm": float("nan"),
            "max_abs_nm": float("nan"),
        }
    squared = np.square(finite)
    return {
        "mse_nm2": float(np.mean(squared)),
        "rmse_nm": float(np.sqrt(np.mean(squared))),
        "mae_nm": float(np.mean(np.abs(finite))),
        "max_abs_nm": float(np.max(np.abs(finite))),
    }


def prediction_metrics(target: Any, prediction: Any, *, sigma: Any = None,
                       baseline: float | None = None) -> dict[str, float]:
    """Unit-neutral errors. Callers label the quantity and unit at the boundary."""
    target = np.asarray(target, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    if not target.size or target.shape != prediction.shape:
        raise ValueError("nonempty, aligned target and prediction arrays are required")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction metrics require finite observations and predictions")
    residual = prediction - target
    centered_target = target - np.mean(target)
    centered_prediction = prediction - np.mean(prediction)
    centered_error = centered_prediction - centered_target
    variation = float(np.mean(centered_target**2))
    denominator = float(np.linalg.norm(centered_target) * np.linalg.norm(centered_prediction))
    if sigma is not None:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), target.shape)
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("measurement uncertainty must be finite and positive")
    mse = float(np.mean(residual**2))
    result = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "observation_count": float(residual.size),
        "mean_bias": float(np.mean(residual)),
        "mean_mse": float(np.mean(residual)**2),
        "centered_mse": float(np.mean(centered_error**2)),
        "centered_rmse": float(np.sqrt(np.mean(centered_error**2))),
        "centered_r2": float(1 - np.mean(centered_error**2) / variation) if variation > 0 else float("nan"),
        "spatial_correlation": float(np.dot(centered_target, centered_prediction) / denominator) if denominator > 0 else float("nan"),
        "normalized_rmse": float(np.sqrt(np.mean((residual / np.asarray(sigma))**2))) if sigma is not None else float("nan"),
        "target_mean": float(np.mean(target)),
        "target_variance": variation,
        "relative_rmse": float(np.sqrt(mse) / abs(np.mean(target))) if np.mean(target) != 0 else float("nan"),
    }
    if baseline is not None:
        result["baseline_mse"] = float(np.mean((target - baseline)**2))
    return result


def observation_metrics(observation: Mapping[str, Any]) -> dict[str, float]:
    """Thickness adapter retaining existing public metric names."""
    metrics = prediction_metrics(observation["target_nm"], observation["prediction_nm"],
                                 sigma=observation.get("sigma_nm"))
    return {**metrics, **{alias: metrics[key] for key, alias in {
        "mse": "mse_nm2", "rmse": "rmse_nm", "mae": "mae_nm", "max_abs": "max_abs_nm",
        "mean_bias": "mean_bias_nm", "centered_rmse": "centered_rmse_nm",
        "target_mean": "target_mean_nm", "target_variance": "target_variance_nm2",
    }.items()}}


def metric_value(metrics: Mapping[str, Any], name: str, default: float = float("nan")) -> float:
    """Read old thickness records at a single compatibility boundary."""
    if name not in metrics and name in {"mean_mse", "centered_mse"}:
        source = "mean_bias" if name == "mean_mse" else "centered_rmse"
        value = metric_value(metrics, source)
        return value**2 if np.isfinite(value) else default
    aliases = {"mse": "mse_nm2", "baseline_mse": "baseline_mse_nm2",
               "target_mean": "target_mean_nm", "target_variance": "target_variance_nm2",
               "mean_bias": "mean_bias_nm", "centered_rmse": "centered_rmse_nm"}
    return float(metrics.get(name, metrics.get(aliases.get(name, name), default)))


def data_loss(
    *,
    residual_nm: Any,
    loss_kind: str,
    huber_delta_nm: float,
    fallback_values_nm: Any | None = None,
) -> float:
    _require_numpy()
    residual = np.asarray(residual_nm, dtype=float)
    finite = np.isfinite(residual)
    if np.any(finite):
        data = residual[finite]
    elif fallback_values_nm is not None:
        fallback = np.asarray(fallback_values_nm, dtype=float)
        finite_fb = np.isfinite(fallback)
        if not np.any(finite_fb):
            raise ValueError("no finite residual_nm values and no finite fallback_values_nm values for objective data loss")
        data = fallback[finite_fb]
    else:
        raise ValueError("no finite residual_nm values for objective data loss; measurement data may be missing or fully masked")

    kind = str(loss_kind).strip().lower()
    if kind == "l1":
        return l1_loss(data)
    return huber_loss(data, delta_nm=float(huber_delta_nm))


def solver_penalty(*, diagnostics: Mapping[str, Any], lambda_solver: float) -> float:
    if float(lambda_solver) <= 0.0:
        return 0.0

    non_bracket_map = diagnostics.get("root_non_bracket_count_map", diagnostics.get("root_status_map", 0.0))
    iter_map = diagnostics.get("root_iteration_count", 0.0)

    arr_non = np.asarray(non_bracket_map, dtype=float)
    arr_iter = np.asarray(iter_map, dtype=float)

    n_pts = max(int(arr_iter.size), 1)
    non_bracket_mean = float(np.nansum(np.clip(arr_non, 0.0, np.inf))) / float(n_pts)
    iter_mean = _mean_finite(arr_iter, fallback=0.0)
    iter_over = max(iter_mean - 4.0, 0.0) / 10.0
    return float(lambda_solver) * (non_bracket_mean + iter_over)


def physics_penalty(
    *,
    fields: Mapping[str, Any],
    role_has_i: bool,
    role_has_b: bool,
    lambda_phys: float,
    phi_b_min: float,
) -> float:
    if float(lambda_phys) <= 0.0:
        return 0.0

    penalty_raw = 0.0
    if role_has_i:
        penalty_raw += np.clip(_mean_finite(fields.get("f_I", 0.0), fallback=0.0), 0.0, np.inf)

    if role_has_b:
        phi_b = _mean_finite(fields.get("phi_B", 0.0), fallback=0.0)
        threshold = max(float(phi_b_min), _EPS)
        penalty_raw += max(threshold - phi_b, 0.0) / threshold

    return float(lambda_phys) * float(penalty_raw)


def prior_penalty(*, lambda_prior: float, prior_terms: Sequence[float] | None) -> float:
    terms = [float(v) for v in list(prior_terms or []) if np.isfinite(float(v))]
    if float(lambda_prior) <= 0.0 or not terms:
        return 0.0
    return float(lambda_prior) * 0.5 * float(np.mean(np.asarray(terms, dtype=float)))


def complexity_penalty(*, lambda_complex: float, role_has_i: bool, role_has_b: bool) -> float:
    if float(lambda_complex) <= 0.0:
        return 0.0
    return float(lambda_complex) * float(int(role_has_i) + int(role_has_b))


def _center_edge_residual_bias(*, fields: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    residual = np.asarray(fields.get("residual_nm", []), dtype=float).reshape(-1)
    xy = np.asarray(diagnostics.get("xy_mm", []), dtype=float)
    if residual.size == 0 or xy.ndim != 2 or xy.shape[0] != residual.shape[0]:
        return 0.0
    finite = np.isfinite(residual)
    if not np.any(finite):
        return 0.0
    r = np.sqrt(np.sum(np.square(xy), axis=1))
    center = finite & (r <= np.nanpercentile(r, 25.0))
    edge = finite & (r >= np.nanpercentile(r, 75.0))
    if not np.any(center) or not np.any(edge):
        return 0.0
    return float(abs(np.nanmean(residual[center]) - np.nanmean(residual[edge])))


def profile_penalty(
    *,
    fields: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> float:
    """Return small process-profile penalties without changing model physics."""

    profile = str(objective.get("profile", "cvd_map")).strip().lower()
    if profile not in {"generic", "cvd_map", "ald_cycle"}:
        raise ValueError("objective.profile must be generic|cvd_map|ald_cycle")

    weights = dict(objective.get("weights", {}) or {})
    if profile == "cvd_map":
        return float(weights.get("spatial_bias", 0.0)) * _center_edge_residual_bias(
            fields=fields,
            diagnostics=diagnostics,
        )

    if profile == "ald_cycle":
        ald = dict(diagnostics.get("ald_metrics", {}) or {})
        plateau_ratio = ald.get("plateau_gain_ratio", diagnostics.get("plateau_gain_ratio"))
        cycle_cv = ald.get("cycle_gpc_cv", diagnostics.get("cycle_gpc_cv"))
        purge_fraction = ald.get("purge_growth_fraction", diagnostics.get("purge_growth_fraction"))

        penalty = 0.0
        if plateau_ratio is not None and np.isfinite(float(plateau_ratio)):
            target = max(float(objective.get("plateau_gain_ratio_target", 0.5)), _EPS)
            penalty += float(weights.get("plateau", 0.0)) * max(float(plateau_ratio) - target, 0.0) / target
        if cycle_cv is not None and np.isfinite(float(cycle_cv)):
            target = max(float(objective.get("cycle_gpc_cv_target", 0.10)), _EPS)
            penalty += float(weights.get("cycle", 0.0)) * max(float(cycle_cv) - target, 0.0) / target
        if purge_fraction is not None and np.isfinite(float(purge_fraction)):
            target = max(float(objective.get("purge_growth_fraction_target", 0.01)), _EPS)
            penalty += float(weights.get("purge", 0.0)) * max(float(purge_fraction) - target, 0.0) / target
        return float(penalty)

    return 0.0


def evaluate_candidate_score(
    *,
    residual_nm: Any,
    fields: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    role_has_i: bool,
    role_has_b: bool,
    objective: Mapping[str, Any] | None,
    lambda_complex: float,
    prior_terms: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute decomposed candidate score components.

    Returned keys are stable and intended for ranking/report tables.
    """

    _require_numpy()
    objective_cfg = dict(objective or {})
    observation = diagnostics.get("observation")
    reported_metrics = residual_metrics(residual_nm)
    if observation is not None:
        residual_nm = np.asarray(observation["residual_nm"], dtype=float)
        reported_metrics = observation_metrics(observation)
    fitting_residual = residual_nm
    sigma = observation.get("sigma_nm") if observation is not None else None
    if sigma is not None:
        fitting_residual = np.asarray(residual_nm) / np.asarray(sigma)
    penalties = dict(objective_cfg.get("penalties", {}) or {})

    huber_delta = float(objective_cfg.get("huber_delta_nm", 10.0))
    if sigma is not None:
        huber_delta = float(objective_cfg.get("huber_delta", 1.345))
    loss_kind = str(objective_cfg.get("loss", "huber"))
    lambda_solver = float(penalties.get("lambda_solver", 0.0))
    lambda_phys = float(penalties.get("lambda_phys", 0.0))
    lambda_prior = float(penalties.get("lambda_prior", 0.0))
    phi_b_min = float(objective_cfg.get("phi_B_min", 0.05))
    weights = dict(objective_cfg.get("weights", {}) or {})
    measurement_weight = float(weights.get("measurement", 1.0))

    allow_prediction_fallback = bool(objective_cfg.get("allow_prediction_fallback_when_no_measurement", False))
    loss_data = 0.0
    if measurement_weight > 0.0:
        loss_data = measurement_weight * data_loss(
            residual_nm=fitting_residual,
            loss_kind=loss_kind,
            huber_delta_nm=huber_delta,
            fallback_values_nm=fields.get("h_nm") if allow_prediction_fallback else None,
        )
    pen_solver = solver_penalty(diagnostics=diagnostics, lambda_solver=lambda_solver)
    pen_phys = physics_penalty(
        fields=fields,
        role_has_i=role_has_i,
        role_has_b=role_has_b,
        lambda_phys=lambda_phys,
        phi_b_min=phi_b_min,
    )
    pen_prior = prior_penalty(lambda_prior=lambda_prior, prior_terms=prior_terms)
    profile_fields = fields if observation is None else {**fields, "residual_nm": observation["residual_nm"]}
    profile_diagnostics = diagnostics if observation is None else {**diagnostics, "xy_mm": observation["xy_mm"]}
    pen_profile = profile_penalty(fields=profile_fields, diagnostics=profile_diagnostics, objective=objective_cfg)
    pen_complex = complexity_penalty(
        lambda_complex=float(lambda_complex),
        role_has_i=role_has_i,
        role_has_b=role_has_b,
    )

    score_total = loss_data + pen_solver + pen_phys + pen_prior + pen_profile + pen_complex
    return {
        **reported_metrics,
        "loss_data": float(loss_data),
        "penalty_solver": float(pen_solver),
        "penalty_phys": float(pen_phys),
        "penalty_prior": float(pen_prior),
        "penalty_profile": float(pen_profile),
        "penalty_complexity": float(pen_complex),
        "score_total": float(score_total),
    }


__all__ = [
    "complexity_penalty",
    "data_loss",
    "evaluate_candidate_score",
    "huber_loss",
    "l1_loss",
    "profile_penalty",
    "physics_penalty",
    "prior_penalty",
    "residual_metrics",
    "prediction_metrics",
    "observation_metrics",
    "metric_value",
    "solver_penalty",
]
