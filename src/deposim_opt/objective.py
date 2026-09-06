"""Compose data loss and physically declared fit penalties."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .losses import data_loss, multi_observation_loss, validate_loss_name
from .metrics import observation_metrics, residual_metrics

def _mean_finite(values: Any, *, fallback: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(fallback)
    return float(np.mean(arr[finite]))


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


def prior_penalty(*, lambda_prior: float, prior_terms: Sequence[float] | None) -> float:
    terms = [float(v) for v in list(prior_terms or []) if np.isfinite(float(v))]
    if float(lambda_prior) <= 0.0 or not terms:
        return 0.0
    return float(lambda_prior) * 0.5 * float(np.mean(np.asarray(terms, dtype=float)))


def evaluate_candidate_score(
    *,
    residual_nm: Any,
    fields: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    objective: Mapping[str, Any] | None,
    prior_terms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute decomposed candidate score components.

    Returned keys are stable and intended for ranking/report tables.
    """

    objective_cfg = dict(objective or {})
    observation = diagnostics.get("observation")
    reported_metrics = residual_metrics(residual_nm)
    if observation is not None:
        residual_nm = np.asarray(observation["residual_nm"], dtype=float)
        reported_metrics = observation_metrics(observation)
    sigma = observation.get("sigma_nm") if observation is not None else None
    penalties = dict(objective_cfg.get("penalties", {}) or {})

    raw_loss = objective_cfg.get("loss", {"name": "mse"})
    if not isinstance(raw_loss, Mapping):
        raise ValueError("objective.loss must be a mapping with a named loss")
    loss_cfg = dict(raw_loss)
    loss_name = validate_loss_name(str(loss_cfg.get("name", "mse")))
    standardized = loss_cfg.get("standardized", "auto")
    if isinstance(standardized, str):
        normalized_standardized = standardized.strip().lower()
        if normalized_standardized not in {"auto", "true", "false"}:
            raise ValueError("loss.standardized must be auto, true, or false")
        use_standardized = (
            sigma is not None
            if normalized_standardized == "auto"
            else normalized_standardized == "true"
        )
    elif isinstance(standardized, (bool, np.bool_)):
        use_standardized = bool(standardized)
    else:
        raise ValueError("loss.standardized must be auto or a boolean")
    if use_standardized and sigma is None:
        raise ValueError("standardized loss requires measurement uncertainty")
    fitting_residual = np.asarray(residual_nm) / np.asarray(sigma) if use_standardized else residual_nm
    huber_delta = float(
        loss_cfg.get("delta", 1.345)
        if use_standardized
        else loss_cfg.get("delta_nm", 10.0)
    )
    lambda_solver = float(penalties.get("lambda_solver", 0.0))
    lambda_prior = float(penalties.get("lambda_prior", 0.0))
    weights = dict(objective_cfg.get("weights", {}) or {})
    measurement_weight = float(weights.get("measurement", 1.0))
    if not np.isfinite(measurement_weight) or measurement_weight < 0.0:
        raise ValueError("objective.weights.measurement must be finite and nonnegative")

    allow_prediction_fallback = bool(objective_cfg.get("allow_prediction_fallback_when_no_measurement", False))
    loss_data = 0.0
    observation_losses: dict[str, float] = {}
    multiple_observations = diagnostics.get("observations")
    if multiple_observations:
        if not isinstance(multiple_observations, Mapping):
            raise ValueError("diagnostics.observations must map names to observations")
        loss_data, observation_losses = multi_observation_loss(
            multiple_observations,
            loss_name=loss_name,
            huber_delta=float(loss_cfg.get("delta", 1.345)),
        )
        use_standardized = True
    elif measurement_weight > 0.0:
        loss_data = measurement_weight * data_loss(
            residual=fitting_residual,
            loss_name=loss_name,
            huber_delta=huber_delta,
            fallback_values=fields.get("h_nm") if allow_prediction_fallback else None,
        )
    pen_solver = solver_penalty(diagnostics=diagnostics, lambda_solver=lambda_solver)
    pen_prior = prior_penalty(lambda_prior=lambda_prior, prior_terms=prior_terms)

    score_total = loss_data + pen_solver + pen_prior
    return {
        **reported_metrics,
        "loss_data": float(loss_data),
        "loss_standardized": float(use_standardized),
        "observation_loss_count": float(len(observation_losses)),
        **{
            f"loss_observation_{name}": float(value)
            for name, value in observation_losses.items()
        },
        "penalty_solver": float(pen_solver),
        "penalty_prior": float(pen_prior),
        "score_total": float(score_total),
    }


__all__ = [
    "evaluate_candidate_score",
    "prior_penalty",
    "solver_penalty",
]
