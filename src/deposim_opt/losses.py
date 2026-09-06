"""Pure residual losses and uncertainty-aware observation aggregation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


_LOSS_NAMES = {"mse", "huber", "l1"}
_WAFER_LOSS_NAMES = {
    "mse",
    "wafer_normalized_mse",
    "wafer_normalized_mae",
    "symmetric_normalized_mse",
}


def mse_loss(residual: np.ndarray) -> float:
    return float(np.mean(np.square(residual)))


def huber_loss(residual: np.ndarray, *, delta: float) -> float:
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("Huber delta must be finite and positive")
    abs_r = np.abs(residual)
    quad = np.minimum(abs_r, delta)
    linear = abs_r - quad
    return float(np.mean(0.5 * quad**2 + delta * linear))


def l1_loss(residual: np.ndarray) -> float:
    return float(np.mean(np.abs(residual)))


def validate_loss_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _LOSS_NAMES:
        raise ValueError(f"loss name must be one of {sorted(_LOSS_NAMES)}, got {name!r}")
    return key


def validate_wafer_loss_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _WAFER_LOSS_NAMES:
        raise ValueError(
            f"wafer loss name must be one of {sorted(_WAFER_LOSS_NAMES)}, got {name!r}"
        )
    return key


def _aligned_wafer_arrays(
    target: Any,
    prediction: Any,
    condition_id: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(target, dtype=float).reshape(-1)
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    groups = np.asarray(condition_id).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if observed.size == 0 or not (
        observed.shape == predicted.shape == groups.shape == mass.shape
    ):
        raise ValueError("wafer loss requires nonempty aligned target, prediction, groups, and weights")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("wafer loss target and prediction must be finite")
    if not np.all(np.isfinite(mass) & (mass >= 0.0)) or float(np.sum(mass)) <= 0.0:
        raise ValueError("wafer loss weights must be finite, nonnegative, and contain positive mass")
    return observed, predicted, groups, mass


def wafer_loss(
    *,
    target: Any,
    prediction: Any,
    condition_id: Any,
    weights: Any,
    loss_name: str,
) -> float:
    """Evaluate one whole-wafer loss with equal aggregation across conditions.

    ``weights`` may express point uncertainty, quadrature area, or both.  Each
    condition is normalized internally, so a dense map or a high-rate process
    condition does not acquire extra voting power merely from its row count.
    """

    observed, predicted, groups, mass = _aligned_wafer_arrays(
        target, prediction, condition_id, weights
    )
    name = validate_wafer_loss_name(loss_name)
    values: list[float] = []
    global_rms = float(np.sqrt(np.average(np.square(observed), weights=mass)))
    scale_floor = max(np.finfo(float).eps * max(global_rms, 1.0), np.finfo(float).tiny)
    for group in np.unique(groups):
        mask = groups == group
        local = mass[mask]
        local_total = float(np.sum(local))
        if local_total <= 0.0:
            continue
        local = local / local_total
        truth = observed[mask]
        estimate = predicted[mask]
        residual = estimate - truth
        if name == "mse":
            values.append(float(np.sum(local * np.square(residual))))
            continue
        target_rms = max(float(np.sqrt(np.sum(local * np.square(truth)))), scale_floor)
        if name == "wafer_normalized_mse":
            values.append(float(np.sum(local * np.square(residual))) / target_rms**2)
        elif name == "wafer_normalized_mae":
            values.append(float(np.sum(local * np.abs(residual))) / target_rms)
        else:
            denominator = max(
                float(np.sum(local * (np.square(truth) + np.square(estimate)))),
                2.0 * scale_floor**2,
            )
            values.append(2.0 * float(np.sum(local * np.square(residual))) / denominator)
    if not values:
        raise ValueError("wafer loss groups contain no positive weight")
    return float(np.mean(values))


def data_loss(
    *,
    residual: Any,
    loss_name: str,
    huber_delta: float = 1.345,
    fallback_values: Any | None = None,
) -> float:
    """Evaluate one named loss after rejecting missing and unknown inputs."""

    values = np.asarray(residual, dtype=float)
    finite = np.isfinite(values)
    if np.any(finite):
        data = values[finite]
    elif fallback_values is not None:
        fallback = np.asarray(fallback_values, dtype=float)
        finite_fallback = np.isfinite(fallback)
        if not np.any(finite_fallback):
            raise ValueError("no finite residual or fallback values for objective data loss")
        data = fallback[finite_fallback]
    else:
        raise ValueError("no finite residual values for objective data loss")

    name = validate_loss_name(loss_name)
    if name == "mse":
        return mse_loss(data)
    if name == "l1":
        return l1_loss(data)
    return huber_loss(data, delta=float(huber_delta))


def multi_observation_loss(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    loss_name: str = "huber",
    huber_delta: float = 1.345,
) -> tuple[float, dict[str, float]]:
    """Combine uncertainty-standardized observation losses with normalized weights."""

    validate_loss_name(loss_name)
    if not observations:
        raise ValueError("at least one observation is required")

    components: dict[str, float] = {}
    weights: dict[str, float] = {}
    for name, observation in observations.items():
        missing = [key for key in ("target", "prediction", "sigma") if key not in observation]
        if missing:
            raise ValueError(f"observation {name!r} is missing: {missing}")
        target = np.asarray(observation["target"], dtype=float).reshape(-1)
        prediction = np.asarray(observation["prediction"], dtype=float).reshape(-1)
        if not target.size or target.shape != prediction.shape:
            raise ValueError(f"observation {name!r} requires nonempty aligned target and prediction")
        sigma = np.broadcast_to(np.asarray(observation["sigma"], dtype=float), target.shape)
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
            raise ValueError(f"observation {name!r} contains nonfinite values")
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError(f"observation {name!r} uncertainty must be finite and positive")
        weight = float(observation.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"observation {name!r} weight must be finite and nonnegative")
        standardized = (prediction - target) / sigma
        components[str(name)] = data_loss(
            residual=standardized,
            loss_name=loss_name,
            huber_delta=huber_delta,
        )
        weights[str(name)] = weight

    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        raise ValueError("observation weights must contain at least one positive value")
    total = sum(weights[name] * components[name] for name in components) / total_weight
    return float(total), components


__all__ = [
    "data_loss",
    "huber_loss",
    "l1_loss",
    "mse_loss",
    "multi_observation_loss",
    "validate_loss_name",
    "validate_wafer_loss_name",
    "wafer_loss",
]
