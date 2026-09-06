"""Prediction metrics used for interpretation and candidate comparison."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def residual_metrics(residual: Any) -> dict[str, float]:
    """Return ordinary error metrics for finite residual values."""

    values = np.asarray(residual, dtype=float)
    finite = values[np.isfinite(values)]
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


def prediction_metrics(
    target: Any,
    prediction: Any,
    *,
    sigma: Any = None,
    baseline: float | None = None,
) -> dict[str, float]:
    """Return unit-neutral prediction, bias, and centered-shape metrics."""

    target_values = np.asarray(target, dtype=float).ravel()
    prediction_values = np.asarray(prediction, dtype=float).ravel()
    if not target_values.size or target_values.shape != prediction_values.shape:
        raise ValueError("nonempty, aligned target and prediction arrays are required")
    if not np.all(np.isfinite(target_values)) or not np.all(np.isfinite(prediction_values)):
        raise ValueError("prediction metrics require finite observations and predictions")

    residual = prediction_values - target_values
    centered_target = target_values - np.mean(target_values)
    centered_prediction = prediction_values - np.mean(prediction_values)
    centered_error = centered_prediction - centered_target
    variation = float(np.mean(centered_target**2))
    denominator = float(np.linalg.norm(centered_target) * np.linalg.norm(centered_prediction))

    uncertainty = None
    if sigma is not None:
        uncertainty = np.broadcast_to(np.asarray(sigma, dtype=float), target_values.shape)
        if not np.all(np.isfinite(uncertainty) & (uncertainty > 0.0)):
            raise ValueError("measurement uncertainty must be finite and positive")

    mse = float(np.mean(residual**2))
    target_mean = float(np.mean(target_values))
    result = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "observation_count": float(residual.size),
        "mean_bias": float(np.mean(residual)),
        "mean_mse": float(np.mean(residual) ** 2),
        "centered_mse": float(np.mean(centered_error**2)),
        "centered_rmse": float(np.sqrt(np.mean(centered_error**2))),
        "centered_r2": float(1 - np.mean(centered_error**2) / variation) if variation > 0 else float("nan"),
        "spatial_correlation": (
            float(np.dot(centered_target, centered_prediction) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "normalized_rmse": (
            float(np.sqrt(np.mean((residual / uncertainty) ** 2)))
            if uncertainty is not None
            else float("nan")
        ),
        "target_mean": target_mean,
        "target_variance": variation,
        "relative_rmse": float(np.sqrt(mse) / abs(target_mean)) if target_mean != 0 else float("nan"),
    }
    if baseline is not None:
        result["baseline_mse"] = float(np.mean((target_values - baseline) ** 2))
    return result


def observation_metrics(observation: Mapping[str, Any]) -> dict[str, float]:
    """Thickness adapter retaining the public nanometer metric names."""

    metrics = prediction_metrics(
        observation["target_nm"],
        observation["prediction_nm"],
        sigma=observation.get("sigma_nm"),
    )
    aliases = {
        "mse": "mse_nm2",
        "rmse": "rmse_nm",
        "mae": "mae_nm",
        "max_abs": "max_abs_nm",
        "mean_bias": "mean_bias_nm",
        "centered_rmse": "centered_rmse_nm",
        "target_mean": "target_mean_nm",
        "target_variance": "target_variance_nm2",
    }
    return {**metrics, **{alias: metrics[key] for key, alias in aliases.items()}}


def metric_value(
    metrics: Mapping[str, Any],
    name: str,
    default: float = float("nan"),
) -> float:
    """Read legacy thickness aliases at one compatibility boundary."""

    if name not in metrics and name in {"mean_mse", "centered_mse"}:
        source = "mean_bias" if name == "mean_mse" else "centered_rmse"
        value = metric_value(metrics, source)
        return value**2 if np.isfinite(value) else default
    aliases = {
        "mse": "mse_nm2",
        "baseline_mse": "baseline_mse_nm2",
        "target_mean": "target_mean_nm",
        "target_variance": "target_variance_nm2",
        "mean_bias": "mean_bias_nm",
        "centered_rmse": "centered_rmse_nm",
    }
    return float(metrics.get(name, metrics.get(aliases.get(name, name), default)))


__all__ = [
    "metric_value",
    "observation_metrics",
    "prediction_metrics",
    "residual_metrics",
]
