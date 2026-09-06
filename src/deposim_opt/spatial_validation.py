"""Small spatial grouping and rate-metric utilities shared by CVD analyses."""

from __future__ import annotations

import math

import numpy as np


EPS = 1.0e-30


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.size < 2 or float(np.std(left)) <= EPS or float(np.std(right)) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    residual = predicted - observed
    mse = float(np.mean(np.square(residual)))
    centered = observed - float(np.mean(observed))
    sst = float(np.sum(np.square(centered)))
    sse = float(np.sum(np.square(residual)))
    return {
        "mse_nm2_s2": mse,
        "rmse_nm_s": float(math.sqrt(max(mse, 0.0))),
        "mae_nm_s": float(np.mean(np.abs(residual))),
        "max_abs_nm_s": float(np.max(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > EPS else float("nan"),
        "bias_nm_s": float(np.mean(residual)),
    }


def angular_groups(xy: np.ndarray, requested_groups: int = 8) -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    groups = np.floor(angles / (2.0 * np.pi / requested_groups)).astype(int)
    radius = np.sqrt(np.sum(np.square(points), axis=1))
    groups[radius <= max(float(np.max(radius)) * 1.0e-10, EPS)] = 0
    return groups


def radial_groups(xy: np.ndarray, max_groups: int = 6) -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    radius = np.sqrt(np.sum(np.square(points), axis=1))
    rounded = np.round(radius, decimals=10)
    unique = np.unique(rounded)
    if unique.size <= max_groups:
        mapping = {float(value): idx for idx, value in enumerate(unique)}
        return np.asarray([mapping[float(value)] for value in rounded], dtype=int)
    quantiles = np.quantile(radius, np.linspace(0.0, 1.0, max_groups + 1))
    return np.digitize(radius, np.unique(quantiles[1:-1]), right=True).astype(int)


__all__ = ["EPS", "angular_groups", "radial_groups", "rate_metrics", "safe_correlation"]
