"""Parameter transforms for constrained optimization variables."""

from __future__ import annotations

import math


def positive_to_unconstrained(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"value must be > 0 for log transform, got {value}")
    return float(math.log(value))


def unconstrained_to_positive(value: float) -> float:
    return float(math.exp(value))


def unit_to_unconstrained(value: float) -> float:
    eps = 1.0e-12
    clipped = min(max(float(value), eps), 1.0 - eps)
    return float(math.log(clipped / (1.0 - clipped)))


def unconstrained_to_unit(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def transform_value(value: float, transform: str) -> float:
    key = str(transform).strip().lower()
    if key in {"identity", "none"}:
        return float(value)
    if key in {"positive", "log"}:
        return positive_to_unconstrained(float(value))
    if key in {"unit", "logit"}:
        return unit_to_unconstrained(float(value))
    raise ValueError(f"Unknown transform: {transform!r}")


def inverse_transform_value(value: float, transform: str) -> float:
    key = str(transform).strip().lower()
    if key in {"identity", "none"}:
        return float(value)
    if key in {"positive", "log"}:
        return unconstrained_to_positive(float(value))
    if key in {"unit", "logit"}:
        return unconstrained_to_unit(float(value))
    raise ValueError(f"Unknown transform: {transform!r}")


__all__ = [
    "inverse_transform_value",
    "positive_to_unconstrained",
    "transform_value",
    "unconstrained_to_positive",
    "unconstrained_to_unit",
    "unit_to_unconstrained",
]
