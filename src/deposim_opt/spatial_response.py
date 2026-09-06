"""Optional spatial residual response applied after chemical-model selection.

This module never enumerates reaction roles and never changes a chemical fit.
It estimates a transferable, mean-preserving residual shape from identification
conditions and applies the frozen shape to a new condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .role_fields import RoleFieldSet


SPATIAL_NONE = "none"
RADIAL_QUADRATIC = "radial_quadratic"
RADIAL_QUARTIC = "radial_quartic"
SPATIAL_RESPONSE_MODES = (SPATIAL_NONE, RADIAL_QUADRATIC, RADIAL_QUARTIC)
_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class SpatialResponseFit:
    """A frozen empirical residual shape, separate from reaction parameters."""

    mode: str
    coefficients: np.ndarray
    center_xy: np.ndarray
    radius_scale: float
    train_condition_ids: tuple[int, ...]
    weighting: str = "condition_balanced_points"

    @property
    def terms(self) -> tuple[str, ...]:
        return _terms(self.mode)


def _terms(mode: str) -> tuple[str, ...]:
    normalized = str(mode).strip().lower()
    if normalized == SPATIAL_NONE:
        return ()
    if normalized == RADIAL_QUADRATIC:
        return ("rho^2",)
    if normalized == RADIAL_QUARTIC:
        return ("rho^2", "rho^4")
    raise ValueError(
        f"spatial response mode must be one of {SPATIAL_RESPONSE_MODES}, got {mode!r}"
    )


def _geometry(xy: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("xy must be a finite [point, 2] array")
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    radius = np.sqrt(np.sum(np.square(points - center), axis=1))
    scale = float(np.max(radius))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("spatial response requires at least two distinct wafer positions")
    return center, scale


def _basis(
    data: RoleFieldSet,
    mode: str,
    *,
    center_xy: np.ndarray,
    radius_scale: float,
) -> np.ndarray:
    terms = _terms(mode)
    if not terms:
        return np.empty((data.rate.size, 0), dtype=float)
    rho = np.sqrt(
        np.sum(np.square(np.asarray(data.xyz[:, :2], dtype=float) - center_xy), axis=1)
    ) / float(radius_scale)
    columns = [np.square(rho)]
    if mode == RADIAL_QUARTIC:
        columns.append(np.power(rho, 4))
    design = np.column_stack(columns)
    # A spatial response has no condition intercept. Centering within each
    # supplied wafer makes that boundary explicit even when point grids differ.
    for condition in np.unique(data.condition_id):
        mask = data.condition_id == condition
        design[mask] -= np.mean(design[mask], axis=0)
    return design


def _center_log_by_condition(values: np.ndarray, condition_id: np.ndarray) -> np.ndarray:
    logged = np.log(np.maximum(np.asarray(values, dtype=float), _TINY))
    centered = np.empty_like(logged)
    for condition in np.unique(condition_id):
        mask = condition_id == condition
        centered[mask] = logged[mask] - np.mean(logged[mask])
    return centered


def fit_spatial_response(
    mode: str,
    data: RoleFieldSet,
    chemical_prediction: np.ndarray,
) -> SpatialResponseFit:
    """Fit shared residual shape after a chemical prediction has been frozen."""

    normalized = str(mode).strip().lower()
    _terms(normalized)
    prediction = np.asarray(chemical_prediction, dtype=float)
    if prediction.shape != data.rate.shape:
        raise ValueError("chemical_prediction must match the assembled observation shape")
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= 0.0):
        raise ValueError("chemical_prediction must be positive and finite")
    if normalized == SPATIAL_NONE:
        center = np.mean(np.asarray(data.xyz[:, :2], dtype=float), axis=0)
        radius_scale = 1.0
    else:
        center, radius_scale = _geometry(data.xyz[:, :2])
    design = _basis(
        data, normalized, center_xy=center, radius_scale=radius_scale
    )
    if design.shape[1] == 0:
        coefficients = np.empty(0, dtype=float)
    else:
        target = _center_log_by_condition(data.rate, data.condition_id)
        chemical = _center_log_by_condition(prediction, data.condition_id)
        residual = target - chemical
        weights = np.empty(data.rate.size, dtype=float)
        conditions = np.unique(data.condition_id)
        for condition in conditions:
            mask = data.condition_id == condition
            weights[mask] = 1.0 / (conditions.size * np.count_nonzero(mask))
        root_weight = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_weight[:, None], residual * root_weight, rcond=None
        )
    return SpatialResponseFit(
        mode=normalized,
        coefficients=np.asarray(coefficients, dtype=float),
        center_xy=np.asarray(center, dtype=float),
        radius_scale=radius_scale,
        train_condition_ids=tuple(int(value) for value in np.unique(data.condition_id)),
    )


def apply_spatial_response(
    fit: SpatialResponseFit,
    data: RoleFieldSet,
    chemical_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen shape while preserving each chemical condition mean."""

    prediction = np.asarray(chemical_prediction, dtype=float)
    if prediction.shape != data.rate.shape:
        raise ValueError("chemical_prediction must match the assembled observation shape")
    design = _basis(
        data,
        fit.mode,
        center_xy=np.asarray(fit.center_xy, dtype=float),
        radius_scale=float(fit.radius_scale),
    )
    raw_factor = (
        np.ones(prediction.shape, dtype=float)
        if not fit.terms
        else np.exp(design @ np.asarray(fit.coefficients, dtype=float))
    )
    corrected = prediction * raw_factor
    factor = np.empty_like(raw_factor)
    for condition in np.unique(data.condition_id):
        mask = data.condition_id == condition
        chemical_mean = float(np.mean(prediction[mask]))
        corrected_mean = float(np.mean(corrected[mask]))
        scale = chemical_mean / max(corrected_mean, _TINY)
        corrected[mask] *= scale
        factor[mask] = raw_factor[mask] * scale
    return corrected, factor


def spatial_coefficient_rows(fit: SpatialResponseFit) -> list[dict[str, object]]:
    return [
        {
            "spatial_model": fit.mode,
            "term": term,
            "coefficient": float(value),
            "unit": "1",
            "train_conditions": "|".join(map(str, fit.train_condition_ids)),
            "weighting": fit.weighting,
        }
        for term, value in zip(fit.terms, fit.coefficients)
    ]


__all__ = [
    "RADIAL_QUADRATIC",
    "RADIAL_QUARTIC",
    "SPATIAL_NONE",
    "SPATIAL_RESPONSE_MODES",
    "SpatialResponseFit",
    "apply_spatial_response",
    "fit_spatial_response",
    "spatial_coefficient_rows",
]
