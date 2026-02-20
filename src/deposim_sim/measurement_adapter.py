"""Measurement map alignment and resampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import DomainGrid

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for measurement adapter utilities.")


@dataclass(frozen=True)
class MeasurementMap:
    x_mm: np.ndarray
    y_mm: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        _require_numpy()
        if self.values.ndim != 2:
            raise ValueError(f"measurement.values must be 2D, got shape {self.values.shape}")
        ny, nx = self.values.shape
        if self.x_mm.ndim != 1 or self.x_mm.shape[0] != nx:
            raise ValueError("measurement.x_mm must be 1D with length matching values.shape[1]")
        if self.y_mm.ndim != 1 or self.y_mm.shape[0] != ny:
            raise ValueError("measurement.y_mm must be 1D with length matching values.shape[0]")
        if self.valid_mask is not None and self.valid_mask.shape != self.values.shape:
            raise ValueError("measurement.valid_mask shape must match values")


def _grid_xy(grid: DomainGrid) -> tuple[np.ndarray, np.ndarray]:
    if grid.x_grid_mm is not None and grid.y_grid_mm is not None:
        return np.asarray(grid.x_grid_mm, dtype=float), np.asarray(grid.y_grid_mm, dtype=float)
    if grid.kind == "wafer_2d_polar":
        if grid.theta_grid_rad is None:
            raise ValueError("polar grid is missing theta_grid_rad")
        x = grid.r_grid_mm * np.cos(grid.theta_grid_rad)
        y = grid.r_grid_mm * np.sin(grid.theta_grid_rad)
        return x, y
    if grid.kind == "wafer_1d_radial":
        x = np.asarray(grid.r_grid_mm, dtype=float)
        y = np.zeros_like(x, dtype=float)
        return x, y
    raise ValueError(f"Unsupported grid kind for XY projection: {grid.kind}")


def _nearest_sample(
    meas: MeasurementMap,
    xq: np.ndarray,
    yq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.abs(meas.x_mm[None, None, :] - xq[..., None]).argmin(axis=-1)
    y_idx = np.abs(meas.y_mm[None, None, :] - yq[..., None]).argmin(axis=-1)
    sampled = meas.values[y_idx, x_idx]
    valid = np.ones(sampled.shape, dtype=bool)
    if meas.valid_mask is not None:
        valid &= meas.valid_mask[y_idx, x_idx]
    in_bounds = (
        (xq >= meas.x_mm.min())
        & (xq <= meas.x_mm.max())
        & (yq >= meas.y_mm.min())
        & (yq <= meas.y_mm.max())
    )
    valid &= in_bounds
    return sampled, valid


def _bilinear_sample(
    meas: MeasurementMap,
    xq: np.ndarray,
    yq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = meas.x_mm
    y = meas.y_mm
    nx = x.shape[0]
    ny = y.shape[0]
    ix_hi = np.searchsorted(x, xq, side="right")
    iy_hi = np.searchsorted(y, yq, side="right")
    ix_hi = np.clip(ix_hi, 1, nx - 1)
    iy_hi = np.clip(iy_hi, 1, ny - 1)
    ix_lo = ix_hi - 1
    iy_lo = iy_hi - 1

    x0 = x[ix_lo]
    x1 = x[ix_hi]
    y0 = y[iy_lo]
    y1 = y[iy_hi]
    tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq, dtype=float), where=np.abs(x1 - x0) > 0)
    ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq, dtype=float), where=np.abs(y1 - y0) > 0)

    v00 = meas.values[iy_lo, ix_lo]
    v01 = meas.values[iy_lo, ix_hi]
    v10 = meas.values[iy_hi, ix_lo]
    v11 = meas.values[iy_hi, ix_hi]
    sampled = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v01
        + (1.0 - tx) * ty * v10
        + tx * ty * v11
    )

    valid = (
        (xq >= x.min())
        & (xq <= x.max())
        & (yq >= y.min())
        & (yq <= y.max())
    )
    if meas.valid_mask is not None:
        valid &= (
            meas.valid_mask[iy_lo, ix_lo]
            & meas.valid_mask[iy_lo, ix_hi]
            & meas.valid_mask[iy_hi, ix_lo]
            & meas.valid_mask[iy_hi, ix_hi]
        )
    return sampled, valid


def align_measurement_to_grid(
    measurement: MeasurementMap,
    grid: DomainGrid,
    *,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    edge_exclusion_mm: float = 0.0,
    interpolation: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Align and resample measurement values onto a simulation grid."""
    _require_numpy()
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if edge_exclusion_mm < 0.0:
        raise ValueError(f"edge_exclusion_mm must be >= 0, got {edge_exclusion_mm}")

    x_sim, y_sim = _grid_xy(grid)
    theta = np.deg2rad(float(rotation_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    x_local = (x_sim - float(dx_mm)) / float(scale)
    y_local = (y_sim - float(dy_mm)) / float(scale)

    x_query = cos_t * x_local + sin_t * y_local
    y_query = -sin_t * x_local + cos_t * y_local

    method = interpolation.strip().lower()
    if method == "nearest":
        sampled, valid = _nearest_sample(measurement, x_query, y_query)
    elif method == "bilinear":
        sampled, valid = _bilinear_sample(measurement, x_query, y_query)
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation!r}")

    radial_mm = np.sqrt(x_sim**2 + y_sim**2)
    max_radius = max(float(grid.wafer_radius_mm) - float(edge_exclusion_mm), 0.0)
    valid &= radial_mm <= (max_radius + 1.0e-12)

    aligned = sampled.astype(float, copy=True)
    aligned[~valid] = np.nan
    return aligned, valid


__all__ = ["MeasurementMap", "align_measurement_to_grid"]
