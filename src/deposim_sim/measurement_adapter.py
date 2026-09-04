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


def compare_point_observations(
    *,
    prediction_nm: Any,
    model_xy_mm: Any,
    measured: Any,
    measurement_xy_mm: Any,
    align: dict[str, Any],
    quantity: str = "thickness",
    duration_s: float = 1.0,
    initial_nm: Any = 0.0,
    sigma: Any | None = None,
) -> dict[str, Any]:
    """Compare at original measurement points, once per observation.

    Map plots may resample measurements onto the simulation mesh; fitting must
    instead sample the prediction at the observations. All returned values are
    in equivalent final-thickness units, including uncertainty for mean rates.
    """
    _require_numpy()
    model_xy = np.asarray(model_xy_mm, dtype=float)
    xy = np.asarray(measurement_xy_mm, dtype=float).copy()
    target = np.asarray(measured, dtype=float).reshape(-1).copy()
    prediction = np.asarray(prediction_nm, dtype=float).reshape(-1)
    if xy.shape != (target.size, 2) or model_xy.shape != (prediction.size, 2):
        raise ValueError("observation/model coordinates must match their values")
    if not target.size or not prediction.size:
        raise ValueError("observation/model points must be nonempty")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(model_xy)):
        raise ValueError("observation/model coordinates must be finite")
    enabled = bool(align.get("enable", False))
    if enabled:
        angle = np.deg2rad(float(align.get("rotate_deg", 0.0)))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        scale = float(align.get("scale", 1.0))
        if scale <= 0.0:
            raise ValueError("measurement alignment scale must be positive")
        xy = scale * (xy @ rotation.T) + np.asarray(align.get("shift_mm", [0.0, 0.0]), dtype=float)
        distances = np.sum((xy[:, None, :] - model_xy[None, :, :]) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)
        nearest = np.sqrt(distances[np.arange(target.size), indices])
    else:
        if target.size != prediction.size:
            raise ValueError("unaligned observation and prediction sizes must match")
        indices = np.arange(target.size)
        nearest = np.zeros(target.size)
    predicted = prediction[indices]
    valid = np.isfinite(target)
    if enabled and align.get("mask_radius_mm") is not None:
        valid &= np.linalg.norm(xy, axis=1) <= float(align["mask_radius_mm"])
    distance_rejected = 0
    if enabled and align.get("max_nearest_distance_mm") is not None:
        max_distance = float(align["max_nearest_distance_mm"])
        if max_distance < 0:
            raise ValueError("max_nearest_distance_mm must be nonnegative")
        distance_rejected = int(np.sum(valid & (nearest > max_distance)))
        valid &= nearest <= max_distance
    if np.any(valid & ~np.isfinite(predicted)):
        raise ValueError("non-finite prediction at a valid observation")
    if quantity not in {"thickness", "mean_rate"}:
        raise ValueError("measurement quantity must be thickness|mean_rate")
    factor = 1.0
    if quantity == "mean_rate":
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("mean_rate observations require positive simulation duration")
        factor = float(duration_s)
        initial = np.broadcast_to(np.asarray(initial_nm, dtype=float), np.asarray(prediction_nm).shape).ravel()
        target = initial[indices] + factor * target
    uncertainty = None
    if sigma is not None:
        uncertainty = factor * np.broadcast_to(np.asarray(sigma, dtype=float), target.shape).copy()
        if np.any(valid & (~np.isfinite(uncertainty) | (uncertainty <= 0.0))):
            raise ValueError("measurement sigma must be finite and positive at valid observations")
        uncertainty = uncertainty[valid]
    return {
        "prediction_nm": predicted[valid],
        "target_nm": target[valid],
        "residual_nm": (predicted - target)[valid],
        "sigma_nm": uncertainty,
        "xy_mm": xy[valid],
        "quantity": quantity,
        "count": int(np.sum(valid)),
        "distance_rejected_count": distance_rejected,
        "mean_distance_mm": float(np.mean(nearest[valid])) if np.any(valid) else float("nan"),
        "max_distance_mm": float(np.max(nearest[valid])) if np.any(valid) else float("nan"),
    }


def observation_residuals(result: Any) -> np.ndarray:
    """The exact residual vector used by fitting and sensitivity calculations."""
    observed = result.diagnostics.get("observation")
    if observed is not None:
        residual = np.asarray(observed["residual_nm"], dtype=float)
        sigma = observed.get("sigma_nm")
        return residual if sigma is None else residual / np.asarray(sigma, dtype=float)
    residual = np.asarray(result.fields.get("residual_nm", []), dtype=float).ravel()
    return residual[np.isfinite(residual)]


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


def align_point_measurement_to_points(
    *,
    values: np.ndarray,
    source_xy_mm: np.ndarray,
    target_xy_mm: np.ndarray,
    shift_mm: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    mask_radius_mm: float | None = None,
    max_nearest_distance_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and nearest-resample unstructured point measurements to target points."""

    _require_numpy()
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")

    src_xy = np.asarray(source_xy_mm, dtype=float)
    tgt_xy = np.asarray(target_xy_mm, dtype=float)
    vals = np.asarray(values, dtype=float).reshape(-1)

    if src_xy.ndim != 2 or src_xy.shape[1] != 2:
        raise ValueError(f"source_xy_mm must be shape [n,2], got {src_xy.shape}")
    if tgt_xy.ndim != 2 or tgt_xy.shape[1] != 2:
        raise ValueError(f"target_xy_mm must be shape [m,2], got {tgt_xy.shape}")
    if vals.shape[0] != src_xy.shape[0]:
        raise ValueError("values length must match source_xy_mm length")
    if src_xy.shape[0] == 0:
        raise ValueError("source_xy_mm must contain at least one point")

    dx_mm, dy_mm = float(shift_mm[0]), float(shift_mm[1])
    theta = np.deg2rad(float(rotation_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    x_local = (tgt_xy[:, 0] - dx_mm) / float(scale)
    y_local = (tgt_xy[:, 1] - dy_mm) / float(scale)
    x_query = cos_t * x_local + sin_t * y_local
    y_query = -sin_t * x_local + cos_t * y_local
    query = np.stack([x_query, y_query], axis=1)

    delta = query[:, None, :] - src_xy[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    nearest = np.argmin(dist2, axis=1)
    nearest_distance_mm = np.sqrt(np.min(dist2, axis=1))

    aligned = vals[nearest].astype(float, copy=True)
    valid = np.isfinite(aligned)
    if mask_radius_mm is not None:
        radius = np.sqrt(np.sum(np.square(tgt_xy), axis=1))
        valid &= radius <= (float(mask_radius_mm) + 1.0e-12)
    if max_nearest_distance_mm is not None:
        if float(max_nearest_distance_mm) < 0.0:
            raise ValueError("max_nearest_distance_mm must be >= 0")
        valid &= nearest_distance_mm <= (float(max_nearest_distance_mm) + 1.0e-12)
    aligned[~valid] = np.nan
    return aligned, valid


def point_alignment_distance_stats(
    *,
    source_xy_mm: np.ndarray,
    target_xy_mm: np.ndarray,
    shift_mm: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    max_nearest_distance_mm: float | None = None,
) -> dict[str, float | int | None]:
    """Summarize nearest-neighbor distances for alignment quality review."""

    _require_numpy()
    src_xy = np.asarray(source_xy_mm, dtype=float)
    tgt_xy = np.asarray(target_xy_mm, dtype=float)
    if src_xy.ndim != 2 or src_xy.shape[1] != 2 or src_xy.shape[0] == 0:
        raise ValueError("source_xy_mm must be non-empty shape [n,2]")
    if tgt_xy.ndim != 2 or tgt_xy.shape[1] != 2:
        raise ValueError("target_xy_mm must be shape [m,2]")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")

    dx_mm, dy_mm = float(shift_mm[0]), float(shift_mm[1])
    theta = np.deg2rad(float(rotation_deg))
    x_local = (tgt_xy[:, 0] - dx_mm) / float(scale)
    y_local = (tgt_xy[:, 1] - dy_mm) / float(scale)
    query = np.stack(
        [
            np.cos(theta) * x_local + np.sin(theta) * y_local,
            -np.sin(theta) * x_local + np.cos(theta) * y_local,
        ],
        axis=1,
    )
    delta = query[:, None, :] - src_xy[None, :, :]
    distances = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    threshold = None if max_nearest_distance_mm is None else float(max_nearest_distance_mm)
    rejected = 0 if threshold is None else int(np.sum(distances > threshold))
    return {
        "mean_distance_mm": float(np.mean(distances)) if distances.size else float("nan"),
        "p95_distance_mm": float(np.percentile(distances, 95.0)) if distances.size else float("nan"),
        "max_distance_mm": float(np.max(distances)) if distances.size else float("nan"),
        "max_nearest_distance_mm": threshold,
        "distance_rejected_count": rejected,
        "target_count": int(distances.size),
    }


__all__ = [
    "compare_point_observations",
    "observation_residuals",
    "MeasurementMap",
    "align_measurement_to_grid",
    "align_point_measurement_to_points",
    "point_alignment_distance_stats",
]
