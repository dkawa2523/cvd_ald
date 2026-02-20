"""KPI metric helpers for run summaries and DOE ranking."""

from __future__ import annotations

from typing import Any

from .domain import DomainGrid

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for KPI metric calculations.")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def compute_kpi_metrics(
    thickness: np.ndarray,
    grid: DomainGrid,
    *,
    spec_min: float | None = None,
    spec_max: float | None = None,
    ring_count: int = 5,
) -> dict[str, Any]:
    """Compute KPI metrics from a thickness map on a wafer grid."""
    _require_numpy()
    data = np.asarray(thickness, dtype=float)
    if data.shape != grid.shape:
        raise ValueError(f"thickness shape {data.shape} does not match grid shape {grid.shape}")

    active = np.asarray(grid.edge_mask, dtype=bool)
    weights = np.asarray(grid.area_weights_mm2, dtype=float)
    weighted_total = float(np.sum(weights[active]))
    if weighted_total <= 0.0:
        raise ValueError("active wafer area must be positive")

    valid = active & np.isfinite(data)
    if not np.any(valid):
        raise ValueError("thickness map has no finite values on active wafer area")

    values = data[valid]
    value_weights = weights[valid]
    mean = float(np.average(values, weights=value_weights))
    t_min = float(np.min(values))
    t_max = float(np.max(values))
    nu_percent = float(((t_max - t_min) / (2.0 * max(abs(mean), 1.0e-30))) * 100.0)

    r = np.asarray(grid.r_grid_mm, dtype=float)
    radius = float(grid.wafer_radius_mm)
    ring_count = max(int(ring_count), 1)
    ring_edges = np.linspace(0.0, radius, ring_count + 1)
    ring_means: list[float | None] = []
    for idx in range(ring_count):
        lo = ring_edges[idx]
        hi = ring_edges[idx + 1]
        shell = valid & (r >= lo) & (r < hi)
        if idx == ring_count - 1:
            shell = valid & (r >= lo) & (r <= hi + 1.0e-12)
        if not np.any(shell):
            ring_means.append(None)
            continue
        ring_means.append(float(np.average(data[shell], weights=weights[shell])))

    out_of_spec = np.zeros(data.shape, dtype=bool)
    if spec_min is not None:
        out_of_spec |= valid & (data < float(spec_min))
    if spec_max is not None:
        out_of_spec |= valid & (data > float(spec_max))
    out_of_spec_fraction = float(np.sum(weights[out_of_spec]) / weighted_total)

    center_mask = valid & (r <= 0.1 * radius)
    edge_mask = valid & (r >= 0.9 * radius)
    center_mean = float(np.average(data[center_mask], weights=weights[center_mask])) if np.any(center_mask) else None
    edge_mean = float(np.average(data[edge_mask], weights=weights[edge_mask])) if np.any(edge_mask) else None

    return {
        "nu_percent": nu_percent,
        "thickness_min": t_min,
        "thickness_mean": mean,
        "thickness_max": t_max,
        "center_mean": center_mean,
        "edge_mean": edge_mean,
        "center_edge_delta": None if center_mean is None or edge_mean is None else float(edge_mean - center_mean),
        "ring_count": ring_count,
        "ring_means": ring_means,
        "out_of_spec_area_fraction": out_of_spec_fraction,
        "spec_min": _safe_float(spec_min),
        "spec_max": _safe_float(spec_max),
    }


__all__ = ["compute_kpi_metrics"]
