"""Domain grid builders and mapping utilities for wafer simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - dependency guard for minimal environments
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore[assignment]


_SUPPORTED_DOMAIN_KINDS = {"wafer_2d_polar", "wafer_1d_radial", "wafer_2d_xy"}


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for deposim_sim.domain grid utilities. "
            "Install numpy to build domain grids."
        )


@dataclass(frozen=True)
class DomainGrid:
    """Materialized coordinate grid for a simulation domain."""

    kind: str
    wafer_radius_mm: float
    r_mm: np.ndarray
    r_edges_mm: np.ndarray
    dr_mm: np.ndarray
    r_grid_mm: np.ndarray
    area_weights_mm2: np.ndarray
    edge_mask: np.ndarray
    theta_rad: np.ndarray | None = None
    theta_edges_rad: np.ndarray | None = None
    dtheta_rad: float | None = None
    theta_grid_rad: np.ndarray | None = None
    x_mm: np.ndarray | None = None
    y_mm: np.ndarray | None = None
    x_grid_mm: np.ndarray | None = None
    y_grid_mm: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape for values aligned with this domain."""
        return tuple(self.r_grid_mm.shape)


def _read_domain_value(domain_spec: Any, name: str) -> Any:
    if not hasattr(domain_spec, name):
        raise ValueError(f"domain spec is missing required field '{name}'")
    return getattr(domain_spec, name)


def _build_radial_axis(wafer_radius_mm: float, nr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_numpy()
    r_edges_mm = np.linspace(0.0, wafer_radius_mm, nr + 1, dtype=float)
    r_mm = 0.5 * (r_edges_mm[:-1] + r_edges_mm[1:])
    dr_mm = np.diff(r_edges_mm)
    return r_edges_mm, r_mm, dr_mm


def _edge_mask_from_radius(r_values_mm: np.ndarray, wafer_radius_mm: float, edge_exclusion_mm: float) -> np.ndarray:
    if edge_exclusion_mm < 0:
        raise ValueError(f"edge_exclusion_mm must be >= 0, got {edge_exclusion_mm}")
    valid_radius_mm = max(wafer_radius_mm - edge_exclusion_mm, 0.0)
    return r_values_mm <= (valid_radius_mm + 1.0e-12)


def build_wafer_2d_polar(
    wafer_radius_mm: float,
    nr: int,
    ntheta: int,
    edge_exclusion_mm: float = 0.0,
) -> DomainGrid:
    """Build a polar wafer grid with area weights and edge mask."""
    _require_numpy()
    if nr < 2:
        raise ValueError(f"nr must be >= 2, got {nr}")
    if ntheta < 2:
        raise ValueError(f"ntheta must be >= 2, got {ntheta}")

    r_edges_mm, r_mm, dr_mm = _build_radial_axis(wafer_radius_mm, nr)
    theta_edges_rad = np.linspace(0.0, 2.0 * np.pi, ntheta + 1, dtype=float)
    theta_rad = 0.5 * (theta_edges_rad[:-1] + theta_edges_rad[1:])
    dtheta_rad = float(theta_edges_rad[1] - theta_edges_rad[0])

    r_grid_mm, theta_grid_rad = np.meshgrid(r_mm, theta_rad, indexing="ij")

    annulus_area_per_radian_mm2 = 0.5 * (r_edges_mm[1:] ** 2 - r_edges_mm[:-1] ** 2)
    sector_area_mm2 = annulus_area_per_radian_mm2[:, None] * dtheta_rad
    area_weights_mm2 = np.broadcast_to(sector_area_mm2, r_grid_mm.shape).copy()

    edge_mask = _edge_mask_from_radius(r_grid_mm, wafer_radius_mm, edge_exclusion_mm)
    return DomainGrid(
        kind="wafer_2d_polar",
        wafer_radius_mm=float(wafer_radius_mm),
        r_mm=r_mm,
        r_edges_mm=r_edges_mm,
        dr_mm=dr_mm,
        r_grid_mm=r_grid_mm,
        theta_rad=theta_rad,
        theta_edges_rad=theta_edges_rad,
        dtheta_rad=dtheta_rad,
        theta_grid_rad=theta_grid_rad,
        area_weights_mm2=area_weights_mm2,
        edge_mask=edge_mask,
    )


def build_wafer_1d_radial(
    wafer_radius_mm: float,
    nr: int,
    edge_exclusion_mm: float = 0.0,
) -> DomainGrid:
    """Build a 1D radial wafer grid with ring-area weights and edge mask."""
    _require_numpy()
    if nr < 2:
        raise ValueError(f"nr must be >= 2, got {nr}")

    r_edges_mm, r_mm, dr_mm = _build_radial_axis(wafer_radius_mm, nr)
    area_weights_mm2 = np.pi * (r_edges_mm[1:] ** 2 - r_edges_mm[:-1] ** 2)

    edge_mask = _edge_mask_from_radius(r_mm, wafer_radius_mm, edge_exclusion_mm)
    return DomainGrid(
        kind="wafer_1d_radial",
        wafer_radius_mm=float(wafer_radius_mm),
        r_mm=r_mm,
        r_edges_mm=r_edges_mm,
        dr_mm=dr_mm,
        r_grid_mm=r_mm.copy(),
        area_weights_mm2=area_weights_mm2,
        edge_mask=edge_mask,
    )


def build_wafer_2d_xy(
    wafer_radius_mm: float,
    nr: int,
    nx: int,
    ny: int,
    edge_exclusion_mm: float = 0.0,
) -> DomainGrid:
    """Build a Cartesian wafer grid clipped by wafer radius and edge exclusion."""
    _require_numpy()
    if nr < 2:
        raise ValueError(f"nr must be >= 2, got {nr}")
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")
    if ny < 2:
        raise ValueError(f"ny must be >= 2, got {ny}")

    r_edges_mm, r_mm, dr_mm = _build_radial_axis(wafer_radius_mm, nr)
    x_edges_mm = np.linspace(-wafer_radius_mm, wafer_radius_mm, nx + 1, dtype=float)
    y_edges_mm = np.linspace(-wafer_radius_mm, wafer_radius_mm, ny + 1, dtype=float)
    x_mm = 0.5 * (x_edges_mm[:-1] + x_edges_mm[1:])
    y_mm = 0.5 * (y_edges_mm[:-1] + y_edges_mm[1:])
    x_grid_mm, y_grid_mm = np.meshgrid(x_mm, y_mm, indexing="xy")
    r_grid_mm = np.sqrt(x_grid_mm**2 + y_grid_mm**2)

    dx_mm = float(x_edges_mm[1] - x_edges_mm[0])
    dy_mm = float(y_edges_mm[1] - y_edges_mm[0])
    area_weights_mm2 = np.full(r_grid_mm.shape, dx_mm * dy_mm, dtype=float)

    inside_wafer = r_grid_mm <= (wafer_radius_mm + 1.0e-12)
    edge_mask = _edge_mask_from_radius(r_grid_mm, wafer_radius_mm, edge_exclusion_mm) & inside_wafer

    return DomainGrid(
        kind="wafer_2d_xy",
        wafer_radius_mm=float(wafer_radius_mm),
        r_mm=r_mm,
        r_edges_mm=r_edges_mm,
        dr_mm=dr_mm,
        r_grid_mm=r_grid_mm,
        area_weights_mm2=area_weights_mm2,
        edge_mask=edge_mask,
        x_mm=x_mm,
        y_mm=y_mm,
        x_grid_mm=x_grid_mm,
        y_grid_mm=y_grid_mm,
    )


def build_domain_grid(domain_spec: Any) -> DomainGrid:
    """Build a DomainGrid from a schema DomainSpec-like object."""
    kind = _read_domain_value(domain_spec, "kind")
    if kind not in _SUPPORTED_DOMAIN_KINDS:
        ordered = ", ".join(sorted(_SUPPORTED_DOMAIN_KINDS))
        raise ValueError(f"Unsupported domain kind '{kind}'. Supported kinds: {{{ordered}}}")

    wafer_radius_mm = float(_read_domain_value(domain_spec, "wafer_radius_mm"))
    edge_exclusion_mm = float(_read_domain_value(domain_spec, "edge_exclusion_mm"))
    nr = int(_read_domain_value(domain_spec, "nr"))

    if kind == "wafer_2d_polar":
        ntheta = int(_read_domain_value(domain_spec, "ntheta"))
        return build_wafer_2d_polar(
            wafer_radius_mm=wafer_radius_mm,
            nr=nr,
            ntheta=ntheta,
            edge_exclusion_mm=edge_exclusion_mm,
        )
    if kind == "wafer_2d_xy":
        nx = int(_read_domain_value(domain_spec, "nx"))
        ny = int(_read_domain_value(domain_spec, "ny"))
        return build_wafer_2d_xy(
            wafer_radius_mm=wafer_radius_mm,
            nr=nr,
            nx=nx,
            ny=ny,
            edge_exclusion_mm=edge_exclusion_mm,
        )

    return build_wafer_1d_radial(
        wafer_radius_mm=wafer_radius_mm,
        nr=nr,
        edge_exclusion_mm=edge_exclusion_mm,
    )


def edge_exclusion_mask(grid: DomainGrid, edge_exclusion_mm: float) -> np.ndarray:
    """Return a mask for a different edge exclusion distance on an existing grid."""
    _require_numpy()
    return _edge_mask_from_radius(grid.r_grid_mm, grid.wafer_radius_mm, edge_exclusion_mm)


def radial_profile(
    values: np.ndarray,
    grid: DomainGrid,
    extra_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a radial profile aligned to `grid.r_mm`."""
    _require_numpy()
    data = np.asarray(values, dtype=float)
    if data.shape != grid.shape:
        raise ValueError(f"values shape {data.shape} does not match grid shape {grid.shape}")

    active_mask = grid.edge_mask
    if extra_mask is not None:
        user_mask = np.asarray(extra_mask, dtype=bool)
        if user_mask.shape != grid.shape:
            raise ValueError(f"extra_mask shape {user_mask.shape} does not match grid shape {grid.shape}")
        active_mask = active_mask & user_mask

    if grid.kind == "wafer_1d_radial":
        profile = data.copy()
        profile[~active_mask] = np.nan
        return grid.r_mm.copy(), profile

    if grid.kind == "wafer_2d_xy":
        profile = np.full(grid.r_mm.shape, np.nan, dtype=float)
        for idx in range(len(grid.r_mm)):
            lo = grid.r_edges_mm[idx]
            hi = grid.r_edges_mm[idx + 1]
            shell_mask = active_mask & (grid.r_grid_mm >= lo) & (grid.r_grid_mm < hi)
            if idx == len(grid.r_mm) - 1:
                shell_mask = active_mask & (grid.r_grid_mm >= lo) & (grid.r_grid_mm <= hi + 1.0e-12)
            if not np.any(shell_mask):
                continue
            shell_weights = grid.area_weights_mm2[shell_mask]
            shell_values = data[shell_mask]
            profile[idx] = float(np.sum(shell_values * shell_weights) / np.sum(shell_weights))
        return grid.r_mm.copy(), profile

    weights = np.where(active_mask, grid.area_weights_mm2, 0.0)
    weighted_sum = np.sum(data * weights, axis=1)
    weight_sum = np.sum(weights, axis=1)
    profile = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(weighted_sum.shape, np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    return grid.r_mm.copy(), profile


__all__ = [
    "DomainGrid",
    "build_domain_grid",
    "build_wafer_1d_radial",
    "build_wafer_2d_xy",
    "build_wafer_2d_polar",
    "edge_exclusion_mask",
    "radial_profile",
]
