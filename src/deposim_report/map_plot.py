"""Common 2D map rendering helpers used by report modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deposim_sim.common.render_tri import render_unstructured_map
from deposim_sim.domain import DomainGrid

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def require_plot_deps() -> None:
    if np is None or plt is None:
        raise RuntimeError("NumPy and Matplotlib are required for report generation.")


def map2d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return np.repeat(arr[:, None], 2, axis=1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array for map plot, got shape {arr.shape}")
    return arr


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("center coordinates must be 1D with at least two points")
    edges = np.empty(arr.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
    edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return edges


def draw_map(
    ax: Any,
    *,
    grid: DomainGrid,
    value: Any,
    cmap: str = "viridis",
    xy_mm: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    discrete: bool = False,
) -> Any:
    if grid.kind == "from_fluent_xy":
        if xy_mm is None and grid.x_grid_mm is not None and grid.y_grid_mm is not None:
            xy_mm = np.stack([np.asarray(grid.x_grid_mm, dtype=float), np.asarray(grid.y_grid_mm, dtype=float)], axis=1)
        if xy_mm is None:
            raise ValueError("xy_mm is required for from_fluent_xy map rendering")
        mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else np.asarray(grid.edge_mask, dtype=bool)
        return render_unstructured_map(
            ax,
            xy_mm=np.asarray(xy_mm, dtype=float),
            values=np.asarray(value, dtype=float),
            valid_mask=mask,
            cmap=cmap,
            discrete=discrete,
        )

    data = map2d(value)
    if grid.kind == "wafer_2d_polar" and grid.theta_edges_rad is not None and data.shape == grid.shape:
        r_edges = np.asarray(grid.r_edges_mm, dtype=float)
        theta_edges = np.asarray(grid.theta_edges_rad, dtype=float)
        rr, tt = np.meshgrid(r_edges, theta_edges, indexing="ij")
        xx = rr * np.cos(tt)
        yy = rr * np.sin(tt)
        masked = np.ma.array(data, mask=~np.asarray(grid.edge_mask, dtype=bool))
        mesh = ax.pcolormesh(xx, yy, masked, shading="auto", cmap=cmap)
        radius = float(grid.wafer_radius_mm)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    if grid.kind == "wafer_2d_xy" and grid.x_mm is not None and grid.y_mm is not None and data.shape == grid.shape:
        x_edges = _centers_to_edges(np.asarray(grid.x_mm, dtype=float))
        y_edges = _centers_to_edges(np.asarray(grid.y_mm, dtype=float))
        xx, yy = np.meshgrid(x_edges, y_edges, indexing="xy")
        masked = np.ma.array(data, mask=~np.asarray(grid.edge_mask, dtype=bool))
        mesh = ax.pcolormesh(xx, yy, masked, shading="auto", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    mesh = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel("grid x-index")
    ax.set_ylabel("grid y-index")
    return mesh


def save_map(
    path: Path,
    *,
    grid: DomainGrid,
    value: Any,
    title: str,
    cmap: str = "viridis",
    xy_mm: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    discrete: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = draw_map(
        ax,
        grid=grid,
        value=value,
        cmap=cmap,
        xy_mm=xy_mm,
        valid_mask=valid_mask,
        discrete=discrete,
    )
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


__all__ = ["draw_map", "map2d", "require_plot_deps", "save_map"]
