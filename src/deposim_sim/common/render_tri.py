"""Triangulation-based rendering helpers for unstructured XY data."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import matplotlib.colors as mcolors
    import matplotlib.tri as mtri
except ModuleNotFoundError:  # pragma: no cover
    mcolors = None  # type: ignore[assignment]
    mtri = None  # type: ignore[assignment]


def tri_quality(x: np.ndarray, y: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Return triangle quality in [0, 1] (1=equilateral)."""

    x1 = x[tri[:, 0]]
    x2 = x[tri[:, 1]]
    x3 = x[tri[:, 2]]
    y1 = y[tri[:, 0]]
    y2 = y[tri[:, 1]]
    y3 = y[tri[:, 2]]
    a = np.hypot(x2 - x1, y2 - y1)
    b = np.hypot(x3 - x2, y3 - y2)
    c = np.hypot(x1 - x3, y1 - y3)
    area2 = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return (2.0 * np.sqrt(3.0) * area2) / np.maximum(a * a + b * b + c * c, 1.0e-30)


def render_unstructured_map(
    ax: Any,
    *,
    xy_mm: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray | None,
    cmap: str,
    discrete: bool = False,
    min_quality: float = 1.0e-3,
) -> Any:
    """Render unstructured point data with triangulation fallback to scatter."""

    if np is None:
        raise RuntimeError("NumPy is required for unstructured rendering")

    x = np.asarray(xy_mm[:, 0], dtype=float)
    y = np.asarray(xy_mm[:, 1], dtype=float)
    v = np.asarray(values, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool).reshape(-1)

    if int(np.sum(valid)) < 3 or mtri is None:
        mesh = ax.scatter(x[valid], y[valid], c=v[valid], cmap=cmap, s=14)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    xv = x[valid]
    yv = y[valid]
    vv = v[valid]
    try:
        triang = mtri.Triangulation(xv, yv)
    except Exception:
        mesh = ax.scatter(xv, yv, c=vv, cmap=cmap, s=14)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    if triang.triangles.size == 0:
        mesh = ax.scatter(xv, yv, c=vv, cmap=cmap, s=14)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    quality = tri_quality(xv, yv, triang.triangles)
    triang.set_mask(quality < float(min_quality))
    if triang.mask is not None and bool(np.all(triang.mask)):
        mesh = ax.scatter(xv, yv, c=vv, cmap=cmap, s=14)
    else:
        if discrete and mcolors is not None:
            finite = np.asarray(vv[np.isfinite(vv)], dtype=float)
            unique = np.unique(finite)
            if unique.size > 0:
                vmin = float(np.min(unique))
                vmax = float(np.max(unique))
                boundaries = np.arange(np.floor(vmin) - 0.5, np.ceil(vmax) + 1.5, 1.0)
                norm = mcolors.BoundaryNorm(boundaries, ncolors=max(len(boundaries) - 1, 1), clip=True)
                mesh = ax.tripcolor(triang, vv, shading="flat", cmap=cmap, norm=norm)
            else:
                mesh = ax.tripcolor(triang, vv, shading="flat", cmap=cmap)
        else:
            mesh = ax.tripcolor(triang, vv, shading="flat", cmap=cmap)
        ax.scatter(xv, yv, c="k", s=4, alpha=0.25, linewidths=0.0)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return mesh
