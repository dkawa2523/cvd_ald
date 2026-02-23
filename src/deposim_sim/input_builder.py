"""Input loading and role mapping for AIB simulation path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

from .domain import DomainGrid

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for deposim_sim.input_builder")


@dataclass(frozen=True)
class FluentData:
    mode: str
    cref: np.ndarray
    xy: np.ndarray
    time: np.ndarray | None
    species: tuple[str, ...]


def _validate_fluent_shapes(mode: str, cref: np.ndarray, xy: np.ndarray, time: np.ndarray | None, species: Sequence[str]) -> None:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"fluent xy must be shape [n_pts,2], got {xy.shape}")
    n_pts = int(xy.shape[0])
    n_species = len(species)
    if mode == "steady":
        if cref.ndim != 2:
            raise ValueError(f"steady cref must be shape [n_pts,n_species], got {cref.shape}")
        if cref.shape != (n_pts, n_species):
            raise ValueError(
                f"steady cref shape mismatch: expected {(n_pts, n_species)} from xy/species, got {cref.shape}"
            )
        if time is not None:
            raise ValueError("steady fluent input must not include time array")
    elif mode == "transient":
        if cref.ndim != 3:
            raise ValueError(f"transient cref must be shape [n_t,n_pts,n_species], got {cref.shape}")
        if cref.shape[1] != n_pts or cref.shape[2] != n_species:
            raise ValueError(
                "transient cref shape mismatch: expected [n_t,n_pts,n_species] aligned to xy/species"
            )
        if time is None:
            raise ValueError("transient fluent input requires time array")
        if time.ndim != 1:
            raise ValueError(f"transient time must be 1D, got {time.shape}")
        if time.shape[0] != cref.shape[0]:
            raise ValueError("transient time length must match cref first axis")
    else:
        raise ValueError("fluent mode must be steady|transient")


def load_fluent_npz_v2(
    *,
    path: str | Path,
    mode: str,
    keys: Any,
    species: Sequence[str],
) -> FluentData:
    """Load Fluent NPZ and validate shape contract for AIB simulation."""

    _require_numpy()
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Fluent NPZ not found: {resolved}")

    with np.load(resolved, allow_pickle=False) as data:
        cref = np.asarray(data[getattr(keys, "cref", "cref")], dtype=float)
        xy = np.asarray(data[getattr(keys, "xy", "xy")], dtype=float)
        time = None
        if mode == "transient":
            time = np.asarray(data[getattr(keys, "time", "time")], dtype=float)

    if np.any(cref < 0.0):
        warnings.warn("Negative Fluent concentration values were clipped to zero.", RuntimeWarning, stacklevel=2)
        cref = np.clip(cref, 0.0, np.inf)

    _validate_fluent_shapes(mode, cref, xy, time, species)
    return FluentData(mode=mode, cref=cref, xy=xy, time=time, species=tuple(str(s) for s in species))


def build_domain_from_fluent_xy(
    *,
    xy: np.ndarray,
    xy_unit: str,
    wafer_radius_mm: float,
) -> DomainGrid:
    """Materialize a lightweight DomainGrid from Fluent XY points."""

    _require_numpy()
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be [n_pts,2]")
    if wafer_radius_mm <= 0.0:
        raise ValueError("wafer_radius_mm must be > 0")

    xy_arr = np.asarray(xy, dtype=float)
    if xy_unit == "m":
        xy_mm = xy_arr * 1000.0
    elif xy_unit == "mm":
        xy_mm = xy_arr
    else:
        raise ValueError("xy_unit must be mm|m")

    x_mm = xy_mm[:, 0]
    y_mm = xy_mm[:, 1]
    r_mm = np.sqrt(x_mm**2 + y_mm**2)
    n_pts = int(r_mm.shape[0])

    if n_pts < 1:
        raise ValueError("xy must contain at least one point")

    order = np.argsort(r_mm)
    r_sorted = r_mm[order]
    r_edges = np.zeros(n_pts + 1, dtype=float)
    if n_pts == 1:
        r_edges[0] = 0.0
        r_edges[1] = max(float(r_sorted[0]), 1.0e-9)
    else:
        r_edges[1:-1] = 0.5 * (r_sorted[:-1] + r_sorted[1:])
        r_edges[0] = 0.0
        r_edges[-1] = max(float(wafer_radius_mm), float(r_sorted[-1]))

    area_each = np.pi * float(wafer_radius_mm) ** 2 / float(n_pts)
    area_weights = np.full((n_pts,), area_each, dtype=float)
    edge_mask = r_mm <= (float(wafer_radius_mm) + 1.0e-12)

    return DomainGrid(
        kind="from_fluent_xy",
        wafer_radius_mm=float(wafer_radius_mm),
        r_mm=r_mm,
        r_edges_mm=r_edges,
        dr_mm=np.diff(r_edges),
        r_grid_mm=r_mm,
        area_weights_mm2=area_weights,
        edge_mask=edge_mask,
        x_mm=x_mm,
        y_mm=y_mm,
        x_grid_mm=x_mm,
        y_grid_mm=y_mm,
    )


def _species_index_map(species: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(species)}


def apply_roles(
    *,
    cref: np.ndarray,
    species: Sequence[str],
    role_a: str,
    role_i: str | None,
    role_b: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map species-major concentration tensor to A/I/B arrays."""

    _require_numpy()
    idx = _species_index_map(species)
    if role_a not in idx:
        raise ValueError(f"role A species {role_a!r} is not in species list")
    if role_i is not None and role_i not in idx:
        raise ValueError(f"role I species {role_i!r} is not in species list")
    if role_b is not None and role_b not in idx:
        raise ValueError(f"role B species {role_b!r} is not in species list")

    if len({x for x in (role_a, role_i, role_b) if x is not None}) != len([x for x in (role_a, role_i, role_b) if x is not None]):
        raise ValueError("Roles A/I/B must be disjoint")

    cref_arr = np.asarray(cref, dtype=float)
    if cref_arr.ndim not in {2, 3}:
        raise ValueError(f"cref must be 2D or 3D, got {cref_arr.shape}")

    c_a = cref_arr[..., idx[role_a]]
    c_i = np.zeros_like(c_a, dtype=float) if role_i is None else cref_arr[..., idx[role_i]]
    c_b = np.zeros_like(c_a, dtype=float) if role_b is None else cref_arr[..., idx[role_b]]
    return c_a, c_i, c_b


__all__ = [
    "FluentData",
    "load_fluent_npz_v2",
    "build_domain_from_fluent_xy",
    "apply_roles",
]
