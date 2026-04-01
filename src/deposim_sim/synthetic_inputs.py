"""Deterministic synthetic input generation for smoke and local tests."""

from __future__ import annotations

from typing import Any
import warnings

from .domain import DomainGrid
from .physics.cvd_steady import FieldBundle

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for synthetic input generation.")


def synthetic_pattern(case: str, grid: DomainGrid, *, random_seed: int = 0) -> Any:
    """Build a deterministic, nonnegative synthetic spatial pattern."""
    _require_numpy()
    name = case.strip().lower()
    r_norm = np.asarray(grid.r_grid_mm, dtype=float) / max(float(grid.wafer_radius_mm), 1.0e-12)

    if name == "uniform":
        return np.ones(grid.shape, dtype=float)
    if name == "radial_gradient":
        return np.clip(1.15 - 0.5 * r_norm, 0.05, np.inf)
    if name == "edge_depleted":
        return np.clip(1.10 - 0.85 * (r_norm**2), 0.05, np.inf)
    if name == "seeded_perturbation":
        rng = np.random.default_rng(int(random_seed))
        perturb = rng.normal(loc=0.0, scale=0.02, size=grid.shape)
        return np.clip(1.0 + perturb, 0.05, np.inf)
    raise ValueError(
        f"Unsupported synthetic_case '{case}'. "
        "Expected one of: uniform, radial_gradient, edge_depleted, seeded_perturbation."
    )


def build_synthetic_field_bundle(run_spec: Any, grid: DomainGrid) -> FieldBundle:
    """Create deterministic synthetic ``FieldBundle`` from RunSpec-like config."""
    _require_numpy()
    warnings.warn(
        "synthetic input bundle path is legacy-oriented; active workflows should use Fluent-backed inputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    pattern = synthetic_pattern(
        run_spec.inputs.synthetic_case,
        grid,
        random_seed=getattr(run_spec, "random_seed", 0),
    )
    c_ref_base = float(run_spec.inputs.c_ref_mol_m3)
    c_ref = {
        species: c_ref_base * pattern * (1.0 + 0.03 * idx)
        for idx, species in enumerate(run_spec.reference_plane.species)
    }
    return FieldBundle(
        C_ref=c_ref,
        T=np.full(grid.shape, float(run_spec.inputs.temperature_k), dtype=float),
        scalars={"omega_rad_s": float(run_spec.inputs.omega_rad_s)},
    )


__all__ = ["build_synthetic_field_bundle", "synthetic_pattern"]
