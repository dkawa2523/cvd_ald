"""Simulation package for deposition models."""

__version__ = "0.1.0"

from .domain import (
    DomainGrid,
    build_domain_grid,
    build_wafer_1d_radial,
    build_wafer_2d_xy,
    build_wafer_2d_polar,
    edge_exclusion_mask,
    radial_profile,
)
from .pipeline import SimRunResult, run_from_run_spec

__all__ = [
    "__version__",
    "DomainGrid",
    "SimRunResult",
    "build_domain_grid",
    "build_wafer_1d_radial",
    "build_wafer_2d_xy",
    "build_wafer_2d_polar",
    "edge_exclusion_mask",
    "radial_profile",
    "run_from_run_spec",
]
