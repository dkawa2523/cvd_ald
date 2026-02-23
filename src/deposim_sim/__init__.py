"""Simulation package for deposition models."""

__version__ = "0.2.0"

from .domain import (
    DomainGrid,
    build_domain_grid,
    build_wafer_1d_radial,
    build_wafer_2d_xy,
    build_wafer_2d_polar,
    edge_exclusion_mask,
    radial_profile,
)
from .pipeline import SimRunResult, compose_aib_spec, run_aib_from_config, run_aib_from_spec, run_from_run_spec

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
    "compose_aib_spec",
    "run_aib_from_config",
    "run_aib_from_spec",
    "run_from_run_spec",
]
