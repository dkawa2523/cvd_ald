"""Validation helpers for AIB simulation configuration."""

from __future__ import annotations

from typing import Any
import warnings

_SUPPORTED_DOMAIN_KINDS = {"from_fluent_xy", "wafer_2d_xy", "wafer_2d_polar", "wafer_1d_radial"}


def _validate_domain(sim: Any) -> None:
    domain = sim.domain
    kind = str(getattr(domain, "kind", "from_fluent_xy"))
    if kind not in _SUPPORTED_DOMAIN_KINDS:
        raise ValueError(f"sim.domain.kind must be one of {_SUPPORTED_DOMAIN_KINDS}")

    wafer_radius = float(getattr(domain, "wafer_radius_mm", 0.0))
    if wafer_radius <= 0.0:
        raise ValueError("sim.domain.wafer_radius_mm must be > 0")

    edge_exclusion = float(getattr(domain, "edge_exclusion_mm", 0.0))
    if edge_exclusion < 0.0:
        raise ValueError("sim.domain.edge_exclusion_mm must be >= 0")

    if kind in {"wafer_2d_xy", "wafer_2d_polar", "wafer_1d_radial"} and int(getattr(domain, "nr", 0)) < 2:
        raise ValueError("sim.domain.nr must be >= 2 for structured domains")
    if kind == "wafer_2d_polar" and int(getattr(domain, "ntheta", 0)) < 2:
        raise ValueError("sim.domain.ntheta must be >= 2 when domain.kind=wafer_2d_polar")
    if kind == "wafer_2d_xy":
        if int(getattr(domain, "nx", 0)) < 2:
            raise ValueError("sim.domain.nx must be >= 2 when domain.kind=wafer_2d_xy")
        if int(getattr(domain, "ny", 0)) < 2:
            raise ValueError("sim.domain.ny must be >= 2 when domain.kind=wafer_2d_xy")


def validate_roles_v2(run_spec: Any) -> None:
    """Validate role contract: A required, I/B optional, all disjoint."""

    sim = getattr(run_spec, "sim", run_spec)
    roles = sim.roles
    species = list(sim.inputs.fluent.species)

    if not roles.A:
        raise ValueError("sim.roles.A is required")
    if roles.A not in species:
        raise ValueError("sim.roles.A must be in sim.inputs.fluent.species")
    if roles.I is not None and roles.I not in species:
        raise ValueError("sim.roles.I must be in sim.inputs.fluent.species")
    if roles.B is not None and roles.B not in species:
        raise ValueError("sim.roles.B must be in sim.inputs.fluent.species")

    selected = [x for x in (roles.A, roles.I, roles.B) if x is not None]
    if len(set(selected)) != len(selected):
        raise ValueError("sim.roles A/I/B must be disjoint")


def validate_sim_spec_v2(run_spec: Any) -> None:
    """Validate AIB config contract and order constraints."""

    sim = getattr(run_spec, "sim", run_spec)
    validate_roles_v2(sim)
    _validate_domain(sim)

    if sim.time_mode not in {"steady", "transient"}:
        raise ValueError("sim.time_mode must be steady|transient")
    if sim.inputs.fluent.mode not in {"steady", "transient"}:
        raise ValueError("sim.inputs.fluent.mode must be steady|transient")
    if sim.time_mode != sim.inputs.fluent.mode:
        raise ValueError("sim.time_mode and sim.inputs.fluent.mode must match")

    transport = dict(getattr(sim.model.params, "transport", {}) or {})
    km_source = str(transport.get("km_source", "fit_scalar")).strip().lower()
    if km_source not in {"fit_scalar", "from_cfd_flux_sink"}:
        raise ValueError("sim.model.params.transport.km_source must be fit_scalar|from_cfd_flux_sink")

    if km_source == "from_cfd_flux_sink":
        from_flux = dict(transport.get("from_cfd_flux_sink", {}) or {})
        policy = str(from_flux.get("flux_negative_policy", "error")).strip().lower()
        if policy not in {"error", "clip_to_zero", "allow"}:
            raise ValueError("flux_negative_policy must be error|clip_to_zero|allow")
        eps_cref = float(from_flux.get("eps_cref", 1.0e-12))
        if eps_cref <= 0.0:
            raise ValueError("from_cfd_flux_sink.eps_cref must be > 0")
        clip = from_flux.get("km_clip", [1.0e-8, 1.0e4])
        if not isinstance(clip, (list, tuple)) or len(clip) < 2:
            raise ValueError("from_cfd_flux_sink.km_clip must be [min,max]")
        clip_min = float(clip[0])
        clip_max = float(clip[1])
        if clip_min > clip_max:
            raise ValueError("from_cfd_flux_sink.km_clip min must be <= max")

    if len(sim.inputs.fluent.species) > 10:
        warnings.warn(
            "sim.inputs.fluent.species > 10 may make role enumeration expensive in v1.",
            RuntimeWarning,
            stacklevel=2,
        )

    orders = sim.model.orders
    has_b = sim.roles.B is not None
    total_order = orders.reaction_site_order_A + orders.reaction_site_order_star + (1 if has_b else 0)
    if total_order > int(orders.enforce_total_order_le):
        raise ValueError("order constraint violated: p_A + p_* + m_B > 3")


# Backward-compatible name used by existing callers.
def validate_run_spec(run_spec: Any) -> None:
    validate_sim_spec_v2(run_spec)


__all__ = ["validate_roles_v2", "validate_sim_spec_v2", "validate_run_spec"]
