"""Validation helpers for AIB simulation configuration."""

from __future__ import annotations

from typing import Any
import warnings


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

    if sim.time_mode not in {"steady", "transient"}:
        raise ValueError("sim.time_mode must be steady|transient")
    if sim.inputs.fluent.mode not in {"steady", "transient"}:
        raise ValueError("sim.inputs.fluent.mode must be steady|transient")
    if sim.time_mode != sim.inputs.fluent.mode:
        raise ValueError("sim.time_mode and sim.inputs.fluent.mode must match")

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
