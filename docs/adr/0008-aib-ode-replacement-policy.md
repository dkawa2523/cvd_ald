# ADR 0008: AIB-ODE Replacement Policy

- Date: 2026-02-22
- Status: Superseded in part by ADR 0019
- Decision Task: D-006

## Context

`model_equation_new.md` defines a unified AIB-ODE model where species roles are constrained to A(required), I(optional <=1), B(optional <=1). The existing runtime is centered on `power_law + root_solve` progression variable routing, which conflicts with the new policy intent.

## Decision

Adopt full replacement to AIB-ODE for the legacy route retirement stage. ADR
0019 supersedes the long-term public model contract: future CVD/ALD work should
treat role-based modeling and role discovery as primary, with `aib_ode` kept as
a compatibility implementation.

1. Runtime public model contract is fixed to `sim.model.name = aib_ode` for
   this migration stage only.
2. Role contract is fixed to:
   - A required, single species
   - I and B each `null` or one species
   - A/I/B disjoint
   - unused species allowed
3. B-order is fixed by role presence:
   - B is null => `m_B = 0`
   - B is set => `m_B = 1`
4. Order constraint is fixed:
   - `p_A + p_* + m_B <= 3`
5. Solver policy:
   - ODE is primary state engine
   - `implicit_euler_bisect` for theta in `[0,1]`
   - non-bracket fallback is clamped explicit update with diagnostics
6. Legacy route (`power_law`, `lhhw`, `root_solve`) is retired from primary path.

## Consequences

- Breaking change for legacy YAML model selectors (`kinetics_name`, `mass_transfer_name`, `root_solver_name`).
- Validation/reporting/optimization contracts are rewritten around AIB roles and A/AI/AB/AIB class comparison.
- Migration is staged by tasks `P3-001..P3-007` after this ADR.
