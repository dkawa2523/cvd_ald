# ADR 0009: AIB Requirements and Traceability Rewrite

- Date: 2026-02-22
- Status: Accepted
- Decision Task: D-007

## Context

After adopting ADR 0008, requirements and traceability entries tied to `power_law/root_solve` no longer represent runtime behavior. Keeping mixed requirement sets would violate `No silent spec changes` and reduce verification trust.

## Decision

1. Rewrite requirement clauses that prescribe legacy kinetics/root routes to AIB-ODE contracts.
2. Add P3 task chain (`P3-001..P3-007`) as the new implementation and verification backbone.
3. Keep deferred model-note items (`D-001..D-005`) unchanged.
4. Keep `POLICY_LOCK` invariants unchanged (scripts/commands as single source of truth, output policies, dependency policies).

## Consequences

- `docs/REQUIREMENTS.md` and `docs/TRACEABILITY.md` are updated to include AIB-specific gates.
- `tasks/tasks.json` gets new decision/implementation entries for D-006/D-007/P3 series.
- `scripts/commands.sh verify_task` includes D-006/D-007/P3 handlers.
