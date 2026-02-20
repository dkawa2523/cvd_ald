# ADR 0007: D-005 Chamber Seasoning Inclusion Policy

- Date: 2026-02-19
- Status: Accepted
- Decision task: `D-005`

## Context

`model_explain.md` mentions chamber seasoning/drift as an advanced extension candidate.
Current platform focuses on wafer-level forward/DOE/assimilation baseline and does not define
a chamber-history state contract, lifecycle policy, or data interface for seasoning effects.

Per AGENTS and POLICY_LOCK, introducing this model family requires explicit requirement and
validation design before implementation.

## Decision

For `D-005`, chamber seasoning inclusion is classified as:

- `DEFERRED`

No promotion to requirement-level implementation is made in this phase.

## Rationale

1. Chamber-level historical state is outside current validated data contract.
2. Adding it now would introduce cross-run state coupling complexity and operational risk.
3. No current milestone gate requires seasoning support to preserve platform readiness.

## Consequences

1. No chamber-history state module is added by this decision.
2. `MODEL_GAP` tracks this as deferred and blocked for silent implementation.
3. Future adoption must include schema, persistence, reproducibility, and rollback policy.

## Trigger To Reopen

Reopen when at least one is true:

1. Product requirement mandates chamber drift compensation in production workflows.
2. Stable chamber-history data inputs become available and quality-controlled.
3. A decision/ADR defines state persistence, reset policy, and acceptance tests.
