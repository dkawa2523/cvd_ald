# ADR 0005: D-003 Purge-Decay Driver Contract Policy

- Date: 2026-02-19
- Status: Accepted
- Decision task: `D-003`

## Context

`model_explain.md` proposes an explicit purge residual driver (`purge_decay`, e.g. `C(t)=C0*exp(-t/tau)`).
Current code supports phase execution and scalar overrides, but does not define a standardized
`purge_decay` input contract or validator-level semantics across ALD workflows.

Per AGENTS and POLICY_LOCK, model-note proposals cannot be silently implemented without formal
requirements, traceability mapping, and deterministic acceptance tests.

## Decision

For `D-003`, purge-decay driver standardization is classified as:

- `DEFERRED`

No new requirement is added in this decision.

## Rationale

1. Existing ALD phase path is stable and does not require immediate contract expansion.
2. A premature driver contract risks incompatibility across existing YAML and validator rules.
3. A proper contract needs explicit parameter schema (`tau`, bounds, phase binding, units) and
   dedicated validation/test gates.

## Consequences

1. No runtime API change is introduced by this decision.
2. `MODEL_GAP` tracks purge-decay as deferred with reopen conditions.
3. Future adoption must start from requirement + traceability + validator/test design.

## Trigger To Reopen

Reopen this decision when at least one is true:

1. ALD purge behavior mismatch is observed and cannot be represented by current phase overrides.
2. Product requirement asks for explicit purge-time residual dynamics with auditable parameters.
3. A decision/ADR proposes a concrete schema and acceptance tests for purge-decay semantics.
