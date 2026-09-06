# ADR 0006: D-004 Incubation/Poisoning State Model Promotion Policy

- Date: 2026-02-19
- Status: Accepted
- Decision task: `D-004`

## Context

`model_explain.md` proposes explicit incubation and poisoning state tracks (MS-11/MS-12).
Current runtime already has basic state-closure scaffolding, but mechanistic contracts
(state variables, driving species, coupling rules, identifiability constraints) are not
standardized as requirement-level interfaces.

Per AGENTS and POLICY_LOCK, promotion from model-note to implementation scope requires a
formal requirement contract and test gate design.

## Decision

For `D-004`, incubation/poisoning promotion is classified as:

- `DEFERRED`

No requirement-level promotion is made in this decision.

## Rationale

1. Partial scaffolding exists, but mechanistic semantics are not fixed enough for stable gates.
2. Direct promotion now would blur boundaries between reduced-state closure and mechanistic states.
3. Identifiability risks are high without explicit experiment/parameter policy.

## Consequences

1. No new state API is introduced by this decision.
2. `docs/GAPS.md` keeps the item deferred with reopen triggers.
3. Future adoption must define state contracts and deterministic acceptance tests first.

## Trigger To Reopen

Reopen when at least one is true:

1. Target processes require explicit incubation or poisoning dynamics to explain residuals.
2. A calibrated dataset with identifiable parameters is available for these states.
3. A decision/ADR defines YAML schema, validator rules, and regression tests.
