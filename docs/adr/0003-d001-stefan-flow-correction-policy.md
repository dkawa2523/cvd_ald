# ADR 0003: D-001 Stefan Flow Correction Policy

- Date: 2026-02-19
- Status: Accepted
- Decision task: `D-001`

## Context

`model_explain.md` includes MS-14 (Stefan flow correction) as an optional physics extension.
Current runtime requirements and task gates are satisfied without Stefan correction, and no
validated input contract exists yet for robust multi-component total-flux correction in this repo.

Per AGENTS and POLICY_LOCK, new physics must not be silently promoted into implementation scope.

## Decision

For `D-001`, Stefan flow correction is classified as:

- `DEFERRED`

It is not promoted to `docs/REQUIREMENTS.md` in the current phase.

## Rationale

1. No immediate gate requires Stefan correction to keep P0/P1/P2 operational integrity.
2. Premature insertion would add model-combination and identifiability complexity without
   fixed acceptance tests.
3. A dedicated requirement/validation contract is needed before implementation.

## Consequences

1. No code changes to mass-transfer solver stack are introduced in this decision.
2. `docs/GAPS.md` keeps Stefan as deferred with trigger conditions.
3. Any future implementation must first add requirement IDs + traceability + acceptance tests.

## Trigger To Reopen

Reopen this decision when at least one is true:

1. A target use-case shows systematic error in strong-consumption regimes that cannot be
   explained by existing Bosanquet/pattern/state options.
2. A measurement/calibration study requires explicit Stefan-term sensitivity.
3. An ADR/decision task promotes Stefan into MUST/SHOULD with concrete verification cases.
