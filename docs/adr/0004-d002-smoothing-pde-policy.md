# ADR 0004: D-002 Smoothing PDE Policy

- Date: 2026-02-19
- Status: Accepted
- Decision task: `D-002`

## Context

`model_explain.md` includes MS-15 (morphology smoothing PDE) as an optional extension.
Current platform gates (P0/P1/P2) are satisfied without PDE-based postprocessing, and there is
no validated numerical/physical contract yet for introducing a smoothing term without changing
interpretability of core deposition-rate outputs.

Per POLICY_LOCK and AGENTS, optional theory should not be promoted to implementation without
explicit requirement, traceability, and acceptance tests.

## Decision

For `D-002`, smoothing PDE is classified as:

- `DEFERRED`

It is not promoted to `docs/REQUIREMENTS.md` in the current phase.

## Rationale

1. Smoothing can mask model mismatch and reduce identifiability if introduced too early.
2. Stable timestep/discretization policy and boundary conditions are not yet standardized.
3. No current gate requires this feature to preserve operational quality.

## Consequences

1. No runtime postprocess PDE module is added in this decision.
2. `MODEL_GAP` keeps smoothing PDE as deferred with reopen triggers.
3. Future adoption requires new requirement IDs, traceability links, and deterministic tests.

## Trigger To Reopen

Reopen when at least one is true:

1. Measured morphology smoothing effect cannot be represented by existing model/state/options.
2. Product requirement explicitly requests PDE-based morphology regularization.
3. A dedicated ADR defines discretization, stability constraints, and acceptance tests.
