# ADR 0018: Operational gates, skip policy, and generated file hygiene

- Date: 2026-04-04
- Status: Accepted
- Decision task: D-014

## Context

Refactor stabilization completed through `D-013` and `P3-058..P3-063`, but day-to-day operation still had three friction points:

1. Optional dependency tests (notably `optuna`) could surface skips that were not explicitly policy-bound.
2. Full `verify_p2` is comprehensive but heavy for iterative developer loops.
3. Generated runtime files (`scripts/env.sh`, `.wslbin/`) created recurring local noise.

## Decision

We fix the following operational policy:

1. **Skip policy**
   - Optional dependency skips are accepted as `skip=warn`.
   - Mandatory checks continue to use fail-fast semantics (`skip=fail` where required).
   - Gate logs must clearly indicate warning-vs-failure skip treatment.

2. **Two-stage P2 gate**
   - `verify_p2_quick` is the default daily gate for rapid iteration.
   - `verify_p2` remains the full pre-release gate and compatibility baseline.
   - Existing command names remain compatible; `verify_p2` semantics stay full.

3. **Generated file boundary**
   - `scripts/env.sh` is treated as generated runtime state and is not tracked.
   - `.wslbin/` is treated as local bootstrap/runtime state and is not tracked.
   - Reference configuration is documented via tracked template (`scripts/env.sh.example`).

## Consequences

- Developer loops become faster by default (`verify_p2_quick`) while preserving a strict full gate (`verify_p2`) before release.
- Optional dependency gaps remain visible as warnings without blocking unrelated work.
- Repository diffs are cleaner by default due to explicit generated-file boundaries.
