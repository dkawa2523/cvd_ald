# ADR 0015: Domain runtime alignment and staged refactor guardrails

- Date: 2026-04-01
- Status: Accepted
- Scope: sim domain/runtime consistency, legacy phase-out policy, refactor boundaries

## Context

Recent improvements require:

- runtime support for `wafer_2d_xy` (not schema-only),
- reduced duplicate output/report plumbing,
- cleaner path for retiring legacy modules/tests without breaking current users.

## Decision

1. Domain kinds are formally supported as:
   - `from_fluent_xy`
   - `wafer_2d_xy`
   - `wafer_2d_polar`
   - `wafer_1d_radial`

2. Runtime is domain-aware:
   - `from_fluent_xy` uses Fluent XY points directly.
   - structured domains (`wafer_2d_xy`, `wafer_2d_polar`, `wafer_1d_radial`) project Fluent point fields onto target grid points with nearest-neighbor mapping.

3. Legacy phase-out is staged:
   - keep legacy APIs temporarily with explicit deprecation direction,
   - exclude legacy-only tests from default verify gates,
   - remove legacy implementations in a dedicated follow-up ADR/task.

4. Refactor guardrail:
   - any logic repeated in 3+ call sites should be extracted into shared utility modules unless there is a documented, intentional divergence.

## Consequences

- Schema, validator, and runtime must be kept in lockstep for domain options.
- New output/report wiring changes should happen in shared helpers first.
- Verification gates must reflect actively supported execution paths.
