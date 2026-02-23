# ADR 0012: `output_viz.md` as Primary Output/Viz Spec Source

- Date: 2026-02-23
- Status: Accepted
- Scope: output/visualization documentation authority

## Context

`output_viz.md` existed as an empty placeholder while runtime behavior had already converged on:

- ADR 0011 (`output.v1`)
- manifest-driven artifacts
- plot catalog + triangulation rendering policy

Without a maintained primary spec file, review and change tracking were fragmented.

## Decision

1. `output_viz.md` is promoted to a first-class spec document for output/visualization behavior.
2. `output_viz.md` must remain consistent with ADR 0011 and runtime contract checks.
3. Schema version remains `output.v1`; no breaking schema bump is introduced by this ADR.
4. Verification gates reference both code and `output_viz.md` content for drift detection.

## Consequences

- Reviewers can validate output/viz behavior from one implementation-aligned document.
- Contract drift between docs and code becomes detectable in task gates.
- Future breaking changes still require a dedicated ADR and new schema version.
