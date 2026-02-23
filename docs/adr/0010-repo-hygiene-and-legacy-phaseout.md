# ADR 0010: Repository Hygiene and Legacy Phase-out (AIB Single Path)

- Date: 2026-02-22
- Status: Accepted
- Scope: Runtime/test/config verification lines and repository asset hygiene

## Context

After migrating core runtime to AIB-ODE, residual legacy assets (old config names, scaffold modules,
old verification branches, duplicated handoff bundles) reduced maintainability and caused stale gate failures
(e.g. outdated smoke artifact assumptions).

## Decision

1. AIB single runtime path is kept as the only supported primary path.
2. Legacy config names are stage-deprecated via alias resolution:
   - `smoke`, `example_cvd`, `multiz` -> `cvd_steady_min`
   - `ald_synthetic` -> `ald_transient_min`
   - `base`, `stub` (opt) -> `fit_cvd_steady_min`
3. Legacy `P1/P2` verify_task branches are remapped to executable AIB gates.
4. Obsolete duplicate/generated repository assets are removed from tracked files when not required for runtime
   (duplicate handoff pack, generated `*.egg-info`, accidental platform metadata).
5. Triangulation/rendering and override/path helpers are centralized under `deposim_sim.common.*`
   to prevent helper drift across runtime modules.

## Consequences

- Existing old config names continue to compose, but are marked deprecated and must migrate to canonical names.
- Verification and reviewer workflow are aligned to AIB tasks and contracts.
- Runtime maintenance cost is reduced by removing duplicate helper implementations and obsolete tracked assets.
- Future removal of alias compatibility can proceed as a bounded follow-up without changing AIB runtime behavior.
