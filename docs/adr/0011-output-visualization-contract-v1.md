# ADR 0011: Output / Visualization Contract V1 (`output.v1`)

- Date: 2026-02-23
- Status: Accepted
- Scope: AIB runtime output contracts, report links, and visualization metadata

## Context

After AIB migration, output layouts diverged by runner (`sim`, `doe`, `benchmark`, `fit`) and
report links were partially hardcoded. This reduced traceability and made review/automation brittle.

## Decision

1. Introduce strict `outputs/manifest.json` for all primary runners with:
   - `schema_version = "output.v1"`
   - `run_id`, `mode`, `created_at_utc`
   - `artifacts[]` records (`id`, `path`, `kind`, `required`)
   - `plots[]` records (`plot_id`, `path`, `source_key`, `cmap`, `discrete`)
2. `summary.json` keeps domain metrics but references artifacts via `manifest_path` and
   manifest-derived `artifact_paths`.
3. Report artifact links are generated from manifest records (no per-run hardcoded link lists).
4. Visualization policy remains static PNG + HTML.
5. For `from_fluent_xy`, tri rendering is the primary map path; fallback is scatter only.

## Breaking-change / compatibility policy

- This is a breaking contract for consumers reading ad-hoc `summary.artifact_paths` layouts.
- Compatibility is provided only by explicit manifest mapping (no long-term duplicate output names).

## Failure policy

- Missing required manifest keys are treated as errors (not warnings).
- Invalid manifest payloads fail verification tasks and tests.
