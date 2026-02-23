# Output / Visualization Contract (`output.v1`)

Status: adopted (code-first reference)
Last updated: 2026-02-23

This document is the primary spec for output/report contracts aligned with:

- `docs/adr/0011-output-visualization-contract-v1.md`
- `src/deposim_sim/output_manifest.py`
- `src/deposim_report/plot_catalog.py`

## 1. Scope

Applies to all primary runners:

- simulation (`run_manager`)
- DOE (`doe`)
- benchmark (`benchmark_wafer2d`)
- fit (`run_fit`)

Visualization policy remains static PNG + HTML (no interactive JS plotting in mainline).

## 2. Required output layout

Each run directory MUST contain:

- `config_resolved.yaml`
- `summary.json`
- `report.html`
- `outputs/manifest.json` (`schema_version=output.v1`)

Additional runner-specific artifacts are declared by `manifest.artifacts`.

Entry points:

- root: `results/index.html`
- project: `results/<project>/index.html`
- run: `results/<project>/runs/<run_id>/report.html`

## 3. Manifest schema (`output.v1`)

`outputs/manifest.json` MUST include:

- `schema_version` (string, exactly `"output.v1"`)
- `run_id` (non-empty string)
- `mode` (non-empty string)
- `created_at_utc` (non-empty string)
- `artifacts` (array)
- `plots` (array; may be empty)

### 3.1 `artifacts[]` record

Required keys per row:

- `id` (non-empty, unique within artifacts)
- `path` (non-empty relative path from run root; absolute/parent traversal forbidden)
- `kind` (non-empty string)
- `required` (boolean recommended; truthy means required on disk by validation stage)

### 3.2 `plots[]` record

Required keys per row:

- `plot_id` (non-empty, unique within plots)
- `path` (non-empty relative path from run root; absolute/parent traversal forbidden)
- `source_key` (non-empty)
- `cmap` (string)
- `discrete` (boolean)

## 4. `summary.json` contract

`summary.json` remains domain/run summary and MUST expose:

- `run_id`
- `timestamp_utc`
- `mode`
- `manifest_path` (fixed: `outputs/manifest.json`)
- `artifact_paths` (map derived from manifest artifacts)

`artifact_paths` is treated as a convenience mirror of manifest records, not an independent source of truth.

## 5. Plot contract

Plot naming/style metadata is centralized in `src/deposim_report/plot_catalog.py`.

Core map classes:

- run report primary maps (`thickness_map`, `phi_B_map`, `f_I_map`, etc.)
- comparison maps (`measurement_map`, `comparison_error_map`)
- solver health maps
- profile plots (`radial_profile`)
- species-keyed maps (`cs_over_cref_*`, `n_app_*`)
- benchmark physviz maps
- DOE KPI/z_ref sensitivity plots

## 6. Unstructured map policy (`from_fluent_xy`)

For `grid.kind == "from_fluent_xy"`:

- triangulation rendering is primary path
- scatter fallback is allowed for degenerate meshes
- `imshow` is not used for unstructured maps
- NaN/invalid points are masked

## 7. Failure policy

The following are errors (not warnings):

- missing required manifest top-level keys
- duplicate `artifacts[].id` or `plots[].plot_id`
- invalid artifact/plot paths (absolute or traversal)
- missing on-disk files for `required=true` artifacts at validation stage

## 8. Compatibility policy

- Contract version stays `output.v1`.
- Backward-compatible strictness additions are allowed.
- Breaking schema updates require ADR and new schema version (e.g., `output.v2`).

## 9. Verification

Primary checks:

- `./scripts/commands.sh verify_task_contracts`
- `./scripts/commands.sh verify_p2`
- task-level `verify_task` for D-009 / P3-039..P3-045
