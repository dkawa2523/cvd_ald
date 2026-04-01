# Evaluation Protocol (EVAL_PROTOCOL)

This document defines the verification gates and commands that MUST pass.

All commands MUST be executed via `scripts/commands.sh` to avoid python command mismatch.

---

## P0 Gate (must pass before checkpoint)

### 1) Import sanity
- Ensure the top-level packages import without side effects.

Command:
- `./scripts/commands.sh import_check`

### 2) Smoke run (synthetic input)
- Run a minimal CVD steady simulation on synthetic fields.
- Confirm output directory layout is created and the fixed entrypoint exists.

Command:
- `./scripts/commands.sh smoke`

Acceptance checks:
- `results/index.html` exists
- run directory contains:
  - `config_resolved.yaml`
  - `outputs/fields.npz` and `outputs/metrics.json`
  - `outputs/manifest.json` (`schema_version=output.v1`)
  - `plots/` and `report.html`

### 3) Unit tests (numerical sanity)
- Non-negativity and bounds:
  - C_s >= 0
  - R in [0, R_max]
- Regime limits:
  - reaction-limited vs transport-limited behavior
- Root solver robustness on monotonic cases

Command:
- `./scripts/commands.sh test`

### 4) Full verify shortcut
Command:
- `./scripts/commands.sh verify_p0`

---

## P1 Gate

Run AIB utility migration gates:

- `./scripts/commands.sh verify_p1`

This gate includes:

- Utility common AIB execution checks
- DOE/z_ref workflows on AIB path
- Physviz benchmark outputs on AIB path
- Identifiability and assimilation checks on AIB path
- Residual legacy-utility migration checks (`multiz`, `benchmark`, `phases_driver`)

## Wafer2D AIB Benchmark

Run the benchmark runner that validates:

- A/AI/AB/AIB class coverage
- AIB diagnostics (`phi_B`, `f_I`, `CsA_over_CrefA`, `CsB_over_CrefB`, `residual_nm`)
- ranking/class comparison outputs (`ranking.csv`, `class_compare.csv`)
- summary trend assertions (`overall_passed`)

Command:

- `./scripts/commands.sh benchmark_wafer2d`

Physviz extension command (time-space maps + term-importance plots):

- `./scripts/commands.sh benchmark_wafer2d_physviz`

Flux-km comparison command (free-km vs flux-km judge):

- `./scripts/commands.sh benchmark_wafer2d_flux_km`

## P2 Gate

Run AIB contract/output gates:

- `./scripts/commands.sh verify_p2`

This gate includes:

- Opt contract completion checks (`role_enumeration`, `selection`, top-k outputs)
- selective optimize.md adoption checks (`sampler/pruner/storage`, decomposed objective, multi-condition/hierarchical fidelity)
- Measurement alignment integration checks
- Output contract strictness checks (`save_fields`, manifest validation, report map generation)
- ranking/full-candidate + tri-rendering + IO/run_manager + non-mainline test convergence checks

Final AIB gate:

- `./scripts/commands.sh verify_task P3-038`

`verify_task P3-038` enforces:

- full P1 + P2 AIB gate replay
- task contract validation

Output/Viz convergence gates:

- `./scripts/commands.sh verify_task D-009`
- `./scripts/commands.sh verify_task P3-039`
- `./scripts/commands.sh verify_task P3-040`
- `./scripts/commands.sh verify_task P3-041`
- `./scripts/commands.sh verify_task P3-042`
- `./scripts/commands.sh verify_task P3-043`
- `./scripts/commands.sh verify_task P3-044`
- `./scripts/commands.sh verify_task P3-045`

Optimization selective-adoption gates:

- `./scripts/commands.sh verify_task D-010`
- `./scripts/commands.sh verify_task P3-046`
- `./scripts/commands.sh verify_task P3-047`
- `./scripts/commands.sh verify_task P3-048`
- `./scripts/commands.sh verify_task P3-049`
- `./scripts/commands.sh verify_task P3-050`
- `./scripts/commands.sh verify_task P3-051`
- `./scripts/commands.sh verify_task P3-052`

improve2 selective-adoption gates:

- `./scripts/commands.sh verify_task D-011`
- `./scripts/commands.sh verify_task P3-053`
- `./scripts/commands.sh verify_task P3-054`
- `./scripts/commands.sh verify_task P3-055`
- `./scripts/commands.sh verify_task P3-056`
- `./scripts/commands.sh verify_task P3-057`

## Contract Gate

Task contract and command centralization checks:

- `./scripts/commands.sh verify_task_contracts`
