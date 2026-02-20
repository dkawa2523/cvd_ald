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
  - `outputs/` with thickness + diagnostics arrays
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

Run all P1 verification tasks:

- `./scripts/commands.sh verify_p1`

This gate includes:

- XY domain runtime checks
- registry metadata + compatibility validator checks
- measurement/KPI/report path tests
- DOE and z_ref workflow tests
- extended kinetics/net, phases/state, bosanquet/pattern, identifiability
- optional JAX selection and benchmark checks

## Wafer2D Trend Benchmark (polar-first)

Run the benchmark runner that validates:

- synthetic + file(npz) input paths
- reaction-limited vs transport-limited trend ordering
- radial and theta transfer trend checks
- solver health (`root_failure_fraction == 0` for all benchmark cases)

Command:

- `./scripts/commands.sh benchmark_wafer2d`

Physviz extension command (time-space maps + term-importance plots):

- `./scripts/commands.sh benchmark_wafer2d_physviz`

## P2 Gate

Run all P2 verification tasks:

- `./scripts/commands.sh verify_p2`

This gate includes:

- `deposim_opt` scaffold + minimal synthetic assimilation loop
- ALD skeleton gate
- optional ClearML leaf-package behavior
- IO plugin selection with smoke end-to-end checks
- optional Zarr storage with DOE/smoke end-to-end checks
- multi-z diagnostics with smoke end-to-end checks

Final P2 consistency gate:

- `./scripts/commands.sh verify_task P2-999`

`verify_task P2-999` enforces:

- autorun state synchronization for P1/P2
- full P1 + P2 gate replay
- repo consistency checks

If state is not synchronized after manual execution, run:

- `./scripts/commands.sh reconcile_state --milestone P1,P2`

## Contract Gate

Task contract and command centralization checks:

- `./scripts/commands.sh verify_task_contracts`
