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

## Post-P0 (P1/P2) gates (planned)

These are not P0 blockers but should be added later:

- DOE run (10+ cases) produces case-dimension output without directory explosion
- JAX engine smoke run (CPU) matches NumPy within tolerance
- Optional GPU run is reproducible when environment supports it
- Assimilation (opt) minimal parameter fit on synthetic dataset
