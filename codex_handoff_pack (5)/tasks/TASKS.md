# Task List
This is a human-readable summary of `tasks/tasks.json`.

---
## P0
### P0-000: Handoff scaffolding self-audit (POLICY_LOCK + scripts + docs)
- Type: review
- Stop after: False
- Estimated: 45 min
- Acceptance:
  - No contradictions between POLICY_LOCK (ADR) and scripts/commands.sh
  - tasks/tasks.json verification_commands reference scripts/commands.sh (no raw python/python3)
  - No extra checkpoints beyond P0-999

### P0-001: Scaffold Python project (pyproject + src layout packages)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Acceptance:
  - src/ contains deposim_schema, deposim_sim, deposim_report packages with __init__.py
  - pyproject.toml exists and supports editable install
  - Basic import test exists under tests/

### P0-002: Implement deposim_schema: structured configs + YAML/Hydra composition for sim
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-001
- Acceptance:
  - Structured config dataclasses exist for sim config, importable from deposim_schema
  - configs/sim/example_cvd.yaml exists (or equivalent) with clear parameters
  - Config can be composed and saved as resolved YAML

### P0-003: Implement domain grids and masks (wafer_2d_polar + wafer_1d_radial)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P0-002
- Acceptance:
  - DomainSpec can generate a grid object with r/theta arrays
  - Masking supports edge exclusion in mm
  - Unit tests cover basic grid shape and mask behavior

### P0-004: Model registry + mass transfer models (stagnant film, rotating disk w/ omega guard)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-003
- Acceptance:
  - Mass transfer registry resolves model by name from config
  - rotating_disk guard on omega=0 is implemented (error/fallback configurable)
  - km output shape matches domain grid

### P0-005: Rate laws (powerlaw + saturation/inhibition) with apparent order diagnostics
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-004
- Acceptance:
  - Rate law registry resolves model by name from config
  - powerlaw supports negative/fractional orders robustly
  - apparent_orders returns maps consistent with analytical expectations for these models

### P0-006: Transport–reaction coupling solver: vectorized bisection on progress variable R
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-005
- Acceptance:
  - solve_progress_R returns R, Cs, iteration_count, status_map
  - Nonnegativity constraints are enforced (Cs>=0, R within bracket)
  - Monotonicity check exists and produces an explicit status/warning

### P0-007: CVD steady simulator: thickness + diagnostics (Cs/Cref, Da proxy, n_app)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-006
- Acceptance:
  - run_cvd_steady returns thickness + diagnostics as arrays aligned with domain grid
  - Diagnostics include Cs/Cref, Da proxy, apparent orders, solver health
  - Sign conventions documented (deposit positive)

### P0-008: RunManager + outputs + plots + HTML report with fixed results/index.html entrypoint
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-007
- Acceptance:
  - results/index.html exists and links to latest run report
  - Run directory contains resolved config, arrays, plots, report.html
  - No file explosion; outputs are organized deterministically

### P0-009: Synthetic input generator + smoke entrypoint (Hydra config) used by scripts/commands.sh
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P0-008
- Acceptance:
  - Running ./scripts/commands.sh smoke produces a new run under results/runs/<run_id>
  - Synthetic input is deterministic for reproducibility
  - Smoke run uses Hydra-composed YAML config

### P0-010: Unit tests + P0 verify gate wiring (import_check, smoke, tests)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-009
- Acceptance:
  - ./scripts/commands.sh verify_p0 passes
  - Tests are deterministic and fast
  - EVAL_PROTOCOL commands are accurate

### P0-011: Repo-wide consistency sweep (commands, docs, old names) [review]
- Type: review
- Stop after: False
- Estimated: 45 min
- Depends on: P0-010
- Acceptance:
  - No stale run/test commands outside scripts/commands.sh
  - Docs match actual code entrypoints
  - .gitignore includes runs/ and results/

### P0-999: P0 CHECKPOINT: run verify_p0, summarize status, stop
- Type: checkpoint
- Stop after: True
- Estimated: 30 min
- Depends on: P0-011
- Acceptance:
  - verify_p0 passes
  - A checkpoint summary is written
  - Autorun stops with exit code 42 after this task

## P1
### P1-001: DOE runner (grid/random) with case-dimension outputs (no directory explosion)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P0-999
- Acceptance:
  - DOE run produces a single run directory with case-dimension outputs
  - No per-case directory explosion
  - Summary metrics (uniformity, center-edge etc.) are produced

### P1-002: z_ref sensitivity workflow (as DOE factor + report)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-001
- Acceptance:
  - Config supports sweeping z_ref
  - Report includes z_ref sensitivity plots/metrics

### P1-003: Additional kinetics + net models (competition/LHHW, dep-etch-loss composition)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-002
- Acceptance:
  - New rate law and net model are selectable via YAML
  - Diagnostics remain available (n_app, Cs/Cref, solver health)

### P1-004: MeasurementAdapter skeleton (2D map alignment + masks)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-003
- Acceptance:
  - MeasurementAdapter can align and mask a 2D thickness map
  - Unit tests cover basic alignment/masking behavior

### P1-005: Optional JAX engine path (CPU first) for core solver and rate laws
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-004
- Acceptance:
  - JAX engine can run a smoke case on CPU when installed
  - NumPy and JAX results match within tolerance
  - Engine selection respects user YAML choice

### P1-006: Benchmark helper (reports throughput; does not override user choice)
- Type: implement
- Stop after: False
- Estimated: 60 min
- Depends on: P1-005
- Acceptance:
  - Benchmark mode produces a report with timings
  - No automatic override of user compute settings

## P2
### P2-001: Create deposim_opt package scaffold (configs/opt split) with parameter transforms
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-006
- Acceptance:
  - deposim_opt imports successfully (when installed)
  - Opt config is separate from sim config

### P2-002: Assimilation minimal loop on synthetic data (CPU) [no heavy deps required]
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P2-001
- Acceptance:
  - Synthetic assimilation reduces loss from initial guess
  - Results and fitted params are saved to results/ with report

### P2-003: ALD phases + state model skeleton (coverage) with stiff-safe solver choices
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P2-002
- Acceptance:
  - ALD phases mode runs a synthetic example
  - Coverage stays within [0,1]

### P2-004: ClearML integration as optional leaf package (no core dependency)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P2-003
- Acceptance:
  - Core simulation imports without ClearML installed
  - ClearML integration is optional and documented

### P2-005: IO plugin system for CFD fields and measurement maps (CSV/NPZ baseline)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P2-004
- Acceptance:
  - Config can select an IO loader by name
  - Basic loaders work on simple sample files

### P2-006: Zarr output option for large DOE (keep NPZ fallback)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P2-005
- Acceptance:
  - Zarr output works when dependency installed
  - NPZ remains default/fallback

### P2-007: Multi-z reference-plane support (optional) + diagnostics
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P2-006
- Acceptance:
  - Single-plane mode remains unchanged
  - Multi-plane mode runs on synthetic example and outputs diagnostics

### P2-999: Final consistency sweep (docs/commands/tasks) [review]
- Type: review
- Stop after: False
- Estimated: 45 min
- Depends on: P2-007
- Acceptance:
  - Repo is internally consistent and traceability remains 100%

