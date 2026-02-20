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
### P1-001: XY domain runtime support (wafer_2d_xy grid + radial profiling)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P0-999
- Acceptance:
  - DomainSpec.kind=wafer_2d_xy builds a valid DomainGrid
  - Edge exclusion mask works on XY grid
  - radial_profile works for XY grids and returns stable values

### P1-002: Registry metadata foundation (requires/excludes/time_modes/governing_class)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-001
- Acceptance:
  - Mass-transfer and rate-law registries expose metadata lookups
  - Metadata schema supports requires/excludes/time_modes/governing_class
  - Existing YAML model selection keeps working

### P1-003: Compatibility validator (preflight/runtime guard)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-002
- Acceptance:
  - Invalid model combinations fail with explicit error
  - Valid baseline smoke config passes validation
  - Validator is callable from scripts/commands verification path

### P1-004: MeasurementAdapter implementation (alignment/mask/resample)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-003
- Acceptance:
  - MeasurementAdapter aligns/masks 2D map deterministically
  - Resampling to simulation grid is available
  - Unit tests cover alignment, masking, and reproducibility

### P1-005: KPI metrics (NU%, center-edge, ring stats, out-of-spec area)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-004
- Acceptance:
  - summary.json includes KPI metrics
  - KPI metrics are deterministic for same inputs
  - Metrics are reusable for single-run and DOE reports

### P1-006: Comparison report extension (sim-vs-meas maps + error stats)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-005
- Acceptance:
  - report.html includes comparison plots when measurement input exists
  - Error statistics are summarized in report and summary artifacts
  - Plot generation remains deterministic and headless-safe

### P1-007: DOE runner with case-dimension outputs and KPI ranking
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-006
- Acceptance:
  - DOE run outputs a single run directory with case-dimension arrays
  - No deep per-case directory explosion
  - KPI ranking/report for DOE cases is produced

### P1-008: z_ref sensitivity workflow (DOE factor + report)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-007
- Acceptance:
  - Config supports z_ref sweeps
  - Report includes z_ref sensitivity metrics and plots

### P1-009: Additional kinetics + net models (competition/LHHW + dep-etch-loss)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-008
- Acceptance:
  - New kinetics/net models selectable via YAML
  - Sign conventions remain consistent for net thickness
  - Diagnostics remain available with new models

### P1-010: Time.phases and driver extension
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-009
- Acceptance:
  - Time mode phases can run synthetic scenario
  - Driver modifications are reflected in phase execution
  - Input preview output exists for phase-driven inputs

### P1-011: State closure modes (dynamic_ode / steady_state) + sticking_flux
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-010
- Acceptance:
  - Coverage state remains within [0,1]
  - dynamic_ode and steady_state closure modes are selectable
  - sticking_flux model runs via YAML selection

### P1-012: Bosanquet diffusivity bridge + pattern loading S(x,y)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-011
- Acceptance:
  - Bosanquet diffusivity selectable in km path
  - Pattern loading S(x,y) can be applied and disabled
  - Baseline behavior remains unchanged when feature is off

### P1-013: Identifiability diagnostics (finite-difference sensitivity + correlation)
- Type: implement
- Stop after: False
- Estimated: 90 min
- Depends on: P1-012
- Acceptance:
  - Sensitivity diagnostics are generated for selected parameters
  - Correlation/degeneracy warning is reported
  - Report artifacts include identifiability section

### P1-014: Optional JAX path (extras-based, no core hard dependency)
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-013
- Acceptance:
  - JAX path runs smoke when extras installed
  - NumPy remains baseline without JAX installed
  - Engine selection respects YAML choice

### P1-015: Benchmark helper (user-choice preserving)
- Type: implement
- Stop after: False
- Estimated: 60 min
- Depends on: P1-014
- Acceptance:
  - Benchmark report includes throughput/timing summaries
  - No automatic override of compute policy

## P2
### P2-001: Create deposim_opt package scaffold (configs/opt split) with parameter transforms
- Type: implement
- Stop after: False
- Estimated: 120 min
- Depends on: P1-015
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

## Post-P2 Decisions
### D-001: Decision: adopt/defer Stefan flow correction (MS-14)
- Type: decision
- Stop after: False
- Estimated: 30 min
- Depends on: P2-999
- Acceptance:
  - Decision is recorded with ADOPT/DEFERRED/ADR_REQUIRED classification
  - MODEL_GAP references D-001 and decision outcome

### D-002: Decision: adopt/defer smoothing PDE postprocess (MS-15)
- Type: decision
- Stop after: False
- Estimated: 30 min
- Depends on: D-001
- Acceptance:
  - Decision is recorded with ADOPT/DEFERRED/ADR_REQUIRED classification
  - MODEL_GAP references D-002 and decision outcome

### D-003: Decision: purge_decay driver contract standardization
- Type: decision
- Stop after: False
- Estimated: 30 min
- Depends on: D-002
- Acceptance:
  - Decision is recorded with ADOPT/DEFERRED/ADR_REQUIRED classification
  - MODEL_GAP references D-003 and decision outcome

### D-004: Decision: incubation/poisoning state model promotion
- Type: decision
- Stop after: False
- Estimated: 30 min
- Depends on: D-003
- Acceptance:
  - Decision is recorded with ADOPT/DEFERRED/ADR_REQUIRED classification
  - MODEL_GAP references D-004 and decision outcome

### D-005: Decision: chamber seasoning model inclusion policy
- Type: decision
- Stop after: False
- Estimated: 30 min
- Depends on: D-004
- Acceptance:
  - Decision is recorded with ADOPT/DEFERRED/ADR_REQUIRED classification
  - MODEL_GAP references D-005 and decision outcome
