# Requirements

This document enumerates requirements extracted from the prior discussion.

Each requirement has an ID and is classified as MUST/SHOULD/COULD.

Traceability is maintained in `docs/TRACEABILITY.md`.
Model-note gap triage is maintained in `docs/MODEL_GAP.md`.

This file is the primary implementation source of truth.
If `model.md` proposes additional items, they must be triaged in `docs/MODEL_GAP.md`
and promoted through ADR/decision task before implementation.


---

## MUST

### MUST-001: TCRSK core (transport–reaction coupling) is implemented

The simulation MUST compute surface-adjacent concentrations C_s via transport–reaction coupling from reference-plane CFD fields C_ref; it MUST NOT treat C_ref as surface concentration.

### MUST-002: Reference plane metadata (z_ref) is configurable and preserved

z_ref_mm MUST be configurable via YAML and stored in run metadata; C_ref is treated as reference-plane not bulk unless explicitly modeled.

### MUST-003: Support CVD (continuous) and ALD (phase) modes in one platform

Time domain MUST support CVD steady/transient and ALD phases; ALD support may be staged but must be planned and tracked.

### MUST-004: Wafer thickness 2D map is a primary output

The platform MUST produce a 2D wafer thickness map (full wafer) as a standard output; 1D radial approximation is optional.

### MUST-005: Rotation is optional; omega=0 must run

The platform MUST run when omega_rad_s=0; rotation affects km models/averaging but must not be required for correctness.

### MUST-006: Mass-transfer coefficient km model is pluggable and configurable

Mass-transfer coefficient (k_m) MUST be selected via registry and YAML, supporting at least stagnant-film and rotating-disk options with omega guards.

### MUST-007: Role-based CVD/ALD model paths are the primary kinetics contract

The primary simulation path MUST use role-based model selection with A/AI/AB/AIB role discovery and validation. `aib_ode` remains the current compatibility implementation, but future CVD/ALD work MUST NOT treat `sim.model.name = aib_ode` as the permanent public contract. Legacy `power_law/lhhw` model selection MUST NOT return as a primary execution route.

### MUST-008: Role contract is fixed and validated

Role assignment MUST satisfy: A required, I/B each optional single species, A/I/B disjoint, and unused species allowed.

### MUST-009: Standard AIB diagnostics are produced (Cs ratios, phi_B, f_I)

Outputs MUST include `CsA_over_CrefA`, `CsB_over_CrefB` (NaN when B absent), `phi_B`, and `f_I`, plus residual map when measurement is provided.

### MUST-010: Physical constraints are enforced

Concentrations must be nonnegative; coverage must be in [0,1]; thickness sign convention must be consistent (deposit positive, etch negative).

### MUST-011: Process-specific bounded state updates are declared and implemented

CVD AIB compatibility models MUST use implicit Euler with bisection in `[0,1]`.
`role_ald_state` MUST declare and use bounded explicit substeps, including state
projection diagnostics. A config MUST NOT advertise a solver different from the
one executed by its process model.

### MUST-012: Non-bracket fallback behavior is mandatory

If implicit bracketing fails, the solver MUST fall back to a clamped explicit update and emit diagnostics (`non_bracketed_count`).

### MUST-013: Order constraints are enforced by validator

The validator MUST enforce integer-order constraints and total-order limit: `p_A + p_* + m_B <= 3`, with `m_B` fixed by role-B presence.

### MUST-014: Regime sanity checks are testable

The code MUST include tests verifying behavior in reaction-limited and transport-limited limits.

### MUST-015: Numerical engine is selectable (NumPy baseline)

NumPy CPU baseline MUST exist. JAX is optional but planned; engine selection must be YAML-controlled.

### MUST-016: Time modes are `steady` and `transient` with shared role-input contract

CVD steady/transient and ALD transient execution MUST share the same Fluent input and role-assignment contract. CVD and ALD MAY dispatch to different role-based process models.

### MUST-017: Drivers: support time/space varying external input modification

The framework MUST support modifying scalar and spatial inputs over time/phases (drivers) and preview them for debugging.

### MUST-018: Initial conditions support scalars and spatially varying fields

Initial conditions for state variables MUST support scalar values and spatial maps (for future ALD/state models).

### MUST-019: Simulation config is YAML-managed and split as `sim` / `opt`

Configs MUST use YAML composition with `configs/sim/` and `configs/opt/`, and runtime contracts MUST be centered on `sim:` and `opt:` blocks.

### MUST-020: Simulation and optimization/ML code is separated into packages

Numerical simulation and optimization/ML MUST be separated into packages/modules with clear dependency direction.

### MUST-021: Model registry supports adding process models without refactor

Mass-transfer models and process models MUST be registered by name and discoverable, allowing later CVD/ALD model extensions with minimal file/directory growth.

### MUST-022: Compute resources are user-selected (no forced auto policy)

CPU/GPU selection and engine selection MUST be user-controlled via YAML; auto-selection may exist but must not override explicit user choice.

### MUST-023: Run/test commands are centralized (Single Source of Truth)

All run/verify commands MUST be centralized in scripts/commands.sh; tasks and docs must reference it.

### MUST-024: Output layout has fixed entrypoint results/index.html

Outputs MUST be organized per project with `results/index.html` as the fixed entrypoint; resolved config and summary must be saved per run. Root and project indexes MUST both resolve to latest run reports.

### MUST-025: No directory explosion for DOE; case dimension storage

DOE outputs MUST be stored without per-case deep directories; store case-dimension arrays (npz/zarr/hdf5) and summary tables.

### MUST-026: Standard plots and HTML report are generated

Generate thickness map, radial profile, Cs ratios, `phi_B`/`f_I` maps, and HTML report linking run artifacts from manifest records.

### MUST-027: Numerical health metrics are logged

Implicit solver non-bracket counts, bounded-state checks, state projections, and
validation violations MUST be recorded and visualizable. Root-solver metrics MUST
state whether they are applicable; non-applicable ALD root metrics MUST NOT be
reported as successful zero counts. Solver health maps MUST be driven by runtime
diagnostics rather than placeholder defaults.

### MUST-061: Output contract must be versioned and machine-readable

Each run MUST emit `outputs/manifest.json` with `schema_version=\"output.v1\"`, required artifact records, and plot metadata; missing required keys MUST fail validation.
`output_viz.md` MUST be maintained as the implementation-aligned contract reference for output/visualization behavior.

### MUST-062: Optimization objective must emit decomposed score components

`deposim_opt` fitting outputs MUST include decomposed score columns (`loss_data`,
`penalty_solver`, `penalty_phys`, `penalty_prior`, `penalty_complexity`,
`score_total`) plus RMSE, MAE, and maximum absolute error in nanometer units.
Role selection MUST distinguish training loss from refitted prediction error.
When condition refits are available, they determine selection, with preference
for simpler candidates within paired error uncertainty. The existing complexity
penalty sweep remains a training-score diagnostic. Adoption MUST be withheld for
failed baseline comparisons, unresolved alternative roles, unsupported spatial
variation, or unassessed/degenerate fitted parameters. Role-stability and local
parameter-identifiability results MUST be reported separately. Local sensitivity
MUST use all estimated parameter directions and all training observations.

### MUST-063: Optimization contract must support multi-condition and staged fidelity

Optimization config MUST support weighted multi-condition fitting, explicit
`train|holdout` condition splits, and staged train-condition-count fidelity
(`coarse -> fine`) with pruning hooks, while preserving backward compatibility
for single-condition YAML files. External holdout data MUST not affect parameter
or role selection. Condition refits MUST reestimate parameters from the remaining
training conditions; an external holdout MUST never become a refit training row.
The no-role reference prediction MUST be estimated using training data only.

### MUST-028: runs/ and results/ are gitignored and managed

Automation state goes in runs/ and simulation outputs in results/ (both gitignored by default).

### MUST-046: Autorun must resume from state without restarting

codex_autorun must record completed tasks and resume; it must not rerun completed tasks.

### MUST-047: Codex CLI flag detection prevents read-only dead-ends

Autorun must detect Codex CLI flag support and always invoke write-enabled mode to avoid read-only defaults.

### MUST-048: Preflight handles python vs python3 and src import issues

Preflight must detect python and pip commands, avoid python/python3 mismatch, and set PYTHONPATH fallback if needed.

### MUST-049: MPLCONFIGDIR=/tmp is enforced in scripts

Run scripts must set MPLCONFIGDIR to /tmp by default to prevent headless permission errors.

### MUST-050: Traceability is 100% (requirements -> tasks)

Every requirement must map to at least one task ID in docs/TRACEABILITY.md.

### MUST-051: CLI is not the primary UX; YAML + Python API must work

The system MUST be operable without a heavy CLI UX; configuration via YAML/Hydra and execution via Python API/notebooks is primary. CLI entrypoints may exist for automation but must not be required for core usage.

### MUST-052: Simulation components are independently runnable and composable

Numerical components (domain, solvers, models, report) MUST be runnable independently for single conditions and composable for DOE/optimization workflows without tight coupling.

### MUST-055: `wafer_2d_xy` must be runtime-supported, not schema-only

If `domain.kind=wafer_2d_xy` is selected, the runtime MUST build a valid simulation grid, masks, and radial summary diagnostics using the same public execution path as polar/radial domains.

### MUST-056: Compatibility metadata and validator must gate invalid model combinations

Model registries MUST expose compatibility metadata (`requires`, `excludes`, `time_modes`, `governing_class`), and a validator MUST stop representative invalid configurations before simulation execution.

### MUST-057: Measurement comparison and role ranking must be first-class in role workflows

Workflow MUST include deterministic measurement alignment, nearest-match distance
diagnostics, KPI generation, role-candidate ranking, and sim-vs-measurement
reporting for role-based CVD/ALD runs.
Fitting MUST score original observations once each, independently of simulation
mesh density. Reports MUST separate mean bias from centered spatial error.
Known measurement uncertainty MUST scale the fitting residual consistently with
thickness or mean-rate conversion; ordinary error metrics retain physical units.

### MUST-058: Verification commands for P1/P2 must be executable gates

`scripts/commands.sh verify_task <task_id>` for P1/P2 MUST run concrete checks/tests; placeholder pass-through verification is not allowed.

### MUST-059: ALD transient execution and ALD metrics must be YAML-selectable

ALD transient execution (time-series concentration input) MUST be selectable through YAML and validated with the same role-assignment contract. ALD model-readiness metrics such as GPC plateau, cycle GPC stability, and purge growth fraction MUST be reportable separately from generic runtime success.

### MUST-060: Heavy dependencies remain optional extras unless a gate requires them

JAX/ClearML/Zarr and similar heavy dependencies MUST remain optional extras by default; core simulation and required gates must continue to run without them unless explicitly promoted by ADR.



## SHOULD

### SHOULD-029: DOE runner supports grid/random sweeps and summary metrics

Provide sweep runners with grid/random sampling, producing summary metrics (uniformity, center-edge, etc.) and comparison plots.

### SHOULD-030: Benchmark helper to evaluate CPU vs JAX CPU vs JAX GPU

Provide a benchmark mode that reports performance; it must not force resource selection.

### SHOULD-031: Additional kinetics models remain optional research tracks

Alternative kinetics (competition/LHHW, ER-like) may exist as optional research tracks but must stay outside the primary AIB runtime path unless promoted by ADR.

### SHOULD-032: Net model supports dep-etch-loss composition

Support multiple channels (deposition, etch, loss) and consistent sign/units for net thickness.

### SHOULD-033: z_ref sensitivity workflow exists as standard diagnostic

Provide a built-in way to assess sensitivity to reference-plane height; optionally via DOE factor.

### SHOULD-034: Structured config (dataclasses) is used for type safety

Use Hydra structured configs (dataclasses) to prevent misconfiguration and to validate YAML inputs.

### SHOULD-035: Output can be stored in Zarr for large DOE

Support Zarr storage as an option for large DOE outputs; keep NPZ as fallback.

### SHOULD-036: Measurement adapter for 2D map alignment exists

Provide alignment/masking tools for measured thickness maps (center shift, rotation, edge exclusion).

### SHOULD-053: Performance targets cover wafer grids (10–few hundred points) and DOE (10–1000 cases)

The design SHOULD explicitly support wafer grids from ~10 to a few hundred points and DOE sweeps from ~10 to ~1000 cases with predictable memory/time behavior and without output directory explosion.



## COULD

### COULD-037: Data assimilation package (deposim_opt) with robust loss and regularization

Implement parameter estimation against 2D thickness maps with robust losses (Huber) and regularization; stage freedom carefully.

### COULD-038: JAXopt implicit differentiation for root solve

Use implicit differentiation (JAXopt) to differentiate through root solves for assimilation.

### COULD-039: Diffrax adjoint for state ODEs (ALD)

Use adjoint methods (e.g., Diffrax) for differentiable ODE solves in ALD/state models.

### COULD-040: Multi-z reference-plane input support

Support multiple z planes from CFD to reduce z_ref sensitivity; include diagnostics.

### COULD-041: ClearML integration as an optional leaf package

Integrate ClearML for run/param/model tracking without coupling core simulation to ClearML.

### COULD-042: Plugin IO for real CFD/measurement formats (CSV/HDF5/Zarr)

Add IO plugins for CFD fields and measurement maps; keep schema stable and decouple from core solvers.

### COULD-043: JAX engine parity with NumPy including float32/float64 switch

Add JAX backend parity tests and precision controls; allow user selection of float32/float64.

### COULD-044: Axisymmetric reduction workflow with diagnostics to justify 1D

Provide automated diagnostics comparing theta variation to justify 1D radial approximation for speed.

### COULD-045: Advanced fallback for non-monotonic multi-root cases

Implement robust multi-root handling strategies and configurable root selection policy when needed.

### COULD-054: JAX JIT compilation caching is leveraged for repeated sweeps when available

When JAX is used, the system COULD support persistent/explicit JIT caching to reduce repeated compilation overhead in DOE runs, while remaining optional and user-controlled.
