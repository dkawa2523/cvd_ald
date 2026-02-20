# Requirements

This document enumerates requirements extracted from the prior discussion.

Each requirement has an ID and is classified as MUST/SHOULD/COULD.

Traceability is maintained in `docs/TRACEABILITY.md`.


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

### MUST-007: Rate law family supports multi-order kinetics including negative/fractional apparent orders

Rate laws MUST support explicit reaction orders and allow negative/fractional values as 'apparent' orders; stable numerical evaluation is required.

### MUST-008: Saturation/inhibition kinetics (denominator form) is supported

At least one rate law MUST support saturation/inhibition via denominator (Langmuir/LHHW-like) to capture order transitions and inhibition effects.

### MUST-009: Standard diagnostics are produced (Cs/Cref, Da proxy, apparent orders)

Outputs MUST include Cs/Cref map, a Damköhler-like proxy map, and apparent reaction order map(s) where possible.

### MUST-010: Physical constraints are enforced

Concentrations must be nonnegative; coverage must be in [0,1]; thickness sign convention must be consistent (deposit positive, etch negative).

### MUST-011: Progress-variable reduction to scalar root solve (R) for dominant reaction

For the common single-dominant-reaction case, the coupled system MUST be reduced to a scalar root solve in progress variable R with a physically safe bracket.

### MUST-012: Vectorized bracketing root solver (bisection) is implemented

A vectorized bisection solver MUST be implemented for CPU and be compatible with a JAX backend later; iteration counts and status must be tracked.

### MUST-013: Monotonicity check + fallback behavior exists

The solver MUST include a monotonicity/shape check for F(R) and define fallback behavior (e.g., warn+interval split or controlled failure) for non-monotonic cases.

### MUST-014: Regime sanity checks are testable

The code MUST include tests verifying behavior in reaction-limited and transport-limited limits.

### MUST-015: Numerical engine is selectable (NumPy baseline)

NumPy CPU baseline MUST exist. JAX is optional but planned; engine selection must be YAML-controlled.

### MUST-016: Time modes: steady is implemented as primary CVD mode

CVD steady mode MUST be implemented with thickness = rate * process_time; transient/phases are planned and tracked.

### MUST-017: Drivers: support time/space varying external input modification

The framework MUST support modifying scalar and spatial inputs over time/phases (drivers) and preview them for debugging.

### MUST-018: Initial conditions support scalars and spatially varying fields

Initial conditions for state variables MUST support scalar values and spatial maps (for future ALD/state models).

### MUST-019: Simulation config is YAML-managed via Hydra, split sim/opt

Configs MUST be YAML-based and composed by Hydra; sim and opt configs are separated into distinct directories.

### MUST-020: Simulation and optimization/ML code is separated into packages

Numerical simulation and optimization/ML MUST be separated into packages/modules with clear dependency direction.

### MUST-021: Model registry supports adding models without refactor

Models MUST be registered by name and discoverable, allowing later extension with minimal file/directory growth.

### MUST-022: Compute resources are user-selected (no forced auto policy)

CPU/GPU selection and engine selection MUST be user-controlled via YAML; auto-selection may exist but must not override explicit user choice.

### MUST-023: Run/test commands are centralized (Single Source of Truth)

All run/verify commands MUST be centralized in scripts/commands.sh; tasks and docs must reference it.

### MUST-024: Output layout has fixed entrypoint results/index.html

Outputs MUST be organized per project with results/index.html as the fixed entrypoint; resolved config and summary must be saved per run.

### MUST-025: No directory explosion for DOE; case dimension storage

DOE outputs MUST be stored without per-case deep directories; store case-dimension arrays (npz/zarr/hdf5) and summary tables.

### MUST-026: Standard plots and HTML report are generated

Generate thickness map, radial profile, Cs/Cref, Da proxy, apparent order maps, plus an HTML report linking outputs.

### MUST-027: Numerical health metrics are logged

Root iteration counts, failure rates, monotonicity fallback usage, and constraint violations MUST be recorded and visualizable.

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



## SHOULD

### SHOULD-029: DOE runner supports grid/random sweeps and summary metrics

Provide sweep runners with grid/random sampling, producing summary metrics (uniformity, center-edge, etc.) and comparison plots.

### SHOULD-030: Benchmark helper to evaluate CPU vs JAX CPU vs JAX GPU

Provide a benchmark mode that reports performance; it must not force resource selection.

### SHOULD-031: Additional kinetics models (competition/LHHW, ER-like) are supported

Add competition adsorption / LHHW-like and ER-like reduced models for broader applicability.

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


