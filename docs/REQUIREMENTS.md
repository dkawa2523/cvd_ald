# Product and scientific requirements

These requirements define the current reaction-role assimilation product. Numeric IDs
provide stable references for code review and scientific discussion; the current model
and workflow documents define their operational meaning.

## Scientific scope

### MUST-007: Role models are the public kinetic interface

Raw Fluent species are assigned to disjoint `A`, optional `B`, optional `I`, or unused
roles. The public dynamic models are `role_cvd_aib`, `role_cvd_mvk`, and
`role_ald_state`. Species names do not carry chemical meaning by themselves.

### MUST-008: Role assignments are validated

`A` is required. `B` and `I` are optional and each names at most one raw species. One
species cannot occupy more than one role. Unused species remain allowed.

### MUST-003 and MUST-016: CVD and ALD share role inputs but retain distinct physics

Steady and transient CVD and transient ALD use the same Fluent field and role-assignment
semantics. Their surface-state equations, numerical updates, and model-readiness metrics
remain process specific.

### MUST-021 and SHOULD-031: Equation families are registered and comparable

New families provide an equation, required roles, exact reductions, exchange symmetry,
parameter bounds, and evidence requirements through the existing registry. A new family
must enter the common fit and validation workflow rather than introduce a second ranking
system.

### MUST-010: Physical bounds and signs are explicit

Concentrations and nonnegative rates remain nonnegative; coverages and capacity fractions
remain in `[0,1]`; deposition is positive and etch/loss subtract from net film rate.

## Transport and process state

### MUST-001 and MUST-002: Concentration location is explicit

Reference-plane concentration `C_ref` and surface concentration `C_s` are different
quantities. The input records the location and reference-plane metadata. The software
may use a declared `bulk_as_surface` approximation, but must report that approximation
and must not describe it as a solved wall transformation.

### MUST-006: Mass-transfer closure is pluggable

The active path accepts a supplied surface concentration, a scalar/field mass-transfer
coefficient, or a documented CFD transport-capacity flux. Stagnant-film, rotating-disk,
and Bosanquet utilities supply candidate coefficients without changing the reaction
model.

### MUST-009: Flux and role diagnostics preserve meaning

Where defined, outputs include `CsA_over_CrefA`, `CsB_over_CrefB`, A/B surface and
transport fluxes, inhibitor factor `f_I`, and B transport-demand ratio `phi_B`. Missing
physical inputs produce unavailable diagnostics rather than fabricated zero values.

### MUST-011 and MUST-012: Dynamic state updates are bounded

Dynamic AIB and MvK use bounded implicit Euler with bisection and a counted fallback for
non-bracketed steps. The ALD state uses bounded explicit substeps and reports projection
and substep diagnostics. The configured solver name must match the executed method.

### MUST-013 and MUST-014: Reaction orders and limiting regimes are tested

Configured reaction orders obey the implemented integer and total-order limits. Tests
cover reaction-limited, transport-limited, zero-co-reactant, state-bound, and time-step
behavior appropriate to each model.

## Estimation and evidence

### MUST-057: Film-map comparison and role ranking are first-class

The workflow aligns each film observation once, records alignment distance, fits all
applicable candidates, and writes role summary, ranking, stability, and per-condition
scores. Mean bias and centered wafer-pattern error are separate quantities.

### MUST-062: Candidate selection estimates predictive error

Training loss, condition-refit prediction error, ordinary RMSE/MAE/max error, exact
reduction evidence, role stability, and local parameter sensitivity are reported
separately. Condition-refit error determines selection when available. Simpler models
are preferred only when their paired error is statistically indistinguishable. A small
training loss alone cannot produce `adopt`.

### MUST-063: Multi-condition validation prevents leakage

Conditions carry explicit train or holdout status and condition-balanced weights.
Normalization, fitting, model selection, and reference baselines use training conditions
only. An external holdout remains untouched until one model and parameter set are fixed.
Outer condition folds repeat the complete selection procedure to measure selection
stability.

### SHOULD-033: Transport and reference-location sensitivity are reported

When the inputs support more than one concentration location or transport closure, the
workflow compares them under the same observation and split. The comparison is a model
sensitivity analysis, not proof of the correct boundary condition.

### MUST-064: An unresolved use produces an experimental requirement

The workflow assesses wafer spatial correction, anonymous-species role assignment, and
elementary kinetic-parameter estimation independently. For each use lacking evidence it
writes a reusable measurement requirement with the controlled variation, ambiguity
resolved, workflow insertion point, and readiness criterion. The output must not stop at
a generic statement that the current data cannot support the use.

## Configuration, code, and outputs

### MUST-019, MUST-020, and MUST-052: Configuration and package boundaries are stable

YAML configuration separates `sim` and `opt`. Simulation code owns physical state and
flux; optimization code owns fitting and selection; report code presents computed
artifacts. Components are usable independently through Python and composable in the
pipeline.

### MUST-056: Compatibility is checked before execution

Model metadata declare required roles, excluded combinations, supported time modes, and
governing class. Invalid model/input combinations stop before numerical execution.

### MUST-004, MUST-024, MUST-026, and MUST-061: Outputs are reviewable

Configured simulation produces a wafer map, resolved configuration, metrics, plots,
report, and a machine-readable `output.v1` manifest. The steady equation census produces
role tables, per-condition predictions, uncertainty diagnostics, target-use data
requirements, figures, source hashes, and its analysis manifest. Human-readable
summaries link to, rather than recalculate, these artifacts.

### MUST-027: Numerical health is observable

State bounds, projection counts, non-bracketed implicit steps, applicable residuals, and
validation violations are recorded. A diagnostic that does not apply is marked
unavailable rather than successful.

### MUST-023, MUST-028, MUST-051, and MUST-060: Operation stays compact

Common run and verification commands remain in `scripts/commands.sh`; Python and YAML
remain the primary programmatic interface. Generated inputs belong in
`runs/generated_inputs/`, run outputs in `results/`, and heavy dependencies remain
optional unless a verified requirement needs them.

## Adoption boundary

The code is responsible for fair candidate enumeration, numerical correctness,
leakage-free validation, transparent approximations, and conservative reporting. The
data are responsible for independent role excitation, required physical metadata,
measurement uncertainty, mechanism-specific observables, and a declared application
tolerance. If either side is missing, the result remains `review` or `reject` even when
the fitted error is small.
