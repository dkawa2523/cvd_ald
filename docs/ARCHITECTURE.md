# Architecture

This document describes the intended package boundaries, dependency direction, and
how configuration and outputs are managed.

---

## Package boundaries (logical separation)

The system is split into top-level Python packages (within a single repo / single editable install),
to match the requirement that numerical simulation and optimization/ML are separated.

Recommended top-level packages:

- `deposim_schema`
  - dataclasses / pydantic models for configs and I/O contracts
  - unit conventions and validation
  - minimal dependencies

- `deposim_sim`
  - numerical simulation core
  - transport–reaction coupling solver
  - domain grids and operators
  - model registry (mass transfer, rate laws, state models, net models)
  - output writing (arrays + metadata)
  - does NOT depend on optimization libraries or ClearML

- `deposim_report`
  - plotting and HTML reporting
  - deterministic outputs for runs

- `deposim_opt` (P2)
  - data assimilation / parameter estimation / surrogate modeling
  - optional JAX engine, implicit differentiation, adjoints
  - depends on (optional) heavy deps: jax, jaxopt, diffrax, etc.

- `deposim_tracking_clearml` (P2)
  - ClearML integration layer
  - optional dependency
  - core simulation must remain independent

**Dependency direction:**
schema -> sim -> report
schema -> opt (and optionally sim)
tracking is leaf-only (depends on outputs, not vice versa)

---

## Source layout

We standardize on **src layout** to avoid import ambiguity:

    src/
      deposim_schema/
      deposim_sim/
      deposim_report/
      deposim_opt/          (later)
      deposim_tracking_clearml/ (later)

Import stability is handled by `scripts/preflight.sh`:
- prefers `pip install -e .`
- otherwise sets `PYTHONPATH=src`

---

## Model registry

To keep the code base small and extensible:

- models are discovered via a registry module rather than deep directory trees.
- the registry provides a stable API for adding new models:
  - mass transfer models (k_m)
  - process models (CVD continuous role models, ALD cycle/state role models)
  - optional internal rate/state kernels used by those process models
  - net models (dep/etch/loss composition)

The registry pattern supports YAML selection of model names. Existing `aib_ode`
configs are compatibility inputs; new CVD/ALD work should add model aliases or
registry entries instead of hard-coding more `aib_ode` checks.

The process-model naming direction is intentionally small:

- `role_cvd_aib`: CVD-facing continuous role model path.
- `role_ald_compat`: diagnostic ALD transient compatibility path.
- `role_ald_state`: minimal ALD role-state assimilation model defined by ADR 0020.

`role_ald_state` uses an A-only event when B is absent and an A+B conversion
event when B is present, as fixed by ADR 0021. This keeps A/AI versus AB/AIB a
fair role-selection comparison without adding another model family.

---

## Configuration management (Hydra + YAML)

- user-facing configuration is YAML-based (no CLI required)
- Hydra is used to compose configs, but we avoid config file explosions by:
  - keeping group count small
  - using `name:` + `params:` patterns rather than per-model YAML files
  - saving exactly one `config_resolved.yaml` in the run directory

YAML is split:
- `configs/sim/` for forward simulation
- `configs/opt/` for assimilation/optimization

---

## Output layout (anti-"迷子" design)

All simulation outputs are stored under a **project** directory (user-chosen), with a fixed entrypoint:

    results/
      index.html        # always the entry point
      summary.json      # quick glance metrics
      runs/
        <run_id>/
          config_resolved.yaml
          inputs_preview/
          outputs/
            thickness.<store>
            diagnostics.<store>
          plots/
          report.html

Key requirements:
- avoid nested case directories for DOE
- store DOE results in a case dimension (zarr/npz/hdf5; chosen by implementation)
- keep run metadata close to run outputs
- record dirty-worktree state in provenance so a commit hash is not mistaken for
  a complete code snapshot
- fit runs expose training, condition-refit, and external-holdout errors, together
  with the reason a role assignment is adopted, reviewed, or rejected

### Responsibilities in role assimilation

The existing CVD/ALD dispatcher remains the only simulator entry point. CVD and
ALD retain their separate process models. Fitting is split into small modules:

| Module | Responsibility |
| --- | --- |
| `deposim_sim/measurement_adapter.py` | Sample model predictions at original observations; apply coordinate and rate/thickness conversion. Mesh-resampled measurements are for plotting. |
| `deposim_opt/fit_conditions.py` | Read condition settings, prepare a simulator input, and evaluate one condition. |
| `deposim_opt/objective.py` | Compute optimization loss and physical error metrics from observations. |
| `deposim_opt/fit_optuna.py` | Fit continuous parameters for one role/order candidate, including cache and fidelity. |
| `deposim_opt/enumerate_roles.py` / `cvd_spatial_analysis.py` | Define role effects and permitted reductions; only the empirical AB product is exchangeable. |
| `deposim_opt/fit_roles.py` | Enumerate candidates, cache simulator refits, and provide the shared condition-fold runner. |
| `deposim_opt/class_compare.py` | Pure ranking, independently refitted reduction comparisons, effective-role stability, and one adoption summary for both paths. |
| `deposim_sim/identifiability.py` | Evaluate all fitted parameter directions across all training observations with a scaled sensitivity SVD. |
| `deposim_opt/run_fit.py` | Write the existing ranking, summary, condition tables, and report. |

With at least two measured training conditions and shared parameters, the primary
`selection_score` is the condition-weighted mean squared prediction error from
condition refits. Only numerical loss ties prefer fewer active effects and
parameters; CV variability is not a performance-equivalence margin. `best_score` remains
the full-training optimization loss, so the selected candidate need not minimize
it. External holdouts do not affect either fit or ranking; they can reject the
chosen candidate without choosing a replacement using test data.
Condition refits always use fresh studies. For persistent full-training Optuna
studies, the configured study name is a prefix; a suffix derived from training
inputs, candidate settings, and objective prevents reuse of unrelated trials.

Role stability repeats the same condition-CV selection on each training-condition
subset, sharing votes among numerical ties and counting duplicate signatures once.
Subset fits are cached. Disabling this optional analysis never disables the
condition CV used for selection. With fewer than three conditions, repeated
selection is unavailable; score ties are explicitly not selection stability.
Prediction status and role support are reported separately in the role summary.

The empirical rate fitter also uses the shared condition-fold runner, metrics,
ranking, stability and summary. Permitted term reductions are enumerated before
fitting and select their own regularization using training folds. A zero effect
on the full fit never filters the fold candidate sets. Effect identity describes
active terms, not equality of coefficients or predictions. The physical models
retain ordered A/B roles and only compare reductions enabled by their existing
configuration; isolated kinetic zeros do not prove that a species is inactive.

Condition tables label quantity/unit and distinguish training fit, inner
selection, fixed-model holdout and outer selection-procedure evaluation. An outer
fold uses its own fitted effects for stability; duplicate aliases share one vote.
The AB product is reported as an undirected pair in stability tables.
No application adoption is inferred without user criteria. Optional
`opt.selection.application` specifies `conditions`, `max_relative_rmse` and
`require_spatial` (default true). The empirical Python entry point accepts the
same mapping as `application`. Criteria do not affect selection. A successful
outer evaluation of model selection cannot certify the primary fixed model.

Local identifiability uses the observation residuals and all estimated parameters,
scaled by parameter magnitude and observation uncertainty or signal magnitude.
SVD exposes dependent combinations involving more than two parameters. A clean
local result does not prove global uniqueness or a chemical mechanism. Disabled
analysis is reported as unassessed, not a successful identifiability result.
Legacy `topk_window` and `max_paths` analysis settings are ignored; examples no
longer advertise them. `role_stability.score_epsilon` is a relative tie tolerance.

---

## Single Source of Truth: run/test commands

All automation and docs MUST reference:

- `scripts/commands.sh`

It defines:
- python executable and env
- smoke run command
- verify gates

This avoids python vs python3 accidents and keeps CI consistent.

---

## GPU / CPU support

The simulation engine must be selectable by YAML:

- `engine: numpy` (CPU robust baseline)
- `engine: jax`   (optional CPU/GPU, differentiable)

**Important:**
- compute resource selection is user-controlled
- no hard-coded "single=CPU, DOE=GPU" assumptions

A benchmark helper may recommend settings, but must not override explicit user choice.
