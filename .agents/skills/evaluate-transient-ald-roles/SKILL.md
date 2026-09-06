---
name: evaluate-transient-ald-roles
description: Fit anonymous Fluent species to the repository's transient ALD storage, conversion, and inhibitor roles across dose, purge, and cycle conditions, then assess condition transfer and role stability. Use for ALD role and parameter evaluation. Do not use for steady CVD equation censuses, simulation-only runs, or model implementation.
---

# Evaluate Transient ALD Roles

Determine which role-state structure transfers across ALD conditions while preserving the process clock. Cycle diagnostics support role judgment; they are not a separate reaction framework.

## Confirm that the data contain a state problem

Read `AGENTS.md`, the simulation config, and the measurement-condition list. Each Fluent input must provide spatial coordinates, a strictly increasing time array, and concentration frames with a documented species order. Identify dose, purge, conversion, and cycle intervals from supplied timing or metadata; do not infer phase chemistry from anonymous species names alone.

Check coordinate alignment, final or time-resolved measurement units, condition weights, concentration contrast, purge duration, dose range, cycle count, replicates, and measurement uncertainty. A final thickness map can constrain an integrated response but generally cannot identify every storage, release, and conversion coefficient.

Confirm `Gamma_s` before interpreting an absolute surface or transport flux. It converts
coverage rate to kmol/(m2 s). `Gamma_s: 1.0` is acceptable for a normalized study, but
its flux magnitude is then normalized and cannot calibrate an elementary constant.

## Use the ALD state path

Use `role_ald_state` for the primary ALD role-state comparison. It represents stored A, optional stored inhibitor I, and A-only or B-assisted conversion. Keep CVD QSS equations and MvK out of this fit unless the user requests a physically distinct ALD model change with suitable observations.

Read [references/run-and-visualize.md](references/run-and-visualize.md) for the current
commands, output tables, and visualization limits. Prefer the multi-condition
configuration for role evidence:

```powershell
uv run python scripts/generate_multicond_fit_inputs.py `
  --output-dir runs/generated_inputs/multicond_fit
uv run python -m deposim_opt.run_fit --config-name fit_ald_state_multicond_min
```

Replace condition file paths through config or Hydra overrides without changing train/holdout labels. Use `fit_ald_state_min` only for smoke, parameter recovery, or explicitly single-condition work; a single condition cannot establish transfer or role stability.

For a reproducible generated-input check from Bash, use the repository command:

```bash
uv sync --extra optuna
bash scripts/commands.sh fit ald
```

Keep generated inputs in `runs/generated_inputs/` and fit outputs in `results/`.

## Keep selection and evaluation separate

Enumerate disjoint A/AI/AB/AIB assignments from configured candidates. Select parameters and roles using measured training conditions only. When multiple measured training conditions are available, use condition refits to assess the selection procedure and keep configured holdouts out of ranking, search bounds, and stopping rules.

Compare added roles against simpler structures on the same folds:

- A versus AI tests whether a retained inhibitor state is needed;
- A versus AB tests whether a B-assisted conversion path is needed;
- AB versus AIB tests whether inhibitor storage adds transferable value.

Do not interpret a selected coefficient as an elementary rate constant unless the site capacity, concentration and flux units, film conversion, and observation timing make that conversion identifiable.

Keep optimizer and scientific choices separate. Select `random`, `tpe`, or `cmaes`
under `opt.parameter_fit.search`; do not change the parameter bounds, Loss, or folds when
comparing samplers. Use at least two recorded seeds when optimizer repeatability is part
of the claim. Inspect `optimization_summary.csv` and `optimization_trace.csv` before
interpreting small score differences. A missing Optuna backend is an execution error,
not permission to substitute random search.

## Read ALD-specific evidence

Read [references/ald-evidence.md](references/ald-evidence.md) before making an adoption statement.

Inspect `role_summary.csv`, `role_ranking.csv`, `role_stability.csv`,
`condition_scores.csv`, `loss_components.csv`, `optimization_summary.csv`,
`optimization_trace.csv`, resolved configs, the fit manifest, and the two generated
optimizer figures. Separate final
thickness prediction from optimizer convergence, cycle repeatability,
saturation/plateau behavior, purge growth, spatial shape, and role stability.

Plateau, cycle, purge, coverage, and pathway terms enter the objective only as measured
observations with uncertainty. Otherwise treat them as diagnostics and do not construct
a heuristic penalty from the simulated value.

Finish by stating the validated condition range, selected observable role structure,
unstable alternatives, and parameter limitations. For every unresolved target use,
state the smallest added measurement and recipe variation that separates the surviving
models and the evidence criterion it must pass. Include time-resolved uptake, calibrated
temperature variation, site density, and wall concentration/flux when elementary
parameters are requested; do not stop at saying that final GPC is insufficient.
