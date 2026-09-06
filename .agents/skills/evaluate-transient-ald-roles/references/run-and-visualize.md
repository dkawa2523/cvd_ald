# ALD role-fit execution and visualization

Use this reference for a multi-condition ALD role fit. Read `ald-evidence.md` for the
scientific claim levels.

## Execution

Generate or supply time-resolved Fluent and measurement inputs, then run the configured
fit:

```powershell
uv run python scripts/generate_multicond_fit_inputs.py `
  --output-dir runs/generated_inputs/multicond_fit
uv run python -m deposim_opt.run_fit --config-name fit_ald_state_multicond_min
```

Use Hydra overrides for source file paths, train/holdout assignment, declared
uncertainties, or a requested sampler. Do not edit generated fixtures into source data.
The multi-condition config is the role-selection path; `fit_ald_state_min` is suitable
for smoke and parameter-recovery work.

## Current fit outputs

The run directory contains resolved configuration and manifest files plus the following
tables under `tables/`:

| Table | Purpose |
| --- | --- |
| `ranking.csv` and `class_compare.csv` | Candidate order and comparable role structures |
| `role_summary.csv` | Adoption or review statement derived from stored evidence |
| `role_stability.csv` | Raw-species and role-structure persistence across condition refits |
| `condition_scores.csv` | Train, condition-CV, and holdout errors by condition |
| `loss_components.csv` | Measured data terms and any configured numerical/prior terms |
| `optimization_summary.csv` | Backend, active dimension, budget, repetitions, convergence, and repeated best-score range |
| `optimization_trace.csv` | Trial-by-trial best-so-far history for train and condition refits |

The current fit report generates `plots/optimization_convergence.png` and
`plots/loss_components.png`. Read the first for numerical search stability and the
second for the composition of the fitted objective. A lower objective is not a role or
mechanism conclusion; role evidence still comes from condition transfer, exact
structure comparisons, and stability.

The forward-simulation report may also contain thickness, measurement-error, radial,
surface-to-reference concentration, identifiability-correlation, and solver-health
figures. Use them only when their source fields are present in the manifest. State and
pathway histories remain numerical predictions unless matched time-resolved
measurements enter the objective with declared uncertainty.

## Figure requirements for an ALD scientific comparison

When the requested evidence exists, prefer a small set of conventional figures:

1. measured and predicted thickness or uptake versus time for dose, purge, and
   conversion segments;
2. heldout final-thickness map and residual map on a common physical scale;
3. stored-A, inhibitor, and free-site histories on a 0–1 axis;
4. A-only and B-assisted conversion rates in compatible units;
5. role-selection frequency across complete recipe holdouts;
6. optimizer convergence only as a numerical-quality panel.

Do not invent target plateau, purge, or pathway curves when no observation exists.
Avoid combining coverage, molar flux, and thickness on one unlabelled axis. A pathway
diagram may state the assumed storage/release/conversion topology, but arrows remain
model terms until a time- or state-sensitive observation distinguishes them.

## Completion

Confirm that a complete recipe or condition is held out, timestamps and coordinates are
aligned, final thickness is not duplicated as the final history point, and every cited
figure resolves through the manifest. Report the fitted range of dose, purge, cycle,
temperature, and concentration rather than claiming transfer beyond it.
