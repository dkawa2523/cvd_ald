# CVD spatial concentration/rate analysis

The production analysis path uses multiple conditions for identification and one
condition for a no-refit transfer test:

```bash
./scripts/commands.sh analyze_cvd_multicond
```

The default split is conditions 1+2+4+5 for identification and condition 3 for
testing. It also reports all five leave-one-condition-out splits. Response
coefficients transfer between conditions; an unseen condition does not receive
a fitted offset or measured-rate calibration.

The production response is `surface_qss`. It assigns raw species to candidate
surface roles and evaluates the quasi-steady site-balance reduction
`theta_* + theta_A + theta_I = 1`. The fitted values are a rate scale and
dimensionless observable groups (`h`, `delta`, `b`, and `kappa` as applicable),
not separate adsorption, desorption, conversion, or site-density constants.
Each candidate is refitted in condition CV using condition-balanced squared
error in deposition-rate units. The rate scale is profiled analytically, leaving
at most three nonlinear shape groups. Identification-data concentration
references are locked for validation and test prediction.

The candidate set includes A, AI, the steady-symmetric AB pair, AIB, a
constant observable-response boundary, and exact no-desorption
reductions. `role_ranking.csv` records the formula, observable groups, kinetic
limit, reduced model IDs, per-condition scores, local sensitivity rank, and
search-boundary warnings. Site coverages are written with test predictions when
the selected reduction identifies them. Mechanism presence, observable
sensitivity, and raw-species assignment remain separate judgments.

Use `--response-model empirical_power` for the previous empirical response.
The sections below document that compatibility model and its
`--response-structure` option.

## Empirical power compatibility model

When the empirical model is requested, `shared` coefficients are its default. The existing Python API
accepts `response_structure="shared"`, `"within_between"`, or `"select"`; the
CLI exposes the same choice through `--response-structure`. `select` compares
both structures using training CV. The 2026-09-03 data evaluation found a 39.2%
increase in outer MSE when separation was added to selection, so it is available
for explicit hypothesis comparisons and is not the production default.

The two response structures are nested. `shared` uses the same
elasticities within and between conditions. `within_between` uses
`log(rate) = intercept + mean_map(x) @ beta_between + (x - mean_map(x)) @ beta_within`,
where x contains the existing signed log-concentration/fraction features.
Means use the full supplied Fluent map, including when predicting a subset or
resampling observations; measured rates never enter input centering. Both
coefficient blocks keep the model's nonnegative effect constraints. These are
empirical responses, not independently modified CVD/ALD kinetic constants.

Estimation minimizes condition-balanced mean squared log-rate error. Shared
fits penalize `lambda * ||beta||^2`. Separated fits penalize
`lambda * ((||between||^2 + ||within||^2)/2 + ||between - within||^2)`.
The shared subspace has exactly the original penalty; the intercept is unpenalized.
The dimensionless elasticities use unit scale. The fixed grid
`[0, 1e-8, 1e-6, 1e-4, 1e-2, 1]` includes unregularized estimation and is selected
jointly with the role and response structure using condition-CV MSE in physical rate units. This choice
makes the relative-error fitting assumption and absolute-error selection target
explicit. Inner scores select; outer condition predictions evaluate the procedure.
The shared CVD/ALD ranking helper prefers fewer active effects only for numerical
loss ties. The compatibility field `equivalent_to_best` now means only a numerical
score tie, never statistical or practical equivalence. Spatial blocked CV remains
a diagnostic conditional on the selected strength. Bootstrap holds that strength
fixed and is also conditional on the selected model and response structure.
Numerically tied structures prefer `shared`; no practical-equivalence threshold is inferred.
If a setting overflows or its linear solve fails on a condition fold, that
setting has infinite selection risk and its reason is retained in
`regularization_scores`. Predictions are never clipped to obtain a finite score.
An entirely failed search raises an error rather than returning a model.

Outputs retain nominal roles and separately show `effective_roles`,
`inactive_roles`, `role_symmetry`, `regularization`, `response_structure`, and
`effect_scopes`. `common_total_order` is the between-condition order in separated
fits; `within_total_order` gives the within-map order. Coefficients and log
contributions carry their scope explicitly. A zero coefficient is not
evidence for that role. AB fraction-product effects are symmetric under A/B
exchange, so they cannot resolve the two roles' direction. This symmetry does
not apply automatically to the process ODEs. Predictive status and role support
are separate columns in the existing role summary.

Permitted reduced terms are enumerated before fitting. Each reduction refits its
remaining coefficients and selects regularization on the same training folds.
`I_response:<species>` is a common-scale plus inhibitory response whose driving
species is unassigned; it is not an I-only film-growth mechanism. A full-fit zero
does not remove a role from other folds. `effect_groups` identifies active terms,
and `reduced_model_comparisons` reports independently refitted, paired CV losses
for total error, squared mean bias and centered spatial error. Their differences
also appear per condition in `condition_scores.csv`.
`role_evidence` separates necessity of an effect from assignment to a raw species.
`consistent_benefit` requires at least two conditions, none with worse total loss
and at least one with better loss, allowing only floating-point roundoff.
Crossing losses are `mixed`; absent comparisons are `not_assessed`. Assignment
requires consistent benefit over each enabled alternative that changes only
that effect. Missing reductions or alternatives leave support unresolved, also
when native model configuration disallows those comparisons. This does not
automatically add physical reaction mechanisms or relax candidate restrictions.
These are descriptive inner-CV comparisons after tuning, not significance tests,
selection-independent confidence intervals or chemical-causality evidence.
Different nominal candidates with the same active effects count once per refit
in `role_stability.csv`; symmetric AB pairs occupy slot `AB`.

A selected candidate must improve on a constant estimated from training data.
Reports separate mean bias from centered spatial error: a close mean with poor
map shape is labeled `review`. A failed prediction is `reject_prediction`;
inconsistent roles across outer condition splits also require review. These
decisions are calculated from the run, not fixed statements about a species or
condition. All candidates may be inadequate. Small test RMSE alone does not
establish a chemical role or transferable kinetic law.

The summary uses the same decision function as the CVD/ALD simulator fitter.
`condition_scores.csv` labels `quantity=deposition_rate`, `unit=nm/s`, and
`evaluation_scope`. Inner selection scores, the primary fixed-model test and the
outer evaluation of the selection procedure are separate evidence. A successful
primary test cannot hide a failure elsewhere in the evaluated procedure.
Optional Python argument `application` accepts `conditions`,
`max_relative_rmse`, and `require_spatial` (default true). Without user-supplied
scope and tolerance, the numerical winner is reported but never automatically
adopted. Good outer-procedure results alone cannot certify a fixed model.

Angular-block bootstrap intervals describe coefficients conditional on the
selected model and the observed conditions. They do not measure uncertainty in
model selection or transfer to an unobserved operating regime.

For the wall-zero diagnostic, the analysis sets each wall species concentration
to zero and reports `C_bulk - C_wall = C_bulk` in `test_predictions.csv`. This is
a concentration driving-force proxy, not an absolute molar flux: diffusivity or
a mass-transfer coefficient and a wall-normal distance/gradient are not present
in the supplied files. Pressure is therefore not required for the current
empirical concentration fit, but is needed together with temperature and
transport properties when the objective is a physical wall-flux or kinetic
transfer model.

Use the following only as a single-condition spatial diagnostic:

```bash
./scripts/commands.sh analyze_cvd_case
```

Single-condition defaults are `data/condition_1.csv`, `data/validation_1.csv`, and
`results/cvd_condition_1_analysis/`.  The command can also be called through
`scripts/analyze_cvd_spatial_case.py` with explicit `--condition`,
`--validation`, and `--output` paths.

The analysis treats `concentration_*` names as raw species.  It compares
baseline, A, AI, AB, and AIB spatial-response candidates, constrains role
effects to their interpretable signs, and ranks them by the worse of angular-
sector and radial-band cross-validation RMSE.  Coefficient uncertainty uses
an angular-block bootstrap.  `molef_*` and `density` are used only to check
input consistency when they repeat information already present in the
concentrations.

Reported coefficients are local effective response slopes around the median
concentration state.  They are not elementary reaction constants.  A single
spatial condition can support an association model, but cannot establish
chemical identities, causal species roles, or out-of-condition kinetics.

Primary artifacts are `role_summary.csv`, `role_ranking.csv`,
`role_stability.csv`, `model_ranking.csv`, `coefficients.csv`,
`condition_scores.csv`, `test_predictions.csv`, `split_sensitivity.csv`,
`candidate_test_diagnostics.csv`, `analysis_summary.json`, `report.md`, and the
executed companion notebook. Candidate test ranks are diagnostic only and never
feed back into model selection.
