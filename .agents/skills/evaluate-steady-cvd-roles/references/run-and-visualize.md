# Steady CVD execution and visualization

Use this reference when running the production census, comparing Loss/sampler choices,
or reviewing its figures. General equations and metric definitions live in
`docs/THEORY.md`; this file records the repository workflow.

## Production command

The current five-condition reference run is:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --response-model surface_compare `
  --reaction-input bulk_concentration `
  --models all `
  --loss mse `
  --sampler pattern `
  --bootstrap-samples 100 `
  --spatial-response radial_quartic `
  --seed 123 `
  --output results/current_cvd_separated
```

Increase bootstrap samples only when interval stability is part of the requested result.
Changing `--reaction-input` requires matching columns documented in
`docs/inputs_fluent.md`; identical algebra does not preserve physical parameter meaning
across concentration and flux modes.

Use `--spatial-response none` for the primary chemistry-only result. Enable a radial
response when the task explicitly asks whether a transferable residual shape exists.
Always report chemical and corrected centered metrics side by side.

## Optimization comparison

Freeze one exact `model_id` from `role_ranking.csv` before comparing numerical methods:

```powershell
uv run python scripts/benchmark_surface_optimization.py `
  --candidate-id "<exact model_id>" `
  --trials 4096 `
  --repetitions 3 `
  --workers 8 `
  --output results/surface_optimization_benchmark_4096
```

Use `--resume` after an interrupted benchmark. Keep bounds, split, Loss list, and trial
budget fixed across samplers. Judge convergence, repeated-seed spread, runtime, boundary
frequency, training-condition CV RMSE, and the fixed holdout audit. Objective values are
comparable only within the same Loss definition.

Available steady Loss values are `mse`, `wafer_normalized_mse`,
`wafer_normalized_mae`, and `symmetric_normalized_mse`. Available samplers are
`pattern`, `random`, `tpe`, `cmaes`, `de`, `pso`, `levy`, and `cma_mae`. Use a selected
combination in the full equation census afterward; a fixed-candidate numerical result
does not establish that family or roles remain selected.

## Evidence groups

| Stage | Primary tables | Question |
| --- | --- | --- |
| Input and provenance | `condition_quality.csv`, `concentration_scaling.csv`, `test_extrapolation.csv`, `analysis_summary.json` | Are locations, units, contrast, normalization, and extrapolation known? |
| Optimization | `optimization_history.csv`, benchmark combination/trace tables | Did the search reach a repeatable minimum under the stated budget? |
| Equation selection | `role_ranking.csv`, `split_sensitivity.csv`, `role_stability.csv` | Which unique equation wins, and does the selection procedure persist across conditions? |
| Effect and role evidence | `best_model_role_assignments.csv`, `condition_mean_input_correlations.csv`, `role_input_sensitivity.csv`, `role_importance_and_stability.csv` | Which role effects matter for prediction, and are the anonymous assignments separated? |
| Mechanism and parameter interpretation | `reaction_model_predictions.csv`, `reaction_model_states.csv`, `reaction_state_summary.csv`, `parameter_sensitivity_correlations.csv`, `parameter_loss_slices.csv` | Do alternative mechanisms change predictions, and which fitted directions are weak or coupled? |
| Prediction | `condition_scores.csv`, `test_predictions.csv`, `model_structure_uncertainty.csv` | Does the frozen workflow transfer the condition mean and wafer shape? |
| Spatial response | `spatial_response_summary.csv`, `spatial_response_coefficients.csv` | Does a post-selection residual shape transfer without altering chemistry? |
| Decision and next data | `role_summary.csv`, `data_requirements.csv`, `report.md` | What is usable now, and which experiment resolves each surviving ambiguity? |

## Figure reading order

| Figure | Read it for | Do not infer |
| --- | --- | --- |
| `condition_reaction_input_contrast.png` | Between-condition range and independence of the selected concentration or flux inputs | Kinetic causality from co-variation |
| `reaction_input_correlation.png` | Pairwise condition-mean input correlation | Stable anonymous-species identity |
| `optimization_convergence.png` | Best-so-far fit error and whether each family stopped on a plateau | Parameter identifiability from a flat trace |
| `equation_family_comparison.png` | Condition-CV error and outer selection frequency | Mechanism proof from the lowest bar |
| `reaction_pathway_models.png` | State, adsorption, blocking, and conversion terms represented by each family | Confirmed elementary steps |
| `reaction_model_prediction_agreement.png` | Whether best family alternatives materially change heldout prediction | Probability that one mechanism is true |
| `best_model_role_assignments.png` | Best raw-species assignment within each family | Chemical species identity |
| `role_selection_stability.png` | Assignment changes across outer condition refits | Importance of an unstable role |
| `role_importance_and_stability.png` | Prediction consequence and assignment frequency together | Additive rate fractions |
| `role_response_curves.png` | Model-conditional response when one input is varied | Response outside the plotted data range |
| `reaction_state_summary.png` | Mean site and pathway fractions for each family | Directly measured coverages |
| `selected_surface_state_maps.png` | Model-conditional spatial state pattern | Surface spectroscopy |
| `kinetic_parameter_sensitivity.png` | Local log-rate sensitivity and correlated parameter directions | Global uniqueness or confidence intervals |
| `parameter_loss_slices.png` | Flat or sharply changing one-parameter directions with rate scale reprofiled | Full profile likelihood; other shape parameters are fixed |
| `test_spatial_maps.png` | Measured, chemical prediction, and residual maps | Mean-rate transfer from visual colour alone |
| `test_radial_profile.png` | Radial means and azimuthal variation | Azimuthally resolved agreement |
| `model_structure_prediction_spread.png` | Prediction sensitivity to outer-selected structures | Calibrated posterior uncertainty |
| `spatial_correction_performance.png` | Chemical versus corrected centered performance across heldout conditions | Chemical evidence from the correction |
| `test_spatial_response.png` | Centered map shape before and after spatial response | A physical cause for the residual basis |
| `spatial_residuals.png` | Residual structure removed and retained | Transfer beyond the tested conditions |
| `spatial_correction_profile.png` | Magnitude and radial form of the learned correction | Temperature or transport causality without corresponding fields |

Read figures with the source CSV, not as standalone evidence. A scientific report should
show one comparison per panel, physical units, conventional labels, and the relevant
holdout scope. Keep colour ranges common for compared maps and use 0–1 for fractions.
The complete illustrated reference is
[`docs/VISUALIZATION_GUIDE.md`](../../../../docs/VISUALIZATION_GUIDE.md); use it for axis,
marker, error-bar, source-table, and interpretation details rather than duplicating
those definitions in an evaluation response.

## Completion checks

Confirm that `manifest.json` resolves every declared artifact, `report.md` performs no
hidden refit, and `analysis_summary.json` records the exact split, reaction input, Loss,
sampler, spatial response, seed, and source hashes. Visually inspect all figures used in
the report for clipped labels, misleading colour ranges, and inconsistent units.
