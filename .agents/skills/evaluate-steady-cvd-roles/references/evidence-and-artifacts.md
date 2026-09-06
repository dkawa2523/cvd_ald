# Evidence and artifact contract

Use this reference when reviewing a completed steady CVD census.

## Artifact roles

- `role_ranking.csv`: comparable fitted candidates, reductions, scores, roles, symmetry, and supported effects.
- `role_summary.csv`: concise adoption or review decision and the evidence behind it.
- `role_stability.csv`: equation-family, structure, and role selection across outer condition refits.
- `condition_scores.csv`: condition-mean and centered spatial errors.
- `split_sensitivity.csv`: complete outer selection-procedure results.
- `test_predictions.csv`: frozen-model prediction and residual at each test location.
- `test_extrapolation.csv`: identification ranges and out-of-range test fractions.
- `data_requirements.csv`: target use, required measurement, experimental design,
  ambiguity resolved, workflow insertion point, and readiness criterion.
- `analysis_summary.json`: machine-readable equations, workflow layers, mechanism assessments, assumptions, and headline metrics.
- `report.md`: presentation of stored evidence; it must not perform hidden refitting.

Use [run-and-visualize.md](run-and-visualize.md) for the complete figure sequence. The
radial error bars are azimuthal standard deviations; surface fractions use the physical
0–1 colour range. Every plotted conclusion must remain traceable to its source CSV.

## Minimum numerical reading

Report RMSE and MAE in nm/s, relative RMSE against the held-out mean, mean bias, centered spatial RMSE and centered R-squared, spatial correlation when meaningful, and prediction-range capture. Pooled metrics weight map points; averages of condition-relative errors weight conditions. State which one is used.

Negative centered R-squared means the predicted map explains less within-condition variation than the correct condition mean. It can coexist with a small relative RMSE when condition-mean transfer is good.

For an assigned role (j), compare its reference-substitution prediction change
(S_j) with heldout RMSE (E) through (Q_j=S_j/E), and compare both with its outer
selection frequency. (Q_j\ll1) indicates that assignment instability has little
prediction consequence in the supplied range. (Q_j\gtrsim1) indicates an influential
unresolved assignment. This ratio is a diagnostic scale comparison rather than a test
threshold or probability.

For an alternative family (m), use the RMS difference from the selected prediction
and its ratio to (E). A small ratio means the mechanism label changes while the tested
prediction barely changes; a large ratio means mechanism ambiguity is also a prediction
risk. Neither result identifies a microscopic pathway.

## Evidence levels

Keep these conclusions independent:

- **Equation evidence:** one observable response predicts better across held-out conditions.
- **Effect evidence:** an added A, B, inhibitor, or finite-loss term consistently improves its exact reduction.
- **Assignment evidence:** one anonymous species assignment is separated from alternatives.
- **Mechanism evidence:** observations distinguish physical interpretations that may share an equation.
- **Application evidence:** the frozen workflow meets a declared tolerance on independent conditions.

Use `review` when prediction improves but the application tolerance, spatial behavior, role assignment, or mechanism remains unresolved. Do not invent a tolerance.

Treat `design_full_rank` as structural information and
`parameter_identifiability_status` as the practical sensitivity assessment. A full-rank
design can remain `weak` when scaled sensitivity directions are nearly collinear or
ill-conditioned. Neither status turns a normalized observable group into an elementary
constant.

## Code and data responsibility

Code is responsible for leakage-free references and splits, condition-balanced selection, unique equation enumeration, stable optimization, exact reductions, equivalence grouping, physical-unit metrics, and distinct workflow layers.

Data are responsible for independent A/B/I perturbations, low-coverage and saturation range, wall concentration or flux when transport is claimed, time response when state memory is claimed, temperature contrast for activation parameters, replicates and uncertainty, and chemistry or surface-state observations for named mechanisms.

For each target use, state which surviving alternatives the requested measurement
separates and the criterion that the added data must pass. Prefer the generated
requirement rows over a generic “insufficient data” statement.
