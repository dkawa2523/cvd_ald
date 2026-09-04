# Artifacts and reporting

Read this reference when implementing outputs, handing off results, or writing a durable report.

## Minimum reusable artifacts

Follow the repository's existing artifact conventions. Prefer a concise set that preserves the decision trail:

- `role_summary.csv`: selected candidate, decision, reason, prediction status, spatial status, and role support;
- `role_ranking.csv`: comparable candidate scores and exact-reduction evidence;
- `role_stability.csv`: role or effect frequency across outer refits;
- `condition_scores.csv`: per-condition mean and spatial metrics;
- `split_sensitivity.csv`: complete outer selection-procedure results;
- `test_predictions.csv`: coordinates, measured response, prediction, and residual;
- `test_extrapolation.csv`: identification ranges and outside-range fractions;
- `analysis_summary.json`: equations, references, selected observable parameters, assumptions, and primary results;
- `manifest.json`: generated files, inputs, hashes or provenance fields required by the repository.

Do not create a second diagnostic hierarchy when equivalent files already exist. Extend existing rows and metadata when semantics remain coherent.

## Content requirements

Every selected-model artifact should make clear:

- process mode and response structure;
- role assignment and unordered effect groups;
- exact formula and observable parameter names;
- identification conditions, test condition, and locked references;
- selection metric and tie rule;
- model boundaries or parameters at the search boundary;
- fixed-model versus outer-procedure scope;
- unsupported interpretations.

Prediction files should contain physical units. Avoid normalized-only output that cannot be checked against measurements.

## Technical report structure

When a durable report is requested, use this order unless the audience needs a different shape:

1. result-first technical summary;
2. data and decision scope;
3. changed equations and implementation responsibilities;
4. comparable baseline versus new evaluation;
5. physical interpretation and supported limiting behavior;
6. identifiability and role stability;
7. code limitations versus data limitations;
8. missing-data matrix and acquisition plan;
9. validated practical uses and prohibited claims;
10. reproducibility and evidence files.

For a separate kinetics paper, include the state balance, reduction derivation, nondimensionalization, limiting regimes, scale profiling, structural invariances, transport closure, temperature dependence, and primary literature. Cite authoritative definitions and primary papers rather than generic web summaries.

## Writing discipline

Use the model's exact evidence level:

- “supports an AB-type effective response” is weaker and often more accurate than “identifies the reaction mechanism”;
- “finite loss group” is safer than “desorption constant” without desorption evidence;
- “bulk-concentration proxy” is different from “surface concentration”;
- “condition-mean transfer” is different from “spatial-map prediction”;
- “outer selection procedure” is different from “one fixed model.”

Keep equations and reported metrics consistent with stored artifacts. If a number appears in prose, derive it from the same source used by the table or chart.

## Verification

Before delivery:

- rerun the production command from a clean or documented state;
- verify artifact completeness and manifest integrity;
- recalculate headline metrics from prediction rows;
- inspect at least one difficult holdout and one representative spatial map;
- confirm that test data were not used for fitting, references, or wording thresholds;
- render reports and inspect equations, tables, units, citations, and page overflow;
- remove temporary QA files and generated caches that are not part of the deliverable.

Report the exact validation performed and any remaining limitation that affects use.
