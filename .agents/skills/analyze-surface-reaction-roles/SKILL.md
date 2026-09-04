---
name: analyze-surface-reaction-roles
description: Analyze anonymous CFD species against measured CVD or ALD film maps by fitting interpretable reduced reaction-role models, testing condition transfer and identifiability, and separating model limitations from missing-data limitations. Use for this repository when raw species names are simulation fields rather than trusted chemistry labels. Do not use for CFD-only validation or for detailed mechanisms whose species and elementary steps are already established.
---

# Analyze Surface Reaction Roles

Build the smallest physically interpretable role model that the supplied conditions can distinguish. Treat predictive accuracy, role evidence, and chemical interpretation as separate conclusions.

## Start from the repository and the decision

Read repository instructions and inspect the existing config, runner, model, and artifact patterns before changing code. Identify:

- the operational decision the model must support;
- the measured response, units, spatial coordinates, and condition identifiers;
- whether the process is steady CVD or cyclic/transient ALD;
- which CFD fields are raw species inputs;
- the current production command and baseline result.

Preserve the existing main path when it is coherent. Add a new abstraction only when the current file cannot own the responsibility without mixing equations, fitting, and reporting.

If the request is analytical only, inspect and report without modifying code. If the user asks to improve the implementation, establish a reproducible baseline on the same condition splits before editing.

## Audit whether the data can answer the role question

Before fitting, verify row counts, finite values, units, coordinate alignment, response resolution, condition coverage, and the exact mapping between simulation and measurement points. Quantify:

- within-condition range of every candidate species;
- between-condition scale and composition changes;
- pairwise dependence or near-collinearity among species;
- test points outside the identification range;
- replicate and measurement-uncertainty availability.

Do not equate many spatial points with many independent kinetic experiments. A species whose field barely varies, or two species that move together, cannot be assigned distinct roles reliably.

Read [references/model-reduction.md](references/model-reduction.md) when selecting or changing equations. Read [references/evaluation-and-decisions.md](references/evaluation-and-decisions.md) when designing splits, metrics, adoption rules, or an experiment plan.

## Choose a process-appropriate reduced model

Keep CVD and ALD as separate process modes while reusing the role-assignment concept.

- For steady CVD maps, prefer a quasi-steady site-balance response when the available data do not resolve surface transients.
- For ALD dose, purge, or cycle data, retain time or cycle state when the observations contain transient information. Do not force a steady CVD response onto ALD.
- Keep raw names such as `s0`, `adn_2`, or `n2` anonymous. Enumerate their possible roles rather than declaring chemistry from the name.
- Fit observable parameter groups. Do not report elementary rate constants when the observation map only identifies ratios or products.
- Include exact, interpretable reductions of the model as competitors. Avoid arbitrary polynomial terms, free condition offsets, and detailed species mechanisms that the data cannot distinguish.

Use the lowest-dimensional state model that represents the physical effects under review: supply/adsorption, finite loss or desorption, co-reactant conversion, site blocking, saturation, and process-specific time dependence. Add another state or pathway only when residual structure or a new measurement demands it.

## Fit without leaking condition information

Separate linear scale from nonlinear shape when possible. Profile the linear nonnegative scale analytically and search only the positive shape parameters in log space. Compute concentration references from identification data and lock them for validation and test prediction.

Give each identification condition equal total weight unless the measurement model justifies another weighting. Use condition-level validation for transfer claims. A typical hierarchy is:

1. inner condition cross-validation for role and reduction selection;
2. blocked spatial validation as a diagnostic of local-map behavior;
3. outer leave-one-condition-out refits to assess the entire selection procedure;
4. one fixed, untouched condition for the primary no-refit claim when the data allow it.

Do not use the test condition to choose candidates, references, parameter bounds, regularization, tie tolerances, or report wording. Collapse exact symmetries and numerical ties into equivalence statements instead of manufacturing a role winner.

## Judge prediction, spatial structure, and roles separately

Always distinguish:

- absolute rate or thickness transfer;
- condition-mean bias;
- within-condition spatial variation after removing the mean;
- role-effect necessity;
- raw-species assignment;
- parameter identifiability;
- application-scope validation.

A low relative RMSE can result from correct condition means while the spatial map remains unsupported. A selected AIB candidate does not establish inhibition when the AB reduction predicts equally well. A stable AB pair does not establish which member is A and which is B when the steady response has exchange symmetry.

State adoption as `adopt`, `review`, or `reject` with a short evidence basis. Use `review` when prediction improves but role assignment, spatial support, or application tolerance remains unresolved.

## Separate code responsibility from data responsibility

Assign an issue to code when the observation contains the relevant contrast but the implementation loses it through leakage, an unsuitable response structure, biased weighting, unstable optimization, duplicate candidates, or conflated metrics.

Assign an issue to data when the required contrast is absent: independent species perturbations, wall-state or flux information, time response, temperature variation, replicates, uncertainty, chemistry identity, or surface-state observations.

Do not propose a more flexible equation to compensate for absent excitation. Recommend the smallest additional measurement that changes the observation map and can distinguish the competing explanations.

## Implement with clear responsibilities

Prefer these ownership boundaries within the repository's existing structure:

- model module: states, equations, candidates, exact reductions, prediction diagnostics;
- analysis or runner: data loading, splits, fitting, selection, refits, metrics;
- artifact layer: rankings, stability, condition scores, predictions, provenance;
- report layer: interpretation of stored evidence, with no hidden refitting.

Avoid building a framework, contract hierarchy, or diagnostic subsystem for a small model change. Add meaningful tests for limiting behavior, leakage prevention, symmetry/reduction handling, and end-to-end artifact generation. Do not add tests that merely mirror implementation details.

## Run the improved code and compare fairly

Execute the production path on the supplied data after the change. Compare old and new methods on identical outer condition splits and report both pooled and condition-balanced results. Investigate any degradation before claiming improvement.

Inspect selected roles across outer refits, boundary parameters, extrapolation fractions, mean bias, centered spatial metrics, and prediction-range capture. Stop optional tuning once the remaining weakness is explained by a known structural or data limit.

Read [references/artifacts-and-reporting.md](references/artifacts-and-reporting.md) when producing durable outputs or a detailed technical report.

## Finish with a decision-ready result

Report:

- what equation or selection logic changed and why;
- the comparable baseline and improved results;
- which physical effect is supported;
- which role assignments remain equivalent or unstable;
- practical uses that are validated now;
- model assumptions and code limits;
- missing data, the question each item resolves, and what cannot be claimed until it exists;
- exact commands and artifact locations needed to reproduce the result.

Use professional scientific prose. Do not overstate mechanism evidence, causal meaning, or generalization beyond the evaluated conditions.
