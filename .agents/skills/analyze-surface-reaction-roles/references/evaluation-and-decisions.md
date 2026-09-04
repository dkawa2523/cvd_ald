# Evaluation, identifiability, and decisions

Read this reference when designing validation, interpreting results, or specifying missing data.

## Data audit

For each condition, record:

- observation count and unique response count;
- mean, range, and resolution of the response;
- nonfinite values and coordinate mismatches;
- median total concentration and composition;
- within-condition relative range for each species;
- coordinate and unit metadata.

Across conditions, calculate concentration scale ratios, composition changes, pairwise correlations, and the fraction of each test field outside the identification range. Flag nearly uniform total-concentration changes separately from composition perturbations.

## Weighting and splits

If condition `j` has `N_j` points and there are `J` identification conditions, condition-balanced squared error uses

```text
w_n = 1/(J N_j)
L = sum_n w_n (y_n - yhat_n)^2
```

This prevents conditions with more map points from dominating role selection. Use another weight only when supported by measurement variance or a stated business loss.

Keep all points from a condition together for transfer evaluation. Spatial blocks within a condition diagnose interpolation or local smoothness; they do not substitute for unseen-condition validation.

Outer refits assess the selection procedure, so each fold may select a different candidate. A fixed holdout assesses one frozen model. Report these as different estimands.

## Required metric separation

For every held-out condition, report at least:

- RMSE and MAE in physical units;
- RMSE relative to the held-out condition mean;
- mean bias;
- centered spatial RMSE and centered spatial R-squared;
- spatial correlation when meaningful;
- observed and predicted range, or range-capture fraction;
- extrapolation fraction for each input.

Pooled RMSE weights observations. Macro relative RMSE weights conditions. Report both when condition scales differ.

Interpret centered spatial R-squared against the held-out condition mean. Negative values mean the prediction explains less within-condition variation than a constant at the correct mean, even if absolute-rate transfer is good.

## Role evidence

Evaluate three questions independently:

1. **Effect necessity:** Does adding A, AB, or I improve condition transfer over an exact reduction?
2. **Assignment:** Are competing raw-species assignments separated beyond numerical and data uncertainty?
3. **Stability:** Is the same effect group selected when each condition is omitted?

Report AB as an unordered pair when direction is structurally symmetric. Use `mixed`, `unresolved`, or `not assessed` rather than forcing a binary conclusion.

## Adoption decision

Use these default meanings:

- `adopt`: the model meets the stated application tolerance on untouched conditions, spatial behavior required by the application is supported, and the claimed roles are stable at the level being adopted;
- `review`: prediction improves materially, but application tolerance, spatial support, assignment, or external validation is incomplete;
- `reject`: the model fails the target, is dominated by a simpler reduction, leaks test information, or relies on non-identifiable interpretation.

The absence of a user-specified tolerance is a reason for `review`, not for inventing a threshold.

## Code-versus-data diagnosis

Typical code responsibilities:

- preserve train/test separation and references;
- use a response structure consistent with the process mode;
- enumerate candidates without duplicates;
- profile separable scales and constrain positive shapes;
- use condition-balanced selection and outer procedure evaluation;
- distinguish condition mean, spatial pattern, role evidence, and application decision;
- report equivalence, boundary behavior, and extrapolation.

Typical data responsibilities:

- independent perturbations of candidate A, B, and I;
- enough range to cross low-coverage and saturation regimes;
- replicates and measurement covariance;
- true wall concentration, flux, diffusivity, gradient, and sampling-plane metadata;
- substrate temperature maps and repeated temperature levels;
- time-resolved startup, step, pulse, exposure, or purge response;
- chemical identity, molar mass, stoichiometry, product/off-gas data;
- site density, film density or molar volume, and surface-state measurements;
- a new condition collected after the model is frozen.

For each missing item, state:

- which competing explanations it separates;
- the measurement or perturbation required;
- what cannot be claimed until it exists;
- priority based on the intended decision.

More points from the same collinear steady map do not resolve structural symmetry or missing time scales.

## Experiment design priorities

Prefer experiments that maximize disagreement between viable candidates rather than uniformly adding data. Common priorities are:

1. replicate current conditions to estimate measurement noise;
2. vary one role candidate independently while holding the others near fixed levels;
3. span both low-response and saturation regimes;
4. vary flow or pressure to expose transport coupling;
5. collect transient response to identify relaxation and role order;
6. vary temperature with the same composition design;
7. freeze the model and collect an external validation condition.

Do not prescribe arbitrary sample counts without a noise model. Specify the contrasts and coverage the experiment must create.
