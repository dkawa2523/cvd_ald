---
name: evaluate-steady-cvd-roles
description: Run and interpret this repository's multi-condition steady CVD equation census, including anonymous-role selection, optimization evidence, mechanism ambiguity, wafer-map prediction, optional spatial residual response, and scientific figures. Use for an existing steady CVD analysis. Do not use for transient state simulation or model implementation.
---

# Evaluate Steady CVD Roles

Determine which registered observable equation transfers across process conditions and
which chemical or spatial claims the supplied data support. Keep prediction accuracy,
raw-species role assignment, kinetic-effect evidence, and microscopic mechanism
interpretation as separate conclusions.

## Fix the observation before fitting

Read `AGENTS.md`, the condition files, and [docs/inputs_fluent.md](../../../docs/inputs_fluent.md).
Treat Fluent species names as anonymous inputs. Record the coordinate system, units,
temperature assumption, observation clock, train conditions, and fixed holdout.

Choose one reaction input before enumerating chemistry:

- `bulk_concentration`: reference-plane concentration used as an explicit surface proxy;
- `surface_concentration`: supplied wall or near-wall concentration;
- `transport_capacity_flux`: independently calculated nonnegative wafer-supply flux.

Never let model ranking choose the input location. A realized reacting-wall flux is a
closure observation and must not be used as a fitted reaction driver. Many points on
one wafer add spatial resolution; they do not add independent kinetic conditions.

## Run one complete comparison

Read [references/run-and-visualize.md](references/run-and-visualize.md) for the current
CLI, optional spatial response, optimization benchmark, artifact groups, and figure
reading order. Inventory the registry before a new campaign:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py --list-models
```

Use all registered steady families unless the user explicitly requests a frozen family
or candidate. Preserve a user-specified split. Otherwise reserve one complete condition
as the fixed audit and evaluate selection stability by outer condition refits.

The current steady census fits one parameter set jointly to all identification wafers.
`--loss` changes the whole-wafer objective; it does not create per-wafer coefficients.
Use radial uncertainty weighting only when uncertainty is measured or explicitly posed
as a sensitivity analysis. Compare different Loss functions with dimensional condition-
CV and holdout metrics, because their objective values have different scales.

An optional `radial_quadratic` or `radial_quartic` response is fitted after chemical
selection. It preserves each chemical condition mean and cannot change role ranking,
exact-reduction evidence, or mechanism claims. Uniform wafer temperature is the default;
recording one temperature does not create a radial temperature correction.

## Interpret equations at the observable level

The optimized steady families are:

- `aib_qss`: adsorbed A followed by B-assisted conversion, with optional blocker I;
- `parallel_a_ab_qss`: A-only and B-assisted paths in parallel;
- `langmuir_hinshelwood_qss`: coadsorbed A and B sharing one site pool.

Constant-rate and total-concentration candidates are nuisance baselines. The steady
Mars-van Krevelen two-reactant limit is observationally equivalent to the sequential AB
`no_desorption` equation and receives one comparison vote. A pathway diagram describes
terms in the candidate equation; it is not evidence that the arrows are elementary
reactions.

Treat fitted steady coefficients as normalized observable groups. Only the rate scale
has unit nm/s. Shape parameters, state fractions, and pathway fractions are
dimensionless and model-conditional; do not report them as elementary rate constants.

## Decide from independent evidence axes

Read [references/evidence-and-artifacts.md](references/evidence-and-artifacts.md) before
writing the conclusion. Assess, in this order:

1. optimizer convergence and repeated-run agreement;
2. transfer of condition mean and absolute deposition rate;
3. centered within-wafer prediction and residual structure;
4. added effects against their exact reductions;
5. equation-family and raw-species assignment stability across outer refits;
6. prediction consequence of alternative families and assignments;
7. local parameter sensitivity, coupling, and loss flatness;
8. whether a mechanism-specific observation distinguishes shared equations.

Use role importance together with selection frequency. An unstable role whose removal
changes prediction by much less than heldout RMSE is predictively negligible over the
tested range. An unstable role whose change exceeds heldout RMSE is influential but
unresolved. Do not call a one-at-a-time prediction change an additive species
contribution or elementary pathway rate.

Finish with `adopt`, `review`, or `reject` for the stated use. Report separately what is
established for rate transfer, wafer shape, role assignment, kinetic effects, and
mechanism. For every unresolved use, take the measurement, controlled perturbation,
ambiguity, workflow insertion point, and pass criterion from `data_requirements.csv`.
Separate a missing code capability from missing experimental contrast.
