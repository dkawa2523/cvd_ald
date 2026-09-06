---
name: extend-reaction-role-models
description: Add, replace, or restructure reaction-role equations, exact reductions, dynamic process-state models, transport coupling, and their comparison workflow in this repository. Use for code changes to model meaning or extensibility. Do not use for a routine run of existing models or for interpretation-only analysis.
---

# Extend Reaction Role Models

Implement a physically distinct model once, place each responsibility in its existing owner, and make the resulting evidence comparable without adding a general framework.

## Classify the requested change

Read `AGENTS.md` and inspect the current registry and main execution path. Classify the change before editing:

- **steady observable equation:** a response fitted directly to steady CVD film maps;
- **dynamic process mechanism:** an ODE/state model that represents history;
- **transport closure:** mapping reference-plane concentration to surface concentration and flux;
- **net-film balance:** signed composition of deposition, etch, and loss;
- **selection/reporting:** comparison and evidence logic, not new physics.

Do not put one model into multiple layers merely to make it visible. Read [references/implementation-boundaries.md](references/implementation-boundaries.md) for current file ownership and model inventory.

## Specify the physical distinction before code

Write down the state balance, event/pathway rates, site or capacity balance, surface-transport closure, film conversion, input location, initial state, and units. Derive the steady limit and meaningful exact reductions.

Check whether the proposed model is observationally identical to an existing equation under the available data. If so, record one equivalence relation and fit the unique equation once. Do not add a renamed family, duplicate candidate, or extra model-selection vote. The implemented MvK steady limit and sequential AB no-desorption response are the reference example.

Keep raw species roles anonymous. A, B, and I describe model effects; they do not establish chemistry. Keep CVD and ALD state clocks separate.

## Extend a steady equation family

Add the descriptor and equation behavior to `src/deposim_sim/models/aib_reductions.py`. The family should declare:

- supported process and required inputs;
- physical question, mechanism interpretation, pathways, and states;
- role classes and exchange symmetries;
- normalized observable parameters and units;
- exact reductions and effects removed by each reduction;
- evaluation function and state/pathway contributions.

Use normalized concentration references computed from identification data. Prefer observable parameter groups and profile the nonnegative rate scale when separable. Keep generic fitting in `deposim_opt/surface_fit.py`; avoid family-name branches in enumeration or reporting when registry metadata can drive the behavior.

## Extend fitting without coupling it to a model

Place a new residual formula in `deposim_opt/losses.py`, a reporting statistic in
`metrics.py`, a search-variable transform in `parameter_space.py`, and a backend or
stopping rule in `samplers.py`. `parameter_fit.py` may orchestrate these pieces but must
not implement their formulas. `fit_conditions.py` remains the single adapter from a
prepared condition to simulator output and observation score.

Keep the sampler, Loss, parameter space, and validation split independently
configurable. Reject an unavailable requested backend instead of changing algorithms.
Do not mix standardized and dimensional losses across conditions. Add a simulated state
or role contribution to the Loss only when it has a measured target and uncertainty;
unmeasured physical expectations belong in evidence reporting. Prefer analytic
profiling for a separable scale and count only nonfixed variables in the search budget.

Every parameter-search extension must preserve `optimization_summary.csv`,
`optimization_trace.csv`, and `loss_components.csv`. These files distinguish failed
optimization, poor predictive fit, and parameter non-identifiability.

## Extend a dynamic process model

Keep the numerical state kernel in `src/deposim_sim/models/`. Then update only the existing integration points that apply:

1. process-model metadata and public name in `process_models.py`;
2. config schema for genuinely new parameters or initial states;
3. compatibility validation for process, time mode, roles, and transport semantics;
4. pipeline dispatch, Fluent adaptation, measurement alignment, fields, diagnostics, and units;
5. one minimal config using existing output conventions.

Use a bounded, balance-consistent update appropriate to the state. Report separate pathway rates and species fluxes when the equation defines them. Do not add contract classes or a directory tree for one state model.

## Connect comparison without conflating evidence

Update `--list-models` and the analysis summary so the new item appears in its physical layer. Add it to the steady optimization census only if the supplied observation directly computes a unique steady equation. Otherwise report `not_evaluated`, `not_applicable`, or an observable equivalence with the missing evidence.

Reports must separate:

- numerical prediction;
- exact-reduction effect evidence;
- anonymous role assignment and stability;
- mechanism discrimination;
- transport and net-film assumptions;
- code limitations and missing-data limitations.

If an intended use remains unresolved, extend the generic mapping in
`deposim_opt/evidence_requirements.py` only when the new model introduces a genuinely new
observable requirement. Each row must state the target use, measurement, controlled
variation, ambiguity resolved, workflow insertion point, and readiness criterion.
Avoid dataset-specific species names or fixed run counts.

Reporting code formats stored evidence and must not refit or choose candidates. Read
[references/visual-evidence-design.md](references/visual-evidence-design.md) when a model,
diagnostic, or artifact change affects scientific figures.

## Clean up and verify

Remove superseded equation implementations, compatibility aliases that still look like the main path, stale documentation, and generated fixtures that are not source inputs. Preserve supported public aliases only when needed for compatibility.

Add meaningful tests for equation limits, state bounds and balances, symmetry or
equivalence, transport flux closure, steady and transient dispatch where supported,
leakage prevention, and end-to-end artifacts. For dynamic history fits, permute
measurement coordinates and check that every time slice aligns; ensure final thickness
is not counted twice. For parameter-search changes, retain or improve a known optimum
and test structural rank separately from practical collinearity. For evidence changes,
assert that unresolved uses produce actionable data-requirement rows. Avoid tests that
only restate metadata strings.

Run targeted tests first. For a change spanning registry, schema, pipeline, analysis, and reporting, run the full `src` and root test suites, compile the source, run `git diff --check`, and execute the production data path on unchanged splits. Report whether accuracy, role stability, spatial prediction, or only model coverage changed.
