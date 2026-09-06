# Documentation map

The documents below describe the current implementation. Relevant design decisions
remain under `docs/adr/`; they explain why a choice was made but do not override the
current model and workflow specifications.

| Document | Responsibility |
| --- | --- |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Integrated scientific report and present operational conclusion |
| [THEORY.md](THEORY.md) | Governing equations, reductions, assumptions, units, and literature |
| [EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md) | Input-dependent execution, optimization, validation, and decision flow |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | Example of every steady CVD figure, axis and colour definitions, source artifacts, and valid scientific interpretations |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Code boundaries, model registries, extension points, and output ownership |
| [CURRENT_DATA_EVALUATION.md](CURRENT_DATA_EVALUATION.md) | Reproducible evaluation of the five steady CVD conditions in `data/` |
| [CONTEXT.md](CONTEXT.md) | Domain vocabulary and claim levels |
| [inputs_fluent.md](inputs_fluent.md) | Fluent field meanings, required units, and supported observation capabilities |
| [transport_km.md](transport_km.md) | Reference-plane to wall transport closures and their limits |
| [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) | Commands and acceptance checks for routine evaluation |
| [GAPS.md](GAPS.md) | Current scientific and software limitations with evidence needed to close them |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Product-level requirements for role assimilation |

For a scientific review, read `TECHNICAL_REPORT.md`, then follow model details in
`THEORY.md`, execution semantics in `EVALUATION_WORKFLOW.md`, and exact current numbers
in `CURRENT_DATA_EVALUATION.md`. Use `VISUALIZATION_GUIDE.md` to audit the axes,
encodings, source tables, and permitted conclusions for each figure. The
[generated report](../results/current_cvd_separated/report.md)
for the documented run is a run artifact rather than a general
specification.

## Repository-local Codex skills

The maintained workflow instructions are stored only under `.agents/skills/` in this
repository. Their responsibilities are deliberately disjoint:

| Skill | Use |
| --- | --- |
| `evaluate-steady-cvd-roles` | Run the steady equation census and interpret optimization, roles, mechanisms, wafer maps, spatial response, and figures |
| `evaluate-transient-ald-roles` | Fit transient ALD storage/conversion roles and assess recipe transfer and fit diagnostics |
| `run-reaction-role-state-models` | Run specified-parameter CVD AIB, CVD MvK, or ALD state simulations and check states, fluxes, units, and plots |
| `extend-reaction-role-models` | Change equations, transport, fitting components, evidence artifacts, or scientific visualizations in the existing architecture |

Detailed execution and figure guides are supporting references inside the relevant skill,
so loading one workflow does not pull in unrelated process modes.

## Authority order

For model behavior, the executable implementation is authoritative. `THEORY.md` records
the implemented equations in readable form. `EVALUATION_WORKFLOW.md` records selection
and validation semantics. `ARCHITECTURE.md` records file responsibilities. Generated
`results/*/report.md` files describe one run and must not be used as a general model
specification.

When a model meaning, public configuration field, or adoption rule changes, update the
corresponding current document in the same change. ADR files are retained as history and
are marked or interpreted as superseded when they conflict with the current documents.
