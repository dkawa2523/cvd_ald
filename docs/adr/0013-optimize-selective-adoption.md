# ADR 0013: Selective Adoption of `optimize.md` for AIB Optimization

- Date: 2026-02-23
- Status: Accepted
- Scope: `deposim_opt` optimization contract (`fit_optuna`, `run_fit`, schema)

## Context

`optimize.md` proposes a broad optimization roadmap. Current runtime already includes:

- role/order enumeration
- full-candidate `ranking.csv`
- `class_compare.csv` and `topk_assignments.csv`

Re-implementing those paths would be redundant and increases maintenance risk.

## Decision

Adopt only the useful, non-redundant subset:

1. Optuna runtime controls (`sampler`, `pruner`, `storage`, resume).
2. Objective decomposition into explicit components:
   - `loss_data`
   - `penalty_solver`
   - `penalty_phys`
   - `penalty_prior`
   - `penalty_complexity`
3. Multi-condition weighted fitting (`opt.measurement.conditions`).
4. Hierarchical per-condition parameterization (`log_offset`) with prior regularization.
5. Coarse-to-fine fidelity by condition-count stages with pruning hooks.
6. Ranking/class-compare explanation columns and tie metadata.

Explicitly out of scope for this decision:

- NSGA-II/BoTorch/EKI/EnKF implementations
- new physics models outside AIB-ODE
- mandatory heavy dependencies beyond optional Optuna

## Consequences

- Existing configs remain backward-compatible when new keys are omitted.
- Optimization behavior becomes auditable from table outputs and score decomposition.
- Further optimizer family expansion requires a new ADR and dedicated tasks.
