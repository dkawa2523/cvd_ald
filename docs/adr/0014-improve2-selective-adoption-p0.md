# ADR 0014: Selective Adoption of `improve2.md` for P0-Focused AIB Improvements

- Date: 2026-02-23
- Status: Accepted
- Scope: AIB optimization explainability/health/performance refinements (`deposim_opt`, `run_report`)

## Context

`improve2.md` proposes broad improvements. Current codebase already includes many optimization baseline capabilities:

- role/order enumeration
- `ranking.csv`, `class_compare.csv`, `topk_assignments.csv`
- multi-condition fitting, hierarchical offsets, pruner/storage/fidelity controls
- AIB diagnostics (`phi_B`, `f_I`, `Cs*` ratios, solver metrics)

Re-introducing these would be redundant and increase maintenance load.

## Decision

Adopt only non-redundant, low-conflict improvements that strengthen P0 operation quality:

1. Role stability + identifiability warning diagnostics in fit outputs.
2. Integration of existing identifiability diagnostics into fit best-candidate flow.
3. Coarse-to-fine duplicate evaluation reduction via cache (trial-local + bounded cross-trial cache).
4. Preflight fail-fast checks for finite ratio and shape consistency.
5. Solver health warning banner in HTML report.

Explicitly excluded from this ADR:

- new physics model extensions (Arrhenius full rollout, dynamic I state, A-set expansion)
- objective registry framework replacement (Loss/Metric/Plot Registry)
- heavy dependency additions (GPU/JAX-first rewrites, new large frameworks)

## Consequences

- AIB single-path policy remains unchanged.
- Existing YAML files remain backward-compatible when `opt.parameter_fit.analysis.*` is omitted.
- Fit artifacts gain additional explainability outputs:
  - `tables/role_stability.csv`
  - `outputs/fit_diagnostics.json`
- Report front page explicitly warns on solver-health risk (`non_bracketed_total` threshold breach).
