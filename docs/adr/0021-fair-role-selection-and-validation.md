# ADR 0021: Fair Role Selection and Minimal Validation Hardening

- Date: 2026-09-02
- Status: Accepted
- Scope: CVD/ALD role fitting, ALD role-state semantics, diagnostics, and provenance

## Context

The first `role_ald_state` implementation made film growth proportional only to
the `A+B` conversion event. That made role `B` optional in configuration but
structurally mandatory for nonzero growth, so A/AI versus AB/AIB was not a fair
data-driven comparison.

The fit path also ranked candidates mainly with Huber loss plus one configured
complexity penalty. It did not expose ordinary error units, a held-out condition,
or whether the winning roles changed when that penalty was rescaled. ALD used a
bounded explicit substep update while its config still advertised the CVD
implicit root solver. A clean commit hash alone also could not identify runs
executed from a dirty worktree.

## Decision

1. Keep one compact ALD state model, but make `B` genuinely optional:
   - without `B`, use `R_event = k_convert_A * theta_A`;
   - with `B`, use `R_event = k_convert_AB * Cs_B * theta_A`;
   - update `theta_A` and film thickness with the active event only.
2. Name the ALD numerical method honestly as `explicit_substep_bounded`.
   CVD compatibility models retain `implicit_euler_bisect`.
3. Mark CVD root metrics as applicable and ALD root metrics as not applicable.
   ALD reports bounded-state violation, projection, and substep diagnostics.
4. Keep Huber loss for optimization, and add RMSE, MAE, and maximum absolute
   error in nanometer units for interpretation.
5. Support `split: train|holdout` on conditions. Holdout conditions are excluded
   from parameter selection and evaluated only with the shared fitted parameters.
   Per-condition parameter search is rejected when holdouts are configured, to
   avoid leakage or undefined holdout offsets.
6. Emit a complexity sensitivity table using penalty multipliers `0`, `1`, and
   `10`. If the winning assignment changes, the nominal winner is labeled
   `review`, not automatically adopted.
7. Run finite-difference identifiability through the common process dispatcher so
   it works for both CVD and `role_ald_state`. If the best-fit parameters are
   degenerate or strongly correlated, label the candidate `review` rather than
   automatically adopting it.
8. Record role-stability and parameter-identifiability warnings separately while
   retaining the former `role_identifiability_warning` field as a compatibility
   alias for role stability.
9. Record measurement nearest-neighbor distance diagnostics and dirty-worktree
   provenance (`code_dirty`, `code_diff_fingerprint`).

## Consequences

September 2026 refinement: condition-refit prediction now takes precedence over
the training complexity sweep for role selection. The sweep remains visible for
training-only comparisons. Scoring uses original observations, with explicit
thickness/mean-rate and uncertainty handling; local identifiability spans all
training conditions and fitted parameters. See `ARCHITECTURE.md` for module
responsibilities and `cvd_spatial_case_analysis.md` for the empirical analysis
selection rule. This supersedes the penalty sweep as an adoption gate when
condition-refit evidence is available.

- A/AI and AB/AIB now answer a real model-selection question in ALD.
- A role assignment must be good on train conditions, interpretable in ordinary
  error units, explicitly reported on holdout, and not overly sensitive to one
  complexity penalty or a degenerate parameter combination before adoption.
- The change adds no detailed chemistry, new state family, heavy dependency, or
  dataset framework.
- Existing compatibility CVD/ALD models remain executable with their existing
  implicit solver contract.

## Non-Goals

- No elementary reaction mechanism is inferred.
- No Bayesian model-selection framework is added.
- No per-condition holdout escape parameters are fitted.
- No concentration-unit inference or temperature mechanism is introduced.
