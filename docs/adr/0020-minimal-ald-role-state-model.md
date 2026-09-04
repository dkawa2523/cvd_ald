# ADR 0020: Minimal ALD Role-State Assimilation Model

- Date: 2026-04-16
- Status: Accepted; extended by ADR 0021
- Scope: ALD role-state model for role-based data assimilation

## Context

The product goal is not a radical-species-first elementary reaction mechanism.
Fluent provides raw species concentration/flux fields, measured wafer thickness
provides the calibration target, and this code must infer which raw species act
as effective reaction roles.

`role_ald_compat` proved that transient input plumbing can run, but it is not the
ALD production model. It should remain a diagnostic/compatibility path only.

The ALD model should help answer:

- Which raw species should be assigned to the growth-driving role?
- Is a second role needed to explain condition changes?
- Is a suppression role actually justified, or only improving the number by
  adding complexity?
- Does the same role assignment explain multiple shared tool/recipe/Fluent
  conditions?

## Decision

Add `role_ald_state` as the ALD production candidate. It is a compact latent
state model for data assimilation, not a detailed ALD chemistry model.

The role contract is:

- `A`: effective growth-driving role
- `B`: optional effective conversion/response-shaping role
- `I`: optional effective suppression role
- unused raw species are allowed

The minimum state variables are:

- `theta_A`: stored growth-driving state from role `A`
- `theta_I`: optional unavailable/suppressed state from role `I`
- `theta_free = max(1 - theta_A - theta_I, 0)`: remaining finite capacity
- `h_nm`: film thickness

The minimum equations are extended by ADR 0021 so that optional role `B` can
actually be tested rather than being structurally required for all growth:

```text
d theta_A / dt =
    k_store_A * C_A * theta_free
  - k_release_A * theta_A
  - R_event

d theta_I / dt =
    k_store_I * C_I * theta_free
  - k_release_I * theta_I

R_event =
    k_convert_A * theta_A                  when B is absent
    k_convert_AB * C_B * theta_A           when B is present

d h_nm / dt = alpha_h * R_event
```

where `C_A`, `C_B`, and `C_I` are role-mapped Fluent inputs. Parameter names must
remain role/assimilation names. They must not be renamed into fixed species
chemistry without a later explicit decision.

## Production Evaluation

Production evaluation should prioritize role interpretability:

1. measured film-map error across conditions
2. role summary, role ranking, and next-best gap
3. role stability across top candidates
4. complexity penalty for unnecessary `I`/`B` roles
5. shared-parameter fit before condition-specific escape routes
6. holdout prediction separated from train-condition scoring
7. RMSE/MAE/max-error reporting alongside the robust fit loss
8. sensitivity of the winning role assignment to complexity-penalty scaling

ALD-specific quantities such as dose response, purge response, and cycle growth
are diagnostics. They are useful only insofar as they help judge whether a role
assignment is stable and meaningful across conditions.

## Non-Goals

- Do not add a radical-species-first mechanism.
- Do not make `role_ald_compat` the production model.
- Do not add a separate purge-decay framework in this model.
- Do not add a broad dataset framework for the first production path.
- Do not accept a more complex role set unless it improves interpretation enough
  to justify the added variables.

## Consequences

Future ALD work should improve `role_ald_state`, role-ranking outputs, and
multi-condition adoption criteria. Compatibility benchmarks may remain for
diagnostics, but reports and user-facing guidance should not present them as the
main ALD model.
