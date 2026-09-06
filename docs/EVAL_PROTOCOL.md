# Verification and evaluation protocol

This protocol defines the checks required for code changes and for a new scientific
evaluation. `scripts/commands.sh` remains the repository entry point for general
verification; the steady role census has one explicit analysis command so that the
train/test split and random seed are visible.

## Routine code gate

Run from a Bash environment:

```bash
./scripts/commands.sh smoke cvd
./scripts/commands.sh smoke ald
./scripts/commands.sh test
./scripts/commands.sh verify
```

The gate passes when imports have no side effects, the minimal configured simulation
completes, numerical tests pass, and a run contains resolved configuration, fields,
metrics, plots, and a valid `output.v1` manifest. Model-specific changes must also run
the closest focused unit modules.

| Change area | Focused verification |
| --- | --- |
| Steady equations or selection | `deposim_opt.test_surface_kinetics`, `deposim_opt.test_cvd_multicond_analysis` |
| Objective and uncertainty scaling | `deposim_opt.test_objective` |
| Dynamic AIB | `deposim_sim.test_aib_ode`, `deposim_sim.test_pipeline_aib` |
| Mars-van Krevelen | `deposim_sim.test_mvk_state`, `deposim_sim.test_process_models` |
| ALD state | `deposim_sim.test_ald_role_state` |
| Transport location or coefficient | `deposim_sim.test_transport_provider` |
| Configuration semantics | `deposim_schema.test_sim_config_v2`, `tests/test_sim_config_compose.py` |

## Current steady-data evaluation

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --response-model surface_compare `
  --reaction-input bulk_concentration `
  --models all `
  --loss mse `
  --sampler pattern `
  --bootstrap-samples 100 `
  --spatial-response radial_quartic `
  --seed 123 `
  --output results/current_cvd_separated
```

Acceptance is based on the generated evidence, not process completion alone:

1. All input rows and coordinates pass the quality checks, or exclusions are listed.
2. The fixed holdout is absent from fitting, normalization, and model selection.
3. Every applicable family, exact reduction, and disjoint role assignment is included.
4. Selection uses condition-refit error and reports an explicit training-only baseline.
5. Mean error, centered spatial error, correlation, bias, and range capture are reported
   separately.
6. Outer condition folds report the stability of the complete selection procedure.
7. The report states extrapolation, reduction evidence, coefficient uncertainty, and
   model-structure spread.
8. `data_requirements.csv` states what measurement and controlled variation would make
   each unresolved target use assessable.
9. Equation-family, condition-contrast, reaction-path, alternative-model prediction,
   role-importance/stability, parameter-sensitivity, heldout spatial, radial-shell,
   model-structure, and optional spatial-response figures are generated and visually
   checked against their source CSV files.
10. The final status is `adopt`, `review`, or `reject`, with the supported use narrower
   than or equal to the tested evidence.

## Dynamic model evaluation

Dynamic AIB, MvK, and ALD models are first checked for state bounds, mass-transfer
closure, time-step convergence, and source/sink sign. Scientific comparison additionally
requires a time-resolved observable. Steady film-rate CSV files may confirm a limiting
response but cannot rank reservoir or storage dynamics.

The following checks are mandatory for a dynamic claim:

- initial state and its preparation are stated;
- input timestamps, concentration location, temperature, and pressure are known;
- the same observation operator is applied to all models;
- at least one full switching, dose/purge, or cycle sequence is withheld;
- fitted state time constants are resolved by the sampling interval;
- A-reduction, B-regeneration, conversion, and transport fluxes are reported in
  compatible units.

## Documentation and artifact gate

Before release:

- `git diff --check` reports no whitespace errors;
- relative links in current Markdown documents resolve;
- equations in `THEORY.md` agree with the executable responses;
- `CURRENT_DATA_EVALUATION.md` is regenerated when source hashes or selection semantics
  change;
- report figures are copied from the declared run and visually checked;
- generated working outputs stay in `results/`, while only selected report figures are
  retained under `docs/assets/`.
