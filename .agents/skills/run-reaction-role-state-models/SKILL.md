---
name: run-reaction-role-state-models
description: Configure, execute, and physically check the repository's CVD or ALD reaction-role state models, including AIB, Mars-van Krevelen, transport closure, initial state, units, and output fluxes. Use for simulation runs from Fluent NPZ inputs. Do not use for steady multi-condition equation fitting or for adding a new kinetic model.
---

# Run Reaction Role State Models

Run one declared process model while preserving its state, transport, and unit meaning. Raw Fluent species are inputs assigned to roles for the run; their names are not chemical identifications.

## Select the process problem first

Read `AGENTS.md`, the chosen config, and `src/deposim_sim/models/process_models.py`. Use the public process model that matches the observation clock:

- `role_cvd_aib`: steady or transient CVD with required A and optional I/B; evolves adsorbed-A coverage;
- `role_cvd_mvk`: steady or transient CVD with required A and B; evolves oxidized reaction capacity;
- `role_ald_state`: transient ALD dose/purge/cycle input; evolves stored A and optional inhibitor coverage.

Do not substitute a steady QSS census equation for a transient state problem. Do not add MvK to a steady candidate list as a separate vote when only its steady projection is observable.

Read [references/process-models-and-units.md](references/process-models-and-units.md) when selecting inputs, parameters, or output fields. Read
[references/run-and-inspect.md](references/run-and-inspect.md) for current commands,
artifacts, and figure interpretation.

## Check the input contract

Require the Fluent NPZ to match `sim.time_mode`:

- steady: `xy` and `cref` with shape `[space, species]`;
- transient: `xy`, strictly increasing `time`, and `cref` with shape `[time, space, species]`.

Confirm the configured species order before assigning A, I, and B. Keep roles disjoint. For MvK, require A as the reduction/growth role and B as the regeneration role; leave I unset so inhibition remains a separate mechanism.

Choose concentration and transport semantics explicitly:

- `direct_surface`: input concentration is already a wall/surface value;
- `fit_scalar`: close each role with configured scalar or mapped film coefficient;
- `from_cfd_flux_sink`: interpret the CFD flux only as transport capacity and provide its boundary-concentration convention.

The present role pipeline closes A and B independently. Do not describe it as Stefan-flow or Maxwell-Stefan multicomponent diffusion.

## Execute through the public runner

Use the existing config and smoke runner rather than calling a model kernel for a production run:

```powershell
uv run python -m deposim_sim.smoke `
  --config-name cvd_mvk_transient_min `
  "sim.inputs.fluent.file=runs/generated_inputs/example/fluent.npz"
```

Use Hydra-style overrides only for real run inputs or declared parameters. Keep generated inputs under `runs/generated_inputs/` and outputs under `results/`. Record the config name and overrides with the run.

## Verify physical outputs

For every state run, check finite thickness, state bounds, time monotonicity, declared solver, role mapping, concentration location, and the units stored in diagnostics.

For MvK, inspect at least:

- `oxidized_fraction` and `reduced_fraction` summing to one;
- separate `reduction_rate_s-1` and `regeneration_rate_s-1`;
- A and B surface fluxes and, when finite transport is used, their matching transport fluxes;
- `redox_balance_rate_s-1` and the relaxation-time field;
- observation-time `oxidized_fraction_history`, pathway-rate histories, and surface-flux histories;
- sensitivity to initial redox fraction or A/B feed history when a memory claim is made.

For CVD AIB and ALD, inspect their declared coverages, free-site balance, event rate,
surface concentrations, and species fluxes. Verify `tau_A_s` and `tau_B_s` against
`z_ref_mm * 1e-3 / km`. For ALD, verify that `Gamma_s` converts coverage storage and
conversion rates to surface flux and that each finite-transport surface flux matches
`km * (C_ref - C_s)`. Treat a projection or fallback count as a numerical result to
investigate, not as chemical evidence.

When MvK history observations are configured, confirm that every simulated history slice
is transformed and sampled to the measurement coordinates. Use the final film map once:
history thickness excludes its final time because the final-film observation already
contains that endpoint.

## Report only what the run resolves

A specified-parameter run verifies equation behavior and predicts under those assumptions. It does not identify parameters or reaction roles. Compare histories only when the input contains a meaningful time perturbation. If measured outputs are present, separate total thickness error, condition mean, spatial shape, and state/pathway interpretation.

If the requested interpretation is not supported, state the additional input or
measurement that would resolve it: switching histories for redox memory, dose/purge
histories for ALD storage, wall concentration or transport-capacity flux for surface
conversion, and calibrated site density for absolute molar flux. If the task becomes
parameter selection across anonymous species or equation families, use the dedicated
evaluation workflow. If the model equation itself must change, use the model-extension
workflow.
