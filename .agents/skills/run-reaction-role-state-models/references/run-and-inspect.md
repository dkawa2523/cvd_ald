# State-model execution and inspection

Use this reference for a specified-parameter forward run. It verifies model behavior;
it does not rank anonymous species or estimate reaction roles.

## Commands

Choose one public configuration and provide a matching Fluent NPZ:

```powershell
uv run python -m deposim_sim.smoke `
  --config-name cvd_transient_min `
  "sim.inputs.fluent.file=<path-to-transient-fluent.npz>"

uv run python -m deposim_sim.smoke `
  --config-name cvd_mvk_transient_min `
  "sim.inputs.fluent.file=<path-to-transient-fluent.npz>"

uv run python -m deposim_sim.smoke `
  --config-name ald_state_min `
  "sim.inputs.fluent.file=<path-to-transient-fluent.npz>"
```

Use the config printed in the run directory as the record of all overrides. Generated
inputs belong under `runs/generated_inputs/`; timestamped outputs belong under
`results/`.

## Artifact reading

Read `summary.json`, the resolved config, array fields, and `manifest.json` before the
HTML report. The report is a view of stored results and should not be the sole evidence
for units or state definitions.

| Figure or field | Physical check |
| --- | --- |
| `thickness_map.png`, `radial_profile.png` | Finite film response and spatial magnitude |
| `measurement_map.png`, `comparison_error_map.png` | Direct comparison when a compatible measurement is supplied |
| `cs_over_cref_<role>.png` | Surface depletion relative to the selected reference location |
| `solver_health_map.png` | Iteration count and fallback/status pattern |
| `identifiability_correlation.png` | Local parameter-direction correlation when sensitivity diagnostics were requested |
| `theta_A`, `theta_free`, `theta_I` histories | AIB or ALD site balance and state bounds |
| `oxidized_fraction`, `reduced_fraction` histories | MvK capacity balance and memory |
| A-only, AB, reduction, and regeneration rates | Pathway bookkeeping in the declared state model |
| surface and transport flux fields | Local closure in kmol/(m2 s) |

Generic maps such as a Damköhler proxy or `phi_B` are meaningful only when the selected
model populates that diagnostic. Do not interpret a placeholder or zero field as a
physical result. Keep compared maps on common colour limits; fractions use 0–1; signed
residuals use a symmetric colour scale.

## Dynamic evidence

For MvK, plot feed switching together with oxidized fraction, reduction rate, and
regeneration rate. For AIB, plot A coverage and A-only/AB event rates. For ALD, align
stored-A, inhibitor, free-site, and conversion histories with dose and purge timing.
Use separate axes or panels for concentration, fraction, flux, and film thickness.

Initial-state sensitivity demonstrates model memory only when the feed history excites
the state and the time step resolves the relaxation time. A model trajectory is not
mechanism evidence until it predicts an independent time- or state-sensitive
observation.

## Completion

Check finite outputs, monotonic time, state conservation, solver fallback counts,
surface/transport flux closure, units, coordinate alignment, and the observation-time
sampling rule. If a numerical or physical check fails, stop interpretation and report
the specific field, condition, and scale of the failure.
