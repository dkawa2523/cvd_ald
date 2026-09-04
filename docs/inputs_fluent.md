# Fluent Input Contract (AIB)

This document defines the repo-native canonical input contract for Fluent-like inputs in the
current `deposim_*` runtime. The canonical contract is the in-memory `FluentData` shape used by
the public loader path in `deposim_sim.input_builder` / `deposim_sim.io_plugins`.

## Required keys
- `xy`: shape `[n_pts, 2]`
- `cref`: shape
  - steady: `[n_pts, n_species]`
  - transient: `[n_t, n_pts, n_species]`
- `time` (transient only): shape `[n_t]`

## Optional key
- `flux_sink`: shape aligned with `cref`
  - steady: `[n_pts, n_species]`
  - transient: `[n_t, n_pts, n_species]`

`flux_sink` is optional for `km_source=fit_scalar`.
`km_source=from_cfd_flux_sink` requires valid `flux_sink` for role `A`.

## Loader selection
- Default loader selection is suffix-based:
  - `.npz` -> `npz`
  - `.csv` -> `csv`
- Explicit loader override is available via:
  - `sim.inputs.fluent.io_loader_name`
  - `sim.measurement.io_loader_name`
- When explicit loader names are set, they take precedence over file suffix.

## Sign and units
- Positive sign means wall-normal sink toward wafer surface.
- Recommended unit consistency: `k_m = flux_sink / cref` has dimension of velocity.
- Fluent concentration/flux units are not auto-converted. Unit consistency must
  be guaranteed by input preparation. Coordinate units are configured explicitly.

## Error/guard behavior
- Missing required NPZ keys raise `ValueError` with the missing key name, file path, and available keys.
- Negative `cref` values are clipped to zero with warning.
- `flux_sink` shape mismatch raises `ValueError`.
- For flux-driven transport:
  - `flux_negative_policy=error`: raises on negative flux
  - `clip_to_zero`: clips negative flux to 0
  - `allow`: keeps sign and applies `km_clip`

## Measurements used for fitting

The measurement loader accepts NPZ or CSV point data. `keys.h` identifies the
observed value column/key and `keys.xy` identifies coordinates (CSV also accepts
`keys.x` and `keys.y`). The following settings belong to `sim.measurement` for a
simulation, or `opt.measurement` / individual condition rows for fitting:

- `quantity: thickness` (default): measured final thickness in nm.
- `quantity: mean_rate`: measured average growth rate in nm/s. Comparison uses
  `initial_thickness + rate * duration`, with the same duration as the simulator.
  This is not an instantaneous rate or growth per ALD cycle.
- `xy_unit: mm` (default) or `m`: measurement coordinate units, independent of
  Fluent coordinate units.
- `sigma`: optional positive measurement standard deviation in the measurement's
  native units. Alternatively, `keys.sigma` identifies a per-point uncertainty
  column/key and takes precedence over the scalar.

For aligned data, coordinates are transformed and the simulated prediction is
sampled at each original observation using the nearest model point. Existing
distance and radial masks apply before scoring. An observation does not receive
extra weight when the simulation mesh is refined. Alignment disabled retains
row-by-row matching and requires equal array lengths. Nearest sampling is an
approximation; the distance columns show its spatial resolution.

With known sigma, the data loss uses residual/sigma and `objective.huber_delta`
(default 1.345). Without sigma it uses nm residuals and the existing
`huber_delta_nm`. RMSE, MAE, mean bias, and centered spatial RMSE remain in
equivalent final-thickness nm. Condition CV compares that physical MSE using
declared condition weights; uncertainty normalization applies to the fitted loss.
Do not mix uncertainty conventions between conditions unintentionally. Missing
sigma is not evidence of zero measurement noise.

`condition_scores.csv` distinguishes `train`, `condition_cv`, and `holdout`, and
includes a training-only constant baseline for predictive comparisons. Centered
R2 measures spatial shape after removing each map's mean; it is undefined for a
constant observed map. These results support role interpretation without treating
species names as chemistry labels.
