# Fluent Input Contract (AIB)

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

## Sign and units
- Positive sign means wall-normal sink toward wafer surface.
- Recommended unit consistency: `k_m = flux_sink / cref` has dimension of velocity.
- This code does not auto-convert units. Unit consistency must be guaranteed by input preparation.

## Error/guard behavior
- Negative `cref` values are clipped to zero with warning.
- `flux_sink` shape mismatch raises `ValueError`.
- For flux-driven transport:
  - `flux_negative_policy=error`: raises on negative flux
  - `clip_to_zero`: clips negative flux to 0
  - `allow`: keeps sign and applies `km_clip`
