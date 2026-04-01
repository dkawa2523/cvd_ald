# Transport-km Policy

## km sources
- `fit_scalar` (default)
  - Uses configured `km_A`, `km_B` as constant maps.
  - Backward-compatible with existing runs.
- `from_cfd_flux_sink`
  - Computes spatial `k_m` from Fluent input:
  - `k_m_cfd = flux_sink / (cref + eps_cref)`
  - `k_m_used = gamma_km * k_m_cfd`

## Runtime controls
`sim.model.params.transport`:
- `km_source`: `fit_scalar | from_cfd_flux_sink`
- `gamma_km_A`, `gamma_km_B`: scale factors for flux-derived `km`
- `from_cfd_flux_sink.eps_cref`
- `from_cfd_flux_sink.km_clip = [min, max]`
- `from_cfd_flux_sink.flux_negative_policy = error|clip_to_zero|allow`
- `from_cfd_flux_sink.units_hint` (shown in diagnostics/report)

## Diagnostics emitted
- `km_A_map`, `km_B_map`
- `km_A_cfd_map`, `km_B_cfd_map` (flux source)
- `tau_A_map`, `tau_B_map` with `tau = z_ref_mm / km`
- `km_source`, `transport_units_hint`

## Optimization constraint (flux source)
When `km_source=from_cfd_flux_sink`:
- Direct optimization of `model.params.transport.km_A/km_B` is forbidden.
- `model.params.transport.gamma_km_A` must be in search space.
- If role `B` exists, `gamma_km_B` must also be in search space.
