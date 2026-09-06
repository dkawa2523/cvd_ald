# Fluent input specification

## General rule

The input contract records what each field physically means before a reaction model is
selected. A raw species name is an array label, not a reaction role. Concentration
location, flux semantics, coordinate unit, and time basis must be explicit.

## Multi-condition steady CSV

The default steady CVD analysis pairs:

```text
data/condition_<id>.csv
data/validation_<id>.csv
```

Required condition fields:

| Field | Meaning | Unit |
| --- | --- | --- |
| `x`, `y`, `z` | sampling coordinate | must be declared by dataset owner |
| `concentration_<species>` | concentration at the supplied Fluent location | kmol m\(^{-3}\) |

Required validation fields:

| Field | Meaning | Unit |
| --- | --- | --- |
| `x`, `y`, `z` | measurement coordinate aligned with the condition table | same as condition coordinates |
| `dr_nm_per_sec` | measured deposition rate | nm s\(^{-1}\) |

Optional condition fields are:

| Field | Location and meaning | Unit |
| --- | --- | --- |
| `surface_concentration_<species>` | concentration adjacent to the wafer reaction surface, mapped to the same in-plane coordinates | kmol m\(^{-3}\) |
| `transport_capacity_flux_<species>` | nonnegative species-supply flux toward the wafer, calculated with a boundary condition independent of the fitted reaction | kmol m\(^{-2}\) s\(^{-1}\) |
| `realized_reactive_flux_<species>` | actual reacting-wall flux from a coupled calculation; closure observation only | kmol m\(^{-2}\) s\(^{-1}\) |
| `molef_<species>` | mole fraction used for consistency checks | 1 |
| `density` | mixture density | kg m\(^{-3}\) |

`sigma_nm_per_sec` may be added to the validation file as pointwise standard
uncertainty. Every optional species field selected for a fit must exist for every species
and every condition.

The default filename convention can be replaced by `--conditions-file`, a JSON manifest
mapping condition IDs to condition and validation paths. Relative paths are resolved
from that manifest.

## Simulation NPZ input

The general simulator uses YAML to map NPZ keys. A typical Fluent block is:

```yaml
inputs:
  fluent:
    mode: transient
    file: data/fluent_cvd_transient.npz
    keys:
      cref: cref
      xy: xy
      time: time
      flux_sink: flux_sink
    species: [s0, s1, s2]
domain:
  kind: from_fluent_xy
  xy_unit: mm
reference_plane:
  z_ref_mm: 1.0
```

Expected array shapes are:

| Quantity | Steady | Transient |
| --- | --- | --- |
| `cref` | `[species, point]` or loader-equivalent documented layout | `[time, species, point]` or loader-equivalent documented layout |
| `xy` | `[point, 2]` | `[point, 2]` |
| `time` | absent | strictly increasing `[time]`, seconds |
| `flux_sink` | species/point field | time/species/point field |

The loader and resolved configuration are authoritative for exact array orientation.
Every time interval must have a concentration frame for a dynamic state model.

### Optional MvK history observations

The MvK simulation stores state and pathway values at the supplied Fluent times. An NPZ
measurement can add an oxidation-state history to the final film observation:

```yaml
measurement:
  enabled: true
  file: data/mvk_measurement.npz
  keys:
    xy: xy
    h: h_nm
    sigma: h_sigma_nm
    time: time_s
    oxidized_fraction_history: oxidized_fraction
    oxidized_fraction_history_sigma: oxidized_fraction_sigma
```

`time_s` must match the Fluent time array. A configured history has shape
`[time, *space]` and requires a correspondingly named `_sigma` key. Film uncertainty is
also required when history observations activate the multi-observation objective. The
same convention supports the emitted MvK rate, surface-concentration, and surface-flux
history field names listed in `configs/sim/cvd_mvk_transient_min.yaml`. State and
thickness values are stored at each supplied time. Rate, surface-concentration, and flux
values at an interval endpoint use the piecewise-constant Fluent frame applied over the
preceding interval; the initial entry uses the first frame.

## Concentration-location capabilities

| Capability | Meaning | Compatible steady transport mode |
| --- | --- | --- |
| `bulk_concentration` | concentration at a reference/bulk extraction location | `bulk_as_surface` approximation |
| `surface_concentration` | supplied value adjacent to the reactive wall | `direct_surface` |
| `transport_capacity_flux` | supply flux toward the wafer under a documented reaction-independent boundary condition | `direct_flux` in the steady census, or used to derive (k_m) in the simulation pipeline |
| `realized_reactive_flux` | actual reactive wall flux from a coupled CFD solution | comparison/closure observation only |

A realized reactive flux must not be used to infer (k_m) for the same reaction model.
Doing so would reuse the reaction response as its own transport boundary.

The steady analysis selects exactly one input representation before enumerating chemical
models:

```bash
--reaction-input bulk_concentration
--reaction-input surface_concentration
--reaction-input transport_capacity_flux
```

The first uses `concentration_<species>` as a surface proxy, the second uses
`surface_concentration_<species>`, and the third uses
`transport_capacity_flux_<species>`. The program does not rank these alternatives as if
they were competing reaction mechanisms. For all three, the role equation receives the
dimensionless local driver \(u_j=X_j/X_{j,\mathrm{ref}}\). Flux-driven fitted groups are
therefore conditional flux-response parameters and must not be reported as concentration
adsorption constants.

The steady workflow assumes a uniform wafer temperature. `--wafer-temperature-k` records
the scalar temperature when it is known; it does not create or fit a radial temperature
correction.

## Alignment and quality checks

Before fitting, the steady adapter checks:

1. condition and validation row counts;
2. finite coordinates, concentrations, and rates;
3. coordinate equality after six-decimal normalization and the maximum raw difference;
4. duplicate coordinates;
5. positive reference concentrations;
6. mole-fraction sum and concentration/mole-fraction consistency when available;
7. unique values and minimum positive increments;
8. within-condition range, between-condition log-span, rank, and species correlation;
9. holdout values outside the identification range.

Passing these checks means the arrays are numerically usable. It does not mean that the
experimental design distinguishes reaction roles.

## Current `data/` capability

The current five-condition dataset provides only `bulk_concentration` and measured
steady deposition rate. It has no time array, coordinate unit, wall concentration,
transport-capacity flux, temperature series, pressure, measurement uncertainty, or
replicate maps. The analysis consequently uses `bulk_as_surface` and cannot calculate
an independently validated wall conversion or absolute flux.

See [CURRENT_DATA_EVALUATION.md](CURRENT_DATA_EVALUATION.md) for the quantitative result
and [transport_km.md](transport_km.md) for transport equations.
