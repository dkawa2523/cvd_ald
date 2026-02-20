# Domain Context (Semiconductor Deposition Surface Modeling)

This repository implements a **surface modeling simulation foundation** for semiconductor
deposition processes (CVD and ALD), where the available upstream simulation (CFD) provides only
gas-phase transport fields near the wafer.

The core objective is to **predict wafer film thickness distribution** (2D map) from:

- CFD-derived **reference-plane** fields: concentration(s), velocity magnitude, temperature
- process scalars: pressure, recipe timing, (optionally) wafer rotation speed

while **not requiring** full gas-phase composition or full surface microkinetics.

---

## Key definitions

### Reference-plane concentration `C_ref`
CFD outputs are assumed to be provided on a fixed plane at height:

- `z_ref_mm` above the wafer surface

The concentration on that plane is **NOT** the surface concentration.
We explicitly call it `C_ref` and treat it as a boundary condition for a reduced transport model.

**Important:**
- `z_ref_mm` is configurable and stored as input metadata.
- Changing `z_ref_mm` may require recalibration of mass-transfer modeling.
- The platform provides a *z_ref sensitivity* workflow (as a DOE factor) to diagnose fragility.

### Surface-adjacent concentration `C_s`
`C_s` is the concentration that enters the surface kinetic model.
It is determined by coupling transport with surface reaction.

### Wafer thickness map `h(x,y)` (a.k.a. film thickness / film pressure)
The deliverable is a 2D thickness distribution:
- either in polar coordinates `(r, theta)` or XY coordinates `(x, y)` depending on the domain spec.

---

## Core physics model: Transport-Coupled Reduced Surface Kinetics (TCRSK)

At each wafer surface location (grid point), we solve a local coupled problem:

### Transport (reference-plane → surface)
For each species `i`:
    J_i = k_m,i * (C_ref,i - C_s,i)

### Reduced surface kinetics (rate law family)
One or more reaction channels:
    r_j = r_j(C_s, state, T; params)

The rate law family must support:
- explicit reaction orders (including fractional/negative "apparent" orders)
- saturation / inhibition (denominator forms; Langmuir/LHHW-like)
- state-coupled kinetics (coverage/site models for ALD and surface modification)

### Species balance / stoichiometry coupling
    J_i = Σ_j ν_ij r_j

### State evolution (optional)
For ALD / modification:
    d state / dt = g(C_s, state, T; params)

### Thickness update
    dh/dt = Σ_j α_j r_j

---

## Computational reduction: progress-variable root solve

For many practical CVD cases, a single dominant deposition reaction is assumed.
Then the coupled system can be reduced to a scalar root solve for a progress variable R:

    C_s,i = C_ref,i - (ν_i / k_m,i) * R

    F(R) = R - r(C_s(R), state, T) = 0

with a physically safe bracket:

    0 ≤ R ≤ R_max = min_i (k_m,i * C_ref,i / ν_i)

This enables robust **bracketing** solvers (bisection as default) and vectorization
over wafer grid points.

---

## Coordinate domains

Supported domain specs (configurable):

- `wafer_2d_polar`: (r, theta) grid (natural for single-wafer systems)
- `wafer_1d_radial`: r-only grid (axisymmetric approximation, optional)
- `wafer_2d_xy`: (x, y) grid (useful when measurement is native XY)

Measured film thickness is assumed to be a **2D map**.
Axisymmetric reduction is optional and should be justified by diagnostics.

---

## Wafer rotation

The platform is "rotating disk oriented" (single-wafer), but rotation is NOT mandatory.

- Rotation enters primarily through mass-transfer modeling (`k_m`) and optional averaging.
- If `omega_rad_s = 0`, the simulation still runs.
- Any rotation-specific correlation must guard `omega=0` (error or fallback) to prevent silent nonsense.

---

## Numerical priorities

The platform must prioritize:

1) Physical constraints:
   - concentrations nonnegative
   - coverage in [0,1]
   - thickness sign convention consistent (deposit positive; etch negative)

2) Robust convergence:
   - bracketing root solvers for the coupled transport-reaction equation
   - monotonicity diagnostics + fallback strategies where needed

3) Reproducible, non-exploding outputs:
   - fixed entrypoint `results/index.html`
   - DOE results stored as case-dimension arrays (avoid per-case directories)

4) Future extensibility:
   - package separation (sim vs opt/ML)
   - model registry for adding new models without refactoring

---

## Configuration approach (Hydra + YAML)

- CLI is not a primary user interface.
- The system is configured via YAML files composed by Hydra.
- YAML configs are split into:
  - `configs/sim/` (numerical simulation)
  - `configs/opt/` (data assimilation / optimization)

`config_resolved.yaml` is always saved into the run directory for reproducibility.

---

## Diagnostics that MUST be produced

To avoid "it matches but we don't know why", the platform standardizes outputs:

- thickness map
- radial profile
- Cs/C_ref map (depletion indicator)
- Da proxy map (reaction vs transport)
- apparent reaction order map (n_app)
- root solver iteration counts / failure rates
- warnings on monotonicity violations / fallback usage

These diagnostics are required for:
- model validation
- extrapolation safety assessment
- later data assimilation and ML residual modeling
