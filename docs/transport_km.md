# Reference-plane to surface transport policy

## Transport balance

For role (j\in\{A,B\}), the reduced local film balance is

\[
J_j=k_{m,j}(C_{j,\mathrm{ref}}-C_{j,s}).
\]

The reaction/state model supplies the wall demand and solves for (C_{j,s}). The
transport provider supplies (k_{m,j}) and records the concentration location. These
responsibilities remain separate so a reaction equation cannot silently redefine a CFD
field.

## Active providers

These providers belong to the dynamic/state-model transport closure. The steady CSV
census may instead choose `transport_capacity_flux` as a normalized `direct_flux`
reaction driver. In that steady mode the code does not infer \(k_m\) or \(C_s\); it fits
the film response to the supplied spatial delivery field. The two uses share an input
field but answer different questions.

| `km_source` | Required input | Meaning | Main limitation |
| --- | --- | --- | --- |
| `direct_surface` | wall/surface concentration | Use the supplied (C_s); internally the local film resistance tends to zero | No transport drop is inferred |
| `fit_scalar` | scalar or field `km_A`, `km_B` | Use a prescribed/fitted independent-film coefficient | (k_m) may absorb reaction or geometry error |
| `from_cfd_flux_sink` | `flux_sink`, reference concentration, documented boundary concentration | Infer a spatial transport-capacity coefficient | Requires capacity-flux semantics and known units/sign |

For CFD capacity flux,

\[
k_{m,j}^{\mathrm{CFD}}=
\frac{J_{j,\mathrm{cap}}}
{C_{j,\mathrm{ref}}-C_{j,\mathrm{boundary}}},
\qquad
k_{m,j}=\operatorname{clip}
\left(\gamma_j k_{m,j}^{\mathrm{CFD}},k_{m,\min},k_{m,\max}\right).
\]

`flux_semantics` must be `transport_capacity`. `flux_negative_policy` is `error`,
`clip_to_zero`, or `allow`; `error` is the safe production default. A positive flux with
nonpositive concentration driving force is rejected.

## Supporting (k_m) utilities

The mass-transfer registry includes:

\[
k_m=\frac{D_{\mathrm{eff}}}{\delta_{\mathrm{eff}}}
\]

for a stagnant film and

\[
k_m=C_kD_{\mathrm{eff}}^{2/3}\omega^{1/2}\nu^{-1/6}
\]

for a Levich-type rotating disk. The latter requires an explicit ω=0 policy. It either
raises an error or uses a configured stagnant-film fallback.

For combined molecular and Knudsen resistance,

\[
\frac{1}{D_{\mathrm{eff}}}=\frac{1}{D_m}+\frac{1}{D_K}.
\]

These utilities calculate (k_m); the role-model pipeline still uses one of the three
active providers above.

## Diagnostics

Dynamic CVD and ALD outputs may include:

- `CsA_over_CrefA`, `CsB_over_CrefB`;
- `J_A_surface`, `J_B_surface`;
- `J_A_transport`, `J_B_transport`;
- CFD and used (k_m) fields;
- boundary and driving concentrations;
- a flux-closure residual when compatible observed flux is supplied.

The reported transport residence proxies use seconds explicitly:

\[
\tau_{j,s}=\frac{z_{\mathrm{ref}}\times10^{-3}}{k_{m,j}},
\]

where \(z_{\mathrm{ref}}\) is configured in millimetres and \(k_m\) in metres per
second. Output fields are named `tau_A_s` and `tau_B_s`; their map diagnostics use
`tau_A_s_map` and `tau_B_s_map`.

For ALD, coverage storage and conversion become molar flux only after multiplication
by site density \(\Gamma_s\). Thus
\(J_{A,s}=\Gamma_s(r_{\mathrm{store},A}-r_{\mathrm{release},A})\) and
\(J_{B,s}=\Gamma_s\nu_Br_{\mathrm{conv}}\). Absolute flux interpretation requires a
physically calibrated site density.

For the CVD AIB model, a useful (B)-transport competition group is

\[
\phi_B=
\frac{\Gamma_s\nu_B k_{\mathrm{rxn}}
\theta_A^{p_A}\theta_*^{p_*}}
{C_{B,\mathrm{scale}}k_{m,B}}.
\]

φB much smaller than one indicates a weak local film drop in the scalar-film model;
large φB indicates transport demand comparable to or greater than the film supply.
This remains a reduced Damköhler-like diagnostic, not a complete reactor-scale
transport number.

## Current limitations

- The active provider treats (A) and (B) with independent scalar film laws.
- Stefan flow, cross-diffusion, thermal diffusion, pressure diffusion, and composition-
  dependent multicomponent coupling are not implemented.
- The rotating-disk relation is a correlation and must be justified for the reactor
  geometry and flow regime.
- `bulk_as_surface` in the current steady CSV analysis performs no wall conversion and
  computes no absolute flux.

Maxwell-Stefan transport becomes necessary when species are not dilute, net molar flux
is significant, or cross-diffusion changes the wall composition. Implementing it
requires binary diffusivities, full composition, temperature, pressure, wall boundary
conditions, and a consistent molar-average velocity or flux convention. See
[THEORY.md](THEORY.md) for references and [GAPS.md](GAPS.md) for the evidence trigger.
