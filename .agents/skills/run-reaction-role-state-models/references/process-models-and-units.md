# Process models and units

## CVD AIB state

State: adsorbed-A coverage `theta_A` with dimensionless free and inhibitor fractions. A supplies the adsorbed state; the event path is A-only or B-assisted. The current B dependence uses the dimensionless ratio `C_B/C_B_scale`.

Core units:

- concentration: kmol/m3;
- `k_ads`: m3/(kmol s);
- `k_des`: 1/s;
- `k_rxn`: 1/s with dimensionless B ratio;
- `Gamma_s`: kmol/m2;
- `alpha_h`: nm m2/kmol;
- time: s.

## Mars-van Krevelen state

Let `chi_ox` be the oxidized fraction of a surface or lattice reaction capacity:

```text
d chi_ox/dt = k_regenerate C_s,B (1-chi_ox)
              - k_reduce C_s,A chi_ox
r_growth = k_reduce C_s,A chi_ox
dh/dt = alpha_h Gamma_s r_growth
```

Species surface fluxes are

```text
J_A = Gamma_s r_growth
J_B = Gamma_s nu_B k_regenerate C_s,B (1-chi_ox)
```

`k_reduce` and `k_regenerate` use m3/(kmol s), `chi_ox` is dimensionless, state rates use 1/s, `J` uses kmol/(m2 s), and `alpha_h` uses nm m2/kmol.

With constant A/B and negligible concentration drop,

```text
r_ss = k_reduce C_A k_regenerate C_B /
       (k_reduce C_A + k_regenerate C_B)
```

This is observationally equivalent to the sequential AB no-desorption steady response. Time-dependent feed or an independent oxidation-state observation is required to distinguish the redox reservoir.

## ALD role state

The ALD model retains the dose/purge clock and stored surface states. Use only transient
input. Its dimensional fluxes are

```text
J_A = Gamma_s * (k_store,A Cs,A theta_free - k_release,A theta_A)
J_B = Gamma_s * nu_B * r_conversion
```

`r_conversion` and the storage/release balance use coverage/s, `Gamma_s` uses kmol/m2,
`J` uses kmol/(m2 s), and `alpha_h` uses nm per unit coverage converted. Current
coefficients are effective state-model parameters; do not infer elementary constants or
absolute molar flux without calibrated site density, wall concentration/flux, and a
time-resolved observation.

## Transport responsibility

The active role pipeline accepts `direct_surface`, `fit_scalar`, and `from_cfd_flux_sink`. The registered stagnant-film and rotating-disk calculators, including the Bosanquet diffusivity option, are supporting `k_m` utilities and are not automatically dispatched by the role pipeline.

`k_m` uses m/s. The film closure `J=k_m(C_b-C_s)` produces kmol/(m2 s). A CFD sink used as transport capacity is not automatically the realized reacting-wall flux.

Transport time outputs use
`tau_A_s = z_ref_mm * 1e-3 / km_A` and the corresponding B expression. The `_s`
suffix is part of the field meaning.

MvK observation histories use the same coordinate transform and sampling as the final
film observation. The history thickness vector omits its final time when final thickness
is also fitted, preventing duplicate weighting of one measurement.

## Net-film responsibility

`deposition_only` and `dep_etch_loss` combine signed rates in nm/s after the reaction model. They do not define a surface mechanism and must not be entered as reaction-role candidates.
