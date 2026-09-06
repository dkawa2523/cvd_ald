# ALD role evidence

## Observation hierarchy

Distinguish these observations:

- final thickness or growth per cycle: integrated process response;
- time-resolved thickness or mass uptake: storage and conversion timing;
- dose plateau: saturation under the supplied exposure range;
- purge response: retained or released surface state;
- cycle-to-cycle change: repeatability, incubation, or deactivation;
- spatial map: local delivery and surface response;
- species flux or surface state: direct pathway or state evidence.

One final map may fit several storage/release histories. Report those histories as unresolved unless a time-sensitive observation separates them.

The dimensional surface balances are

```text
J_A = Gamma_s * (r_store,A - r_release,A)
J_B = Gamma_s * nu_B * r_conversion
km * (C_ref - C_s) = J_surface
```

State rates and coverages use 1/s and dimensionless units, `Gamma_s` uses kmol/m2,
`J` uses kmol/(m2 s), and `alpha_h` uses nm per unit coverage converted. Check these
units before using a transport or flux diagnostic.

## Role evidence

- A is supported when a stored-A structure transfers better than a simpler observation baseline.
- B is supported as an effective conversion role only when AB improves A-only conversion consistently across conditions with independent B excitation.
- I is supported as a retained inhibitor only when AIB improves AB and the purge or time response is consistent with storage and release.
- Anonymous species assignment requires stable selection across condition refits and independent concentration contrast.

Do not infer dose chemistry or surface identity from the role label.

## Validation scope

Training loss checks parameter fit. Condition refits check the selection procedure. A configured holdout checks the frozen selected model. Keep these scopes distinct in artifacts and prose.

Report thickness error in nm, relative error against each held-out condition, mean bias, centered spatial behavior when maps are available, and the selected role frequency. Cycle GPC variation, purge-growth fraction, and plateau metrics are evidence only when the objective has nonzero weights tied to declared targets.

## Missing information

- Vary dose to separate unsaturated delivery from surface saturation.
- Vary purge duration to identify retention and release.
- Vary B timing independently to separate stored-A conversion from simultaneous reaction.
- Collect time-resolved mass or thickness to identify state time scales.
- Add surface or exhaust measurements to distinguish storage from unobserved loss.
- Use repeated cycles and fresh substrates to separate repeatable kinetics from incubation or site evolution.

Ask for the contrast that separates the surviving models; do not prescribe an arbitrary number of runs without a noise model.

Map each request to a decision: dose range tests saturation, purge range tests release,
B timing tests conversion topology, surface or exhaust response tests storage versus
loss, multiple temperatures test Arrhenius separation, site density plus wall flux sets
the absolute molar scale, and a frozen full recipe tests transfer. State the expected
success criterion with the requested data.
