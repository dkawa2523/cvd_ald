# Evidence gaps and extension triggers

This document separates limitations of the executable method from information that the
experiment must supply. A missing observation is not repaired by adding another fitted
coefficient. A new model is warranted only when the available observations can test the
new state or pathway.

## Present interpretation boundary

The current five-condition steady CVD dataset supports screening of condition-mean
deposition rate. For uses that are not yet established, the analysis emits an
experimental route forward in `data_requirements.csv`: the measurement to add, how to
vary it, which ambiguity it resolves, and where it enters the workflow. The table below
explains the present example; the executable rules apply to other datasets without
depending on these species names.

## Code-side limitations

| Limitation | Present effect | General extension | Evidence required before implementation |
| --- | --- | --- | --- |
| Independent scalar film transport | Cross-species diffusion and convective molar flux are omitted | Maxwell-Stefan wall-film closure with Stefan flow correction | Species diffusivities, temperature, pressure, boundary-layer geometry, wall-normal flux, and a validated CFD boundary condition |
| Steady census has no latent state | It cannot distinguish a memoryless AB response from a redox reservoir with the same steady reduction | Compare the existing dynamic AIB and MvK state models against step/pulse data | Time-resolved inlet and surface response, initial state, and preferably an oxidation-state observable |
| One uniform site pool per reduced model | Site heterogeneity and lateral interaction can appear as fitted saturation or loss | Add a tested multisite or interaction term only after residual structure demands it | Coverage-sensitive data spanning low and high occupancy |
| Deposition response is positive-only | Etching and parasitic loss cannot be attributed from net film data alone | Use the existing net-rate composition with independently constrained loss terms | Separate deposition/etch information or a designed condition that isolates removal |
| Deterministic log-space multistart search | Confidence intervals reflect resampling and candidate structure, not a full posterior | Profile likelihood or Bayesian inference after the observation model is known | Replicate maps and measurement uncertainty |
| Current random/TPE/CMA-ES/DE/PSO/Lévy state-model search is point estimation | Repeated seeds diagnose numerical variability but do not provide parameter or model probability | Add profile likelihood, bootstrap refits, or posterior sampling behind the same Loss and parameter-space interfaces | Replicate measurements, calibrated uncertainty, and an identifiable observation model |
| Empirical radial residual has no causal label | The optional post-selection correction can transfer a wafer pattern but cannot assign it to chemistry, delivery, or metrology | Replace the radial residual with a measured local driver when evidence identifies one | Co-located wall concentration or reaction-independent supply flux, repeat maps, and metrology controls; spatial temperature only if the uniform-temperature assumption fails |

## Data-side limitations in `data/`

| Missing or weak information | Consequence | Minimum useful addition |
| --- | --- | --- |
| `idn_2` and `n2` vary almost together | Their A/I/B assignments are not separable | Independent factorial perturbations at fixed `adn_2` and total concentration |
| No low-B or B-off condition | Sequential and parallel pathways are poorly separated | At least one B-free condition and several low-B levels |
| No targeted inhibitor sweep | The fitted inhibitor term reduces without measurable loss | Near-zero, intermediate, and strongly suppressing inhibitor conditions |
| No wall concentration or transport-capacity flux | `bulk_as_surface` cannot support an absolute surface-flux claim | Wall/near-wall concentration or a capacity-flux field with sign, units, and boundary condition |
| No temperature, pressure, coordinate unit, or reference-plane metadata | Kinetic constants and transport conversion cannot be assigned physical units | Per-condition metadata and the Fluent sampling-plane definition |
| No time-resolved dose, purge, or switching | Dynamic AIB, MvK, and ALD states cannot be fitted | Synchronized inlet/near-wall histories and film or surface-state response |
| No replicate film maps or uncertainty | Statistical weight and practical acceptance cannot be calibrated | Replicates, instrument precision, spatial registration uncertainty, and a process tolerance |
| Sparse independent condition directions | Outer-fold equation selection is unstable | Conditions selected to maximize disagreement among surviving equation families |

## Decision status of older proposals

The following proposals remain deferred. Their task identifiers are retained for
historical verification. `ADR_REQUIRED` means that changing the present policy requires
a new decision record because it changes model meaning or an input contract.

| Historical proposal | Task | Status and reopening trigger |
| --- | --- | --- |
| Stefan flow correction | D-001 | Deferred; reopen with non-dilute mixtures and documented wall-normal total molar flux. `ADR_REQUIRED`. |
| Smoothing PDE | D-002 | Rejected as an implicit repair of unexplained maps; reopen only with a physical lateral-transport equation and identifiable boundary conditions. `ADR_REQUIRED`. |
| Purge residual driver | D-003 | Deferred; reopen with time-resolved purge tails that the current ALD state cannot explain. `ADR_REQUIRED`. |
| Incubation/poisoning | D-004 | Deferred; reopen with cycle-dependent memory beyond storage/release and a discriminating observable. `ADR_REQUIRED`. |
| Chamber seasoning | D-005 | Deferred; reopen with run-order drift, chamber-state records, and repeated reference conditions. `ADR_REQUIRED`. |

## Highest-value next work

1. Acquire independent species perturbations and a frozen external condition.
2. Add wall-location/flux semantics and physical metadata to the input data.
3. Acquire transient switching data before using MvK or ALD states for mechanism
   selection.
4. Define a process tolerance and measurement uncertainty before changing `review` to
   `adopt`.
5. Extend the equations only when the residual or new observable distinguishes the
   added state from existing reductions.
