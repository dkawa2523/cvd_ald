# Implementation boundaries

Use this reference to locate an extension and avoid duplicating responsibilities.

## Current model layers

| Layer | Current models | Responsibility |
| --- | --- | --- |
| Steady surface response | `aib_qss`, `parallel_a_ab_qss`, `langmuir_hinshelwood_qss` | Unique normalized equations and exact reductions fitted to steady maps |
| Steady reaction input | `bulk_as_surface`, `direct_surface`, `direct_flux` | Select one location/quantity before chemical enumeration; preserve its unit and semantics |
| Spatial residual response | `none`, `radial_quadratic`, `radial_quartic` | Post-selection mean-preserving map correction; no role or mechanism evidence |
| CVD process state | `role_cvd_aib`, `role_cvd_mvk` | Time integration, initial state, surface concentrations, pathway rates, film growth |
| ALD process state | `role_ald_state` | Dose/purge/cycle storage and conversion state |
| Role-pipeline transport | `direct_surface`, `fit_scalar`, `from_cfd_flux_sink` | Concentration location and realized surface flux closure |
| Supporting mass transfer | stagnant-film, rotating-disk, Bosanquet diffusivity option | Calculate a possible `k_m`; currently not automatic role-pipeline dispatch |
| Net film | `deposition_only`, `dep_etch_loss` | Combine signed deposition, etch, and loss rates |
| Evaluation | inner condition CV, exact reductions, outer condition CV, fixed holdout | Rank observable equations and state evidence limits |

An independent Langmuir-Rideal label is not a missing family: the sequential AIB AB pathway already represents that collision/conversion topology. Add a separate family only if its state balance or observable equation differs.

## File ownership

- `src/deposim_sim/models/aib_reductions.py`: steady family descriptors, candidates, equations, reductions, symmetries, contributions.
- `src/deposim_sim/models/process_models.py`: public process names, supported modes, mechanisms, pathways, states, required roles, units, equivalences.
- `src/deposim_sim/models/aib_ode.py`: continuous CVD AIB state kernel.
- `src/deposim_sim/models/mvk_state.py`: MvK oxidized-capacity state kernel.
- `src/deposim_sim/models/ald_role_state.py`: ALD stored-role state kernel.
- `src/deposim_sim/transport_provider.py`: active role-pipeline transport providers.
- `src/deposim_sim/models/mass_transfer.py`: supporting `k_m` calculators.
- `src/deposim_sim/models/net_models.py`: signed rate composition.
- `src/deposim_schema/sim_config.py`: public simulation config fields and composition.
- `src/deposim_sim/validation/compatibility.py`: cross-field compatibility.
- `src/deposim_sim/pipeline.py`: process dispatch and common input/output adaptation.
- `src/deposim_opt/surface_fit.py`: family-independent steady fitting.
- `src/deposim_opt/evidence_requirements.py`: capability readiness and generic
  measurement/experimental-design requirements.
- `src/deposim_opt/cvd_analysis_io.py`: format-level CSV reading, coordinate matching, and artifact serialization.
- `src/deposim_opt/cvd_conditions.py`: condition discovery, column interpretation, quality facts, and role-field assembly.
- `src/deposim_opt/spatial_validation.py`: spatial block construction and ordinary rate metrics.
- `src/deposim_opt/spatial_response.py`: post-selection residual basis fitting and mean-preserving transfer.
- `src/deposim_opt/empirical_response.py`: optional empirical compatibility candidates and constrained fitting.
- `src/deposim_opt/cvd_multicond_analysis.py`: unique-equation census orchestration, split evaluation, and evidence calculation.
- `src/deposim_opt/cvd_multicond_report.py`: report, notebook, and plot rendering from stored evidence.

Observation-only nuisance baselines live in the steady equation registry but use the
`observation_baseline` family in ranking and reporting. They test whether total
concentration alone explains the response and do not receive a reaction-mechanism
interpretation.

## Equivalence and evidence rule

Two mechanisms that reduce to the same observable equation under the supplied observation belong to one observable-equation group. Store all physical interpretations, choose one fitted representative, and state the data needed to separate them.

For MvK with constant A/B and negligible concentration drop:

```text
r_ss = k_reduce C_A k_regenerate C_B /
       (k_reduce C_A + k_regenerate C_B)
```

This is the sequential AB no-desorption response after nondimensionalization. MvK becomes distinguishable through state memory, independent oxidation-state measurement, or another observation that breaks that reduction.

## Common remaining boundaries

Stefan-flow and Maxwell-Stefan multicomponent wall diffusion are not implemented. Dynamic inhibitor comparison, thermodynamically constrained reversible elementary steps, active-site creation/deactivation, and nucleation states also remain outside the current main path. Add one only when the data or requested simulation supplies the variables needed to exercise and verify it.
