# Domain context and terminology

## Product question

The repository asks whether anonymous Fluent species fields can be assigned to a small
set of transferable surface-reaction roles when predicting measured CVD or ALD film
maps. A result is valuable when it states both the supported role-level claim and the
evidence still required.

## Role vocabulary

| Role | Operational meaning | Evidence required for adoption |
| --- | --- | --- |
| (A) | species associated with adsorption, storage, or growth-related supply | independent variation and improvement over the model without (A) |
| (B) | conversion partner or MvK regeneration species | low-(B) contrast, independent variation, or a time response specific to conversion/regeneration |
| (I) | species that reduces available sites or capacity | independent inhibitor sweep and consistent benefit over the no-(I) reduction |
| none | species not used by one candidate | exclusion from one selected model is not proof of inertness |

Names such as `s0`, `adn_2`, and `n2` remain raw identifiers. Chemical identity,
stoichiometry, and feed/byproduct status must come from external process knowledge or
independent measurement.

## Spatial and physical locations

`C_ref` denotes the concentration at the Fluent extraction plane. `C_s` denotes the
concentration adjacent to the reactive wall. They are equal only when a supplied wall
field is used directly or when the transport drop is deliberately neglected. The
selected concentration location is part of the fitted candidate and the output
provenance.

The measured response is deposition rate in nm s\(^{-1}\) for the current CVD CSV
workflow and thickness in nm for general simulations. Coordinates must carry a declared
unit before gradients, wafer radius, or transport length scales are interpreted
dimensionally.

## Claim vocabulary

- `improves_baseline`: every evaluated condition has lower MSE than its training-only
  constant reference.
- `spatial supported`: centered (R^2>0) on every required condition.
- `consistent_benefit`: an effect's parent model does not lose to its independently
  refitted reduction on any condition and improves at least one above roundoff.
- `distinguished`: alternative raw-species assignments are consistently worse for the
  same effect structure.
- `adopt_candidate`: fixed independent prediction meets declared application criteria
  and role/model ambiguity is absent.
- `review`: a narrower predictive use may be valid, but one or more role, structure,
  spatial, or application requirements remain unresolved.
- `reject_prediction`: independent prediction does not beat its declared baseline.

These are evidence states. They do not establish a named chemical species or elementary
reaction without the mechanism-specific observations described in
[THEORY.md](THEORY.md).
