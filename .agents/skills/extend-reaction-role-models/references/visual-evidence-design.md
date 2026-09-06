# Visual evidence design for model extensions

Use this reference when a code change adds or changes a model, fitted diagnostic, or
public artifact. A figure belongs to the layer that computes its source rows; rendering
must not select, refit, or reinterpret a candidate.

## Map each scientific question to one source artifact

| Question | Source data | Preferred figure |
| --- | --- | --- |
| Did optimization converge? | objective-evaluation trace | best-so-far error versus evaluation |
| Which observable equation transfers? | dimensional condition-CV error and outer selection count | paired error and selection-frequency panels |
| Which reaction stages does a family assume? | registered states, pathways, and reductions | compact reaction-path diagram |
| Does mechanism ambiguity affect prediction? | co-located heldout predictions for each family | measured/predicted profiles or pairwise prediction difference |
| Is a raw-species role stable and important? | outer assignment frequency plus reference-substitution prediction change | importance versus selection frequency |
| Which fitted direction is weak or coupled? | scaled local derivatives and parameter correlation | sensitivity magnitude plus correlation matrix |
| Is the Loss flat in one direction? | fixed-parameter scan with documented reprofiling | relative parameter versus dimensional error |
| Does a spatial residual response transfer? | chemical and corrected heldout predictions | centered performance plus before/after residual maps |
| Does a dynamic state close physically? | state, pathway, and flux histories | synchronized time histories with physical units |

## Extension rules

- Put model meaning, pathway labels, states, exact reductions, and units in registry
  metadata when they are common to fitting and reporting.
- Write numerical source rows before rendering a figure. Include them in the manifest.
- Use the same heldout coordinates and observation operator for compared models.
- Use dimensional rate or thickness for cross-Loss comparisons.
- Keep common colour limits for measured and predicted maps, 0–1 for fractions, and a
  symmetric scale for signed residuals.
- State whether a curve is measured, fitted, heldout, model-conditional, or a
  sensitivity perturbation.
- Prefer conventional physical names over repository-specific abbreviations in titles
  and axes. Put detailed caveats in the caption or report text rather than inside the
  plotting area.

## Evidence boundaries

A reaction-path diagram displays the candidate topology. State fractions derived from a
fit are latent model quantities. One-at-a-time input substitution is a prediction
sensitivity. Local derivative correlations and partial Loss slices diagnose weak
directions but are not posterior probabilities or profile-likelihood intervals. Spatial
residual bases are empirical transfer models and do not identify temperature,
transport, or reactor geometry as a cause.

When adding a new family, update the equation-family comparison, pathway diagram,
alternative-family prediction table, state/pathway summary, and data-requirement mapping
only where the new source rows are applicable. Do not create empty panels or duplicate
an observationally equivalent equation.

## Verification

Test numerical row generation and manifest linkage. Run the production path on an
unchanged split, open every new or altered figure, and check labels, units, clipping,
legends, common scales, and behaviour in exact-reduction limits. Update `THEORY.md`,
`EVALUATION_WORKFLOW.md`, and the current data report when scientific meaning or a
reading rule changes.
