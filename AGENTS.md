# AGENTS.md

This repo is now focused on one product idea:

> Map Fluent raw species to interpretable reaction roles, fit those role models
> to measured film-thickness maps, and avoid numerically good but
> uninterpretable combinations.

## Development Focus

- Keep CVD and ALD as separate process modes, but keep the same role-assimilation
  concept in both.
- Treat Fluent species names such as `s0`, `s1`, and `s2` as raw inputs, not as
  fixed chemistry labels.
- Prefer outputs that explain role selection:
  - `role_summary.csv`
  - `role_ranking.csv`
  - `role_stability.csv`
  - per-condition scores
- Do not add detailed species-first reaction mechanisms unless explicitly asked.
- Do not add broad frameworks, heavy dependencies, or new directory trees for
  small model changes.

## Implementation Rules

- Use existing config, runner, and artifact patterns before adding new ones.
- Keep user-facing commands few and clear. Compatibility or diagnostic commands
  may remain, but should not look like the main production path.
- Keep generated inputs under `runs/generated_inputs/` and run outputs under
  `results/`.
- Do not commit or preserve regenerated fixtures unless they are source fixtures.
- If a change mainly affects model meaning, role adoption rules, or public config
  semantics, update the relevant docs briefly in the same change.

## Main Path

The main implementation path should stay:

```text
Fluent raw species
-> role assignment candidates
-> CVD or ALD role model
-> measured thickness-map fit
-> role ranking and stability diagnosis
-> concise adoption / rejection summary
```

ALD dose, purge, and cycle metrics are useful diagnostics, but they are not the
primary product goal. They should support role-assignment judgment rather than
become a separate modeling framework.
