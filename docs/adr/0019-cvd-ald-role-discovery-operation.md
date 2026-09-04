# ADR 0019: CVD/ALD Role Discovery Operation

- Date: 2026-04-15
- Status: Accepted
- Scope: Future CVD/ALD execution and role-discovery implementation

## Context

ADR 0008 retired the legacy `power_law/lhhw/root_solve` route and made AIB-ODE the
single primary implementation path. That was useful for removing old competing
physics paths, but it now creates a misleading implementation signal: future
work should not force CVD and ALD into one `aib_ode` contract.

The product goal is still role-based data assimilation. Fluent provides raw
species concentration and flux-like fields. The code must help find a reasonable
mapping from raw species to reaction roles (`A`, optional `I`, optional `B`) and
fit those role-based models to measured wafer thickness maps under matching
tool/recipe/Fluent conditions.

## Decision

1. The primary concept is **role-based modeling**, not `aib_ode` as a permanent
   public model name.
2. CVD and ALD must be executable as separate process modes:
   - CVD uses a continuous role-based model path.
   - ALD uses a transient role-state model path.
3. Existing `aib_ode` behavior remains a compatibility implementation for the
   current CVD-like role model and existing tests.
4. Implementations should use the small process-model registry and common
   dispatcher rather than adding more `aib_ode` special cases.
5. Role discovery remains first-class:
   - users may fix roles when known;
   - users may enumerate candidate raw-species assignments when roles are unknown;
   - rankings must be based first on measured thickness maps, role stability,
     next-best gaps, and complexity.
6. Keep the first implementation simple:
   - preserve discrete A/I/B role enumeration before adding weighted role exposure;
   - support optional multi-case train/holdout configs with minimal YAML;
   - avoid a large dataset framework unless later justified.

## Consequences

- ADR 0008 remains valid only for the retirement of legacy `power_law/lhhw/root_solve`.
- Requirements and docs must not describe `sim.model.name = aib_ode` as the final
  public runtime contract.
- Existing configs may continue to use `aib_ode` or compatibility aliases, but
  new user-facing CVD/ALD work should prefer process-specific names.
- New Codex tasks should prefer:
  1. `fixed|enumerate` role mode cleanup,
  2. CVD/ALD command separation,
  3. role-summary and role-ranking outputs,
  4. multi-condition role stability,
  5. ALD state-model changes following ADR 0020.
