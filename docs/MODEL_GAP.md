# MODEL_GAP

This document fixes the triage process between:

- `docs/REQUIREMENTS.md` (implementation source of truth)
- `model.md` (extended design notes / proposals)
- `model2.md` (integration-oriented roadmap update)

POLICY_LOCK in `docs/adr/0001-initial-decisions.md` always has higher priority than both.

---

## Governance Rules

1. Primary implementation spec is `docs/REQUIREMENTS.md`.
2. `model.md` and `model2.md` can propose extensions, but proposals are not implemented silently.
3. Any new requirement found only in model notes must be handled by one of:
   - a new ADR under `docs/adr/`
   - a new `type="decision"` task in `tasks/tasks.json`
4. Until approved, model-only proposals remain `DEFERRED`.

---

## Initial Gap Triage (2026-02-19)

| Item from `model.md` | Current status in requirements/tasks | Action |
|---|---|---|
| Single-runner lock (`runs/autorun.lock`) | Not previously explicit | Implemented in `scripts/codex_autorun.py` |
| Non-git autorun execution policy | Not previously explicit | Implemented (`--git-check`, auto skip-git) |
| Scope-limit mechanical enforcement per task | Implied by AGENTS/task scope only | Implemented in autorun runtime |
| Model-compare runner (`model_compare`) | Not explicit in tasks | DEFERRED (needs decision/ADR) |
| Input preview as mandatory artifact (`inputs_preview`) | Architecture mentions output shape | DEFERRED (track under P1/P2 task split) |
| `wafer_2d_xy` as equal-priority domain | Mentioned in architecture context only | DEFERRED (requires requirement/task update) |
| `rotation_average` operator policy | Mentioned in model notes only | DEFERRED (decision needed) |

---

## model2 Gap Triage (2026-02-20)

| Item from `model2.md` | Current status in requirements/tasks | Action |
|---|---|---|
| `wafer_2d_xy` runtime as first-class domain | Schema support existed; runtime grid/profile missing | ADOPTED via P1-001 |
| Registry metadata (`requires/excludes/time_modes/governing_class`) | Not formalized in runtime API | ADOPTED via P1-002 |
| Compatibility validator as preflight gate | Only partial guard behavior existed | ADOPTED via P1-003 |
| Early MeasurementAdapter + KPI + comparison report | Scheduled late and fragmented | ADOPTED via P1-004/P1-005/P1-006 |
| Strong P1/P2 verification gates | Placeholder verification remained in commands | ADOPTED via P1/P2 verify wiring updates |
| ALD phases/state/sticking earlier in roadmap | Previously centered in P2 | ADOPTED by moving feature tasks to P1-010/P1-011 |
| Bosanquet bridge + pattern loading + identifiability | Planned but not chained for early readiness | ADOPTED via P1-012/P1-013 |
| Heavy dependencies policy | Could conflict with acceleration goals | KEPT as optional extras per POLICY_LOCK |

---

## model_explain Gap Triage (2026-02-20)

| Item from `model_explain.md` | Current status in requirements/tasks | Action |
|---|---|---|
| Stefan flow correction (MS-14) | Not implemented in mass-transfer core; only noted as optional theory | DEFERRED (D-001, ADR 0003) |
| Smoothing PDE postprocess (MS-15) | Not implemented; no PDE postprocess module in runtime | DEFERRED (D-002, ADR 0004) |
| Purge residual driver (`purge_decay`) | ALD phases exist, but explicit purge-decay contract is not standardized | DEFERRED (D-003, ADR 0005) |
| Incubation/poisoning state models (MS-11/MS-12) | Partial state-closure scaffolding exists, but mechanistic contracts are not fixed | DEFERRED (D-004, ADR 0006) |
| Chamber seasoning state/drift | No requirement/task contract for seasoning state yet | DEFERRED (D-005, ADR 0007) |

Decision outcome policy:

- `ADOPT`: promote to `docs/REQUIREMENTS.md` and map in `docs/TRACEABILITY.md`
- `DEFERRED`: keep in MODEL_GAP with explicit trigger condition
- `ADR_REQUIRED`: implementation blocked until ADR/decision task is completed

Decision records:

- `D-001`: `DEFERRED` by `docs/adr/0003-d001-stefan-flow-correction-policy.md`
- `D-002`: `DEFERRED` by `docs/adr/0004-d002-smoothing-pde-policy.md`
- `D-003`: `DEFERRED` by `docs/adr/0005-d003-purge-decay-driver-policy.md`
- `D-004`: `DEFERRED` by `docs/adr/0006-d004-incubation-poisoning-policy.md`
- `D-005`: `DEFERRED` by `docs/adr/0007-d005-chamber-seasoning-policy.md`

---

## How To Update This File

When `model.md` changes:

1. Add newly proposed items to this table.
2. Map each item to one of:
   - existing requirement/task IDs, or
   - `DEFERRED` with required ADR/decision task.
3. Never mark `implemented` unless code/tests/tasks are aligned.
