# ADR 0017: Staged Refactor Policy for Compact/Readable AIB Codebase

- Date: 2026-04-02
- Status: Accepted
- Decision task: D-013

## Context

Recent P3 changes improved capability but increased local complexity and duplicated logic in several paths (`benchmark_wafer2d`, output finalization across run modes, dot-path utilities).

Repository policy (`AGENTS.md` + `POLICY_LOCK`) requires:

1. No silent spec/API changes.
2. Task-scoped, auditable refactors.
3. Stable `scripts/commands.sh` execution path and output contracts.

## Decision

For `D-013`, the refactor chain (`P3-058..P3-063`) adopts these fixed rules:

1. **Deletion boundary = code-only**.
   - Delete only source-level unused helpers proven by search/tests.
   - Do not remove top-level artifacts (zip/docs snapshots) or local untracked directories in this chain.
2. **Compatibility is strict**.
   - Keep CLI commands/flags stable.
   - Keep output contract stable (`summary.json`, `outputs/manifest.json`, `report.html`, `results/index.html` entry flow).
3. **Commonization threshold is explicit**.
   - Promote shared helper extraction where equivalent logic appears in at least 3 locations or causes repeated contract drift risk.
4. **Staged rollout only**.
   - Execute as small, reviewable tasks with gate checks after each task.

## Consequences

1. Refactor focuses on readability/maintenance without changing public behavior.
2. Output contract updates become centralized and easier to audit.
3. Duplicate map/path helper logic is reduced with lower drift risk.
4. Cleanup remains conservative and evidence-based.

## Trigger To Reopen

Reopen this decision when at least one is true:

1. Product/API contract intentionally changes.
2. Cleanup scope needs to include non-code assets.
3. A broader restructuring requires relaxing task-level scope limits.
