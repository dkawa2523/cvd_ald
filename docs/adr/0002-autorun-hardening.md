# ADR 0002: Autorun hardening for CLI compatibility and task contract enforcement

- Date: 2026-02-19
- Status: Accepted
- Scope: `scripts/codex_autorun.py`, `scripts/commands.sh`, `tasks/tasks.json`

## Context

Observed issues in current handoff:

1. `codex exec --config <file>` usage is incompatible with CLI expecting `--config <key=value>`.
2. Write-enabled mode was not using the current `--sandbox workspace-write` path.
3. Multiple autorun processes could race on shared state.
4. `scope_limits` and dependency constraints were documented but not mechanically enforced.
5. `verification_commands` were inconsistent, with empty entries in later milestones.

## Decision

We adopt the following hardening decisions:

1. Autorun CLI invocation
   - Prefer `--sandbox workspace-write` when available.
   - Use `--config key=value` form (no config-file argument).
   - Preserve compatibility fallbacks for older workspace flags.

2. Git workspace handling
   - Add autorun option `--git-check {auto,strict,skip}`.
   - In `auto`, when outside git repo, append `--skip-git-repo-check` if supported.

3. Single-runner guarantee
   - Enforce lock file: `runs/autorun.lock`.
   - Recover stale lock by PID liveness check.
   - Fail with explicit timeout code on lock contention.

4. Contract enforcement
   - Validate task contract before execution:
     - non-empty verification commands
     - verification commands reference `scripts/commands.sh`
     - dependency IDs exist
     - single stop-after checkpoint task
   - Enforce per-task dependency completion before execution.
   - Enforce `scope_limits` after each task via workspace snapshots.

5. Verification command unification
   - Standardize `tasks/tasks.json` verification entries via:
     - `./scripts/commands.sh verify_task <task_id>`
   - Add `verify_autorun` and `verify_task_contracts` commands.

## Consequences

- Autorun behavior is deterministic and safer in mixed environments (git/non-git).
- Concurrent invocations no longer silently corrupt state.
- Scope drift is actively rejected instead of being advisory only.
- Task contract quality is checked continuously, including P1/P2 placeholders.
