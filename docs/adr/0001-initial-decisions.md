# ADR 0001: Initial decisions (POLICY_LOCK)

- Date: 2026-01-25
- Status: Accepted
- Scope: Automation + repository execution invariants

## Context

This repo will be implemented by Codex CLI via an autorun system.
To avoid "it doesn't run" accidents, we lock down a small set of operational decisions.

## Decision

We adopt the following POLICY_LOCK:

# POLICY_LOCK (v1)

These are the fixed operational decisions for this repository automation.
They are recorded in `docs/adr/0001-initial-decisions.md` and MUST be treated as the
single source of truth by Codex CLI tasks and humans.

1) Python command selection
   - Default: **environment detection** (prefer `python3`, fallback to `python`).
   - Implementation: `scripts/preflight.sh` writes `scripts/env.sh` exporting `PYTHON`.

2) Package execution / import strategy
   - Prefer `pip install -e .` when a `pyproject.toml` exists and pip is available.
   - Fallback: `PYTHONPATH=src` when using src-layout and editable install is unavailable.
   - `scripts/preflight.sh` resolves this and writes it into `scripts/env.sh`.

3) Single Source of Truth for run/test commands
   - All run/verify commands MUST be invoked via `scripts/commands.sh`.
   - Examples:
     - Run smoke: `./scripts/commands.sh smoke`
     - Verify P0 gate: `./scripts/commands.sh verify_p0`

4) Dependency additions
   - Allowed: **YES**, but must be minimal and staged.
   - Heavy/optional deps (JAX, JAXopt, Diffrax, ClearML, Zarr, etc.) MUST be added as optional extras
     unless required by a P0 gate.
   - If a new dependency is required but uncertain, create a task-local ADR note and/or a decision task.

5) Output directories and git management
   - `runs/` is reserved for automation state and logs (gitignored).
   - `results/` is reserved for simulation outputs (gitignored by default; can be changed later).
   - `.gitignore` MUST include these paths.

6) Checkpoint policy
   - Only **one** checkpoint task at the end of P0: `type="checkpoint", stop_after=true`.
   - No additional checkpoints unless explicitly added in a later ADR.

7) Codex CLI auto-exec policy
   - Autorun MUST use workspace write mode (never read-only).
   - `scripts/codex_autorun.py` MUST detect CLI flags via `codex exec --help` and choose the best
     supported write-enabled invocation.

8) Matplotlib / runtime stability
   - `MPLCONFIGDIR=/tmp` MUST be exported by default in `scripts/run.sh` and `scripts/commands.sh`
     to avoid runtime permission issues in headless environments.


## Consequences

- All tasks MUST reference `scripts/commands.sh` for run/test commands.
- All tasks MUST respect `scripts/preflight.sh` outputs (`scripts/env.sh`).
- Any future change to these policies requires a new ADR.
