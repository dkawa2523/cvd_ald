# Codex CLI Handoff Pack

This repository contains a **Codex CLI autorun system** + **implementation plan** for a
semiconductor deposition surface modeling platform (CVD/ALD) based on the discussions in this chat.

The goal is that **Codex CLI (gpt-5.2-codex, reasoning=high)** can run:

- automatic execution of tasks
- stop only at major milestones (checkpoint/decision)
- resume from state without starting over

---

## Quick start

### 1) Unpack

Unzip this handoff pack into the **repository root**.

### 2) Run autorun

```bash
chmod +x ./scripts/*.sh
./scripts/run.sh
```

`run.sh` will:

1. run `scripts/preflight.sh` (detect python/pip, set `PYTHONPATH`, write `scripts/env.sh`)
2. run `scripts/codex_autorun.py` which executes `tasks/tasks.json` sequentially
3. record task completion in `runs/autorun_state.json`

### 3) Checkpoint / decision behavior (exit code 42)

Autorun exits with **code 42** when it reaches a task with `stop_after=true` (checkpoint or decision).

When that happens:

1. open the task output logs under `runs/`
2. read the referenced docs (often `docs/GAPS.md` or `docs/adr/...`)
3. update the necessary decision/inputs (as instructed)
4. re-run:

```bash
./scripts/run.sh
```

Autorun will resume from the next incomplete task.

---

## Important conventions

### POLICY_LOCK is the law

Before editing anything, read:

- `docs/adr/0001-initial-decisions.md`

The current POLICY_LOCK is also summarized here:

- `docs/adr/0001-initial-decisions.md`
- `scripts/commands.sh`

### runs/ and results/

- `runs/` is for automation state + logs and is **gitignored**
- `results/` is for simulation outputs and is **gitignored** by default

You should always start from `results/index.html` (fixed entry point) when inspecting simulation outputs.

### Dependency installation

This pack assumes dependencies can be installed via `pip` (editable install) **after** Codex generates
a `pyproject.toml`.

If your environment **cannot** install dependencies automatically:
- install required deps manually, or
- run in an environment where NumPy/SciPy/Matplotlib/Hydra are preinstalled.

---

## Single Source of Truth: commands

Use these instead of ad-hoc commands:

- Smoke run: `./scripts/commands.sh smoke`
- Test/verify: `./scripts/commands.sh verify_p0`

Do not hardcode `python` vs `python3`. Preflight writes `scripts/env.sh` and `commands.sh` uses it.

---

## Where to look for requirements / plan

- Requirements: `docs/REQUIREMENTS.md`
- Traceability (100% mapping requirements → tasks): `docs/TRACEABILITY.md`
- Architecture: `docs/ARCHITECTURE.md`
- Domain context: `docs/CONTEXT.md`
- Execution plan: `plans/EXECPLAN.md`
- Tasks: `tasks/TASKS.md` and `tasks/tasks.json`

---

## Troubleshooting

### Matplotlib errors
We set:

- `MPLCONFIGDIR=/tmp`

in `scripts/run.sh` and `scripts/commands.sh`.

### Import errors (src-layout)
Preflight tries:

1. `python -c "import <pkg>"`
2. `pip install -e .`
3. fallback `PYTHONPATH=src`

See `scripts/preflight.sh` and the generated `scripts/env.sh`.

---

## POLICY_LOCK (reference)

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

