# AGENTS.md (Rules for Codex CLI and humans)

This file defines non-negotiable rules for agents executing tasks in this repo.

## Primary rule
- **Obey POLICY_LOCK** in `docs/adr/0001-initial-decisions.md` above all else.

## Task discipline
- **1 task = 1 논点** (one coherent change set; think "one commit").
- Respect task `scope_limits` strictly:
  - max_changed_files
  - max_diff_lines
  - allowed_dirs
  - forbidden_actions

If a task cannot be done within scope, split it by creating follow-up tasks (do not expand scope silently).

## No silent spec changes
- Do NOT invent new product requirements or new physics models beyond what is already specified.
- If something is unclear or undecided, create a **decision task** and stop (do not guess),
  unless the spec explicitly provides a safe default.

## ADR discipline
- Important decisions or deviations MUST be written as an ADR under `docs/adr/`.
- Do not hide major design choices inside chat logs.

## Dependency policy
- Dependency additions are allowed, but must be minimal and staged.
- Heavy dependencies (JAX, JAXopt, Diffrax, ClearML, Zarr, etc.) should be optional extras unless
  required by a P0 gate.

## Network / external actions
- Do not enable network access from within tasks.
- Do not download large datasets.
- If external resources are required, create a decision task and stop.

## Output discipline
- Do not create file/directory explosions.
- Follow the fixed output entrypoint: `results/index.html`.
- Store automation state under `runs/` only.

## Checkpoints
- Only one checkpoint at end of P0 by default.
- Checkpoint tasks must:
  - run the P0 verify gate
  - summarize what is done
  - list next steps
  - stop_after=true and exit code 42 (handled by autorun)
