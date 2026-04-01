# ADR 0016: Runtime Dependency Bootstrap Policy for Benchmark Execution

- Date: 2026-04-01
- Status: Accepted
- Decision task: D-012

## Context

Production benchmark commands (`benchmark_wafer2d`, `benchmark_wafer2d --compare-flux-km --with-physviz`) run through `scripts/commands.sh` and require a Python runtime with core dependencies, including `hydra-core` and `omegaconf`.

In this repository, AGENTS/POLICY_LOCK requires controlled dependency handling and disallows enabling network access from tasks. When required dependencies are missing in local interpreters, benchmark reruns can be blocked even if code is ready.

## Decision

For `D-012`, we adopt the following runtime dependency bootstrap policy:

1. `hydra-core` and `omegaconf` are mandatory runtime prerequisites for benchmark execution paths.
2. Dependency provisioning for task execution must use pre-provisioned/local sources (existing interpreter, offline wheelhouse, or environment prepared outside the task).
3. Tasks MUST NOT perform network-based package installation as part of repository work.
4. If prerequisites are missing, the task records the block explicitly and stops benchmark execution rather than silently bypassing verification.

## Rationale

1. Keeps benchmark verification reproducible and auditable.
2. Preserves AGENTS network policy and avoids hidden external state changes.
3. Separates environment bootstrap responsibility from model/runtime code changes.

## Consequences

1. `scripts/commands.sh` core dependency checks remain strict.
2. Benchmark rerun is considered valid only after local runtime prerequisites are satisfied.
3. Missing-runtime incidents are handled as explicit operational blockers, not code regressions.

## Trigger To Reopen

Reopen this decision when at least one is true:

1. Repository policy allows controlled network installs during tasks.
2. A standard offline wheelhouse/bootstrap artifact is introduced and versioned.
3. Runtime packaging strategy changes (for example, bundled executable environments).
