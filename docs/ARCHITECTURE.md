# Architecture

This document describes the intended package boundaries, dependency direction, and
how configuration and outputs are managed.

---

## Package boundaries (logical separation)

The system is split into top-level Python packages (within a single repo / single editable install),
to match the requirement that numerical simulation and optimization/ML are separated.

Recommended top-level packages:

- `deposim_schema`
  - dataclasses / pydantic models for configs and I/O contracts
  - unit conventions and validation
  - minimal dependencies

- `deposim_sim`
  - numerical simulation core
  - transport–reaction coupling solver
  - domain grids and operators
  - model registry (mass transfer, rate laws, state models, net models)
  - output writing (arrays + metadata)
  - does NOT depend on optimization libraries or ClearML

- `deposim_report`
  - plotting and HTML reporting
  - deterministic outputs for runs

- `deposim_opt` (P2)
  - data assimilation / parameter estimation / surrogate modeling
  - optional JAX engine, implicit differentiation, adjoints
  - depends on (optional) heavy deps: jax, jaxopt, diffrax, etc.

- `deposim_tracking_clearml` (P2)
  - ClearML integration layer
  - optional dependency
  - core simulation must remain independent

**Dependency direction:**
schema -> sim -> report
schema -> opt (and optionally sim)
tracking is leaf-only (depends on outputs, not vice versa)

---

## Source layout

We standardize on **src layout** to avoid import ambiguity:

    src/
      deposim_schema/
      deposim_sim/
      deposim_report/
      deposim_opt/          (later)
      deposim_tracking_clearml/ (later)

Import stability is handled by `scripts/preflight.sh`:
- prefers `pip install -e .`
- otherwise sets `PYTHONPATH=src`

---

## Model registry

To keep the code base small and extensible:

- models are discovered via a registry module rather than deep directory trees.
- the registry provides a stable API for adding new models:
  - mass transfer models (k_m)
  - rate law models (kinetics)
  - state models (ALD/poisoning/coverage)
  - net models (dep/etch/loss composition)

The "registry" pattern also supports YAML selection of model names.

---

## Configuration management (Hydra + YAML)

- user-facing configuration is YAML-based (no CLI required)
- Hydra is used to compose configs, but we avoid config file explosions by:
  - keeping group count small
  - using `name:` + `params:` patterns rather than per-model YAML files
  - saving exactly one `config_resolved.yaml` in the run directory

YAML is split:
- `configs/sim/` for forward simulation
- `configs/opt/` for assimilation/optimization

---

## Output layout (anti-"迷子" design)

All simulation outputs are stored under a **project** directory (user-chosen), with a fixed entrypoint:

    results/
      index.html        # always the entry point
      summary.json      # quick glance metrics
      runs/
        <run_id>/
          config_resolved.yaml
          inputs_preview/
          outputs/
            thickness.<store>
            diagnostics.<store>
          plots/
          report.html

Key requirements:
- avoid nested case directories for DOE
- store DOE results in a case dimension (zarr/npz/hdf5; chosen by implementation)
- keep run metadata close to run outputs

---

## Single Source of Truth: run/test commands

All automation and docs MUST reference:

- `scripts/commands.sh`

It defines:
- python executable and env
- smoke run command
- verify gates

This avoids python vs python3 accidents and keeps CI consistent.

---

## GPU / CPU support

The simulation engine must be selectable by YAML:

- `engine: numpy` (CPU robust baseline)
- `engine: jax`   (optional CPU/GPU, differentiable)

**Important:**
- compute resource selection is user-controlled
- no hard-coded "single=CPU, DOE=GPU" assumptions

A benchmark helper may recommend settings, but must not override explicit user choice.
