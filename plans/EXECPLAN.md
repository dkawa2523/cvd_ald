# Execution Plan (Detailed)

This plan is designed for Codex CLI autorun with minimal checkpoints.

## P0 (Baseline runnable system)

Goal: a runnable, physically constrained baseline that can compute a 2D wafer thickness map for
**CVD steady** using synthetic inputs, including standard diagnostics and deterministic outputs.

Scope:
- src-layout Python packages:
  - deposim_schema
  - deposim_sim
  - deposim_report
- Hydra-based YAML config composition (sim side)
- Transport–reaction coupling with scalar progress-variable root solve (vectorized bisection)
- Rate laws:
  - powerlaw
  - saturation/inhibition
- Mass-transfer models:
  - stagnant film
  - rotating disk correlation (guard omega=0)
- Output layout:
  - results/index.html entry point
  - run_id folder with resolved config, arrays, plots, report.html
- Tests and a smoke run

Checkpoint:
- one checkpoint task at end of P0: runs the P0 verify gate and stops.

## P1 (Early integration and model breadth)

Goal: make the system usable for practical studies.

- Runtime `wafer_2d_xy` domain support (not schema-only)
- Registry metadata and compatibility validator gate
- MeasurementAdapter + KPI + comparison report path (front-loaded)
- DOE runner and z_ref sensitivity workflow
- Additional kinetics/net models
- ALD front-load:
  - `time.phases` + driver extension
  - state closure modes (`dynamic_ode` / `steady_state`)
  - sticking-flux pathway
- Bosanquet diffusivity + pattern loading
- Identifiability diagnostics (finite-difference sensitivity/correlation)
- Optional JAX path remains extras-based (no hard core dependency)
- Benchmark helper that preserves user-selected compute policy

No checkpoint by default (unless later ADR changes this).

## P2 (Assimilation/optimization and ALD)

Goal: enable parameter estimation and ALD recipe modeling.

- deposim_opt:
  - parameter transforms and constraints
  - robust losses
  - optional acceleration hooks (kept extras-based)
- ALD skeleton hardening and integration tests (after P1 front-load)
- ClearML integration (optional separate package)
- Advanced IO:
  - real CFD field loaders (HDF5/Zarr/CSV plugins)
  - measurement ingestion and coordinate transforms
- Optional multi-z reference-plane support
