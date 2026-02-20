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

## P1 (Scalability and model breadth)

Goal: make the system usable for practical studies.

- DOE runner:
  - grid/random sweeps
  - case-dimension storage (avoid directory explosion)
- Additional kinetics:
  - competition / LHHW-like
  - net dep-etch-loss composition
- z_ref sensitivity workflow as a DOE factor and diagnostic
- Optional JAX engine path (CPU first; GPU only when user chooses)
- Measurement adapter skeleton (2D map alignment, masks)

No checkpoint by default (unless later ADR changes this).

## P2 (Assimilation/optimization and ALD)

Goal: enable parameter estimation and ALD recipe modeling.

- deposim_opt:
  - parameter transforms and constraints
  - robust losses
  - JAXopt implicit-diff root solving and/or compatible custom root
  - Diffrax adjoint for state ODEs (if included)
- ALD phases/time spec:
  - state models (coverage/poisoning)
  - phase scheduling and event-driven updates
- ClearML integration (optional separate package)
- Advanced IO:
  - real CFD field loaders (HDF5/Zarr/CSV plugins)
  - measurement ingestion and coordinate transforms
