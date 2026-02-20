"""Benchmark helpers that preserve user-selected compute policy."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .compute_engine import build_engine_context
from .domain import build_domain_grid
from .physics.cvd_steady import run_cvd_steady
from .synthetic_inputs import build_synthetic_field_bundle

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for benchmark helper.")


def run_benchmark(
    run_spec: Any,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run a small throughput/timing benchmark without engine auto-override."""

    _require_numpy()
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    engine_ctx = build_engine_context(run_spec.compute.engine)
    grid = build_domain_grid(run_spec.domain)
    fields = build_synthetic_field_bundle(run_spec, grid)
    cell_count = int(np.prod(grid.shape))

    timings: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        start = perf_counter()
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=run_spec.model,
            process_time_s=run_spec.time.process_time_s,
            solver_config=run_spec.solver,
        )
        elapsed = perf_counter() - start
        timings.append(float(elapsed))
        checksum += float(np.mean(np.asarray(result.thickness, dtype=float)))

    best = float(min(timings))
    mean = float(np.mean(timings))
    throughput = float(cell_count / max(best, 1.0e-12))
    return {
        "engine_requested": engine_ctx["requested_engine"],
        "engine_selected": engine_ctx["selected_engine"],
        "engine_execution_backend": engine_ctx["execution_backend"],
        # Backward compatibility keys.
        "requested_engine": engine_ctx["requested_engine"],
        "engine_used": engine_ctx["selected_engine"],
        "available_engines": engine_ctx["available_engines"],
        "repeats": int(repeats),
        "grid_cells": cell_count,
        "best_timing_sec": best,
        "mean_timing_sec": mean,
        "throughput_cells_per_s": throughput,
        "result_checksum": float(checksum),
    }


__all__ = ["run_benchmark"]
