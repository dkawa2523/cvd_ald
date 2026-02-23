"""AIB benchmark helpers for repeated throughput/timing checks."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .pipeline import run_aib_from_spec
from .validation import validate_run_spec

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
    """Run repeated AIB solves and summarize runtime + key diagnostics."""

    _require_numpy()
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)
    engine_requested = str(getattr(getattr(sim, "compute", object()), "engine", "numpy"))
    cell_count = 0

    timings: list[float] = []
    phi_b_values: list[float] = []
    f_i_values: list[float] = []
    cs_a_values: list[float] = []
    cs_b_values: list[float] = []
    residual_values: list[float] = []
    non_bracketed_total = 0
    checksum = 0.0
    for _ in range(repeats):
        start = perf_counter()
        result = run_aib_from_spec(run_spec)
        elapsed = perf_counter() - start
        timings.append(float(elapsed))
        thickness = np.asarray(result.thickness, dtype=float)
        cell_count = int(thickness.size)
        checksum += float(np.mean(thickness))
        non_bracketed_total += int(result.diagnostics.get("non_bracketed_total", 0))

        phi_b = np.asarray(result.fields.get("phi_B", np.full(thickness.shape, np.nan)), dtype=float)
        f_i = np.asarray(result.fields.get("f_I", np.full(thickness.shape, np.nan)), dtype=float)
        cs_a = np.asarray(result.fields.get("CsA_over_CrefA", np.full(thickness.shape, np.nan)), dtype=float)
        cs_b = np.asarray(result.fields.get("CsB_over_CrefB", np.full(thickness.shape, np.nan)), dtype=float)
        residual = np.asarray(result.fields.get("residual_nm", np.full(thickness.shape, np.nan)), dtype=float)

        if np.any(np.isfinite(phi_b)):
            phi_b_values.append(float(np.nanmean(phi_b)))
        if np.any(np.isfinite(f_i)):
            f_i_values.append(float(np.nanmean(f_i)))
        if np.any(np.isfinite(cs_a)):
            cs_a_values.append(float(np.nanmean(cs_a)))
        if np.any(np.isfinite(cs_b)):
            cs_b_values.append(float(np.nanmean(cs_b)))
        if np.any(np.isfinite(residual)):
            residual_values.append(float(np.nanmean(np.abs(residual))))

    best = float(min(timings))
    mean = float(np.mean(timings))
    throughput = float(cell_count / max(best, 1.0e-12))
    return {
        "engine_requested": engine_requested,
        "engine_selected": "numpy",
        "engine_execution_backend": "numpy",
        # Backward compatibility keys.
        "requested_engine": engine_requested,
        "engine_used": "numpy",
        "available_engines": ["numpy"],
        "repeats": int(repeats),
        "grid_cells": int(cell_count),
        "best_timing_sec": best,
        "mean_timing_sec": mean,
        "throughput_cells_per_s": throughput,
        "result_checksum": float(checksum),
        "phi_B_mean": float(np.mean(phi_b_values)) if phi_b_values else float("nan"),
        "f_I_mean": float(np.mean(f_i_values)) if f_i_values else float("nan"),
        "CsA_over_CrefA_mean": float(np.mean(cs_a_values)) if cs_a_values else float("nan"),
        "CsB_over_CrefB_mean": float(np.mean(cs_b_values)) if cs_b_values else float("nan"),
        "residual_nm_mae": float(np.mean(residual_values)) if residual_values else float("nan"),
        "non_bracketed_total": int(non_bracketed_total),
    }


__all__ = ["run_benchmark"]
