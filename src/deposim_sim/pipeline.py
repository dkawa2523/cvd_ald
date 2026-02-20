"""Unified simulation pipeline dispatch for CVD/ALD/multi-z."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compute_engine import build_engine_context
from .domain import build_domain_grid
from .input_builder import build_field_bundle
from .multiz import run_multi_z_synthetic
from .physics.ald import run_ald_synthetic
from .physics.cvd_steady import run_cvd_steady

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SimRunResult:
    thickness: np.ndarray
    deposition_rate: np.ndarray
    R: np.ndarray
    Cs: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for pipeline execution.")


def run_from_run_spec(run_spec: Any) -> SimRunResult:
    """Run the correct simulator for a RunSpec and normalize the result shape."""

    _require_numpy()
    engine_ctx = build_engine_context(run_spec.compute.engine)
    grid = build_domain_grid(run_spec.domain)
    fields = build_field_bundle(run_spec, grid)
    mode = str(getattr(run_spec.time, "mode", "")).strip().lower()
    z_ref_list = list(getattr(run_spec.reference_plane, "z_ref_mm_list", []) or [])

    if mode == "ald_cycle":
        raw = run_ald_synthetic(run_spec, grid=grid, fields=fields)
    elif z_ref_list:
        raw = run_multi_z_synthetic(run_spec, grid=grid, base_fields=fields)
    else:
        raw = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=run_spec.model,
            process_time_s=run_spec.time.process_time_s,
            solver_config=run_spec.solver,
        )

    diagnostics = dict(getattr(raw, "diagnostics", {}))
    diagnostics.setdefault("dispatch_mode", mode)
    diagnostics.setdefault("dispatch_has_multiz", bool(z_ref_list))
    diagnostics.setdefault("engine_requested", engine_ctx["requested_engine"])
    diagnostics.setdefault("engine_selected", engine_ctx["selected_engine"])
    diagnostics.setdefault("engine_execution_backend", engine_ctx["execution_backend"])
    # Backward compatibility keys.
    diagnostics.setdefault("compute_engine", engine_ctx["selected_engine"])
    diagnostics.setdefault("compute_engine_requested", engine_ctx["requested_engine"])
    return SimRunResult(
        thickness=np.asarray(raw.thickness, dtype=float),
        deposition_rate=np.asarray(raw.deposition_rate, dtype=float),
        R=np.asarray(raw.R, dtype=float),
        Cs={name: np.asarray(value, dtype=float) for name, value in getattr(raw, "Cs", {}).items()},
        diagnostics=diagnostics,
    )


__all__ = ["SimRunResult", "run_from_run_spec"]
