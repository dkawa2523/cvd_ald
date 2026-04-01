"""ALD phased synthetic runner with bounded coverage dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from deposim_schema import RunSpec

from ..domain import build_domain_grid
from ..synthetic_inputs import build_synthetic_field_bundle

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ALDResult:
    thickness: np.ndarray
    deposition_rate: np.ndarray
    R: np.ndarray
    Cs: dict[str, np.ndarray]
    coverage: np.ndarray
    diagnostics: dict[str, Any]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for ALD simulation.")


def _phase_list(run_spec: RunSpec) -> list[dict[str, Any]]:
    phases = list(run_spec.time.phases or [])
    if phases:
        return phases
    return [{"name": "default", "duration_s": run_spec.time.process_time_s, "scalar_overrides": {}}]


def run_ald_synthetic(
    run_spec: RunSpec,
    *,
    grid: Any | None = None,
    fields: Any | None = None,
) -> ALDResult:
    """Execute a bounded ALD-like synthetic coverage loop."""

    _require_numpy()
    warnings.warn(
        "run_ald_synthetic is a legacy compatibility path; use deposim_sim.pipeline AIB execution for active workflows.",
        DeprecationWarning,
        stacklevel=2,
    )
    if run_spec.time.mode != "ald_cycle":
        raise ValueError(f"run_ald_synthetic requires time.mode='ald_cycle', got {run_spec.time.mode!r}")

    if grid is None:
        grid = build_domain_grid(run_spec.domain)
    if fields is None:
        fields = build_synthetic_field_bundle(run_spec, grid)
    c_ref = np.asarray(next(iter(fields.C_ref.values())), dtype=float)

    params = getattr(run_spec.model, "state_params", {}) or {}
    sticking_coeff = float(params.get("sticking_coeff", 0.3))
    desorption_rate_s = float(params.get("desorption_rate_s", 0.05))
    growth_nm_s = float(params.get("growth_rate_nm_s", 0.8))

    theta = np.zeros(grid.shape, dtype=float)
    thickness = np.zeros(grid.shape, dtype=float)
    coverage_history: list[np.ndarray] = []

    for phase in _phase_list(run_spec):
        duration_s = float(phase.get("duration_s", run_spec.time.dt_s))
        if duration_s <= 0.0:
            raise ValueError("ALD phase duration must be > 0")
        scalar_overrides = phase.get("scalar_overrides", {})
        precursor_scale = float(scalar_overrides.get("precursor_scale", 1.0))
        react_scale = float(scalar_overrides.get("react_scale", 1.0))

        adsorption = sticking_coeff * precursor_scale * c_ref * (1.0 - theta)
        desorption = desorption_rate_s * (1.0 - react_scale) * theta
        theta = np.clip(theta + duration_s * (adsorption - desorption), 0.0, 1.0)
        thickness = thickness + growth_nm_s * theta * duration_s
        coverage_history.append(theta.copy())

    diagnostics = {
        "phase_count": len(coverage_history),
        "coverage_history": np.stack(coverage_history, axis=0),
        "coverage_min": float(np.min(theta)),
        "coverage_max": float(np.max(theta)),
        "mode": "ald_cycle",
    }
    deposition_rate = thickness / float(run_spec.time.process_time_s)
    return ALDResult(
        thickness=thickness,
        deposition_rate=deposition_rate,
        R=deposition_rate.copy(),
        Cs={},
        coverage=theta,
        diagnostics=diagnostics,
    )


__all__ = ["ALDResult", "run_ald_synthetic"]
