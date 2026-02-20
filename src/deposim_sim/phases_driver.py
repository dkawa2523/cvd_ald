"""Phase-mode execution helpers and scalar driver preview for synthetic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deposim_sim.domain import build_domain_grid
from deposim_sim.physics.cvd_steady import FieldBundle, run_cvd_steady
from deposim_sim.synthetic_inputs import synthetic_pattern
from deposim_sim.validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for phase driver execution.")


@dataclass(frozen=True)
class PhaseRunResult:
    total_thickness: np.ndarray
    phase_thickness: list[np.ndarray]
    input_preview: list[dict[str, Any]]


def _base_scalars(run_spec: Any) -> dict[str, float]:
    return {
        "c_ref_mol_m3": float(run_spec.inputs.c_ref_mol_m3),
        "temperature_k": float(run_spec.inputs.temperature_k),
        "pressure_pa": float(run_spec.inputs.pressure_pa),
        "omega_rad_s": float(run_spec.inputs.omega_rad_s),
    }


def _phase_list(run_spec: Any) -> list[dict[str, Any]]:
    phases = list(getattr(run_spec.time, "phases", []) or [])
    if phases:
        out: list[dict[str, Any]] = []
        for idx, phase in enumerate(phases):
            name = str(phase.get("name", f"phase_{idx+1:02d}"))
            duration = float(phase.get("duration_s", 0.0))
            if duration <= 0.0:
                raise ValueError(f"phase '{name}' must define duration_s > 0")
            scalar_overrides = phase.get("scalar_overrides", {})
            if scalar_overrides is None:
                scalar_overrides = {}
            if not isinstance(scalar_overrides, dict):
                raise ValueError(f"phase '{name}' scalar_overrides must be a mapping")
            out.append({"name": name, "duration_s": duration, "scalar_overrides": dict(scalar_overrides)})
        return out
    return [{"name": "single_phase", "duration_s": float(run_spec.time.process_time_s), "scalar_overrides": {}}]


def build_phase_input_preview(run_spec: Any) -> list[dict[str, Any]]:
    """Return per-phase effective scalar inputs after applying overrides/drivers."""
    preview: list[dict[str, Any]] = []
    base = _base_scalars(run_spec)
    schedule = getattr(run_spec.drivers, "scalar_schedule", {}) or {}
    if not isinstance(schedule, dict):
        raise ValueError("drivers.scalar_schedule must be a mapping")

    for idx, phase in enumerate(_phase_list(run_spec)):
        scalars = dict(base)
        scalars.update({k: float(v) for k, v in phase["scalar_overrides"].items() if k in scalars})
        phase_schedule = schedule.get(phase["name"])
        if isinstance(phase_schedule, dict):
            scalars.update({k: float(v) for k, v in phase_schedule.items() if k in scalars})
        preview.append(
            {
                "phase_index": idx,
                "phase_name": phase["name"],
                "duration_s": phase["duration_s"],
                "effective_scalars": scalars,
            }
        )
    return preview


def run_phased_synthetic(run_spec: Any) -> PhaseRunResult:
    """Run phased synthetic CVD workflow and return total thickness + preview."""
    _require_numpy()
    validate_run_spec(run_spec)
    grid = build_domain_grid(run_spec.domain)
    pattern = synthetic_pattern(
        run_spec.inputs.synthetic_case,
        grid,
        random_seed=getattr(run_spec, "random_seed", 0),
    )
    preview = build_phase_input_preview(run_spec)
    phase_thickness: list[np.ndarray] = []
    total = np.zeros(grid.shape, dtype=float)

    for item in preview:
        scalars = item["effective_scalars"]
        c_ref = {
            species: float(scalars["c_ref_mol_m3"]) * pattern * (1.0 + 0.03 * idx)
            for idx, species in enumerate(run_spec.reference_plane.species)
        }
        fields = FieldBundle(
            C_ref=c_ref,
            T=np.full(grid.shape, float(scalars["temperature_k"]), dtype=float),
            scalars={"omega_rad_s": float(scalars["omega_rad_s"]), "pressure_pa": float(scalars["pressure_pa"])},
        )
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=run_spec.model,
            process_time_s=float(item["duration_s"]),
            solver_config=run_spec.solver,
        )
        phase_thickness.append(result.thickness)
        total = total + result.thickness

    return PhaseRunResult(total_thickness=total, phase_thickness=phase_thickness, input_preview=preview)


__all__ = ["PhaseRunResult", "build_phase_input_preview", "run_phased_synthetic"]
