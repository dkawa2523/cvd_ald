"""Phase-mode execution helpers for AIB workflows."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from deposim_sim.input_builder import load_fluent_npz_v2
from deposim_sim.pipeline import run_aib_from_spec
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
    phase_diagnostics: list[dict[str, Any]]


def _set_attr_path(root: Any, key_path: str, value: Any) -> None:
    parts = [p for p in str(key_path).split(".") if p]
    if not parts:
        return
    cursor = root
    for key in parts[:-1]:
        if isinstance(cursor, dict):
            cursor = cursor.setdefault(key, {})
        else:
            cursor = getattr(cursor, key)
    if isinstance(cursor, dict):
        cursor[parts[-1]] = value
    else:
        setattr(cursor, parts[-1], value)


def _legacy_fluent_scale(run_spec: Any, phase: dict[str, Any]) -> float:
    scalar_overrides = phase.get("scalar_overrides", {})
    if not isinstance(scalar_overrides, dict) or "c_ref_mol_m3" not in scalar_overrides:
        return 1.0
    base_c = getattr(getattr(run_spec, "inputs", object()), "c_ref_mol_m3", None)
    if base_c is None:
        return float(scalar_overrides["c_ref_mol_m3"])
    base = max(float(base_c), 1.0e-12)
    return float(scalar_overrides["c_ref_mol_m3"]) / base


def _phase_list(run_spec: Any) -> list[dict[str, Any]]:
    sim = getattr(run_spec, "sim", run_spec)
    raw = getattr(sim, "phase_schedule", None)
    if raw is None:
        raw = getattr(sim.time, "phases", None)

    if not raw:
        return [
            {
                "phase_index": 0,
                "phase_name": "single_phase",
                "duration_s": float(sim.time.t_proc_s),
                "fluent_scale": 1.0,
                "overrides": {},
            }
        ]

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(list(raw)):
        if not isinstance(item, dict):
            raise ValueError(f"phase[{idx}] must be a mapping")
        name = str(item.get("name", f"phase_{idx+1:02d}"))
        duration = float(item.get("duration_s", 0.0))
        if duration <= 0.0:
            raise ValueError(f"phase '{name}' must define duration_s > 0")
        fluent_scale = float(item.get("fluent_scale", _legacy_fluent_scale(run_spec, item)))
        overrides = item.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"phase '{name}' overrides must be a mapping")
        out.append(
            {
                "phase_index": idx,
                "phase_name": name,
                "duration_s": duration,
                "fluent_scale": fluent_scale,
                "overrides": dict(overrides),
            }
        )
    return out


def build_phase_input_preview(run_spec: Any) -> list[dict[str, Any]]:
    """Return normalized phase schedule for AIB execution."""
    return _phase_list(run_spec)


def run_phased_synthetic(run_spec: Any) -> PhaseRunResult:
    """Run phased AIB workflow and accumulate total thickness."""
    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)
    preview = build_phase_input_preview(run_spec)

    fluent = load_fluent_npz_v2(
        path=sim.inputs.fluent.file,
        mode=sim.inputs.fluent.mode,
        keys=sim.inputs.fluent.keys,
        species=sim.inputs.fluent.species,
    )
    phase_thickness: list[np.ndarray] = []
    phase_diagnostics: list[dict[str, Any]] = []
    total: np.ndarray | None = None

    with TemporaryDirectory(prefix="deposim_phases_") as tmp:
        tmp_root = Path(tmp)
        for item in preview:
            scale = float(item["fluent_scale"])
            scaled_cref = np.asarray(fluent.cref, dtype=float) * scale
            fluent_path = tmp_root / f"phase_{int(item['phase_index']):03d}.npz"
            payload: dict[str, Any] = {
                str(sim.inputs.fluent.keys.xy): np.asarray(fluent.xy, dtype=float),
                str(sim.inputs.fluent.keys.cref): scaled_cref,
            }
            if fluent.time is not None:
                payload[str(sim.inputs.fluent.keys.time)] = np.asarray(fluent.time, dtype=float)
            np.savez(fluent_path, **payload)

            phase_spec = deepcopy(run_spec)
            phase_sim = getattr(phase_spec, "sim", phase_spec)
            phase_sim.inputs.fluent.file = str(fluent_path)
            phase_sim.time.t_proc_s = float(item["duration_s"])
            for key, value in item["overrides"].items():
                _set_attr_path(phase_sim, str(key), value)

            out = run_aib_from_spec(phase_spec)
            thickness = np.asarray(out.thickness, dtype=float)
            phase_thickness.append(thickness)
            if total is None:
                total = np.zeros_like(thickness)
            total += thickness
            phase_diagnostics.append(
                {
                    "phase_name": str(item["phase_name"]),
                    "duration_s": float(item["duration_s"]),
                    "fluent_scale": float(item["fluent_scale"]),
                    "non_bracketed_total": int(out.diagnostics.get("non_bracketed_total", 0)),
                }
            )

    if total is None:
        raise ValueError("phase schedule produced zero phases")
    return PhaseRunResult(
        total_thickness=total,
        phase_thickness=phase_thickness,
        input_preview=preview,
        phase_diagnostics=phase_diagnostics,
    )


__all__ = ["PhaseRunResult", "build_phase_input_preview", "run_phased_synthetic"]
