"""Multi-z reference-plane utility built on AIB execution."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .input_builder import load_fluent_npz_v2
from .pipeline import run_aib_from_spec
from .validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MultiZResult:
    thickness: np.ndarray
    deposition_rate: np.ndarray
    R: np.ndarray
    Cs: dict[str, np.ndarray]
    plane_thickness: np.ndarray
    diagnostics: dict[str, Any]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for multi-z simulation.")


def _resolve_z_refs_mm(sim: Any) -> list[float]:
    # Compatibility path: accept optional legacy z_ref_mm_list attribute if present.
    raw = getattr(sim.reference_plane, "z_ref_mm_list", None)
    if raw is not None:
        values = [float(v) for v in list(raw)]
        if values:
            return values
    return [float(sim.reference_plane.z_ref_mm)]


def run_multi_z_synthetic(
    run_spec: Any,
    *,
    grid: Any | None = None,
    base_fields: Any | None = None,
) -> MultiZResult:
    """Run multi-z AIB workflow by scaling Fluent concentration inputs per z plane."""

    del grid, base_fields  # kept for call-site compatibility
    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)
    validate_run_spec(sim)
    base_z = float(sim.reference_plane.z_ref_mm)
    z_refs = _resolve_z_refs_mm(sim)

    fluent = load_fluent_npz_v2(
        path=sim.inputs.fluent.file,
        mode=sim.inputs.fluent.mode,
        keys=sim.inputs.fluent.keys,
        species=sim.inputs.fluent.species,
    )

    plane_results: list[np.ndarray] = []
    plane_rates: list[np.ndarray] = []
    plane_r: list[np.ndarray] = []
    cs_a_planes: list[np.ndarray] = []
    cs_b_planes: list[np.ndarray] = []
    per_plane_mean: list[float] = []
    plane_meta: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="deposim_multiz_") as tmp:
        tmp_root = Path(tmp)
        for idx, z in enumerate(z_refs):
            scale = base_z / max(float(z), 1.0e-12)
            scaled_cref = np.asarray(fluent.cref, dtype=float) * float(scale)
            fluent_path = tmp_root / f"multiz_plane_{idx:03d}.npz"
            payload: dict[str, Any] = {
                str(sim.inputs.fluent.keys.xy): np.asarray(fluent.xy, dtype=float),
                str(sim.inputs.fluent.keys.cref): scaled_cref,
            }
            if fluent.time is not None:
                payload[str(sim.inputs.fluent.keys.time)] = np.asarray(fluent.time, dtype=float)
            np.savez(fluent_path, **payload)

            plane_spec = deepcopy(run_spec)
            plane_sim = getattr(plane_spec, "sim", plane_spec)
            plane_sim.inputs.fluent.file = str(fluent_path)
            plane_sim.reference_plane.z_ref_mm = float(z)
            plane_out = run_aib_from_spec(plane_spec)

            thickness = np.asarray(plane_out.thickness, dtype=float)
            plane_results.append(thickness)
            plane_rates.append(np.asarray(plane_out.deposition_rate, dtype=float))
            plane_r.append(np.asarray(plane_out.R, dtype=float))
            cs_a_planes.append(np.asarray(plane_out.Cs.get("A", np.zeros_like(thickness)), dtype=float))
            cs_b_planes.append(np.asarray(plane_out.Cs.get("B", np.full_like(thickness, np.nan)), dtype=float))
            per_plane_mean.append(float(np.mean(thickness)))
            plane_meta.append(
                {
                    "z_ref_mm": float(z),
                    "scale": float(scale),
                    "non_bracketed_total": int(plane_out.diagnostics.get("non_bracketed_total", 0)),
                }
            )

    plane_stack = np.stack(plane_results, axis=0)
    rate_stack = np.stack(plane_rates, axis=0)
    r_stack = np.stack(plane_r, axis=0)
    cs_a_stack = np.stack(cs_a_planes, axis=0)
    cs_b_stack = np.stack(cs_b_planes, axis=0)
    if len(z_refs) == 1:
        combined = plane_stack[0]
        combined_rate = rate_stack[0]
        combined_r = r_stack[0]
        cs_a = cs_a_stack[0]
        cs_b = cs_b_stack[0]
    else:
        combined = np.mean(plane_stack, axis=0)
        combined_rate = np.mean(rate_stack, axis=0)
        combined_r = np.mean(r_stack, axis=0)
        cs_a = np.mean(cs_a_stack, axis=0)
        if np.any(np.isfinite(cs_b_stack)):
            cs_b = np.nanmean(cs_b_stack, axis=0)
        else:
            cs_b = np.full(combined.shape, np.nan, dtype=float)

    diagnostics = {
        "z_ref_mm_list": z_refs,
        "plane_count": len(z_refs),
        "plane_thickness_mean": per_plane_mean,
        "single_plane_compat_mode": len(z_refs) == 1,
        "plane_metadata": plane_meta,
    }
    return MultiZResult(
        thickness=combined,
        deposition_rate=combined_rate,
        R=combined_r,
        Cs={"A": cs_a, "B": cs_b},
        plane_thickness=plane_stack,
        diagnostics=diagnostics,
    )


__all__ = ["MultiZResult", "run_multi_z_synthetic"]
