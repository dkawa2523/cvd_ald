"""Optional multi-z reference-plane synthetic runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deposim_schema import RunSpec

from .domain import build_domain_grid
from .physics.cvd_steady import CVDSteadyResult, FieldBundle, run_cvd_steady
from .synthetic_inputs import build_synthetic_field_bundle

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


def _resolve_z_refs_mm(run_spec: RunSpec) -> list[float]:
    z_list = [float(v) for v in (run_spec.reference_plane.z_ref_mm_list or [])]
    if z_list:
        return z_list
    return [float(run_spec.reference_plane.z_ref_mm)]


def _scaled_fields(base: FieldBundle, *, scale: float) -> FieldBundle:
    scaled_c_ref = {name: np.asarray(value, dtype=float) * float(scale) for name, value in base.C_ref.items()}
    return FieldBundle(C_ref=scaled_c_ref, U=base.U, T=base.T, scalars=base.scalars)


def _run_single(run_spec: RunSpec, fields: FieldBundle) -> CVDSteadyResult:
    grid = build_domain_grid(run_spec.domain)
    return run_cvd_steady(
        grid=grid,
        fields=fields,
        model_config=run_spec.model,
        process_time_s=run_spec.time.process_time_s,
        solver_config=run_spec.solver,
    )


def run_multi_z_synthetic(
    run_spec: RunSpec,
    *,
    grid: Any | None = None,
    base_fields: FieldBundle | None = None,
) -> MultiZResult:
    """Run single-plane compatible or multi-plane synthetic reference-plane workflow."""

    _require_numpy()
    if grid is None:
        grid = build_domain_grid(run_spec.domain)
    if base_fields is None:
        base_fields = build_synthetic_field_bundle(run_spec, grid)
    base_z = float(run_spec.reference_plane.z_ref_mm)
    z_refs = _resolve_z_refs_mm(run_spec)

    plane_results: list[np.ndarray] = []
    per_plane_mean: list[float] = []
    for z in z_refs:
        scale = base_z / float(z)
        result = _run_single(run_spec, _scaled_fields(base_fields, scale=scale))
        arr = np.asarray(result.thickness, dtype=float)
        plane_results.append(arr)
        per_plane_mean.append(float(np.mean(arr)))

    plane_stack = np.stack(plane_results, axis=0)
    if len(z_refs) == 1:
        combined = plane_stack[0]
    else:
        combined = np.mean(plane_stack, axis=0)

    diagnostics = {
        "z_ref_mm_list": z_refs,
        "plane_count": len(z_refs),
        "plane_thickness_mean": per_plane_mean,
        "single_plane_compat_mode": len(z_refs) == 1,
    }
    deposition_rate = combined / float(run_spec.time.process_time_s)
    return MultiZResult(
        thickness=combined,
        deposition_rate=deposition_rate,
        R=deposition_rate.copy(),
        Cs={},
        plane_thickness=plane_stack,
        diagnostics=diagnostics,
    )


__all__ = ["MultiZResult", "run_multi_z_synthetic"]
