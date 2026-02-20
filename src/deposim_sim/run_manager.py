"""Run output management with deterministic layout and fixed entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deposim_report import write_run_report
from deposim_schema import RunSpec, compose_and_save_sim_config

from .domain import DomainGrid, radial_profile
from .metrics import compute_kpi_metrics
from .results_index import next_run_dir, update_project_files
from .zarr_output import save_array_store

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _collect_npz_arrays(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}__{key}" if prefix else str(key)
            _collect_npz_arrays(child, value[key], out)
        return
    try:
        arr = np.asarray(value)
    except Exception:
        return
    if arr.dtype.kind in {"U", "S", "O"}:
        return
    out[prefix] = arr


def _run_summary(
    run_id: str,
    thickness: np.ndarray,
    diagnostics: Mapping[str, Any],
    grid: DomainGrid,
    *,
    kpi_config: Any | None = None,
) -> dict[str, Any]:
    r_mm, profile = radial_profile(thickness, grid)
    valid = np.isfinite(profile)
    center = float(profile[valid][0]) if np.any(valid) else float("nan")
    edge = float(profile[valid][-1]) if np.any(valid) else float("nan")
    status = np.asarray(diagnostics.get("root_status_map", np.zeros_like(thickness, dtype=int)))
    failures = np.asarray(diagnostics.get("root_failure_mask", np.zeros_like(thickness, dtype=bool)))
    iters = np.asarray(diagnostics.get("root_iteration_count", np.zeros_like(thickness, dtype=int)))
    failure_fraction = diagnostics.get("root_failure_fraction")
    if failure_fraction is None:
        failure_fraction = float(np.mean(failures))
    spec_min = getattr(kpi_config, "spec_min", None) if kpi_config is not None else None
    spec_max = getattr(kpi_config, "spec_max", None) if kpi_config is not None else None
    ring_count = getattr(kpi_config, "ring_count", 5) if kpi_config is not None else 5
    kpis = compute_kpi_metrics(
        thickness,
        grid,
        spec_min=spec_min,
        spec_max=spec_max,
        ring_count=int(ring_count),
    )
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine_requested": diagnostics.get("engine_requested"),
        "engine_selected": diagnostics.get("engine_selected"),
        "engine_execution_backend": diagnostics.get("engine_execution_backend"),
        "grid_shape": list(thickness.shape),
        "thickness_min": float(np.nanmin(thickness)),
        "thickness_mean": float(np.nanmean(thickness)),
        "thickness_max": float(np.nanmax(thickness)),
        "center_thickness": center,
        "edge_thickness": edge,
        "center_edge_delta": edge - center,
        "root_failure_fraction": float(failure_fraction),
        "root_failure_count": int(np.sum(failures)),
        "root_status_nonzero_fraction": float(np.mean(status != 0)),
        "root_iteration_max": int(np.max(iters)),
        "radial_profile_points": int(np.sum(valid)),
        "radial_profile_r_mm_min": float(np.nanmin(r_mm[valid])) if np.any(valid) else None,
        "radial_profile_r_mm_max": float(np.nanmax(r_mm[valid])) if np.any(valid) else None,
        "kpi": kpis,
    }


def save_run_outputs(
    *,
    run_spec: RunSpec,
    config_name: str,
    config_overrides: Sequence[str] | None,
    grid: DomainGrid,
    result: Any,
) -> Path:
    """Persist run artifacts under `results/runs/<run_id>` and update results index."""

    if np is None:
        raise RuntimeError("NumPy is required for run output management.")
    project_dir = Path(run_spec.output.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    run_id = ""
    run_dir = project_dir / "runs"
    for _ in range(5):
        run_id, run_dir = next_run_dir(project_dir, run_spec.output.run_dir_name)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Failed to allocate a unique run directory after multiple attempts.")
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    compose_and_save_sim_config(
        run_dir / run_spec.output.resolved_config_filename,
        config_name=config_name,
        overrides=config_overrides,
    )

    thickness = np.asarray(result.thickness, dtype=float)
    store_requested = str(run_spec.output.array_store)
    thickness_store = save_array_store(
        base_path=outputs_dir / "thickness",
        arrays={
            "thickness": thickness,
            "deposition_rate": np.asarray(result.deposition_rate, dtype=float),
            "R": np.asarray(result.R, dtype=float),
        },
        store=store_requested,
    )
    cs_store = save_array_store(
        base_path=outputs_dir / "cs_fields",
        arrays={f"Cs__{k}": np.asarray(v, dtype=float) for k, v in sorted(result.Cs.items())},
        store=store_requested,
    )

    diag_arrays: dict[str, np.ndarray] = {}
    _collect_npz_arrays("", result.diagnostics, diag_arrays)
    diagnostics_store = save_array_store(
        base_path=outputs_dir / "diagnostics",
        arrays=diag_arrays,
        store=store_requested,
    )

    r_mm, profile = radial_profile(thickness, grid)
    radial_store = save_array_store(
        base_path=outputs_dir / "radial_profile",
        arrays={"r_mm": r_mm, "thickness_radial": profile},
        store=store_requested,
    )

    summary = _run_summary(
        run_id,
        thickness,
        result.diagnostics,
        grid,
        kpi_config=getattr(run_spec, "kpi", None),
    )
    summary["array_store_requested"] = store_requested
    summary["artifact_store"] = {
        "thickness": thickness_store["store_used"],
        "cs_fields": cs_store["store_used"],
        "diagnostics": diagnostics_store["store_used"],
        "radial_profile": radial_store["store_used"],
    }
    summary["artifact_paths"] = {
        "thickness": Path(thickness_store["path"]).relative_to(run_dir).as_posix(),
        "cs_fields": Path(cs_store["path"]).relative_to(run_dir).as_posix(),
        "diagnostics": Path(diagnostics_store["path"]).relative_to(run_dir).as_posix(),
        "radial_profile": Path(radial_store["path"]).relative_to(run_dir).as_posix(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if run_spec.output.write_report:
        write_run_report(
            run_dir=run_dir,
            run_id=run_id,
            grid=grid,
            thickness=thickness,
            diagnostics=result.diagnostics,
            summary=summary,
            output_links=[
                summary["artifact_paths"]["thickness"],
                summary["artifact_paths"]["cs_fields"],
                summary["artifact_paths"]["diagnostics"],
                summary["artifact_paths"]["radial_profile"],
                "summary.json",
                run_spec.output.resolved_config_filename,
            ],
        )
    update_project_files(project_dir, summary)
    return run_dir


__all__ = ["save_run_outputs"]
