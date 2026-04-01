"""Run output management for AIB simulation outputs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import warnings

from deposim_report import write_run_report
from deposim_schema import compose_and_save_sim_config

from .metrics import compute_kpi_metrics
from .common.run_artifacts import create_run_layout, finalize_run_outputs
from .output_manifest import artifact_paths, build_manifest

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for run output management")


def _normalize_field_names(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            return [tok.strip().strip("'\"") for tok in text[1:-1].split(",") if tok.strip()]
        if text:
            return [text]
        return []
    if isinstance(raw, Sequence):
        return [str(name) for name in raw]
    return []


def save_run_outputs(
    *,
    run_spec: Any,
    config_name: str,
    config_overrides: Sequence[str] | None,
    result: Any,
) -> Path:
    """Persist run artifacts and update project index files."""

    _require_numpy()
    sim = getattr(run_spec, "sim", run_spec)

    layout = create_run_layout(
        root_dir=Path(sim.output.root_dir),
        project=str(sim.output.project),
        run_name=str(sim.output.run_name),
        with_inputs_dir=False,
    )
    run_id = layout.run_id
    run_dir = layout.run_dir
    outputs_dir = layout.outputs_dir

    compose_and_save_sim_config(
        run_dir / "config_resolved.yaml",
        config_name=config_name,
        overrides=config_overrides,
    )

    fields_path = outputs_dir / "fields.npz"
    requested_fields = _normalize_field_names(getattr(sim.output, "save_fields", []))
    if requested_fields:
        missing = [k for k in requested_fields if k not in result.fields]
        if missing:
            warnings.warn(
                f"Requested save_fields were not produced and will be skipped: {missing}",
                RuntimeWarning,
                stacklevel=2,
            )
        payload = {k: np.asarray(result.fields[k]) for k in requested_fields if k in result.fields}
    else:
        payload = {k: np.asarray(v) for k, v in result.fields.items()}
    np.savez(fields_path, **payload)

    kpi = compute_kpi_metrics(np.asarray(result.thickness, dtype=float), result.grid)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    metrics = {
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "dispatch_mode": result.diagnostics.get("dispatch_mode"),
        "non_bracketed_total": int(result.diagnostics.get("non_bracketed_total", 0)),
        "kpi": kpi,
    }
    metrics_path = outputs_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_enabled = bool(sim.output.report.get("enabled", True))
    artifact_rows = [
        {"id": "config", "path": "config_resolved.yaml", "kind": "yaml", "required": True},
        {"id": "summary", "path": "summary.json", "kind": "json", "required": True},
        {"id": "manifest", "path": "outputs/manifest.json", "kind": "json", "required": True},
        {"id": "fields", "path": "outputs/fields.npz", "kind": "npz", "required": True},
        {"id": "metrics", "path": "outputs/metrics.json", "kind": "json", "required": True},
    ]
    if report_enabled:
        artifact_rows.append({"id": "report", "path": "report.html", "kind": "html", "required": True})
    provisional_manifest = build_manifest(
        run_id=run_id,
        mode="simulation",
        created_at_utc=timestamp_utc,
        artifacts=artifact_rows,
        plots=[],
        metadata={
            "dispatch_mode": result.diagnostics.get("dispatch_mode"),
            "non_bracketed_total": int(result.diagnostics.get("non_bracketed_total", 0)),
        },
    )
    plot_records: list[dict[str, Any]] = []
    if report_enabled:
        report_diagnostics = dict(result.diagnostics)
        report_threshold = float(sim.output.report.get("solver_warning_non_bracketed_threshold", 0))
        report_diagnostics["solver_warning_non_bracketed_threshold"] = report_threshold
        plot_records = write_run_report(
            run_dir=run_dir,
            run_id=run_id,
            grid=result.grid,
            thickness=result.thickness,
            diagnostics=report_diagnostics,
            summary={
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "thickness_min": float(np.nanmin(result.thickness)),
                "thickness_mean": float(np.nanmean(result.thickness)),
                "thickness_max": float(np.nanmax(result.thickness)),
                "kpi": kpi,
            },
            manifest=provisional_manifest,
        )

    manifest = build_manifest(
        run_id=run_id,
        mode="simulation",
        created_at_utc=timestamp_utc,
        artifacts=artifact_rows,
        plots=plot_records,
        metadata={
            "dispatch_mode": result.diagnostics.get("dispatch_mode"),
            "non_bracketed_total": int(result.diagnostics.get("non_bracketed_total", 0)),
        },
    )
    artifact_map = artifact_paths(manifest)
    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "mode": "simulation",
        "thickness_min": float(np.nanmin(result.thickness)),
        "thickness_mean": float(np.nanmean(result.thickness)),
        "thickness_max": float(np.nanmax(result.thickness)),
        "kpi": kpi,
        "manifest_path": "outputs/manifest.json",
        "artifact_paths": artifact_map,
    }
    finalize_run_outputs(layout=layout, summary=summary, manifest=manifest)
    return run_dir


__all__ = ["save_run_outputs"]
