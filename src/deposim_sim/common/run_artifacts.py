"""Shared run-directory and output finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from ..output_manifest import artifact_paths, build_manifest, write_manifest
from ..results_index import next_run_dir, update_project_files


@dataclass(frozen=True)
class RunLayout:
    project_dir: Path
    run_id: str
    run_dir: Path
    outputs_dir: Path
    plots_dir: Path
    inputs_dir: Path | None = None


def create_run_layout(
    *,
    root_dir: Path,
    project: str,
    run_name: str,
    with_inputs_dir: bool = False,
) -> RunLayout:
    project_dir = root_dir / project
    project_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = next_run_dir(project_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = None
    if with_inputs_dir:
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        project_dir=project_dir,
        run_id=run_id,
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        plots_dir=plots_dir,
        inputs_dir=inputs_dir,
    )


def finalize_run_outputs(
    *,
    layout: RunLayout,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    (layout.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(layout.run_dir, manifest)
    update_project_files(layout.project_dir, summary)


def build_manifest_and_summary(
    *,
    run_id: str,
    mode: str,
    artifacts: Sequence[Mapping[str, Any]],
    summary_fields: Mapping[str, Any] | None = None,
    plots: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a validated output manifest plus canonical summary envelope."""

    created_at = str(timestamp_utc or datetime.now(timezone.utc).isoformat())
    manifest = build_manifest(
        run_id=run_id,
        mode=mode,
        created_at_utc=created_at,
        artifacts=artifacts,
        plots=plots or [],
        metadata=metadata or {},
    )
    summary: dict[str, Any] = {
        "run_id": str(run_id),
        "timestamp_utc": created_at,
        "mode": str(mode),
        **dict(summary_fields or {}),
        "manifest_path": "outputs/manifest.json",
        "artifact_paths": artifact_paths(manifest),
    }
    return manifest, summary


def standard_artifact_rows(
    *,
    include_report: bool = True,
    extra_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build standard artifact rows with optional report and extra artifacts."""

    rows: list[dict[str, Any]] = [
        {"id": "config", "path": "config_resolved.yaml", "kind": "yaml", "required": True},
        {"id": "summary", "path": "summary.json", "kind": "json", "required": True},
    ]
    if include_report:
        rows.append({"id": "report", "path": "report.html", "kind": "html", "required": True})
    rows.append({"id": "manifest", "path": "outputs/manifest.json", "kind": "json", "required": True})
    for row in extra_rows or ():
        rows.append(dict(row))
    return rows


__all__ = [
    "RunLayout",
    "create_run_layout",
    "finalize_run_outputs",
    "build_manifest_and_summary",
    "standard_artifact_rows",
]
