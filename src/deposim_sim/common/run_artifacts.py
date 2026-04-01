"""Shared run-directory and output finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..output_manifest import write_manifest
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


__all__ = ["RunLayout", "create_run_layout", "finalize_run_outputs"]
