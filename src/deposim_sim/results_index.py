"""Shared helpers for results run-id allocation and index/summary updates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def next_run_dir(project_dir: Path, run_dir_name: str) -> tuple[str, Path]:
    """Allocate a unique run directory path under `<project_dir>/runs`."""
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_dir_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    candidate = runs_dir / stem
    idx = 1
    while candidate.exists():
        candidate = runs_dir / f"{stem}_{idx:02d}"
        idx += 1
    return candidate.name, candidate


def update_project_files(project_dir: Path, latest_summary: Mapping[str, Any]) -> None:
    """Update top-level `summary.json` and `index.html` for result browsing."""
    runs_dir = project_dir / "runs"
    run_ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    latest_run_id = str(latest_summary["run_id"])
    project_summary = {
        "latest_run_id": latest_run_id,
        "run_count": len(run_ids),
        "latest_metrics": dict(latest_summary),
    }
    (project_dir / "summary.json").write_text(json.dumps(project_summary, indent=2), encoding="utf-8")

    items = "\n".join(
        f'<li><a href="runs/{run_id}/report.html">{run_id}</a></li>'
        for run_id in run_ids
    )
    index_html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>deposim results</title></head>
<body>
  <h1>deposim results</h1>
  <p>Latest run: <a href="runs/{latest_run_id}/report.html">{latest_run_id}</a></p>
  <p><a href="summary.json">summary.json</a></p>
  <h2>Runs</h2>
  <ul>{items}</ul>
</body>
</html>
"""
    (project_dir / "index.html").write_text(index_html, encoding="utf-8")


__all__ = ["next_run_dir", "update_project_files"]
