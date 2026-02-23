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
    update_root_files(project_dir.parent)


def update_root_files(root_dir: Path) -> None:
    """Update root-level `results/index.html` and `results/summary.json`."""

    projects: list[dict[str, Any]] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        index_path = child / "index.html"
        if not summary_path.exists() or not index_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            run_count = int(payload.get("run_count", 0))
        except Exception:
            run_count = 0
        latest = payload.get("latest_metrics", {})
        timestamp = str(latest.get("timestamp_utc", ""))
        projects.append(
            {
                "project": child.name,
                "timestamp_utc": timestamp,
                "run_count": run_count,
                "latest_run_id": str(payload.get("latest_run_id", "")),
                "index_path": f"{child.name}/index.html",
            }
        )

    projects.sort(key=lambda row: (row.get("timestamp_utc", ""), row.get("project", "")), reverse=True)
    latest_project = projects[0]["project"] if projects else None
    root_summary = {
        "latest_project": latest_project,
        "project_count": len(projects),
        "projects": projects,
    }
    (root_dir / "summary.json").write_text(json.dumps(root_summary, indent=2), encoding="utf-8")

    if projects:
        latest = projects[0]
        latest_line = (
            f'Latest project: <a href="{latest["index_path"]}">{latest["project"]}</a> '
            f'({latest["latest_run_id"]})'
        )
    else:
        latest_line = "No projects yet."
    items = "\n".join(
        f'<li><a href="{row["index_path"]}">{row["project"]}</a> '
        f'(runs={row["run_count"]}, latest={row["latest_run_id"]})</li>'
        for row in projects
    )
    index_html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>deposim results</title></head>
<body>
  <h1>deposim results</h1>
  <p>{latest_line}</p>
  <p><a href="summary.json">summary.json</a></p>
  <h2>Projects</h2>
  <ul>{items}</ul>
</body>
</html>
"""
    (root_dir / "index.html").write_text(index_html, encoding="utf-8")


__all__ = ["next_run_dir", "update_project_files", "update_root_files"]
