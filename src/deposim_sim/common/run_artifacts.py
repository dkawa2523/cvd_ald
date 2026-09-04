"""Shared run-directory and output finalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
import subprocess
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


_FINGERPRINT_SKIP_KEYS = {
    "artifact_paths",
    "file",
    "fluent_file",
    "manifest_path",
    "measurement_file",
    "output",
    "project",
    "root_dir",
    "run_name",
}


def _stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sanitize_for_fingerprint(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    if isinstance(payload, Mapping):
        out: dict[str, Any] = {}
        for raw_key in sorted(payload):
            key = str(raw_key)
            if key in _FINGERPRINT_SKIP_KEYS or key.endswith("_file"):
                continue
            out[key] = _sanitize_for_fingerprint(payload[raw_key])
        return out
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_sanitize_for_fingerprint(item) for item in payload]
    if isinstance(payload, Path):
        return payload.name
    return payload


def _digest_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 64), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": int(path.stat().st_size)}


def resolve_code_version(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return str(out.stdout).strip() or "unknown"


def resolve_code_worktree_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Record whether the saved commit fully represents the executing worktree."""

    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except Exception:
        return {"code_dirty": None, "code_diff_fingerprint": "unknown"}
    payload = status + b"\n" + diff
    return {
        "code_dirty": bool(status.strip()),
        "code_diff_fingerprint": _sha256_hex(payload),
    }


def build_provenance_metadata(
    *,
    workflow_name: str,
    config_payload: Any,
    input_paths: Sequence[str | Path] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    digests: list[dict[str, Any]] = []
    for raw_path in input_paths or ():
        path = Path(raw_path)
        if path.exists() and path.is_file():
            digests.append(_digest_file(path))
        else:
            digests.append({"missing": True})
    digest_rows = sorted(digests, key=lambda row: json.dumps(row, sort_keys=True))
    metadata = {
        "workflow_name": str(workflow_name),
        "input_fingerprint": _sha256_hex(_stable_json_bytes(digest_rows)),
        "config_fingerprint": _sha256_hex(_stable_json_bytes(_sanitize_for_fingerprint(config_payload))),
        "code_version": resolve_code_version(repo_root=repo_root),
        **resolve_code_worktree_state(repo_root=repo_root),
    }
    metadata.update(dict(extra_metadata or {}))
    return metadata


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
    "build_provenance_metadata",
    "create_run_layout",
    "finalize_run_outputs",
    "build_manifest_and_summary",
    "resolve_code_version",
    "resolve_code_worktree_state",
    "standard_artifact_rows",
]
