"""Output manifest contract utilities (output.v1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "output.v1"


class ManifestError(ValueError):
    """Raised when output manifest contract is violated."""


def _ensure(cond: bool, message: str) -> None:
    if not cond:
        raise ManifestError(message)


def _validate_rel_path(raw: Any, *, label: str) -> str:
    text = str(raw).strip()
    _ensure(bool(text), f"{label} must be non-empty")
    path = Path(text)
    _ensure(not path.is_absolute(), f"{label} must be a relative path")
    _ensure(".." not in path.parts, f"{label} must not contain parent traversal")
    return text


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate strict output.v1 manifest contract."""

    _ensure(str(manifest.get("schema_version", "")) == SCHEMA_VERSION, "manifest.schema_version must be 'output.v1'")
    _ensure(bool(str(manifest.get("run_id", ""))), "manifest.run_id must be non-empty")
    _ensure(bool(str(manifest.get("mode", ""))), "manifest.mode must be non-empty")
    _ensure(bool(str(manifest.get("created_at_utc", ""))), "manifest.created_at_utc must be non-empty")

    artifacts = manifest.get("artifacts")
    _ensure(isinstance(artifacts, Sequence), "manifest.artifacts must be a sequence")
    seen_ids: set[str] = set()
    for row in artifacts:
        _ensure(isinstance(row, Mapping), "each artifact row must be a mapping")
        art_id = str(row.get("id", ""))
        path = _validate_rel_path(row.get("path", ""), label="artifact.path")
        kind = str(row.get("kind", ""))
        _ensure(bool(art_id), "artifact.id must be non-empty")
        _ensure(bool(kind), "artifact.kind must be non-empty")
        _ensure(art_id not in seen_ids, f"duplicate artifact.id: {art_id}")
        seen_ids.add(art_id)

    plots = manifest.get("plots", [])
    _ensure(isinstance(plots, Sequence), "manifest.plots must be a sequence")
    seen_plot_ids: set[str] = set()
    for row in plots:
        _ensure(isinstance(row, Mapping), "each plot row must be a mapping")
        plot_id = str(row.get("plot_id", ""))
        _ensure(bool(plot_id), "plot.plot_id must be non-empty")
        _ensure(plot_id not in seen_plot_ids, f"duplicate plot.plot_id: {plot_id}")
        seen_plot_ids.add(plot_id)
        _validate_rel_path(row.get("path", ""), label="plot.path")
        _ensure(bool(str(row.get("source_key", ""))), "plot.source_key must be non-empty")


def build_manifest(
    *,
    run_id: str,
    mode: str,
    created_at_utc: str,
    artifacts: Sequence[Mapping[str, Any]],
    plots: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate output.v1 manifest payload."""

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "mode": str(mode),
        "created_at_utc": str(created_at_utc),
        "artifacts": [dict(row) for row in artifacts],
        "plots": [dict(row) for row in list(plots or [])],
        "metadata": dict(metadata or {}),
    }
    validate_manifest(manifest)
    return manifest


def artifact_paths(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return `{artifact_id: path}` map from manifest."""

    out: dict[str, str] = {}
    for row in list(manifest.get("artifacts", [])):
        if not isinstance(row, Mapping):
            continue
        art_id = str(row.get("id", "")).strip()
        path = str(row.get("path", "")).strip()
        if art_id and path:
            out[art_id] = path
    return out


def artifact_links(manifest: Mapping[str, Any]) -> list[str]:
    """Return artifact relative paths in declaration order."""

    links: list[str] = []
    for row in list(manifest.get("artifacts", [])):
        if not isinstance(row, Mapping):
            continue
        path = str(row.get("path", "")).strip()
        if path:
            links.append(path)
    return links


def validate_manifest_files(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Validate that required artifacts declared by manifest exist on disk."""

    validate_manifest(manifest)
    for row in list(manifest.get("artifacts", [])):
        if not isinstance(row, Mapping):
            continue
        required = bool(row.get("required", False))
        if not required:
            continue
        rel = str(row.get("path", "")).strip()
        path = run_dir / rel
        _ensure(path.exists(), f"required artifact is missing: {rel}")


def write_manifest(run_dir: Path, manifest: Mapping[str, Any], *, validate_files: bool = True) -> Path:
    """Write validated manifest to `outputs/manifest.json` and return the path."""

    validate_manifest(manifest)
    out_path = run_dir / "outputs" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    if validate_files:
        validate_manifest_files(run_dir, manifest)
    return out_path


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate output manifest from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError("manifest payload must be a mapping")
    validate_manifest(payload)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "ManifestError",
    "validate_manifest",
    "build_manifest",
    "artifact_paths",
    "artifact_links",
    "validate_manifest_files",
    "write_manifest",
    "load_manifest",
]
