"""Path mutation helpers for dict/dataclass hybrid objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_dot_path(path: str, *, strip_sim_prefix: bool = False) -> str:
    text = str(path).strip()
    if strip_sim_prefix and text.startswith("sim."):
        text = text[4:]
    return ".".join(tok for tok in text.split(".") if tok)


def set_attr_path(root: Any, path: str, value: Any, *, strip_sim_prefix: bool = False) -> None:
    """Set dot path into object/mapping tree.

    Mapping nodes are created on-demand; object nodes must already exist.
    """

    cleaned = normalize_dot_path(path, strip_sim_prefix=strip_sim_prefix)
    if not cleaned:
        raise ValueError(f"invalid path: {path!r}")

    parts = cleaned.split(".")
    cursor = root
    for key in parts[:-1]:
        if isinstance(cursor, dict):
            if key not in cursor:
                cursor[key] = {}
            elif not isinstance(cursor[key], Mapping):
                raise ValueError(f"path not writable: {path!r}")
            cursor = cursor[key]
            continue
        if not hasattr(cursor, key):
            raise ValueError(f"path not found: {path!r}")
        cursor = getattr(cursor, key)

    leaf = parts[-1]
    if isinstance(cursor, dict):
        cursor[leaf] = value
        return
    if not hasattr(cursor, leaf):
        raise ValueError(f"path not writable: {path!r}")
    setattr(cursor, leaf, value)


__all__ = ["normalize_dot_path", "set_attr_path"]
