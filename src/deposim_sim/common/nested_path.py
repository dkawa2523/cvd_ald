"""Backward-compatible wrappers over canonical dot-path utilities."""

from __future__ import annotations

from typing import Any

from .path_tools import get_attr_path, set_attr_path


def get_nested(root: Any, path: str) -> Any:
    """Resolve dot-path from mapping/object tree."""

    return get_attr_path(root, path, strip_sim_prefix=False)


def set_nested(root: Any, path: str, value: Any) -> None:
    """Set dot-path without creating missing mapping nodes."""

    set_attr_path(
        root,
        path,
        value,
        strip_sim_prefix=False,
        create_missing_mappings=False,
    )


__all__ = ["get_nested", "set_nested"]
