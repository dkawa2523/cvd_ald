"""Nested path access helpers for dataclass/dict hybrid configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_nested(root: Any, path: str) -> Any:
    """Resolve dot-path from mapping/object tree."""

    current = root
    tokens = [tok for tok in str(path).split(".") if tok]
    for token in tokens:
        if hasattr(current, token):
            current = getattr(current, token)
            continue
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"parameter path not found: {path!r}")
            current = current[token]
            continue
        raise ValueError(f"parameter path not found: {path!r}")
    return current


def set_nested(root: Any, path: str, value: Any) -> None:
    """Set dot-path into mapping/object tree without creating new object attributes."""

    tokens = [tok for tok in str(path).split(".") if tok]
    if not tokens:
        raise ValueError("parameter path must be non-empty")

    current = root
    for token in tokens[:-1]:
        if hasattr(current, token):
            current = getattr(current, token)
            continue
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"parameter path not found: {path!r}")
            current = current[token]
            continue
        raise ValueError(f"parameter path not found: {path!r}")

    leaf = tokens[-1]
    if hasattr(current, leaf):
        setattr(current, leaf, value)
        return
    if isinstance(current, dict):
        current[leaf] = value
        return
    raise ValueError(f"parameter path not writable: {path!r}")
