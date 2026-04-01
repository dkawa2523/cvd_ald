"""Small parsing helpers for override-like literal strings."""

from __future__ import annotations

import ast
from typing import Any


def parse_literal_value(raw: Any) -> Any:
    """Parse common scalar/list literals while keeping unknown text as-is."""

    if isinstance(raw, (bool, int, float)) or raw is None:
        return raw

    text = str(raw).strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(text)
    except Exception:
        return raw


__all__ = ["parse_literal_value"]
