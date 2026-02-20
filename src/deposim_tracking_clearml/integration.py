"""Optional ClearML integration helpers."""

from __future__ import annotations

import importlib.util
from typing import Any


def is_clearml_available() -> bool:
    return importlib.util.find_spec("clearml") is not None


def create_task(*, project_name: str, task_name: str, **kwargs: Any) -> Any:
    """Create ClearML task when optional dependency is installed."""

    if not is_clearml_available():
        raise RuntimeError(
            "ClearML is not available. Install optional extras (e.g., pip install 'deposim[clearml]')."
        )
    from clearml import Task  # type: ignore

    return Task.init(project_name=project_name, task_name=task_name, **kwargs)


__all__ = ["create_task", "is_clearml_available"]
