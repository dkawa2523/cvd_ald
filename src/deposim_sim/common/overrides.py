"""Override normalization helpers shared by CLI-oriented modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_LEGACY_PREFIX_MAP: dict[str, str] = {
    "output.project_dir": "sim.output.root_dir",
    "output.run_dir_name": "sim.output.run_name",
    "output.array_store": "sim.output.store.format",
    "time.process_time_s": "sim.time.t_proc_s",
    "time.mode": "sim.time_mode",
}

def _map_override(
    override: str,
    *,
    prefix_sim: bool,
) -> str | None:
    text = str(override).strip()
    if not text:
        return None

    for old_prefix, new_prefix in _LEGACY_PREFIX_MAP.items():
        needle = f"{old_prefix}="
        if text.startswith(needle):
            return f"{new_prefix}={text.split('=', 1)[1]}"

    if text.startswith("sim.") or not prefix_sim:
        return text
    return f"sim.{text}"


def normalize_overrides(
    overrides: Sequence[str] | None,
    *,
    prefix_sim: bool = False,
) -> list[str]:
    """Normalize override strings to the SimSpecV2 contract."""

    normalized: list[str] = []
    for item in list(overrides or []):
        mapped = _map_override(str(item), prefix_sim=prefix_sim)
        if mapped is None:
            continue
        normalized.append(mapped)
    return normalized


def normalize_sweep(sweep: Mapping[str, Sequence[Any]]) -> dict[str, Sequence[Any]]:
    """Normalize DOE sweep keys to sim.* paths while preserving value lists."""

    normalized: dict[str, Sequence[Any]] = {}
    for key, values in sweep.items():
        path = str(key).strip()
        if not path:
            continue
        if not path.startswith("sim."):
            path = f"sim.{path}"
        normalized[path] = values
    return normalized


def as_bool(value: Any) -> bool:
    """Parse common bool-like CLI/YAML values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
