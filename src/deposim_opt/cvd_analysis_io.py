"""CSV observation loading, coordinate alignment, and artifact serialization."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_numeric_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    columns: dict[str, np.ndarray] = {}
    for name in headers:
        values: list[float] = []
        for row_index, row in enumerate(rows, start=2):
            raw = str(row.get(name, "")).strip()
            if raw == "":
                values.append(float("nan"))
                continue
            try:
                values.append(float(raw))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in {path}:{row_index}, column {name!r}: {raw!r}"
                ) from exc
        columns[name] = np.asarray(values, dtype=float)
    return headers, columns


def coordinate_matrix(columns: dict[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in ("x", "y", "z") if name not in columns]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    return np.column_stack([columns["x"], columns["y"], columns["z"]])


def _coordinate_keys(xyz: np.ndarray) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in row) for row in np.asarray(xyz, dtype=float)]


def align_validation(
    condition_xyz: np.ndarray,
    validation_xyz: np.ndarray,
    validation_values: np.ndarray,
    *,
    coordinate_decimals: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    condition_match_xyz = (
        np.round(condition_xyz, decimals=coordinate_decimals)
        if coordinate_decimals is not None
        else condition_xyz
    )
    validation_match_xyz = (
        np.round(validation_xyz, decimals=coordinate_decimals)
        if coordinate_decimals is not None
        else validation_xyz
    )
    condition_keys = _coordinate_keys(condition_match_xyz)
    validation_keys = _coordinate_keys(validation_match_xyz)
    duplicate_condition = len(condition_keys) - len(set(condition_keys))
    duplicate_validation = len(validation_keys) - len(set(validation_keys))
    if duplicate_condition or duplicate_validation:
        raise ValueError(
            "Coordinate keys must be unique; "
            f"condition duplicates={duplicate_condition}, "
            f"validation duplicates={duplicate_validation}"
        )

    validation_lookup = {key: idx for idx, key in enumerate(validation_keys)}
    condition_key_set = set(condition_keys)
    missing = [key for key in condition_keys if key not in validation_lookup]
    extra = [key for key in validation_keys if key not in condition_key_set]
    if missing or extra:
        raise ValueError(
            "Condition/validation coordinates do not match at the configured precision; "
            f"missing_in_validation={len(missing)}, extra_in_validation={len(extra)}"
        )
    order = np.asarray([validation_lookup[key] for key in condition_keys], dtype=int)
    aligned_xyz = validation_xyz[order]
    max_diff = (
        float(np.max(np.abs(condition_xyz - aligned_xyz))) if condition_xyz.size else 0.0
    )
    return validation_values[order], {
        "condition_row_count": int(condition_xyz.shape[0]),
        "validation_row_count": int(validation_xyz.shape[0]),
        "coordinate_exact_match": bool(max_diff == 0.0),
        "coordinate_match_decimals": coordinate_decimals,
        "coordinate_tolerance_match": True,
        "coordinate_max_abs_difference": max_diff,
        "condition_duplicate_coordinates": int(duplicate_condition),
        "validation_duplicate_coordinates": int(duplicate_validation),
    }


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = [
    "align_validation",
    "coordinate_matrix",
    "json_safe",
    "read_numeric_csv",
    "sha256_file",
    "write_json",
    "write_rows",
]
