"""Unified field input builder for synthetic/file input sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .domain import DomainGrid
from .io_plugins import load_inputs_from_run_spec
from .physics.cvd_steady import FieldBundle
from .synthetic_inputs import build_synthetic_field_bundle

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for field input building.")


def _grid_align(value: Any, shape: tuple[int, ...], name: str, *, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = np.full(shape, float(arr), dtype=float)
    else:
        try:
            out = np.broadcast_to(arr, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to grid shape {shape}") from exc
    if nonnegative and bool(np.any(out < 0.0)):
        raise ValueError(f"{name} must be >= 0 everywhere")
    return out


def _extract_c_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    c_ref: dict[str, Any] = {}
    raw_c_ref = payload.get("C_ref")
    if isinstance(raw_c_ref, Mapping):
        for species, value in raw_c_ref.items():
            c_ref[str(species)] = value
    for key, value in payload.items():
        if str(key).startswith("C_ref__"):
            species = str(key).split("__", 1)[1]
            if species:
                c_ref[species] = value
    return c_ref


def build_field_bundle(run_spec: Any, grid: DomainGrid) -> FieldBundle:
    """Build field bundle from run_spec.inputs using one canonical entrypoint."""

    _require_numpy()
    source_kind = str(getattr(run_spec.inputs, "source_kind", "synthetic")).strip().lower()
    if source_kind == "synthetic":
        return build_synthetic_field_bundle(run_spec, grid)

    if source_kind != "file":
        raise ValueError(f"Unsupported inputs.source_kind: {source_kind!r}")

    field_path = str(getattr(run_spec.inputs, "field_path", "")).strip()
    if not field_path:
        raise ValueError("inputs.field_path must be set when inputs.source_kind='file'")
    payload = load_inputs_from_run_spec(run_spec, Path(field_path))
    if not isinstance(payload, Mapping):
        raise ValueError("IO loader output must be a mapping")

    c_ref_raw = _extract_c_ref(payload)
    if not c_ref_raw:
        raise ValueError("file input must include at least one 'C_ref__<species>' field (or C_ref mapping).")

    declared_species = [str(name) for name in getattr(run_spec.reference_plane, "species", [])]
    missing = [name for name in declared_species if name not in c_ref_raw]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"file input is missing declared species: {missing_txt}")

    c_ref = {species: _grid_align(value, grid.shape, f"C_ref[{species}]", nonnegative=True) for species, value in c_ref_raw.items()}
    temperature = _grid_align(payload.get("T", payload.get("temperature_k", run_spec.inputs.temperature_k)), grid.shape, "T")

    scalars: dict[str, Any] = {}
    scalars["omega_rad_s"] = float(getattr(run_spec.inputs, "omega_rad_s", 0.0))
    for key, value in payload.items():
        if str(key).startswith("scalar__"):
            scalar_name = str(key).split("__", 1)[1]
            scalars[scalar_name] = value

    return FieldBundle(
        C_ref=c_ref,
        U=payload.get("U"),
        T=temperature,
        scalars=scalars,
    )


__all__ = ["build_field_bundle"]
