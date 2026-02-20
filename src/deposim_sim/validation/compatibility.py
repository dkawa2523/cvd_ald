"""Compatibility validation for model/time/input combinations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deposim_sim.models.mass_transfer import get_mass_transfer_metadata
from deposim_sim.models.rate_laws import get_rate_law_metadata


_DYNAMIC_STATE_NAMES = {"dynamic_ode", "coverage_dynamic", "coverage_ode"}


def _entry(
    metadata: Mapping[str, Mapping[str, Any]],
    model_name: str,
    kind: str,
) -> Mapping[str, Any]:
    if model_name not in metadata:
        supported = ", ".join(sorted(metadata))
        raise ValueError(f"Unknown {kind} model '{model_name}'. Supported models: {{{supported}}}")
    entry = metadata[model_name]
    for key in ("requires", "excludes", "time_modes", "governing_class"):
        if key not in entry:
            raise ValueError(f"{kind} metadata for '{model_name}' is missing key '{key}'")
    return entry


def _check_time_mode(entry: Mapping[str, Any], *, model_name: str, kind: str, mode: str) -> None:
    allowed = {str(value) for value in entry.get("time_modes", [])}
    if allowed and mode not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"{kind} model '{model_name}' is not compatible with time.mode='{mode}'. "
            f"Allowed modes: {{{allowed_text}}}"
        )


def _check_rotating_disk_guard(run_spec: Any) -> None:
    if str(run_spec.model.mass_transfer_name) != "rotating_disk":
        return
    omega = float(getattr(run_spec.inputs, "omega_rad_s", 0.0))
    params = getattr(run_spec.model, "mass_transfer_params", {}) or {}
    if not isinstance(params, Mapping):
        raise ValueError("model.mass_transfer_params must be a mapping")
    guard = str(params.get("omega_zero_guard", "error")).strip().lower()
    if guard in {"fallback", "fallback_stagnant_film", "stagnant_film"}:
        return
    if omega <= 0.0:
        raise ValueError(
            "Invalid configuration: rotating_disk with omega_rad_s<=0 requires "
            "omega_zero_guard='fallback_stagnant_film' or positive omega."
        )


def _check_state_time_mode(run_spec: Any) -> None:
    state_name = str(getattr(run_spec.model, "state_name", "none")).strip().lower()
    mode = str(getattr(run_spec.time, "mode", "")).strip().lower()
    if state_name in _DYNAMIC_STATE_NAMES and mode == "cvd_steady":
        raise ValueError(
            "Invalid configuration: dynamic state model cannot run with time.mode='cvd_steady'. "
            "Use time.mode in {'cvd_transient','ald_cycle'} or a steady_state closure."
        )


def validate_run_spec(run_spec: Any) -> None:
    """Raise ValueError when run_spec contains incompatible model/time/input combinations."""
    if not hasattr(run_spec, "model") or not hasattr(run_spec, "time") or not hasattr(run_spec, "inputs"):
        raise ValueError("run_spec must include model, time, and inputs sections")

    mode = str(getattr(run_spec.time, "mode", "")).strip()
    if not mode:
        raise ValueError("time.mode must be non-empty")

    mass_name = str(getattr(run_spec.model, "mass_transfer_name", "")).strip()
    if not mass_name:
        raise ValueError("model.mass_transfer_name must be non-empty")
    mass_meta = _entry(get_mass_transfer_metadata(), mass_name, "mass_transfer")
    _check_time_mode(mass_meta, model_name=mass_name, kind="mass_transfer", mode=mode)

    kinetics_name = str(getattr(run_spec.model, "kinetics_name", "")).strip()
    if not kinetics_name:
        raise ValueError("model.kinetics_name must be non-empty")
    rate_meta = _entry(get_rate_law_metadata(), kinetics_name, "rate_law")
    _check_time_mode(rate_meta, model_name=kinetics_name, kind="rate_law", mode=mode)

    _check_rotating_disk_guard(run_spec)
    _check_state_time_mode(run_spec)

