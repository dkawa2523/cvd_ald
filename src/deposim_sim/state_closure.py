"""State closure helpers for dynamic and steady-state coverage modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for state closure utilities.")


def _align(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    try:
        return np.broadcast_to(arr, shape).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to {shape}") from exc


def _primary_species(Cs: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    if "species" in params:
        species = str(params["species"])
        if species not in Cs:
            raise ValueError(f"state closure species '{species}' not found in Cs")
        return species
    if len(Cs) != 1:
        raise ValueError("state closure requires 'species' when Cs has multiple species")
    return next(iter(Cs))


def dynamic_ode_closure(
    *,
    Cs: Mapping[str, Any],
    params: Mapping[str, Any],
    dt_s: float,
    initial_state: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    _require_numpy()
    species = _primary_species(Cs, params)
    cs_arr = np.asarray(Cs[species], dtype=float)
    shape = cs_arr.shape
    A = float(params.get("A", 1.0))
    B = float(params.get("B", 0.0))
    m = float(params.get("m", 1.0))
    theta0_default = float(params.get("theta0", 0.0))
    theta0 = theta0_default if initial_state is None else initial_state.get("theta", theta0_default)
    theta = _align(theta0, shape, "theta0")
    rhs = A * np.clip(cs_arr, 0.0, np.inf) * np.power(np.clip(1.0 - theta, 0.0, 1.0), m) - B * theta
    theta_next = np.clip(theta + float(dt_s) * rhs, 0.0, 1.0)
    return {"theta": theta_next}


def steady_state_closure(
    *,
    Cs: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    _require_numpy()
    species = _primary_species(Cs, params)
    cs_arr = np.asarray(Cs[species], dtype=float)
    A = float(params.get("A", 1.0))
    B = float(params.get("B", 1.0))
    if A <= 0.0 or B < 0.0:
        raise ValueError("state steady_state requires A>0 and B>=0")
    theta = np.divide(A * cs_arr, A * cs_arr + B, out=np.zeros_like(cs_arr, dtype=float), where=(A * cs_arr + B) > 0.0)
    return {"theta": np.clip(theta, 0.0, 1.0)}


def resolve_state_from_model_config(
    model_config: Any,
    *,
    Cs: Mapping[str, Any],
    dt_s: float,
    initial_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    name = str(getattr(model_config, "state_name", "none")).strip().lower()
    params = getattr(model_config, "state_params", {}) or {}
    if not isinstance(params, Mapping):
        raise ValueError("model_config.state_params must be a mapping")
    if name in {"none", ""}:
        return initial_state
    if name in {"dynamic_ode", "coverage_dynamic", "coverage_ode"}:
        return dynamic_ode_closure(Cs=Cs, params=params, dt_s=dt_s, initial_state=initial_state)
    if name in {"steady_state", "coverage_steady"}:
        return steady_state_closure(Cs=Cs, params=params)
    raise ValueError(f"Unknown state closure mode: {name!r}")


__all__ = [
    "dynamic_ode_closure",
    "steady_state_closure",
    "resolve_state_from_model_config",
]
