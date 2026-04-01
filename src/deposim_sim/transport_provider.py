"""Transport coefficient providers for AIB execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_ALLOWED_ROLES = {"A", "B"}


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for transport provider execution")


def _as_float_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    return np.broadcast_to(arr, shape).astype(float, copy=True)


def _parse_param(
    value: Any,
    *,
    spatial_shape: tuple[int, ...],
    default: float,
    allow_time_axis: bool,
) -> np.ndarray:
    if value is None:
        return _as_float_array(default, spatial_shape)
    if isinstance(value, dict):
        mode = str(value.get("mode", "constant"))
        if mode != "constant":
            raise ValueError("Only constant transport parameter mode is supported")
        return _as_float_array(float(value.get("value", default)), spatial_shape)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return _as_float_array(float(arr), spatial_shape)
    if arr.shape == spatial_shape:
        return arr.astype(float, copy=True)
    if allow_time_axis and arr.ndim == len(spatial_shape) + 1 and arr.shape[1:] == spatial_shape:
        return arr.astype(float, copy=True)
    return _as_float_array(arr, spatial_shape)


def _time_slice(values: np.ndarray, t_index: int | None, *, spatial_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == spatial_shape:
        return arr
    if arr.ndim == len(spatial_shape) + 1 and arr.shape[1:] == spatial_shape:
        if t_index is None:
            idx = arr.shape[0] - 1
        else:
            idx = int(np.clip(int(t_index), 0, arr.shape[0] - 1))
        return arr[idx]
    raise ValueError(
        f"km field shape mismatch: expected {spatial_shape} or [n_t,{spatial_shape}], got shape={arr.shape}"
    )


def _resolve_clip(raw: Any) -> tuple[float, float]:
    if raw is None:
        return (0.0, float("inf"))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lo = float(raw[0])
        hi = float(raw[1])
    else:
        raise ValueError("km_clip must be [min, max]")
    if lo > hi:
        raise ValueError(f"km_clip bounds invalid: [{lo}, {hi}]")
    return lo, hi


class TransportProvider:
    """Role-aware source of km arrays."""

    def get_km(self, role: str, *, t_index: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def get_diagnostics(self, role: str, *, t_index: int | None = None) -> dict[str, Any]:
        return {"km_used": self.get_km(role, t_index=t_index)}


@dataclass(frozen=True)
class FitScalarKmProvider(TransportProvider):
    km_a: np.ndarray
    km_b: np.ndarray
    spatial_shape: tuple[int, ...]

    @classmethod
    def from_transport_params(
        cls,
        *,
        transport: dict[str, Any],
        reference_shape: tuple[int, ...],
        time_dependent: bool = False,
    ) -> "FitScalarKmProvider":
        if not reference_shape:
            raise ValueError("reference_shape must be non-empty")
        spatial_shape = tuple(reference_shape[1:]) if time_dependent else tuple(reference_shape)
        if not spatial_shape:
            raise ValueError("spatial shape must be non-empty")
        km_a = _parse_param(
            transport.get("km_A"),
            spatial_shape=spatial_shape,
            default=0.02,
            allow_time_axis=bool(time_dependent),
        )
        km_b = _parse_param(
            transport.get("km_B"),
            spatial_shape=spatial_shape,
            default=0.02,
            allow_time_axis=bool(time_dependent),
        )
        return cls(km_a=km_a, km_b=km_b, spatial_shape=spatial_shape)

    def get_km(self, role: str, *, t_index: int | None = None) -> np.ndarray:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of {_ALLOWED_ROLES}")
        base = self.km_a if role == "A" else self.km_b
        return np.asarray(_time_slice(base, t_index, spatial_shape=self.spatial_shape), dtype=float)


@dataclass(frozen=True)
class CfdFluxSinkKmProvider(TransportProvider):
    cref_a: np.ndarray
    cref_b: np.ndarray
    flux_a: np.ndarray | None
    flux_b: np.ndarray | None
    gamma_km_a: np.ndarray
    gamma_km_b: np.ndarray
    spatial_shape: tuple[int, ...]
    eps_cref: float
    km_clip: tuple[float, float]
    flux_negative_policy: str
    units_hint: str

    @classmethod
    def from_arrays(
        cls,
        *,
        cref_a: np.ndarray,
        cref_b: np.ndarray,
        flux_a: np.ndarray | None,
        flux_b: np.ndarray | None,
        transport: dict[str, Any],
        time_dependent: bool = False,
    ) -> "CfdFluxSinkKmProvider":
        cfg = dict(transport.get("from_cfd_flux_sink", {}) or {})
        eps_cref = float(cfg.get("eps_cref", 1.0e-12))
        if eps_cref <= 0.0:
            raise ValueError("eps_cref must be > 0")
        km_clip = _resolve_clip(cfg.get("km_clip", [1.0e-8, 1.0e4]))
        policy = str(cfg.get("flux_negative_policy", "error")).strip().lower()
        if policy not in {"error", "clip_to_zero", "allow"}:
            raise ValueError("flux_negative_policy must be error|clip_to_zero|allow")

        cref_a_arr = np.asarray(cref_a, dtype=float)
        cref_b_arr = np.asarray(cref_b, dtype=float)
        spatial_shape = tuple(cref_a_arr.shape[1:]) if time_dependent else tuple(cref_a_arr.shape)
        if not spatial_shape:
            raise ValueError("cref_a must include at least one spatial dimension")

        gamma_km_a = _parse_param(
            transport.get("gamma_km_A", 1.0),
            spatial_shape=spatial_shape,
            default=1.0,
            allow_time_axis=bool(time_dependent),
        )
        gamma_km_b = _parse_param(
            transport.get("gamma_km_B", 1.0),
            spatial_shape=spatial_shape,
            default=1.0,
            allow_time_axis=bool(time_dependent),
        )

        return cls(
            cref_a=cref_a_arr,
            cref_b=cref_b_arr,
            flux_a=None if flux_a is None else np.asarray(flux_a, dtype=float),
            flux_b=None if flux_b is None else np.asarray(flux_b, dtype=float),
            gamma_km_a=np.asarray(gamma_km_a, dtype=float),
            gamma_km_b=np.asarray(gamma_km_b, dtype=float),
            spatial_shape=spatial_shape,
            eps_cref=eps_cref,
            km_clip=km_clip,
            flux_negative_policy=policy,
            units_hint=str(cfg.get("units_hint", "")),
        )

    def _compute_km_pair(self, role: str, *, t_index: int | None) -> tuple[np.ndarray, np.ndarray]:
        if role == "A":
            cref_all = self.cref_a
            flux_all = self.flux_a
            gamma_all = self.gamma_km_a
        elif role == "B":
            cref_all = self.cref_b
            flux_all = self.flux_b
            gamma_all = self.gamma_km_b
        else:
            raise ValueError(f"role must be one of {_ALLOWED_ROLES}")

        if flux_all is None:
            raise ValueError(f"flux_sink for role {role} is required when km_source=from_cfd_flux_sink")

        cref = _time_slice(np.asarray(cref_all, dtype=float), t_index, spatial_shape=self.spatial_shape)
        flux_raw = _time_slice(np.asarray(flux_all, dtype=float), t_index, spatial_shape=self.spatial_shape)
        gamma = _time_slice(np.asarray(gamma_all, dtype=float), t_index, spatial_shape=self.spatial_shape)

        if self.flux_negative_policy == "error" and np.any(flux_raw < 0.0):
            raise ValueError(f"negative flux_sink detected for role {role}")
        if self.flux_negative_policy == "clip_to_zero":
            flux_work = np.clip(flux_raw, 0.0, np.inf)
        else:
            flux_work = flux_raw

        km_cfd = flux_work / np.maximum(cref, float(self.eps_cref))
        km_cfd = np.clip(km_cfd, float(self.km_clip[0]), float(self.km_clip[1]))
        km_used = np.clip(gamma * km_cfd, float(self.km_clip[0]), float(self.km_clip[1]))
        return np.asarray(km_cfd, dtype=float), np.asarray(km_used, dtype=float)

    def get_km(self, role: str, *, t_index: int | None = None) -> np.ndarray:
        _km_cfd, km_used = self._compute_km_pair(role, t_index=t_index)
        return km_used

    def get_diagnostics(self, role: str, *, t_index: int | None = None) -> dict[str, Any]:
        km_cfd, km_used = self._compute_km_pair(role, t_index=t_index)
        return {
            "km_cfd": km_cfd,
            "km_used": km_used,
            "flux_negative_policy": self.flux_negative_policy,
            "units_hint": self.units_hint,
        }


__all__ = [
    "TransportProvider",
    "FitScalarKmProvider",
    "CfdFluxSinkKmProvider",
]
