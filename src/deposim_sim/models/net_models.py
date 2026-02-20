"""Net-thickness models and registry utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


NetModel = Callable[..., tuple[np.ndarray, dict[str, np.ndarray]]]
_NET_MODEL_REGISTRY: dict[str, NetModel] = {}


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for net model evaluation.")


def _align(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    _require_numpy()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    try:
        return np.broadcast_to(arr, shape).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to {shape}") from exc


def _merge(params: Mapping[str, Any] | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(params) if params is not None else {}
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def deposition_only(
    *,
    deposition_rate: Any,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep deposition-only sign convention: net = deposition."""
    _ = _merge(params, overrides)
    dep = np.asarray(deposition_rate, dtype=float)
    return dep, {"dep_rate": dep, "etch_rate": np.zeros(dep.shape, dtype=float), "loss_rate": np.zeros(dep.shape, dtype=float)}


def dep_etch_loss(
    *,
    deposition_rate: Any,
    params: Mapping[str, Any] | None = None,
    etch_rate: Any = None,
    loss_rate: Any = None,
    **overrides: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compose net thickness as deposition - etch - loss."""
    merged = _merge(params, overrides)
    dep = np.asarray(deposition_rate, dtype=float)
    shape = dep.shape

    if etch_rate is None:
        etch_rate = merged.get("etch_rate")
    if loss_rate is None:
        loss_rate = merged.get("loss_rate")

    if etch_rate is None:
        etch_fraction = float(merged.get("etch_fraction", 0.0))
        etch = np.clip(etch_fraction, 0.0, np.inf) * dep
    else:
        etch = _align(etch_rate, shape, "etch_rate")

    if loss_rate is None:
        loss_fraction = float(merged.get("loss_fraction", 0.0))
        loss = np.clip(loss_fraction, 0.0, np.inf) * dep
    else:
        loss = _align(loss_rate, shape, "loss_rate")

    etch = np.clip(etch, 0.0, np.inf)
    loss = np.clip(loss, 0.0, np.inf)
    net = dep - etch - loss
    return net, {"dep_rate": dep, "etch_rate": etch, "loss_rate": loss}


def register_net_model(name: str, model: NetModel, *, overwrite: bool = False) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("net model name must be non-empty")
    if not overwrite and key in _NET_MODEL_REGISTRY:
        raise ValueError(f"net model '{key}' is already registered")
    _NET_MODEL_REGISTRY[key] = model


def available_net_models() -> tuple[str, ...]:
    return tuple(sorted(_NET_MODEL_REGISTRY))


def resolve_net_model(name: str) -> NetModel:
    key = str(name).strip()
    try:
        return _NET_MODEL_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(available_net_models())
        raise ValueError(f"Unknown net model '{name}'. Supported models: {{{supported}}}") from exc


def compute_net_rate(
    name: str,
    *,
    deposition_rate: Any,
    params: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model_fn = resolve_net_model(name)
    return model_fn(deposition_rate=deposition_rate, params=params, **model_kwargs)


def compute_net_rate_from_model_config(
    model_config: Any,
    *,
    deposition_rate: Any,
    **model_kwargs: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not hasattr(model_config, "net_name"):
        raise ValueError("model_config must define 'net_name'")
    params = getattr(model_config, "net_params", {})
    if params is None:
        params_map: Mapping[str, Any] = {}
    elif isinstance(params, Mapping):
        params_map = params
    else:
        raise ValueError("model_config.net_params must be a mapping")
    return compute_net_rate(getattr(model_config, "net_name"), deposition_rate=deposition_rate, params=params_map, **model_kwargs)


register_net_model("deposition_only", deposition_only)
register_net_model("dep_etch_loss", dep_etch_loss)


__all__ = [
    "NetModel",
    "available_net_models",
    "register_net_model",
    "resolve_net_model",
    "compute_net_rate",
    "compute_net_rate_from_model_config",
    "deposition_only",
    "dep_etch_loss",
]
