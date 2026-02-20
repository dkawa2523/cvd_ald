"""Mass-transfer models and registry utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from deposim_sim.domain import DomainGrid

try:  # pragma: no cover - dependency guard for minimal environments
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore[assignment]


MassTransferModel = Callable[..., Any]
_MASS_TRANSFER_REGISTRY: dict[str, MassTransferModel] = {}
_MASS_TRANSFER_METADATA: dict[str, dict[str, Any]] = {}


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for deposim_sim.models.mass_transfer. "
            "Install numpy to evaluate mass-transfer models."
        )


def _merge_params(params: Mapping[str, Any] | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(params) if params is not None else {}
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _as_grid_array(
    value: Any,
    grid_shape: tuple[int, ...],
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    _require_numpy()
    if value is None:
        raise ValueError(f"{name} is required")

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        aligned = np.full(grid_shape, float(array), dtype=float)
    else:
        try:
            aligned = np.broadcast_to(array, grid_shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(
                f"{name} with shape {array.shape} cannot be broadcast to grid shape {grid_shape}"
            ) from exc

    if positive and bool(np.any(aligned <= 0.0)):
        raise ValueError(f"{name} must be > 0 everywhere")
    if nonnegative and bool(np.any(aligned < 0.0)):
        raise ValueError(f"{name} must be >= 0 everywhere")
    return aligned


def _normalize_omega_zero_guard(policy: str | None) -> str:
    if policy is None:
        return "error"

    lowered = str(policy).strip().lower()
    if lowered == "error":
        return "error"
    if lowered in {"fallback", "fallback_stagnant_film", "stagnant_film"}:
        return "fallback_stagnant_film"

    raise ValueError(
        "omega_zero_guard must be 'error' or 'fallback_stagnant_film' "
        f"(got {policy!r})"
    )


def _resolve_diffusivity_field(
    *,
    grid: DomainGrid,
    merged: Mapping[str, Any],
    diffusivity_m2_s: Any,
) -> np.ndarray:
    if diffusivity_m2_s is not None:
        return _as_grid_array(diffusivity_m2_s, grid.shape, "diffusivity_m2_s", positive=True)

    model = str(merged.get("diffusivity_model", "direct")).strip().lower()
    if model in {"direct", "constant"}:
        return _as_grid_array(merged.get("diffusivity_m2_s"), grid.shape, "diffusivity_m2_s", positive=True)
    if model == "bosanquet":
        d_m = merged.get("d_m_m2_s", merged.get("diffusivity_molecular_m2_s"))
        d_k = merged.get("d_k_m2_s", merged.get("diffusivity_knudsen_m2_s"))
        d_m_arr = _as_grid_array(d_m, grid.shape, "d_m_m2_s", positive=True)
        d_k_arr = _as_grid_array(d_k, grid.shape, "d_k_m2_s", positive=True)
        return 1.0 / ((1.0 / d_m_arr) + (1.0 / d_k_arr))
    raise ValueError(
        "diffusivity_model must be one of {'direct','constant','bosanquet'} "
        f"(got {merged.get('diffusivity_model')!r})"
    )


def stagnant_film(
    *,
    grid: DomainGrid,
    params: Mapping[str, Any] | None = None,
    diffusivity_m2_s: Any = None,
    delta_eff_m: Any = None,
    **overrides: Any,
) -> np.ndarray:
    """Compute km using stagnant-film transport: km = D / delta_eff.

    Legacy compatibility:
    - If only `k_m_m_s` is provided, it is treated as a direct km field.
    """
    merged = _merge_params(params, overrides)
    if diffusivity_m2_s is None:
        diffusivity_m2_s = merged.get("diffusivity_m2_s")
    if delta_eff_m is None:
        delta_eff_m = merged.get("delta_eff_m")

    legacy_km = merged.get("k_m_m_s")
    if legacy_km is not None and (diffusivity_m2_s is None or delta_eff_m is None):
        return _as_grid_array(legacy_km, grid.shape, "k_m_m_s", positive=True)

    diffusivity = _resolve_diffusivity_field(
        grid=grid,
        merged=merged,
        diffusivity_m2_s=diffusivity_m2_s,
    )
    delta_eff = _as_grid_array(delta_eff_m, grid.shape, "delta_eff_m", positive=True)
    return diffusivity / delta_eff


def rotating_disk(
    *,
    grid: DomainGrid,
    params: Mapping[str, Any] | None = None,
    diffusivity_m2_s: Any = None,
    omega_rad_s: Any = None,
    nu_m2_s: Any = None,
    ck: Any = None,
    **overrides: Any,
) -> np.ndarray:
    """Compute km using Levich-like rotating-disk scaling.

    km = Ck * D^(2/3) * omega^(1/2) * nu^(-1/6)

    omega=0 guard behavior is configured by `omega_zero_guard`:
    - "error": raise an exception when any omega is zero.
    - "fallback_stagnant_film": use stagnant-film km where omega==0.
    """
    merged = _merge_params(params, overrides)

    if diffusivity_m2_s is None:
        diffusivity_m2_s = merged.get("diffusivity_m2_s")
    if omega_rad_s is None:
        omega_rad_s = merged.get("omega_rad_s")
    if nu_m2_s is None:
        nu_m2_s = merged.get("nu_m2_s", merged.get("kinematic_viscosity_m2_s"))
    if ck is None:
        ck = merged.get("ck", merged.get("sh_prefactor", 0.62))

    ck_value = float(ck)
    if ck_value <= 0.0:
        raise ValueError(f"ck must be > 0, got {ck_value}")

    diffusivity = _resolve_diffusivity_field(
        grid=grid,
        merged=merged,
        diffusivity_m2_s=diffusivity_m2_s,
    )
    omega = _as_grid_array(omega_rad_s, grid.shape, "omega_rad_s", nonnegative=True)
    nu = _as_grid_array(nu_m2_s, grid.shape, "nu_m2_s", positive=True)

    km = ck_value * (diffusivity ** (2.0 / 3.0)) * np.sqrt(omega) * (nu ** (-1.0 / 6.0))

    zero_mask = omega <= 0.0
    if not bool(np.any(zero_mask)):
        return km

    guard_policy = _normalize_omega_zero_guard(merged.get("omega_zero_guard", "error"))
    if guard_policy == "error":
        raise ValueError(
            "rotating_disk received omega_rad_s=0 while omega_zero_guard='error'. "
            "Set omega_zero_guard='fallback_stagnant_film' to enable fallback."
        )

    fallback_params: dict[str, Any] = {}
    raw_fallback_params = merged.get("omega_zero_fallback_params")
    if isinstance(raw_fallback_params, Mapping):
        fallback_params.update(dict(raw_fallback_params))

    fallback_params.setdefault("diffusivity_m2_s", diffusivity)
    if "delta_eff_m" in merged and "delta_eff_m" not in fallback_params:
        fallback_params["delta_eff_m"] = merged["delta_eff_m"]
    if "k_m_m_s" in merged and "k_m_m_s" not in fallback_params:
        fallback_params["k_m_m_s"] = merged["k_m_m_s"]

    fallback_km = stagnant_film(grid=grid, params=fallback_params)
    km = km.copy()
    km[zero_mask] = fallback_km[zero_mask]
    return km


def register_mass_transfer_model(
    name: str,
    model: MassTransferModel,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a mass-transfer model by name."""
    key = str(name).strip()
    if not key:
        raise ValueError("mass-transfer model name must be non-empty")
    if not overwrite and key in _MASS_TRANSFER_REGISTRY:
        raise ValueError(f"mass-transfer model '{key}' is already registered")
    _MASS_TRANSFER_REGISTRY[key] = model
    _MASS_TRANSFER_METADATA[key] = {
        "requires": list(metadata.get("requires", [])) if metadata is not None else [],
        "excludes": list(metadata.get("excludes", [])) if metadata is not None else [],
        "time_modes": list(metadata.get("time_modes", ["cvd_steady", "cvd_transient", "ald_cycle"]))
        if metadata is not None
        else ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": str(metadata.get("governing_class", "mass_transfer"))
        if metadata is not None
        else "mass_transfer",
    }


def available_mass_transfer_models() -> tuple[str, ...]:
    """Return sorted registered model names."""
    return tuple(sorted(_MASS_TRANSFER_REGISTRY))


def get_mass_transfer_metadata() -> dict[str, dict[str, Any]]:
    """Return compatibility metadata keyed by registered model name."""
    return deepcopy(_MASS_TRANSFER_METADATA)


def resolve_mass_transfer_model(name: str) -> MassTransferModel:
    """Resolve a mass-transfer model by name."""
    key = str(name).strip()
    try:
        return _MASS_TRANSFER_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(available_mass_transfer_models())
        raise ValueError(
            f"Unknown mass-transfer model '{name}'. Supported models: {{{supported}}}"
        ) from exc


def compute_km(
    name: str,
    *,
    grid: DomainGrid,
    params: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> np.ndarray:
    """Resolve and evaluate km for a named mass-transfer model."""
    model_fn = resolve_mass_transfer_model(name)
    return model_fn(grid=grid, params=params, **model_kwargs)


def compute_km_from_model_config(
    model_config: Any,
    *,
    grid: DomainGrid,
    **model_kwargs: Any,
) -> np.ndarray:
    """Resolve and evaluate km from an object with model config attributes."""
    if not hasattr(model_config, "mass_transfer_name"):
        raise ValueError("model_config must define 'mass_transfer_name'")

    model_name = getattr(model_config, "mass_transfer_name")
    params = getattr(model_config, "mass_transfer_params", {})
    if params is None:
        params_map: Mapping[str, Any] = {}
    elif isinstance(params, Mapping):
        params_map = params
    else:
        raise ValueError("model_config.mass_transfer_params must be a mapping")

    return compute_km(model_name, grid=grid, params=params_map, **model_kwargs)


def compute_km_from_run_config(
    run_config: Any,
    *,
    grid: DomainGrid,
    **model_kwargs: Any,
) -> np.ndarray:
    """Resolve and evaluate km from a run config object.

    Expects `run_config.model.mass_transfer_*` fields.
    Automatically forwards `run_config.inputs.omega_rad_s` unless explicitly overridden.
    """
    if not hasattr(run_config, "model"):
        raise ValueError("run_config must define a 'model' field")

    kwargs = dict(model_kwargs)
    if "omega_rad_s" not in kwargs and hasattr(run_config, "inputs"):
        omega_value = getattr(run_config.inputs, "omega_rad_s", None)
        if omega_value is not None:
            kwargs["omega_rad_s"] = omega_value

    return compute_km_from_model_config(run_config.model, grid=grid, **kwargs)


register_mass_transfer_model(
    "stagnant_film",
    stagnant_film,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "mass_transfer",
    },
)
register_mass_transfer_model(
    "rotating_disk",
    rotating_disk,
    metadata={
        "requires": ["inputs.omega_rad_s"],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "mass_transfer",
    },
)


__all__ = [
    "MassTransferModel",
    "available_mass_transfer_models",
    "get_mass_transfer_metadata",
    "register_mass_transfer_model",
    "resolve_mass_transfer_model",
    "compute_km",
    "compute_km_from_model_config",
    "compute_km_from_run_config",
    "stagnant_film",
    "rotating_disk",
]
