"""Rate-law models and registry utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


RateLawModel = Callable[..., Any]
ApparentOrderModel = Callable[..., dict[str, Any]]

_RATE_LAW_REGISTRY: dict[str, RateLawModel] = {}
_APPARENT_ORDER_REGISTRY: dict[str, ApparentOrderModel] = {}
_RATE_LAW_METADATA: dict[str, dict[str, Any]] = {}

_LOG_FLOOR = 1.0e-30
_EXP_MIN = -745.0
_EXP_MAX = 700.0
_R_GAS = 8.31446261815324


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for deposim_sim.models.rate_laws. Install numpy to evaluate rate laws."
        )


def _merge(params: Mapping[str, Any] | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(params) if params is not None else {}
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _shape(Cs: Mapping[str, Any], state: Mapping[str, Any] | None, T: Any) -> tuple[int, ...]:
    _require_numpy()
    if not isinstance(Cs, Mapping):
        raise ValueError("Cs must be a mapping")
    out: tuple[int, ...] = ()
    for value in Cs.values():
        out = np.broadcast_shapes(out, np.asarray(value, dtype=float).shape)
    if T is not None:
        out = np.broadcast_shapes(out, np.asarray(T, dtype=float).shape)
    if isinstance(state, Mapping):
        for value in state.values():
            try:
                out = np.broadcast_shapes(out, np.asarray(value, dtype=float).shape)
            except Exception:
                continue
    return out


def _align(
    value: Any,
    out_shape: tuple[int, ...],
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = np.full(out_shape, float(arr), dtype=float)
    else:
        try:
            out = np.broadcast_to(arr, out_shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to {out_shape}") from exc
    if positive and bool(np.any(out <= 0.0)):
        raise ValueError(f"{name} must be > 0 everywhere")
    if nonnegative and bool(np.any(out < 0.0)):
        raise ValueError(f"{name} must be >= 0 everywhere")
    return out


def _exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, _EXP_MIN, _EXP_MAX))


def _fmap(value: Any, name: str, *, allow_empty: bool = False) -> dict[str, float]:
    if value is None:
        if allow_empty:
            return {}
        raise ValueError(f"{name} is required")
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out = {str(k): float(v) for k, v in value.items()}
    if not out and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return out


def _orders(
    merged: Mapping[str, Any],
    species_set: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> dict[str, float]:
    if merged.get(key) is not None:
        return _fmap(merged[key], key, allow_empty=allow_empty)
    if allow_empty:
        return {}
    if "order" not in merged:
        raise ValueError(f"{key} is required")
    species = merged.get("species")
    if species is None:
        if len(species_set) != 1:
            raise ValueError(f"{key} is required when multiple species are provided")
        species = next(iter(species_set))
    return {str(species): float(merged["order"])}


def _log_floor(merged: Mapping[str, Any]) -> float:
    floor = float(merged.get("log_c_floor", merged.get("concentration_floor", _LOG_FLOOR)))
    if floor <= 0.0:
        raise ValueError("log concentration floor must be > 0")
    return floor


def _species_terms(
    Cs: Mapping[str, Any],
    out_shape: tuple[int, ...],
    floor: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for species, value in Cs.items():
        raw = _align(value, out_shape, f"Cs[{species}]", nonnegative=True)
        eff = np.maximum(raw, floor)
        terms[str(species)] = (np.log(eff), raw > floor)
    return terms


def _log_k(merged: Mapping[str, Any], T: Any, out_shape: tuple[int, ...]) -> np.ndarray:
    if merged.get("log_k") is not None:
        return _align(merged["log_k"], out_shape, "log_k")
    if merged.get("k") is not None:
        return np.log(_align(merged["k"], out_shape, "k", positive=True))
    k0 = _align(merged.get("k0", 1.0), out_shape, "k0", positive=True)
    ea = float(merged.get("ea_j_mol", merged.get("ea", 0.0)))
    if ea == 0.0:
        return np.log(k0)
    if T is None:
        raise ValueError("T is required when ea_j_mol is non-zero")
    gas_constant = float(merged.get("gas_constant_j_mol_k", _R_GAS))
    if gas_constant <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be > 0")
    return np.log(k0) - ea / (gas_constant * _align(T, out_shape, "T", positive=True))


def powerlaw(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> np.ndarray:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    terms = _species_terms(Cs, out_shape, _log_floor(merged))
    ord_map = _orders(merged, terms, "orders")
    ln_r = _log_k(merged, T, out_shape)
    for species, order in ord_map.items():
        if species not in terms:
            raise ValueError(f"orders references unknown species '{species}'")
        ln_r = ln_r + float(order) * terms[species][0]
    return _exp(ln_r)


def powerlaw_apparent_orders(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, np.ndarray]:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    terms = _species_terms(Cs, out_shape, _log_floor(merged))
    ord_map = _orders(merged, terms, "orders")
    return {s: np.where(terms[s][1], float(n), 0.0) for s, n in ord_map.items()}


def _sat_params(
    merged: Mapping[str, Any],
    terms: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float]:
    num = _orders(merged, terms, "numerator_orders", allow_empty=True)
    if not num and (merged.get("orders") is not None or "order" in merged):
        num = _orders(merged, terms, "orders")
    den_c = _fmap(merged.get("denominator_coeffs", merged.get("denom_coeffs")), "denominator_coeffs", allow_empty=True)
    den_n = _fmap(merged.get("denominator_orders", merged.get("denom_orders")), "denominator_orders", allow_empty=True)
    for species in den_c:
        den_n.setdefault(species, 1.0)
    den_p = float(merged.get("denominator_power", merged.get("denom_power", 1.0)))
    den_b = float(merged.get("denominator_base", 1.0))
    if den_p < 0.0:
        raise ValueError("denominator_power must be >= 0")
    if den_b <= 0.0:
        raise ValueError("denominator_base must be > 0")
    for species in set(num) | set(den_c):
        if species not in terms:
            raise ValueError(f"rate law references unknown species '{species}'")
    return num, den_c, den_n, den_p, den_b


def _sat_denom(
    terms: Mapping[str, tuple[np.ndarray, np.ndarray]],
    den_c: Mapping[str, float],
    den_n: Mapping[str, float],
    den_b: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    out_shape = next(iter(terms.values()))[0].shape
    den = np.full(out_shape, den_b, dtype=float)
    den_terms: dict[str, np.ndarray] = {}
    for species, coeff in den_c.items():
        term = float(coeff) * _exp(float(den_n[species]) * terms[species][0])
        den = den + term
        den_terms[species] = term
    if bool(np.any(den <= 0.0)):
        raise ValueError("saturation_inhibition denominator must stay > 0 everywhere")
    return den, den_terms


def saturation_inhibition(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> np.ndarray:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    terms = _species_terms(Cs, out_shape, _log_floor(merged))
    num, den_c, den_n, den_p, den_b = _sat_params(merged, terms)
    ln_r = _log_k(merged, T, out_shape)
    for species, order in num.items():
        ln_r = ln_r + float(order) * terms[species][0]
    den, _ = _sat_denom(terms, den_c, den_n, den_b)
    if den_p != 0.0:
        ln_r = ln_r - den_p * np.log(den)
    return _exp(ln_r)


def saturation_inhibition_apparent_orders(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, np.ndarray]:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    terms = _species_terms(Cs, out_shape, _log_floor(merged))
    num, den_c, den_n, den_p, den_b = _sat_params(merged, terms)
    den, den_terms = _sat_denom(terms, den_c, den_n, den_b)
    n_app: dict[str, np.ndarray] = {}
    for species in set(num) | set(den_c):
        app = np.full(out_shape, float(num.get(species, 0.0)), dtype=float)
        if species in den_c and den_p != 0.0:
            app = app - np.where(terms[species][1], den_p * float(den_n[species]) * den_terms[species] / den, 0.0)
        n_app[species] = app
    return n_app


def lhhw_competition(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> np.ndarray:
    """Competition/LHHW-like reduced model (wrapper of saturation-inhibition form)."""
    return saturation_inhibition(Cs=Cs, state=state, T=T, params=params, **overrides)


def lhhw_competition_apparent_orders(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, np.ndarray]:
    return saturation_inhibition_apparent_orders(Cs=Cs, state=state, T=T, params=params, **overrides)


def sticking_flux(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> np.ndarray:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    floor = _log_floor(merged)
    species = str(merged.get("species", next(iter(Cs)) if len(Cs) == 1 else ""))
    if not species or species not in Cs:
        raise ValueError("sticking_flux requires valid 'species' in params when Cs has multiple species")
    T_arr = _align(T if T is not None else float(merged.get("temperature_k", 700.0)), out_shape, "T", positive=True)
    cs_arr = _align(Cs[species], out_shape, f"Cs[{species}]", nonnegative=True)
    molar_mass = float(merged.get("molar_mass_kg_mol", 0.1))
    if molar_mass <= 0.0:
        raise ValueError("molar_mass_kg_mol must be > 0 for sticking_flux")
    alpha = float(merged.get("alpha_stick", merged.get("alpha", 1.0)))
    if alpha < 0.0:
        raise ValueError("alpha_stick must be >= 0")
    theta_pow = float(merged.get("theta_exponent", 1.0))
    theta = 0.0
    if isinstance(state, Mapping):
        theta = state.get("theta", 0.0)
    theta_arr = np.clip(_align(theta, out_shape, "state.theta"), 0.0, 1.0)
    stick = np.power(np.clip(1.0 - theta_arr, 0.0, 1.0), theta_pow)
    cs_eff = np.maximum(cs_arr, floor)
    pressure = cs_eff * _R_GAS * T_arr
    flux = alpha * pressure / np.sqrt(2.0 * np.pi * molar_mass * _R_GAS * T_arr)
    return np.clip(stick * flux, 0.0, np.inf)


def sticking_flux_apparent_orders(
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, np.ndarray]:
    merged = _merge(params, overrides)
    out_shape = _shape(Cs, state, T)
    species = str(merged.get("species", next(iter(Cs)) if len(Cs) == 1 else ""))
    if not species or species not in Cs:
        raise ValueError("sticking_flux requires valid 'species' in params when Cs has multiple species")
    cs_arr = _align(Cs[species], out_shape, f"Cs[{species}]", nonnegative=True)
    floor = _log_floor(merged)
    return {species: np.where(cs_arr > floor, 1.0, 0.0)}


def register_rate_law_model(
    name: str,
    model: RateLawModel,
    *,
    apparent_order_model: ApparentOrderModel | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("rate-law model name must be non-empty")
    if not overwrite and key in _RATE_LAW_REGISTRY:
        raise ValueError(f"rate-law model '{key}' is already registered")
    _RATE_LAW_REGISTRY[key] = model
    if apparent_order_model is not None:
        _APPARENT_ORDER_REGISTRY[key] = apparent_order_model
    _RATE_LAW_METADATA[key] = {
        "requires": list(metadata.get("requires", [])) if metadata is not None else [],
        "excludes": list(metadata.get("excludes", [])) if metadata is not None else [],
        "time_modes": list(metadata.get("time_modes", ["cvd_steady", "cvd_transient", "ald_cycle"]))
        if metadata is not None
        else ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": str(metadata.get("governing_class", "rate_law"))
        if metadata is not None
        else "rate_law",
    }


def available_rate_law_models() -> tuple[str, ...]:
    return tuple(sorted(_RATE_LAW_REGISTRY))


def get_rate_law_metadata() -> dict[str, dict[str, Any]]:
    """Return compatibility metadata keyed by registered rate-law model name."""
    return deepcopy(_RATE_LAW_METADATA)


def resolve_rate_law_model(name: str) -> RateLawModel:
    key = str(name).strip()
    try:
        return _RATE_LAW_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(available_rate_law_models())
        raise ValueError(f"Unknown rate-law model '{name}'. Supported models: {{{supported}}}") from exc


def compute_rate(
    name: str,
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> np.ndarray:
    return resolve_rate_law_model(name)(Cs=Cs, state=state, T=T, params=params, **model_kwargs)


def apparent_orders(
    name: str,
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    params: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> dict[str, np.ndarray]:
    key = str(name).strip()
    if key not in _APPARENT_ORDER_REGISTRY:
        supported = ", ".join(sorted(_APPARENT_ORDER_REGISTRY))
        raise ValueError(f"Unknown apparent-order model '{name}'. Supported models: {{{supported}}}")
    return _APPARENT_ORDER_REGISTRY[key](Cs=Cs, state=state, T=T, params=params, **model_kwargs)


def compute_rate_from_model_config(
    model_config: Any,
    *,
    Cs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    **model_kwargs: Any,
) -> np.ndarray:
    if not hasattr(model_config, "kinetics_name"):
        raise ValueError("model_config must define 'kinetics_name'")
    params = getattr(model_config, "kinetics_params", {})
    if params is None:
        params_map: Mapping[str, Any] = {}
    elif isinstance(params, Mapping):
        params_map = params
    else:
        raise ValueError("model_config.kinetics_params must be a mapping")
    return compute_rate(getattr(model_config, "kinetics_name"), Cs=Cs, state=state, T=T, params=params_map, **model_kwargs)


register_rate_law_model(
    "powerlaw",
    powerlaw,
    apparent_order_model=powerlaw_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)
register_rate_law_model(
    "power_law",
    powerlaw,
    apparent_order_model=powerlaw_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)
register_rate_law_model(
    "saturation_inhibition",
    saturation_inhibition,
    apparent_order_model=saturation_inhibition_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)
register_rate_law_model(
    "lhhw_competition",
    lhhw_competition,
    apparent_order_model=lhhw_competition_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)
register_rate_law_model(
    "competition_lhhw",
    lhhw_competition,
    apparent_order_model=lhhw_competition_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)
register_rate_law_model(
    "sticking_flux",
    sticking_flux,
    apparent_order_model=sticking_flux_apparent_orders,
    metadata={
        "requires": [],
        "excludes": [],
        "time_modes": ["cvd_steady", "cvd_transient", "ald_cycle"],
        "governing_class": "rate_law",
    },
)


__all__ = [
    "RateLawModel",
    "ApparentOrderModel",
    "available_rate_law_models",
    "get_rate_law_metadata",
    "register_rate_law_model",
    "resolve_rate_law_model",
    "compute_rate",
    "apparent_orders",
    "compute_rate_from_model_config",
    "powerlaw",
    "saturation_inhibition",
    "lhhw_competition",
    "sticking_flux",
]
