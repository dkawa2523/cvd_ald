"""CVD steady forward simulator with transport-reaction diagnostics.

Sign convention:
- Deposition thickness is positive.
- Etch (if a negative net rate is used in future extensions) is negative.

Da proxy definition used here:
- ``Da_proxy = max_i( nu_i * R / (k_m,i * C_ref,i + eps) )`` over consumed species (``nu_i > 0``).
- This is dimensionless and interpretable as transport-capacity usage.
- Values near 0 are reaction-limited; values near 1 indicate transport-limited behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

from deposim_sim.domain import DomainGrid
from deposim_sim.models import mass_transfer, net_models, rate_laws
from deposim_sim.solvers import root_solve
from deposim_sim.state_closure import resolve_state_from_model_config

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-30
_FAILURE_MASK = (
    root_solve.STATUS_NON_MONOTONIC_FAILURE
    | root_solve.STATUS_BRACKET_NOT_FOUND
    | root_solve.STATUS_MAX_ITER_REACHED
)


@dataclass(frozen=True)
class FieldBundle:
    """Spatial input bundle for steady CVD.

    ``C_ref`` is reference-plane concentration (not surface concentration).
    ``U`` and scalar values are accepted for compatibility with future models.
    """

    C_ref: Mapping[str, Any]
    U: Any | None = None
    T: Any | None = None
    scalars: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVDSteadyResult:
    """Steady CVD forward result with standard diagnostics."""

    thickness: np.ndarray
    deposition_rate: np.ndarray
    R: np.ndarray
    Cs: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for deposim_sim.physics.cvd_steady. "
            "Install numpy to run the steady simulator."
        )


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


def _resolve_pattern_loading(
    *,
    shape: tuple[int, ...],
    kinetics_params: Mapping[str, Any],
) -> np.ndarray:
    raw = kinetics_params.get("pattern_loading", kinetics_params.get("S_xy", 1.0))
    return _grid_align(raw, shape, "pattern_loading", nonnegative=True)


def _mapping_from_object(value: Any, name: str, *, allow_empty: bool = True) -> dict[str, Any]:
    if value is None:
        if allow_empty:
            return {}
        raise ValueError(f"{name} must be provided")
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out = {str(key): item for key, item in value.items()}
    if not out and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return out


def _resolve_nu(
    *,
    species: tuple[str, ...],
    nu: Mapping[str, Any] | None,
    kinetics_params: Mapping[str, Any] | None,
) -> dict[str, float]:
    if nu is not None:
        nu_map = _mapping_from_object(nu, "nu", allow_empty=False)
    else:
        nu_from_params: Mapping[str, Any] | None = None
        if isinstance(kinetics_params, Mapping):
            if isinstance(kinetics_params.get("nu"), Mapping):
                nu_from_params = kinetics_params.get("nu")
            elif isinstance(kinetics_params.get("stoichiometry"), Mapping):
                nu_from_params = kinetics_params.get("stoichiometry")
        if nu_from_params is not None:
            nu_map = _mapping_from_object(nu_from_params, "kinetics_params.nu", allow_empty=False)
        elif len(species) == 1:
            nu_map = {species[0]: 1.0}
        else:
            raise ValueError(
                "nu must be provided for multi-species C_ref (or set kinetics_params.nu/stoichiometry)."
            )

    out: dict[str, float] = {}
    for name in species:
        if name not in nu_map:
            raise ValueError(f"nu is missing species '{name}'")
        out[name] = float(nu_map[name])
    return out


def _cs_over_cref(Cs: Mapping[str, np.ndarray], C_ref: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ratio: dict[str, np.ndarray] = {}
    for species, cs_arr in Cs.items():
        cref_arr = C_ref[species]
        ratio[species] = np.divide(
            cs_arr,
            cref_arr,
            out=np.zeros(cs_arr.shape, dtype=float),
            where=cref_arr > _EPS,
        )
    return ratio


def _da_proxy(
    *,
    R: np.ndarray,
    C_ref: Mapping[str, np.ndarray],
    km: Mapping[str, np.ndarray],
    nu: Mapping[str, float],
) -> np.ndarray:
    terms: list[np.ndarray] = []
    for species, nu_value in nu.items():
        if nu_value <= 0.0:
            continue
        cap = km[species] * C_ref[species]
        term = np.divide(
            nu_value * R,
            cap + _EPS,
            out=np.zeros(R.shape, dtype=float),
            where=cap > _EPS,
        )
        terms.append(term)
    if not terms:
        return np.zeros(R.shape, dtype=float)
    da = np.maximum.reduce(terms)
    return np.clip(da, 0.0, np.inf)


def run_cvd_steady(
    *,
    grid: DomainGrid,
    fields: FieldBundle,
    model_config: Any,
    process_time_s: float,
    solver_config: Any | None = None,
    nu: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> CVDSteadyResult:
    """Run a steady CVD forward solve and return thickness with diagnostics.

    ``thickness = deposition_rate * process_time_s`` with deposit-positive sign.
    ``R`` is the gross surface progress variable from the transport-reaction solve.
    ``deposition_rate`` is the net thickness-rate channel selected by ``model_config.net_name``.
    """

    _require_numpy()
    if process_time_s <= 0.0:
        raise ValueError(f"process_time_s must be > 0, got {process_time_s}")
    if not isinstance(fields.C_ref, Mapping) or not fields.C_ref:
        raise ValueError("fields.C_ref must be a non-empty species mapping")

    shape = grid.shape
    species = tuple(str(name) for name in fields.C_ref)
    c_ref = {name: _grid_align(fields.C_ref[name], shape, f"fields.C_ref[{name}]", nonnegative=True) for name in species}
    temperature = None if fields.T is None else _grid_align(fields.T, shape, "fields.T")

    kinetics_params = getattr(model_config, "kinetics_params", {})
    if kinetics_params is None:
        kinetics_params = {}
    if not isinstance(kinetics_params, Mapping):
        raise ValueError("model_config.kinetics_params must be a mapping")

    nu_map = _resolve_nu(species=species, nu=nu, kinetics_params=kinetics_params)
    pattern_loading = _resolve_pattern_loading(shape=shape, kinetics_params=kinetics_params)
    nu_effective = {name: float(coeff) * pattern_loading for name, coeff in nu_map.items()}
    resolved_state = resolve_state_from_model_config(
        model_config,
        Cs=c_ref,
        dt_s=float(process_time_s),
        initial_state=state,
    )
    if fields.scalars is None:
        scalar_inputs: Mapping[str, Any] = {}
    elif isinstance(fields.scalars, Mapping):
        scalar_inputs = fields.scalars
    else:
        raise ValueError("fields.scalars must be a mapping when provided")
    omega_raw = scalar_inputs.get("omega_rad_s", scalar_inputs.get("omega"))
    km_base = mass_transfer.compute_km_from_model_config(model_config, grid=grid, omega_rad_s=omega_raw)
    km_species = {name: _grid_align(km_base, shape, "k_m", nonnegative=True) for name in species}

    kinetics_name = getattr(model_config, "kinetics_name", None)
    if not kinetics_name:
        raise ValueError("model_config.kinetics_name must be non-empty")
    rate_fn = rate_laws.resolve_rate_law_model(kinetics_name)

    solve_kwargs: dict[str, Any] = {}
    if solver_config is not None:
        for key in ("max_iter", "rtol", "atol", "monotonicity_check"):
            value = getattr(solver_config, key, None)
            if value is not None:
                solve_kwargs[key] = value

    R, Cs, iteration_count, status_map = root_solve.solve_progress_R(
        c_ref=c_ref,
        k_m=km_species,
        nu=nu_effective,
        rate_fn=rate_fn,
        state=resolved_state,
        T=temperature,
        rate_params=kinetics_params,
        **solve_kwargs,
    )

    gross_rate = np.asarray(R, dtype=float)
    deposition_rate, net_components = net_models.compute_net_rate_from_model_config(
        model_config,
        deposition_rate=gross_rate,
        Cs=Cs,
        state=resolved_state,
        T=temperature,
    )
    thickness = deposition_rate * float(process_time_s)
    n_app = rate_laws.apparent_orders(
        kinetics_name,
        Cs=Cs,
        state=resolved_state,
        T=temperature,
        params=kinetics_params,
    )

    failure_mask = (status_map & _FAILURE_MASK) != 0
    diagnostics: dict[str, Any] = {
        "thickness": thickness,
        "Cs_over_Cref": _cs_over_cref(Cs, c_ref),
        "Da_proxy": _da_proxy(R=R, C_ref=c_ref, km=km_species, nu=nu_map),
        "n_app": n_app,
        "apparent_orders": n_app,
        "root_iteration_count": iteration_count,
        "root_iteration_counts": iteration_count,
        "root_status_map": status_map,
        "root_failure_mask": failure_mask,
        "root_failure_fraction": float(np.mean(failure_mask)),
        "root_status_flags": {
            "non_monotonic": int(root_solve.STATUS_NON_MONOTONIC),
            "fallback_interval_split": int(root_solve.STATUS_FALLBACK_INTERVAL_SPLIT),
            "non_monotonic_failure": int(root_solve.STATUS_NON_MONOTONIC_FAILURE),
            "bracket_not_found": int(root_solve.STATUS_BRACKET_NOT_FOUND),
            "max_iter_reached": int(root_solve.STATUS_MAX_ITER_REACHED),
            "cs_clipped": int(root_solve.STATUS_CS_CLIPPED),
        },
        "da_proxy_definition": (
            "max_i(nu_i*R/(k_m_i*C_ref_i+eps)) for consumed species (nu_i>0), dimensionless."
        ),
        "sign_convention": "deposit_positive_etch_negative",
        "gross_deposition_rate": gross_rate,
        "net_model_name": getattr(model_config, "net_name", "deposition_only"),
        "net_rate_components": net_components,
        "state_snapshot": resolved_state,
        "pattern_loading": pattern_loading,
        "pattern_loading_enabled": bool(np.any(np.abs(pattern_loading - 1.0) > 1.0e-12)),
    }
    return CVDSteadyResult(
        thickness=thickness,
        deposition_rate=deposition_rate,
        R=R,
        Cs=Cs,
        diagnostics=diagnostics,
    )


__all__ = ["FieldBundle", "CVDSteadyResult", "run_cvd_steady"]
