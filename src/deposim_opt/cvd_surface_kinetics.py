"""Reduced quasi-steady surface-kinetic response models for CVD maps.

The raw Fluent species remain anonymous inputs.  Candidates assign them to an
adsorbing supply (A), a co-reactant pair (AB), and/or a site blocker (I).  The
fitted parameters are the observable dimensionless groups of a site-balance
model, not elementary rate constants::

    theta_* + theta_A + theta_I = 1
    0 = a theta_* - (d + q) theta_A
    theta_I = i theta_*
    v = H q theta_A

This gives ``theta_A = a / (a + (d + q) (1 + i))``.  Concentration references
are calculated from identification data and stored in the fit, so prediction
never re-centres on a validation or test batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Any, Iterable

import numpy as np


_TINY = np.finfo(float).tiny
_LOG_PARAMETER_MIN = -10.0
_LOG_PARAMETER_MAX = 10.0
_INITIAL_LOG_GRID = (-8.0, -4.0, 0.0, 4.0, 8.0)
_REFINEMENT_STEPS = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class SurfaceKineticCandidate:
    """One role assignment and one exact kinetic boundary."""

    class_id: str
    A: str | None = None
    I: str | None = None
    B: str | None = None
    kinetic_limit: str = "general"

    @property
    def model_id(self) -> str:
        if self.class_id == "baseline":
            return "surface_baseline"
        roles = "|".join(str(value) for value in (self.A, self.I, self.B) if value)
        suffix = ":no_desorption" if self.kinetic_limit == "no_desorption" else ""
        return f"surface_{self.class_id}:{roles}{suffix}"

    @property
    def effect_groups(self) -> dict[str, list[str]]:
        if self.class_id == "baseline":
            return {}
        if self.class_id == "A":
            return {"A": [str(self.A)]}
        if self.class_id == "AI":
            return {"A": [str(self.A)], "I": [str(self.I)]}
        groups = {"AB": sorted([str(self.A), str(self.B)])}
        if self.class_id == "AIB":
            groups["I"] = [str(self.I)]
        return groups

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self.class_id == "baseline":
            return ()
        if self.class_id == "A":
            return ("half_saturation_ratio",)
        if self.class_id == "AI":
            return ("half_saturation_ratio", "inhibition_ratio")
        names: list[str] = []
        if self.kinetic_limit != "no_desorption":
            names.append("desorption_ratio")
        names.append("conversion_ratio")
        if self.class_id == "AIB":
            names.append("inhibition_ratio")
        return tuple(names)

    @property
    def role_symmetry(self) -> str:
        return "A/B exchange in the no-inhibitor steady response" if self.class_id in {"AB", "AIB"} else ""

    def reductions(self) -> tuple["SurfaceKineticCandidate", ...]:
        baseline = SurfaceKineticCandidate("baseline")
        if self.class_id == "baseline":
            return ()
        if self.class_id == "A":
            return (baseline,)
        if self.class_id == "AI":
            return (SurfaceKineticCandidate("A", A=self.A),)
        pair = tuple(sorted((str(self.A), str(self.B))))
        ab = SurfaceKineticCandidate("AB", A=pair[0], B=pair[1])
        ab_zero = SurfaceKineticCandidate(
            "AB", A=pair[0], B=pair[1], kinetic_limit="no_desorption"
        )
        if self.class_id == "AB":
            if self.kinetic_limit == "general":
                return (ab_zero, baseline)
            return (baseline,)
        if self.kinetic_limit == "general":
            zero = SurfaceKineticCandidate(
                "AIB", A=self.A, I=self.I, B=self.B,
                kinetic_limit="no_desorption",
            )
            return (zero, ab)
        return (ab_zero,)


@dataclass(frozen=True)
class SurfaceKineticFit:
    """A fitted observable reduction with identification-data references."""

    candidate: SurfaceKineticCandidate
    rate_scale_nm_s: float
    shape_parameters: dict[str, float]
    reference_concentrations: dict[str, float]
    prediction: np.ndarray
    design: np.ndarray
    objective_mse: float
    boundary_parameters: tuple[str, ...]

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(
            [self.rate_scale_nm_s]
            + [self.shape_parameters[name] for name in self.candidate.parameter_names],
            dtype=float,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("rate_scale_nm_s", *self.candidate.parameter_names)

    @property
    def reference_rate_nm_s(self) -> float:
        return self.rate_scale_nm_s

    @property
    def regularization(self) -> float:
        return 0.0

    @property
    def response_structure(self) -> str:
        return "surface_qss"

    @property
    def effect_names(self) -> tuple[str, ...]:
        return self.candidate.parameter_names

    @property
    def active_effects(self) -> tuple[bool, ...]:
        return tuple(True for _ in self.candidate.effect_groups)

    @property
    def effect_groups(self) -> dict[str, list[str]]:
        return self.candidate.effect_groups

    @property
    def effect_scopes(self) -> dict[str, list[str]]:
        return {name: ["quasi_steady_surface"] for name in self.effect_groups}

    @property
    def effective_roles(self) -> dict[str, str | None]:
        roles = {"A": None, "I": None, "B": None}
        for group in self.effect_groups:
            for slot in (("A", "B") if group == "AB" else (group,)):
                roles[slot] = getattr(self.candidate, slot)
        return roles

    @property
    def reference_total_concentration(self) -> float:
        return float(sum(self.reference_concentrations.values()))

    @property
    def reference_species_fractions(self) -> dict[str, float]:
        total = max(self.reference_total_concentration, _TINY)
        return {name: value / total for name, value in self.reference_concentrations.items()}

    @property
    def common_order(self) -> float:
        return float("nan")

    @property
    def within_order(self) -> float:
        return float("nan")

    @property
    def coefficient_terms(self) -> list[tuple[str, str]]:
        return [("observable_reduction", name) for name in self.candidate.parameter_names]


def enumerate_surface_kinetic_candidates(
    species: Iterable[str], *, include_boundaries: bool = True
) -> list[SurfaceKineticCandidate]:
    """Enumerate chemistry-agnostic role assignments without duplicate AB swaps."""
    names = tuple(sorted(str(name) for name in species))
    candidates: list[SurfaceKineticCandidate] = [SurfaceKineticCandidate("baseline")]
    candidates.extend(SurfaceKineticCandidate("A", A=name) for name in names)
    candidates.extend(
        SurfaceKineticCandidate("AI", A=a, I=i)
        for a, i in permutations(names, 2)
    )
    for a, b in combinations(names, 2):
        candidates.append(SurfaceKineticCandidate("AB", A=a, B=b))
        if include_boundaries:
            candidates.append(
                SurfaceKineticCandidate("AB", A=a, B=b, kinetic_limit="no_desorption")
            )
    # Inhibitor occupation breaks the numerical A/B swap in the full
    # expression, so retain both orientations.  Evidence is still reported
    # conservatively as an AB pair plus I.
    for a, inhibitor, b in permutations(names, 3):
        candidates.append(SurfaceKineticCandidate("AIB", A=a, I=inhibitor, B=b))
        if include_boundaries:
            candidates.append(
                SurfaceKineticCandidate(
                    "AIB", A=a, I=inhibitor, B=b,
                    kinetic_limit="no_desorption",
                )
            )
    return candidates


def _reference_concentrations(data: Any, train_indices: np.ndarray) -> dict[str, float]:
    refs: dict[str, float] = {}
    for name in data.species:
        value = float(np.median(np.asarray(data.concentrations[name])[train_indices]))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Reference concentration for {name} must be positive")
        refs[name] = value
    return refs


def _parameter_dict(candidate: SurfaceKineticCandidate, log_values: np.ndarray) -> dict[str, float]:
    return {
        name: float(10.0 ** value)
        for name, value in zip(candidate.parameter_names, np.asarray(log_values, dtype=float))
    }


def _normalized(data: Any, refs: dict[str, float], role: str | None) -> np.ndarray:
    if role is None:
        return np.ones(np.asarray(data.rate).shape, dtype=float)
    return np.asarray(data.concentrations[role], dtype=float) / refs[role]


def response_shape(
    candidate: SurfaceKineticCandidate,
    data: Any,
    refs: dict[str, float],
    parameters: dict[str, float],
) -> np.ndarray:
    """Return the dimensionless response multiplied by the fitted rate scale."""
    if candidate.class_id == "baseline":
        return np.ones(np.asarray(data.rate).shape, dtype=float)
    ui = _normalized(data, refs, candidate.I)
    kappa = parameters.get("inhibition_ratio", 0.0)
    inhibition = 1.0 + kappa * ui
    ua = _normalized(data, refs, candidate.A)
    if candidate.class_id in {"A", "AI"}:
        half = parameters["half_saturation_ratio"]
        return ua / (ua + half * inhibition)
    ub = _normalized(data, refs, candidate.B)
    delta = parameters.get("desorption_ratio", 0.0)
    conversion = parameters["conversion_ratio"]
    denominator = ua + (delta + conversion * ub) * inhibition
    return ua * conversion * ub / denominator


def surface_state(
    candidate: SurfaceKineticCandidate,
    data: Any,
    refs: dict[str, float],
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Reconstruct QSS coverage diagnostics where the reduction identifies them."""
    count = np.asarray(data.rate).size
    nan = np.full(count, np.nan, dtype=float)
    if candidate.class_id == "baseline":
        return {"theta_free": nan, "theta_A": nan, "theta_I": nan}
    ua = _normalized(data, refs, candidate.A)
    ui = _normalized(data, refs, candidate.I)
    kappa = parameters.get("inhibition_ratio", 0.0)
    inhibition = 1.0 + kappa * ui
    if candidate.class_id in {"A", "AI"}:
        loss = np.full(count, parameters["half_saturation_ratio"], dtype=float)
    else:
        ub = _normalized(data, refs, candidate.B)
        loss = parameters.get("desorption_ratio", 0.0) + parameters["conversion_ratio"] * ub
    denominator = ua + loss * inhibition
    theta_a = ua / denominator
    theta_free = loss / denominator
    theta_i = (kappa * ui) * theta_free
    return {"theta_free": theta_free, "theta_A": theta_a, "theta_I": theta_i}


def condition_balanced_weights(condition_id: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Give every identification condition equal total objective weight."""
    labels = np.asarray(condition_id)[indices]
    unique = np.unique(labels)
    weights = np.zeros(indices.size, dtype=float)
    for label in unique:
        mask = labels == label
        weights[mask] = 1.0 / (unique.size * np.count_nonzero(mask))
    return weights


def _profile_rate_scale(shape: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    denominator = float(np.sum(weights * shape * shape))
    if denominator <= _TINY:
        return 0.0, float("inf")
    scale = max(0.0, float(np.sum(weights * shape * target) / denominator))
    residual = scale * shape - target
    return scale, float(np.sum(weights * residual * residual))


def _search_shape_parameters(
    candidate: SurfaceKineticCandidate,
    data: Any,
    train_indices: np.ndarray,
    refs: dict[str, float],
    weights: np.ndarray,
    *,
    initial_log_parameters: np.ndarray | None = None,
    local_only: bool = False,
) -> tuple[np.ndarray, float, float]:
    dimension = len(candidate.parameter_names)
    target = np.asarray(data.rate, dtype=float)[train_indices]
    if dimension == 0:
        scale, loss = _profile_rate_scale(np.ones(target.shape), target, weights)
        return np.empty(0, dtype=float), scale, loss

    def evaluate(values: tuple[float, ...] | np.ndarray) -> tuple[float, float]:
        params = _parameter_dict(candidate, np.asarray(values, dtype=float))
        shape = response_shape(candidate, data, refs, params)[train_indices]
        return _profile_rate_scale(shape, target, weights)

    best_values: np.ndarray | None = None
    best_scale = 0.0
    best_loss = float("inf")

    seeds: list[tuple[float, ...]] = []
    if initial_log_parameters is not None:
        seeds.append(tuple(np.asarray(initial_log_parameters, dtype=float)))
    if not local_only:
        seeds.extend(product(_INITIAL_LOG_GRID, repeat=dimension))
    if not seeds:
        seeds.append(tuple(np.zeros(dimension)))
    for values in seeds:
        scale, loss = evaluate(values)
        key = (loss, tuple(values))
        incumbent = (best_loss, tuple(best_values) if best_values is not None else tuple())
        if key < incumbent:
            best_values = np.asarray(values, dtype=float)
            best_scale = scale
            best_loss = loss

    assert best_values is not None
    for step in _REFINEMENT_STEPS:
        choices = [
            tuple(np.clip((value - step, value, value + step), _LOG_PARAMETER_MIN, _LOG_PARAMETER_MAX))
            for value in best_values
        ]
        round_best = best_values
        round_scale = best_scale
        round_loss = best_loss
        for values in product(*choices):
            scale, loss = evaluate(values)
            if (loss, tuple(values)) < (round_loss, tuple(round_best)):
                round_best = np.asarray(values, dtype=float)
                round_scale = scale
                round_loss = loss
        best_values, best_scale, best_loss = round_best, round_scale, round_loss
    return best_values, best_scale, best_loss


def _sensitivity_design(
    candidate: SurfaceKineticCandidate,
    data: Any,
    refs: dict[str, float],
    parameters: dict[str, float],
    indices: np.ndarray,
) -> np.ndarray:
    """Local log-response Jacobian used only for identifiability diagnostics."""
    base = np.maximum(response_shape(candidate, data, refs, parameters)[indices], _TINY)
    columns = [np.ones(indices.size, dtype=float)]
    step = 1.0e-5
    for name in candidate.parameter_names:
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] *= np.exp(step)
        minus[name] *= np.exp(-step)
        upper = np.maximum(response_shape(candidate, data, refs, plus)[indices], _TINY)
        lower = np.maximum(response_shape(candidate, data, refs, minus)[indices], _TINY)
        columns.append((np.log(upper) - np.log(lower)) / (2.0 * step))
    return np.column_stack(columns)


def fit_surface_kinetic(
    candidate: SurfaceKineticCandidate,
    data: Any,
    train_indices: np.ndarray,
    *,
    reference_concentrations: dict[str, float] | None = None,
    initial_fit: SurfaceKineticFit | None = None,
    local_only: bool = False,
) -> SurfaceKineticFit:
    """Fit a candidate by profiling its linear rate scale and searching shape groups."""
    train_indices = np.asarray(train_indices, dtype=int)
    if train_indices.size == 0:
        raise ValueError("At least one identification row is required")
    refs = (
        dict(reference_concentrations)
        if reference_concentrations is not None
        else _reference_concentrations(data, train_indices)
    )
    weights = condition_balanced_weights(np.asarray(data.condition_id), train_indices)
    initial = None
    if initial_fit is not None:
        initial = np.log10(
            [initial_fit.shape_parameters[name] for name in candidate.parameter_names]
        )
    log_values, rate_scale, objective = _search_shape_parameters(
        candidate,
        data,
        train_indices,
        refs,
        weights,
        initial_log_parameters=initial,
        local_only=local_only,
    )
    parameters = _parameter_dict(candidate, log_values)
    prediction = rate_scale * response_shape(candidate, data, refs, parameters)
    design = _sensitivity_design(candidate, data, refs, parameters, train_indices)
    boundary = tuple(
        name
        for name, value in zip(candidate.parameter_names, log_values)
        if abs(value - _LOG_PARAMETER_MIN) < 1.0e-10
        or abs(value - _LOG_PARAMETER_MAX) < 1.0e-10
    )
    return SurfaceKineticFit(
        candidate=candidate,
        rate_scale_nm_s=rate_scale,
        shape_parameters=parameters,
        reference_concentrations=refs,
        prediction=np.asarray(prediction, dtype=float),
        design=design,
        objective_mse=objective,
        boundary_parameters=boundary,
    )


def predict_surface_kinetic(fit: SurfaceKineticFit, data: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict with locked references and return coverage/sensitivity diagnostics."""
    missing = sorted(set(fit.reference_concentrations) - set(data.species))
    if missing:
        raise ValueError(f"Prediction data are missing species: {missing}")
    shape = response_shape(
        fit.candidate, data, fit.reference_concentrations, fit.shape_parameters
    )
    prediction = fit.rate_scale_nm_s * shape
    state = surface_state(
        fit.candidate, data, fit.reference_concentrations, fit.shape_parameters
    )
    state["dimensionless_response"] = shape
    return np.asarray(prediction, dtype=float), state


def surface_formula(candidate: SurfaceKineticCandidate) -> str:
    """Human-readable formula in observable dimensionless groups."""
    if candidate.class_id == "baseline":
        return "v = R"
    if candidate.class_id == "A":
        return "v = Vmax*uA / (uA + h)"
    if candidate.class_id == "AI":
        return "v = Vmax*uA / (uA + h*(1 + kappa*uI))"
    delta = "0" if candidate.kinetic_limit == "no_desorption" else "delta"
    inhibitor = "*(1 + kappa*uI)" if candidate.class_id == "AIB" else ""
    return f"v = R*uA*b*uB / (uA + ({delta} + b*uB){inhibitor})"


__all__ = [
    "SurfaceKineticCandidate",
    "SurfaceKineticFit",
    "condition_balanced_weights",
    "enumerate_surface_kinetic_candidates",
    "fit_surface_kinetic",
    "predict_surface_kinetic",
    "response_shape",
    "surface_formula",
    "surface_state",
]
