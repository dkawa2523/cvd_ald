"""Parameter estimation for observable steady surface reductions."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping
import warnings
import numpy as np

from deposim_sim.models.aib_reductions import (
    DIRECT_FLUX,
    DIRECT_SURFACE,
    SurfaceKineticCandidate,
    response_shape,
    surface_state,
)
from .role_fields import RoleFieldSet
from .losses import validate_wafer_loss_name, wafer_loss
from .samplers import SearchSettings, run_search


_TINY = np.finfo(float).tiny
_LOG_PARAMETER_MIN = -10.0
_LOG_PARAMETER_MAX = 10.0
_INITIAL_LOG_GRID = (-8.0, -4.0, 0.0, 4.0, 8.0)
_REFINEMENT_STEPS = (
    2.0,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
)
_MULTISTART_COUNT = 5
_PRACTICAL_CONDITION_LIMIT = 1.0e4
_PRACTICAL_CORRELATION_LIMIT = 0.995
_SURFACE_SAMPLERS = {
    "pattern",
    "random",
    "tpe",
    "cmaes",
    "de",
    "pso",
    "levy",
    "cma_mae",
}


@dataclass(frozen=True)
class SurfaceOptimizationSettings:
    """Independent choices for map loss, shape sampler, and point uncertainty."""

    loss_name: str = "mse"
    sampler: str = "pattern"
    trials: int = 256
    seed: int = 123
    sampler_options: Mapping[str, Any] = field(default_factory=dict)
    edge_uncertainty_ratio: float = 1.0
    radial_power: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss_name", validate_wafer_loss_name(self.loss_name))
        method = str(self.sampler).strip().lower()
        if method not in _SURFACE_SAMPLERS:
            raise ValueError(
                f"surface sampler must be one of {sorted(_SURFACE_SAMPLERS)}, got {self.sampler!r}"
            )
        object.__setattr__(self, "sampler", method)
        if int(self.trials) < 1:
            raise ValueError("surface sampler trials must be >= 1")
        if not np.isfinite(self.edge_uncertainty_ratio) or self.edge_uncertainty_ratio <= 0.0:
            raise ValueError("edge uncertainty ratio must be finite and positive")
        if not np.isfinite(self.radial_power) or self.radial_power <= 0.0:
            raise ValueError("radial uncertainty power must be finite and positive")


@dataclass(frozen=True)
class SurfaceKineticFit:
    candidate: SurfaceKineticCandidate
    rate_scale_nm_s: float
    shape_parameters: dict[str, float]
    reference_concentrations: dict[str, float]
    prediction: np.ndarray
    design: np.ndarray
    objective_value: float
    loss_name: str
    boundary_parameters: tuple[str, ...]
    optimizer_method: str
    optimizer_trial_count: int
    optimization_history: tuple[dict[str, float | int | str], ...] = ()

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
        return self.candidate.model_family

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
        return {name: [self.candidate.family] for name in self.effect_groups}

    @property
    def effective_roles(self) -> dict[str, str | None]:
        roles = {"A": None, "I": None, "B": None}
        for group in self.effect_groups:
            for slot in (("A", "B") if group == "AB" else (group,)):
                roles[slot] = getattr(self.candidate, slot)
        return roles

    @property
    def reference_input_total(self) -> float:
        """Sum of the selected reaction-driver references in their native unit."""

        return float(sum(self.reference_concentrations.values()))

    @property
    def reference_input_shares(self) -> dict[str, float]:
        """Dimensionless shares of the selected reaction driver at its reference."""

        total = max(self.reference_input_total, _TINY)
        return {name: value / total for name, value in self.reference_concentrations.items()}

    @property
    def reference_total_concentration(self) -> float:
        """Compatibility alias; use ``reference_input_total`` in new outputs."""

        return self.reference_input_total

    @property
    def reference_species_fractions(self) -> dict[str, float]:
        """Compatibility alias; use ``reference_input_shares`` in new outputs."""

        return self.reference_input_shares

    @property
    def common_order(self) -> float:
        return float("nan")

    @property
    def within_order(self) -> float:
        return float("nan")

    @property
    def coefficient_terms(self) -> list[tuple[str, str]]:
        return [("observable_reduction", name) for name in self.candidate.parameter_names]


def condition_balanced_weights(
    condition_id: np.ndarray,
    indices: np.ndarray,
    sigma: np.ndarray | None = None,
    *,
    xyz: np.ndarray | None = None,
    edge_uncertainty_ratio: float = 1.0,
    radial_power: float = 2.0,
) -> np.ndarray:
    labels = np.asarray(condition_id)[indices]
    unique = np.unique(labels)
    weights = np.zeros(indices.size, dtype=float)
    uncertainty = None
    if sigma is not None:
        uncertainty = np.asarray(sigma, dtype=float)[indices]
        if not np.all(np.isfinite(uncertainty) & (uncertainty > 0.0)):
            raise ValueError("rate uncertainty must be finite and positive")
    if not np.isfinite(edge_uncertainty_ratio) or edge_uncertainty_ratio <= 0.0:
        raise ValueError("edge uncertainty ratio must be finite and positive")
    if not np.isfinite(radial_power) or radial_power <= 0.0:
        raise ValueError("radial uncertainty power must be finite and positive")
    coordinates = None
    if xyz is not None:
        coordinates = np.asarray(xyz, dtype=float)[indices, :2]
        if coordinates.shape != (indices.size, 2) or not np.all(np.isfinite(coordinates)):
            raise ValueError("wafer coordinates must be finite x-y pairs")
    for label in unique:
        mask = labels == label
        local = (
            1.0 / np.square(uncertainty[mask])
            if uncertainty is not None
            else np.ones(np.count_nonzero(mask), dtype=float)
        )
        if edge_uncertainty_ratio != 1.0:
            if coordinates is None:
                raise ValueError("wafer coordinates are required for radial uncertainty weighting")
            local_xy = coordinates[mask]
            center = 0.5 * (np.min(local_xy, axis=0) + np.max(local_xy, axis=0))
            radius = np.linalg.norm(local_xy - center, axis=1)
            radius_max = float(np.max(radius))
            rho = radius / radius_max if radius_max > 0.0 else np.zeros_like(radius)
            relative_sigma = 1.0 + (edge_uncertainty_ratio - 1.0) * rho**radial_power
            local = local / np.square(relative_sigma)
        # Every condition retains the same total mass.  Measurement
        # uncertainty only redistributes that mass among points in the
        # condition; it must not make one process condition dominate another.
        weights[mask] = local / (unique.size * np.sum(local))
    return weights


def parameter_design_diagnostics(design: np.ndarray) -> dict[str, object]:
    """Separate structural rank from practical parameter resolution.

    Sensitivity columns are norm-scaled before the singular-value test.  A
    full-rank design can still be weak when its scaled condition number is
    extreme or two non-intercept sensitivity directions are nearly parallel.
    """

    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("parameter design must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("parameter design must be finite")
    scale = np.linalg.norm(matrix, axis=0)
    scaled = np.zeros_like(matrix)
    nonzero = scale > 0.0
    scaled[:, nonzero] = matrix[:, nonzero] / scale[nonzero]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = singular[0] * np.sqrt(np.finfo(float).eps)
    rank = int(np.count_nonzero(singular > tolerance))
    full_rank = bool(rank == matrix.shape[1])
    condition_number = (
        float(singular[0] / singular[-1])
        if singular[-1] > np.finfo(float).eps * singular[0]
        else float("inf")
    )

    max_correlation = 0.0
    parameter_columns = matrix[:, 1:] if matrix.shape[1] > 1 else np.empty((matrix.shape[0], 0))
    for left in range(parameter_columns.shape[1]):
        a = parameter_columns[:, left] - float(np.mean(parameter_columns[:, left]))
        for right in range(left + 1, parameter_columns.shape[1]):
            b = parameter_columns[:, right] - float(np.mean(parameter_columns[:, right]))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 0.0:
                max_correlation = max(max_correlation, abs(float(np.dot(a, b) / denominator)))

    reasons: list[str] = []
    if not full_rank:
        reasons.append(f"only {rank} of {matrix.shape[1]} sensitivity directions are resolved")
    if full_rank and condition_number >= _PRACTICAL_CONDITION_LIMIT:
        reasons.append(
            f"scaled sensitivity condition number {condition_number:.3g} exceeds "
            f"{_PRACTICAL_CONDITION_LIMIT:.0e}"
        )
    if max_correlation >= _PRACTICAL_CORRELATION_LIMIT:
        reasons.append(
            f"parameter sensitivity correlation {max_correlation:.5f} exceeds "
            f"{_PRACTICAL_CORRELATION_LIMIT:.3f}"
        )
    status = "unresolved" if not full_rank else "weak" if reasons else "sufficient"
    return {
        "status": status,
        "rank": rank,
        "direction_count": int(matrix.shape[1]),
        "full_rank": full_rank,
        "condition_number": condition_number,
        "max_abs_parameter_correlation": max_correlation,
        "reasons": reasons,
    }


def _reference_concentrations(
    data: RoleFieldSet, train_indices: np.ndarray, transport_mode: str
) -> dict[str, float]:
    concentrations = data.reaction_inputs_for(transport_mode)
    refs: dict[str, float] = {}
    for name in data.species:
        value = float(np.median(np.asarray(concentrations[name])[train_indices]))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Reference concentration for {name} must be positive")
        refs[name] = value
    return refs


def _parameter_dict(candidate: SurfaceKineticCandidate, log_values: np.ndarray) -> dict[str, float]:
    return {
        name: float(10.0**value)
        for name, value in zip(candidate.parameter_names, np.asarray(log_values, dtype=float))
    }


def _log_bounds(candidate: SurfaceKineticCandidate) -> tuple[np.ndarray, np.ndarray]:
    declared = candidate.parameter_log10_bounds
    lower = np.asarray([declared[name][0] for name in candidate.parameter_names], dtype=float)
    upper = np.asarray([declared[name][1] for name in candidate.parameter_names], dtype=float)
    return lower, upper


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def _symmetric_scale(
    shape: np.ndarray,
    target: np.ndarray,
    condition_id: np.ndarray,
    weights: np.ndarray,
) -> float:
    positive = (shape > _TINY) & (target > 0.0)
    if not np.any(positive):
        return 0.0
    ratios = target[positive] / shape[positive]
    lower_log = float(np.log(max(float(np.min(ratios)) * 1.0e-2, _TINY)))
    upper_log = float(np.log(max(float(np.max(ratios)) * 1.0e2, np.exp(lower_log))))

    # For each condition the symmetric loss depends on the scale only through
    # three weighted moments.  Precomputing them is algebraically identical to
    # rebuilding every point residual during the one-dimensional search and is
    # much cheaper when a sampler evaluates thousands of shape vectors.
    shape_moment: list[float] = []
    cross_moment: list[float] = []
    target_moment: list[float] = []
    for group in np.unique(condition_id):
        mask = condition_id == group
        local = weights[mask]
        local = local / max(float(np.sum(local)), _TINY)
        shape_moment.append(float(np.sum(local * np.square(shape[mask]))))
        cross_moment.append(float(np.sum(local * shape[mask] * target[mask])))
        target_moment.append(float(np.sum(local * np.square(target[mask]))))
    moment_a = np.asarray(shape_moment, dtype=float)
    moment_b = np.asarray(cross_moment, dtype=float)
    moment_c = np.asarray(target_moment, dtype=float)

    def evaluate(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        numerator = np.maximum(
            moment_a * scale**2 - 2.0 * moment_b * scale + moment_c,
            0.0,
        )
        denominator = np.maximum(
            moment_c + moment_a * scale**2,
            2.0 * np.finfo(float).tiny,
        )
        return float(np.mean(2.0 * numerator / denominator))

    grid = np.linspace(lower_log, upper_log, 65)
    values = np.asarray([evaluate(value) for value in grid])
    best = int(np.argmin(values))
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, grid.size - 1)]

    def derivative(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        denominator = np.maximum(
            moment_c + moment_a * scale**2,
            2.0 * np.finfo(float).tiny,
        )
        return float(
            np.mean(
                4.0
                * moment_b
                * (moment_a * scale**2 - moment_c)
                / np.square(denominator)
            )
        )

    left_derivative = derivative(left)
    right_derivative = derivative(right)
    if left_derivative <= 0.0 <= right_derivative:
        for _ in range(64):
            middle = 0.5 * (left + right)
            if derivative(middle) <= 0.0:
                left = middle
            else:
                right = middle
        return float(np.exp(0.5 * (left + right)))

    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = evaluate(x1), evaluate(x2)
    for _ in range(48):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = evaluate(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = evaluate(x2)
    return float(np.exp(x1 if f1 <= f2 else x2))


def _profile_rate_scale(
    shape: np.ndarray,
    target: np.ndarray,
    condition_id: np.ndarray,
    weights: np.ndarray,
    loss_name: str,
) -> tuple[float, float]:
    name = validate_wafer_loss_name(loss_name)
    effective = np.asarray(weights, dtype=float).copy()
    if name == "wafer_normalized_mse":
        for group in np.unique(condition_id):
            mask = condition_id == group
            local = weights[mask] / max(float(np.sum(weights[mask])), _TINY)
            target_rms2 = max(float(np.sum(local * np.square(target[mask]))), _TINY)
            effective[mask] /= target_rms2
    if name in {"mse", "wafer_normalized_mse"}:
        denominator = float(np.sum(effective * shape * shape))
        if denominator <= _TINY:
            return 0.0, float("inf")
        scale = max(0.0, float(np.sum(effective * shape * target) / denominator))
    elif name == "wafer_normalized_mae":
        positive = shape > _TINY
        if not np.any(positive):
            return 0.0, float("inf")
        scale_weight = effective.copy()
        for group in np.unique(condition_id):
            mask = condition_id == group
            local = weights[mask] / max(float(np.sum(weights[mask])), _TINY)
            target_rms = max(float(np.sqrt(np.sum(local * np.square(target[mask])))), _TINY)
            scale_weight[mask] /= target_rms
        scale = max(
            0.0,
            _weighted_median(
                target[positive] / shape[positive],
                scale_weight[positive] * shape[positive],
            ),
        )
    else:
        scale = _symmetric_scale(shape, target, condition_id, weights)
    objective = wafer_loss(
        target=target,
        prediction=scale * shape,
        condition_id=condition_id,
        weights=weights,
        loss_name=name,
    )
    return scale, objective


def _search_shape_parameters(
    candidate: SurfaceKineticCandidate,
    data: RoleFieldSet,
    train_indices: np.ndarray,
    refs: dict[str, float],
    weights: np.ndarray,
    loss_name: str,
    *,
    initial_log_parameters: np.ndarray | None = None,
    local_only: bool = False,
    record_history: bool = False,
) -> tuple[np.ndarray, float, float, int, tuple[dict[str, float | int | str], ...]]:
    dimension = len(candidate.parameter_names)
    lower, upper = _log_bounds(candidate)
    target = np.asarray(data.rate, dtype=float)[train_indices]
    condition_id = np.asarray(data.condition_id)[train_indices]
    concentrations = data.reaction_inputs_for(candidate.transport_mode)
    if dimension == 0:
        scale, loss = _profile_rate_scale(
            np.ones(target.shape), target, condition_id, weights, loss_name
        )
        history = (
            ({"trial": 1, "score": float(loss), "best_score": float(loss)},)
            if record_history
            else ()
        )
        return np.empty(0, dtype=float), scale, loss, 1, history

    evaluation_count = 0
    best_score = float("inf")
    history_rows: list[dict[str, float | int | str]] = []

    def evaluate(values: tuple[float, ...] | np.ndarray) -> tuple[float, float]:
        nonlocal evaluation_count, best_score
        evaluation_count += 1
        params = _parameter_dict(candidate, np.asarray(values, dtype=float))
        shape = response_shape(candidate, concentrations, refs, params)[train_indices]
        scale, score = _profile_rate_scale(shape, target, condition_id, weights, loss_name)
        best_score = min(best_score, float(score))
        if record_history:
            history_rows.append(
                {
                    "trial": evaluation_count,
                    "score": float(score),
                    "best_score": best_score,
                }
            )
        return scale, score

    seeds: list[tuple[float, ...]] = []
    if initial_log_parameters is not None:
        seeds.append(tuple(np.clip(np.asarray(initial_log_parameters, dtype=float), lower, upper)))
    if not local_only:
        choices = [
            tuple(sorted(set(float(np.clip(value, lower[index], upper[index])) for value in _INITIAL_LOG_GRID)))
            for index in range(dimension)
        ]
        seeds.extend(product(*choices))
    if not seeds:
        seeds.append(tuple(np.zeros(dimension)))

    evaluated: list[tuple[float, tuple[float, ...], float]] = []
    for values in seeds:
        scale, loss = evaluate(values)
        evaluated.append((loss, tuple(float(value) for value in values), scale))

    unique: dict[tuple[float, ...], tuple[float, tuple[float, ...], float]] = {}
    for item in sorted(evaluated):
        unique.setdefault(item[1], item)
    starts = list(unique.values())[: min(_MULTISTART_COUNT, len(unique))]

    refined: list[tuple[float, tuple[float, ...], float]] = []
    for start_index, (loss, values, scale) in enumerate(starts):
        current = np.asarray(values, dtype=float)
        current_loss = float(loss)
        current_scale = float(scale)
        # A deterministic coordinate pattern search is inexpensive for the
        # low-dimensional observable groups and can follow correlated valleys
        # from more than one coarse-grid basin.
        for step in _REFINEMENT_STEPS:
            if start_index == 0:
                # Preserve the original full-neighborhood refinement for the
                # best coarse seed so the new multistart search cannot degrade
                # an established solution through coordinate coupling.
                choices = [
                    tuple(
                        np.clip(
                            (value - step, value, value + step),
                            lower[index],
                            upper[index],
                        )
                    )
                    for index, value in enumerate(current)
                ]
                alternatives = []
                for trial_tuple in product(*choices):
                    trial_scale, trial_loss = evaluate(trial_tuple)
                    alternatives.append((trial_loss, tuple(trial_tuple), trial_scale))
                current_loss, current_tuple, current_scale = min(alternatives)
                current = np.asarray(current_tuple, dtype=float)
                continue

            for _ in range(max(2, dimension)):
                sweep_start = current_loss
                for index in range(dimension):
                    alternatives = [(current_loss, tuple(current), current_scale)]
                    for direction in (-1.0, 1.0):
                        trial = current.copy()
                        trial[index] = np.clip(
                            trial[index] + direction * step,
                            lower[index],
                            upper[index],
                        )
                        trial_scale, trial_loss = evaluate(trial)
                        alternatives.append((trial_loss, tuple(trial), trial_scale))
                    current_loss, current_tuple, current_scale = min(alternatives)
                    current = np.asarray(current_tuple, dtype=float)
                if current_loss >= sweep_start:
                    break
        refined.append((current_loss, tuple(current), current_scale))

    best_loss, best_tuple, best_scale = min(refined)
    return (
        np.asarray(best_tuple, dtype=float),
        best_scale,
        best_loss,
        evaluation_count,
        tuple(history_rows),
    )


def _sampler_shape_parameters(
    candidate: SurfaceKineticCandidate,
    data: RoleFieldSet,
    train_indices: np.ndarray,
    refs: dict[str, float],
    weights: np.ndarray,
    settings: SurfaceOptimizationSettings,
    *,
    record_history: bool = False,
) -> tuple[np.ndarray, float, float, int, tuple[dict[str, float | int | str], ...]]:
    dimension = len(candidate.parameter_names)
    lower, upper = _log_bounds(candidate)
    target = np.asarray(data.rate, dtype=float)[train_indices]
    condition_id = np.asarray(data.condition_id)[train_indices]
    concentrations = data.reaction_inputs_for(candidate.transport_mode)
    if dimension == 0:
        scale, loss = _profile_rate_scale(
            np.ones(target.shape), target, condition_id, weights, settings.loss_name
        )
        history = (
            ({"trial": 1, "score": float(loss), "best_score": float(loss)},)
            if record_history
            else ()
        )
        return np.empty(0, dtype=float), scale, loss, 1, history

    def evaluate(values: np.ndarray) -> tuple[float, dict[str, Any]]:
        params = _parameter_dict(candidate, values)
        shape = response_shape(candidate, concentrations, refs, params)[train_indices]
        scale, objective = _profile_rate_scale(
            shape, target, condition_id, weights, settings.loss_name
        )
        prediction = scale * shape
        means: list[float] = []
        cvs: list[float] = []
        for group in np.unique(condition_id):
            local_prediction = prediction[condition_id == group]
            mean = max(float(np.mean(local_prediction)), _TINY)
            means.append(mean)
            cvs.append(float(np.std(local_prediction)) / mean)
        measures = {
            "mean_cv": float(np.mean(cvs)),
            "log10_rate_ratio": float(
                np.log10(max(means) / max(min(means), _TINY))
            ),
        }
        return objective, {
            "log_values": [float(value) for value in values],
            "rate_scale": scale,
            "sampler_measures": measures,
        }

    search = SearchSettings(
        method=settings.sampler,
        seed=int(settings.seed),
        min_trials=int(settings.trials),
        max_trials=int(settings.trials),
        trials_per_dimension=int(settings.trials),
        patience=int(settings.trials),
        relative_improvement=0.0,
        repetitions=1,
        pruner="none",
        sampler_options=dict(settings.sampler_options),
        storage={},
    )

    def random_objective(rng: np.random.Generator) -> tuple[float, dict[str, Any]]:
        return evaluate(rng.uniform(lower, upper))

    latent_names = [f"z__{name}" for name in candidate.parameter_names]
    log_names = [f"log10__{name}" for name in candidate.parameter_names]

    def optuna_objective(trial: Any) -> tuple[float, dict[str, Any]]:
        if settings.sampler == "cma_mae":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Fixed parameter .* is out of range for distribution.*",
                    category=UserWarning,
                )
                latent = np.asarray(
                    [trial.suggest_float(name, -1.0e9, 1.0e9) for name in latent_names],
                    dtype=float,
                )
            unit = 0.5 * (np.tanh(0.5 * latent) + 1.0)
            values = lower + (upper - lower) * unit
        else:
            values = np.asarray(
                [
                    trial.suggest_float(name, float(low), float(high))
                    for name, low, high in zip(log_names, lower, upper)
                ],
                dtype=float,
            )
        return evaluate(values)

    search_space = None
    qd_context = None
    if settings.sampler in {"de", "pso"}:
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError(
                f"surface sampler {settings.sampler} requires Optuna"
            ) from exc
        search_space = {
            name: optuna.distributions.FloatDistribution(float(low), float(high))
            for name, low, high in zip(log_names, lower, upper)
        }
    elif settings.sampler == "cma_mae":
        qd_context = {
            "param_names": latent_names,
            "measure_names": ["mean_cv", "log10_rate_ratio"],
            "archive_ranges": [(0.0, 1.0), (0.0, 3.0)],
            "emitter_x0": {name: 0.0 for name in latent_names},
        }
    run = run_search(
        search,
        seed=int(settings.seed),
        dimension=dimension,
        random_objective=random_objective,
        optuna_objective=optuna_objective,
        search_space=search_space,
        qd_context=qd_context,
    )
    payload = run.best_payload
    return (
        np.asarray(payload["log_values"], dtype=float),
        float(payload["rate_scale"]),
        float(run.best_score),
        int(run.trial_count),
        tuple(dict(row) for row in run.trace) if record_history else (),
    )


def _sensitivity_design(
    candidate: SurfaceKineticCandidate,
    data: RoleFieldSet,
    refs: dict[str, float],
    parameters: dict[str, float],
    indices: np.ndarray,
) -> np.ndarray:
    concentrations = data.reaction_inputs_for(candidate.transport_mode)
    base = np.maximum(
        response_shape(candidate, concentrations, refs, parameters)[indices], _TINY
    )
    columns = [np.ones(indices.size, dtype=float)]
    step = 1.0e-5
    for name in candidate.parameter_names:
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] *= np.exp(step)
        minus[name] *= np.exp(-step)
        upper = np.maximum(
            response_shape(candidate, concentrations, refs, plus)[indices], _TINY
        )
        lower = np.maximum(
            response_shape(candidate, concentrations, refs, minus)[indices], _TINY
        )
        columns.append((np.log(upper) - np.log(lower)) / (2.0 * step))
    return np.column_stack(columns)


def fit_surface_kinetic(
    candidate: SurfaceKineticCandidate,
    data: RoleFieldSet,
    train_indices: np.ndarray,
    *,
    reference_concentrations: dict[str, float] | None = None,
    initial_fit: SurfaceKineticFit | None = None,
    local_only: bool = False,
    optimization: SurfaceOptimizationSettings | None = None,
    record_optimization_history: bool = False,
) -> SurfaceKineticFit:
    train_indices = np.asarray(train_indices, dtype=int)
    if train_indices.size == 0:
        raise ValueError("At least one identification row is required")
    refs = (
        dict(reference_concentrations)
        if reference_concentrations is not None
        else _reference_concentrations(data, train_indices, candidate.transport_mode)
    )
    settings = optimization or SurfaceOptimizationSettings()
    weights = condition_balanced_weights(
        np.asarray(data.condition_id),
        train_indices,
        data.rate_sigma,
        xyz=data.xyz,
        edge_uncertainty_ratio=settings.edge_uncertainty_ratio,
        radial_power=settings.radial_power,
    )
    initial = None
    if initial_fit is not None:
        initial = np.log10(
            [initial_fit.shape_parameters[name] for name in candidate.parameter_names]
        )
    if settings.sampler == "pattern" or local_only:
        (
            log_values,
            rate_scale,
            objective,
            optimizer_trial_count,
            optimization_history,
        ) = _search_shape_parameters(
            candidate,
            data,
            train_indices,
            refs,
            weights,
            settings.loss_name,
            initial_log_parameters=initial,
            local_only=local_only,
            record_history=record_optimization_history,
        )
        optimizer_method = "pattern"
    else:
        (
            log_values,
            rate_scale,
            objective,
            optimizer_trial_count,
            optimization_history,
        ) = _sampler_shape_parameters(
            candidate,
            data,
            train_indices,
            refs,
            weights,
            settings,
            record_history=record_optimization_history,
        )
        optimizer_method = settings.sampler
    parameters = _parameter_dict(candidate, log_values)
    concentrations = data.reaction_inputs_for(candidate.transport_mode)
    prediction = rate_scale * response_shape(
        candidate, concentrations, refs, parameters
    )
    design = _sensitivity_design(candidate, data, refs, parameters, train_indices)
    boundary = tuple(
        name
        for name, value, low, high in zip(
            candidate.parameter_names, log_values, *_log_bounds(candidate)
        )
        if abs(value - low) < 1.0e-10 or abs(value - high) < 1.0e-10
    )
    return SurfaceKineticFit(
        candidate=candidate,
        rate_scale_nm_s=rate_scale,
        shape_parameters=parameters,
        reference_concentrations=refs,
        prediction=np.asarray(prediction, dtype=float),
        design=design,
        objective_value=objective,
        loss_name=settings.loss_name,
        boundary_parameters=boundary,
        optimizer_method=optimizer_method,
        optimizer_trial_count=optimizer_trial_count,
        optimization_history=optimization_history,
    )


def role_response_curve_rows(
    fit: SurfaceKineticFit,
    data: RoleFieldSet,
    *,
    points: int = 80,
) -> list[dict[str, float | str]]:
    """Vary one assigned input over its observed range at other reference values."""

    if points < 2:
        raise ValueError("role response curves require at least two points")
    inputs = data.reaction_inputs_for(fit.candidate.transport_mode)
    metadata = data.reaction_input_metadata(fit.candidate.transport_mode)
    rows: list[dict[str, float | str]] = []
    for role, species in fit.effective_roles.items():
        if species is None:
            continue
        observed = np.asarray(inputs[species], dtype=float)
        low, high = float(np.min(observed)), float(np.max(observed))
        grid = np.linspace(low, high, points) if high > low else np.full(points, low)
        reference_inputs = {
            name: np.full(points, float(reference), dtype=float)
            for name, reference in fit.reference_concentrations.items()
        }
        reference_inputs[species] = grid
        prediction = fit.rate_scale_nm_s * response_shape(
            fit.candidate,
            reference_inputs,
            fit.reference_concentrations,
            fit.shape_parameters,
        )
        for input_value, rate in zip(grid, prediction):
            rows.append(
                {
                    "role": role,
                    "species": species,
                    "reaction_input": float(input_value),
                    "reaction_input_unit": metadata["unit"],
                    "predicted_rate_nm_s": float(rate),
                    "reference_input": float(fit.reference_concentrations[species]),
                }
            )
    return rows


def role_input_sensitivity_rows(
    fit: SurfaceKineticFit,
    data: RoleFieldSet,
) -> list[dict[str, float | str]]:
    """Quantify prediction change when one local role input is set to its reference."""

    inputs = data.reaction_inputs_for(fit.candidate.transport_mode)
    metadata = data.reaction_input_metadata(fit.candidate.transport_mode)
    prediction = fit.rate_scale_nm_s * response_shape(
        fit.candidate,
        inputs,
        fit.reference_concentrations,
        fit.shape_parameters,
    )
    rows: list[dict[str, float | str]] = []
    for role, species in fit.effective_roles.items():
        if species is None:
            continue
        counterfactual_inputs = {
            name: np.asarray(values, dtype=float).copy() for name, values in inputs.items()
        }
        counterfactual_inputs[species].fill(float(fit.reference_concentrations[species]))
        counterfactual = fit.rate_scale_nm_s * response_shape(
            fit.candidate,
            counterfactual_inputs,
            fit.reference_concentrations,
            fit.shape_parameters,
        )
        difference = prediction - counterfactual
        condition_mse = [
            float(np.mean(np.square(difference[data.condition_id == condition])))
            for condition in np.unique(data.condition_id)
        ]
        rows.append(
            {
                "role": role,
                "species": species,
                "reaction_input_quantity": metadata["quantity"],
                "reaction_input_unit": metadata["unit"],
                "rms_prediction_change_nm_s": float(np.sqrt(np.mean(condition_mse))),
                "mean_prediction_change_nm_s": float(np.mean(difference)),
                "minimum_prediction_change_nm_s": float(np.min(difference)),
                "maximum_prediction_change_nm_s": float(np.max(difference)),
            }
        )
    return rows


def parameter_sensitivity_rows(
    fit: SurfaceKineticFit,
) -> list[dict[str, float | str]]:
    """Return local log-rate sensitivities and their pairwise correlations."""

    columns = np.asarray(fit.design[:, 1:], dtype=float)
    names = tuple(fit.candidate.parameter_names)
    if columns.shape[1] != len(names):
        raise ValueError("parameter sensitivity design does not match parameter names")
    rms = np.sqrt(np.mean(np.square(columns), axis=0))
    means = np.mean(columns, axis=0)
    rows: list[dict[str, float | str]] = []
    for left, name_left in enumerate(names):
        centered_left = columns[:, left] - means[left]
        for right, name_right in enumerate(names):
            centered_right = columns[:, right] - means[right]
            denominator = float(
                np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
            )
            correlation = (
                float(np.dot(centered_left, centered_right) / denominator)
                if denominator > 0.0
                else (1.0 if left == right and rms[left] > 0.0 else float("nan"))
            )
            rows.append(
                {
                    "parameter_1": name_left,
                    "parameter_2": name_right,
                    "rms_log_rate_sensitivity_1": float(rms[left]),
                    "mean_log_rate_sensitivity_1": float(means[left]),
                    "pearson_correlation": correlation,
                }
            )
    return rows


def parameter_loss_slice_rows(
    fit: SurfaceKineticFit,
    data: RoleFieldSet,
    *,
    optimization: SurfaceOptimizationSettings | None = None,
    span_decades: float = 3.0,
    points: int = 61,
) -> list[dict[str, float | str]]:
    """Vary one kinetic ratio while profiling only the overall rate scale.

    Other kinetic ratios remain fixed at their fitted values.  The result is a
    one-parameter loss slice, not a joint confidence interval.
    """

    if points < 3:
        raise ValueError("parameter loss slices require at least three points")
    if not np.isfinite(span_decades) or span_decades <= 0.0:
        raise ValueError("span_decades must be finite and positive")
    settings = optimization or SurfaceOptimizationSettings(loss_name=fit.loss_name)
    indices = np.arange(data.rate.size, dtype=int)
    weights = condition_balanced_weights(
        np.asarray(data.condition_id),
        indices,
        data.rate_sigma,
        xyz=data.xyz,
        edge_uncertainty_ratio=settings.edge_uncertainty_ratio,
        radial_power=settings.radial_power,
    )
    lower, upper = _log_bounds(fit.candidate)
    fitted_logs = np.log10(
        [fit.shape_parameters[name] for name in fit.candidate.parameter_names]
    )
    inputs = data.reaction_inputs_for(fit.candidate.transport_mode)
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(fit.candidate.parameter_names):
        low = max(float(lower[index]), float(fitted_logs[index] - span_decades))
        high = min(float(upper[index]), float(fitted_logs[index] + span_decades))
        grid = np.unique(
            np.concatenate(
                ([fitted_logs[index]], np.linspace(low, high, int(points)))
            )
        )
        for value in grid:
            trial_logs = fitted_logs.copy()
            trial_logs[index] = value
            parameters = _parameter_dict(fit.candidate, trial_logs)
            shape = response_shape(
                fit.candidate,
                inputs,
                fit.reference_concentrations,
                parameters,
            )
            rate_scale, objective = _profile_rate_scale(
                np.asarray(shape, dtype=float),
                np.asarray(data.rate, dtype=float),
                np.asarray(data.condition_id),
                weights,
                settings.loss_name,
            )
            is_mse = settings.loss_name in {
                "mse",
                "wafer_normalized_mse",
                "symmetric_normalized_mse",
            }
            rows.append(
                {
                    "parameter": name,
                    "parameter_value": float(10.0**value),
                    "factor_from_fitted_value": float(10.0 ** (value - fitted_logs[index])),
                    "profiled_rate_scale_nm_s": float(rate_scale),
                    "objective": float(objective),
                    "fitting_error": (
                        float(np.sqrt(max(objective, 0.0))) if is_mse else float(objective)
                    ),
                    "fitting_error_name": (
                        "training_rmse_nm_s" if settings.loss_name == "mse" else "normalized_error"
                    ),
                    "loss": settings.loss_name,
                }
            )
    return rows


def predict_surface_kinetic(
    fit: SurfaceKineticFit, data: RoleFieldSet
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    concentrations = data.reaction_inputs_for(fit.candidate.transport_mode)
    missing = sorted(set(fit.reference_concentrations) - set(concentrations))
    if missing:
        raise ValueError(f"Prediction data are missing species: {missing}")
    shape = response_shape(
        fit.candidate,
        concentrations,
        fit.reference_concentrations,
        fit.shape_parameters,
    )
    prediction = fit.rate_scale_nm_s * shape
    state = surface_state(
        fit.candidate,
        concentrations,
        fit.reference_concentrations,
        fit.shape_parameters,
    )
    state["dimensionless_response"] = shape
    for role, species in fit.effective_roles.items():
        if species is None:
            continue
        supplied = np.asarray(concentrations[species], dtype=float)
        state[f"reaction_input_{role}"] = supplied
        state[f"normalized_reaction_input_{role}"] = (
            supplied / max(float(fit.reference_concentrations[species]), _TINY)
        )
        if fit.candidate.transport_mode != DIRECT_FLUX:
            bulk = np.asarray(data.bulk_concentrations[species], dtype=float)
            ratio = supplied / np.maximum(bulk, _TINY)
            state[f"surface_to_bulk_{role}"] = ratio
            state[f"transport_utilization_{role}"] = 1.0 - ratio
            if (
                fit.candidate.transport_mode == DIRECT_SURFACE
                and species in data.transport_capacity_flux
            ):
                capacity = np.asarray(data.transport_capacity_flux[species], dtype=float)
                state[f"transport_flux_{role}"] = capacity * (1.0 - ratio)
        else:
            state[f"transport_capacity_flux_{role}"] = supplied
        if species in data.realized_reactive_flux:
            realized = np.asarray(data.realized_reactive_flux[species], dtype=float)
            state[f"realized_reactive_flux_{role}"] = realized
            if (
                fit.candidate.transport_mode == DIRECT_SURFACE
                and species in data.transport_capacity_flux
            ):
                state[f"flux_closure_residual_{role}"] = (
                    state[f"transport_flux_{role}"] - realized
                )
            elif fit.candidate.transport_mode == DIRECT_FLUX:
                state[f"reactive_to_capacity_flux_{role}"] = (
                    realized / np.maximum(supplied, _TINY)
                )
    return np.asarray(prediction, dtype=float), state


__all__ = [
    "SurfaceKineticFit",
    "SurfaceOptimizationSettings",
    "condition_balanced_weights",
    "fit_surface_kinetic",
    "parameter_loss_slice_rows",
    "parameter_sensitivity_rows",
    "parameter_design_diagnostics",
    "predict_surface_kinetic",
    "role_input_sensitivity_rows",
    "role_response_curve_rows",
]
