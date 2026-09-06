"""Sampler backends, trial budgets, and convergence tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from deposim_sim.common.literals import parse_literal_value


_METHODS = {"random", "tpe", "cmaes", "de", "pso", "levy", "cma_mae"}
_PRUNERS = {"none", "median", "hyperband"}
_HUB_PACKAGES = {
    "de": ("samplers/differential_evolution", "DESampler"),
    "pso": ("samplers/pso", "PSOSampler"),
    "levy": ("samplers/levy_flight_sampler", "LevyFlightSampler"),
    "cma_mae": ("samplers/cmamae", "CmaMaeSampler"),
}
_OPTUNAHUB_REGISTRY_REF = "abac65b96fef633bba86b85544da12ee473ffdd4"


class PruneRequested(RuntimeError):
    """Signal that an Optuna intermediate evaluation requested pruning."""


@dataclass(frozen=True)
class SearchSettings:
    method: str
    seed: int
    min_trials: int
    max_trials: int
    trials_per_dimension: int
    patience: int
    relative_improvement: float
    repetitions: int
    pruner: str
    sampler_options: dict[str, Any]
    storage: dict[str, Any]


@dataclass(frozen=True)
class SearchRun:
    seed: int
    method: str
    best_score: float
    best_payload: dict[str, Any]
    trial_count: int
    converged: bool
    termination_reason: str
    trace: list[dict[str, Any]]


def _normalize_mapping_values(values: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(values or {}).items():
        normalized[str(key)] = parse_literal_value(value) if isinstance(value, str) else value
    return normalized


def parse_search_settings(parameter_fit: Any) -> SearchSettings:
    raw = dict(getattr(parameter_fit, "search", {}) or {})
    method = str(raw.get("method", "random")).strip().lower()
    pruner = str(raw.get("pruner", "none")).strip().lower()
    if method not in _METHODS:
        raise ValueError(f"search.method must be one of {sorted(_METHODS)}, got {method!r}")
    if pruner not in _PRUNERS:
        raise ValueError(f"search.pruner must be one of {sorted(_PRUNERS)}, got {pruner!r}")

    min_trials = int(raw.get("min_trials", 20))
    max_trials = int(raw.get("max_trials", 120))
    trials_per_dimension = int(raw.get("trials_per_dimension", 20))
    patience = int(raw.get("patience", 30))
    repetitions = int(raw.get("repetitions", 1))
    relative_improvement = float(raw.get("relative_improvement", 1.0e-4))
    if min_trials < 1 or max_trials < min_trials:
        raise ValueError("search requires 1 <= min_trials <= max_trials")
    if trials_per_dimension < 1:
        raise ValueError("search.trials_per_dimension must be >= 1")
    if patience < 1:
        raise ValueError("search.patience must be >= 1")
    if repetitions < 1:
        raise ValueError("search.repetitions must be >= 1")
    if not np.isfinite(relative_improvement) or relative_improvement < 0.0:
        raise ValueError("search.relative_improvement must be finite and nonnegative")
    if method not in {"tpe", "cmaes"} and pruner != "none":
        raise ValueError("search.pruner is only available for tpe or cmaes")

    return SearchSettings(
        method=method,
        seed=int(raw.get("seed", 123)),
        min_trials=min_trials,
        max_trials=max_trials,
        trials_per_dimension=trials_per_dimension,
        patience=patience,
        relative_improvement=relative_improvement,
        repetitions=repetitions,
        pruner=pruner,
        sampler_options=_normalize_mapping_values(raw.get("sampler_options", {})),
        storage=_normalize_mapping_values(raw.get("storage", {})),
    )


def repetition_seeds(settings: SearchSettings) -> list[int]:
    return [int(settings.seed + 104729 * index) for index in range(settings.repetitions)]


def trial_budget(settings: SearchSettings, dimension: int) -> int:
    requested = max(settings.min_trials, settings.trials_per_dimension * max(int(dimension), 1))
    return min(requested, settings.max_trials)


def _material_improvement(score: float, reference: float, relative: float) -> bool:
    if not np.isfinite(reference):
        return True
    threshold = relative * max(abs(reference), np.finfo(float).tiny)
    return score < reference - threshold


def _load_hub_sampler(method: str) -> Any:
    try:
        import optunahub
    except Exception as exc:
        raise RuntimeError(
            f"search.method={method} requires OptunaHub; install the project with the optuna extra"
        ) from exc
    package, class_name = _HUB_PACKAGES[method]
    try:
        module = optunahub.load_module(package=package, ref=_OPTUNAHUB_REGISTRY_REF)
        return getattr(module, class_name)
    except Exception as exc:
        raise RuntimeError(
            f"search.method={method} could not load OptunaHub package {package!r}"
        ) from exc


def _seed_cma_mae(sampler: Any, seed: int) -> None:
    """Seed pyribs objects omitted from the current OptunaHub constructor."""

    scheduler = sampler.scheduler
    scheduler.archive._rng = np.random.default_rng(seed)
    result_archive = getattr(scheduler, "result_archive", None)
    if result_archive is not None and hasattr(result_archive, "_rng"):
        result_archive._rng = np.random.default_rng(seed + 1)
    for index, emitter in enumerate(scheduler._emitters):
        emitter._opt._rng = np.random.default_rng(seed + 2 + 2 * index)
        emitter._ranker._rng = np.random.default_rng(seed + 3 + 2 * index)


def _resolve_optuna_sampler(
    optuna: Any,
    settings: SearchSettings,
    seed: int,
    *,
    search_space: Mapping[str, Any] | None = None,
    qd_context: Mapping[str, Any] | None = None,
) -> Any:
    options = dict(settings.sampler_options)
    if settings.method == "tpe":
        return optuna.samplers.TPESampler(seed=seed, **options)
    if settings.method == "cmaes":
        try:
            return optuna.samplers.CmaEsSampler(seed=seed, **options)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CMA-ES search requires the optional 'cmaes' package; "
                "install the project with the optuna extra"
            ) from exc
    if settings.method == "de":
        sampler_class = _load_hub_sampler("de")
        return sampler_class(search_space=dict(search_space or {}) or None, seed=seed, **options)
    if settings.method == "pso":
        sampler_class = _load_hub_sampler("pso")
        return sampler_class(search_space=dict(search_space or {}) or None, seed=seed, **options)
    if settings.method == "levy":
        sampler_class = _load_hub_sampler("levy")
        return sampler_class(seed=seed, **options)
    if settings.method == "cma_mae":
        context = dict(qd_context or {})
        required = {"param_names", "measure_names", "archive_ranges", "emitter_x0"}
        missing = sorted(required - set(context))
        if missing:
            raise ValueError(f"CMA-MAE search context is missing: {missing}")
        archive_ranges = [
            tuple(item) for item in options.pop("archive_ranges", context["archive_ranges"])
        ]
        defaults = {
            "archive_dims": [10 for _ in archive_ranges],
            "archive_learning_rate": 0.1,
            "archive_threshold_min": -1.0e12,
            "n_emitters": 1,
            "emitter_sigma0": 1.5,
            "emitter_batch_size": 8,
        }
        defaults.update(options)
        sampler_class = _load_hub_sampler("cma_mae")
        sampler = sampler_class(
            param_names=list(context["param_names"]),
            measure_names=list(context["measure_names"]),
            archive_ranges=archive_ranges,
            emitter_x0=dict(context["emitter_x0"]),
            **defaults,
        )
        _seed_cma_mae(sampler, seed)
        return sampler
    raise ValueError(f"Optuna cannot resolve search method {settings.method!r}")


def _resolve_optuna_pruner(optuna: Any, name: str) -> Any:
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()


def _dispose_storage(storage: Any | None) -> None:
    if storage is None:
        return
    storage.remove_session()
    storage.engine.dispose()


def run_search(
    settings: SearchSettings,
    *,
    seed: int,
    dimension: int,
    random_objective: Callable[[np.random.Generator], tuple[float, dict[str, Any]]],
    optuna_objective: Callable[[Any], tuple[float, dict[str, Any]]],
    study_kwargs: Mapping[str, Any] | None = None,
    search_space: Mapping[str, Any] | None = None,
    qd_context: Mapping[str, Any] | None = None,
) -> SearchRun:
    """Run one sampler repetition against a scalar objective."""

    budget = trial_budget(settings, dimension)
    if settings.method in {"cmaes", "cma_mae"} and int(dimension) < 2:
        raise ValueError(f"{settings.method} requires at least two active parameters")
    if settings.method == "random":
        rng = np.random.default_rng(seed)
        best_score = float("inf")
        best_payload: dict[str, Any] = {}
        material_reference = float("inf")
        last_material_trial = 0
        trace: list[dict[str, Any]] = []
        termination = "max_trials"
        for trial_index in range(1, budget + 1):
            score, payload = random_objective(rng)
            score = float(score)
            if not np.isfinite(score):
                raise ValueError("sampler objective must return a finite score")
            if _material_improvement(score, material_reference, settings.relative_improvement):
                material_reference = score
                last_material_trial = trial_index
            if score < best_score:
                best_score = score
                best_payload = dict(payload)
            trace.append(
                {
                    "seed": seed,
                    "trial": trial_index,
                    "score": score,
                    "best_score": best_score,
                    "state": "complete",
                }
            )
            if trial_index >= settings.min_trials and trial_index - last_material_trial >= settings.patience:
                termination = "score_plateau"
                break
        return SearchRun(
            seed=seed,
            method="random",
            best_score=best_score,
            best_payload=best_payload,
            trial_count=len(trace),
            converged=termination == "score_plateau",
            termination_reason=termination,
            trace=trace,
        )

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            f"search.method={settings.method} requires Optuna; "
            "install the project with the optuna extra"
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kwargs = dict(study_kwargs or {})
    owned_storage = None
    storage_value = kwargs.get("storage")
    if isinstance(storage_value, str) and storage_value.strip():
        owned_storage = optuna.storages.RDBStorage(url=storage_value)
        kwargs["storage"] = owned_storage
    kwargs.update(
        direction="minimize",
        sampler=_resolve_optuna_sampler(
            optuna,
            settings,
            seed,
            search_space=search_space,
            qd_context=qd_context,
        ),
        pruner=_resolve_optuna_pruner(optuna, settings.pruner),
    )
    study = optuna.create_study(**kwargs)
    material_reference = float("inf")
    last_material_trial = 0
    completed_in_run = 0
    attempted_in_run = 0
    termination = "max_trials"

    def wrapped(trial: Any) -> float:
        try:
            score, payload = optuna_objective(trial)
        except PruneRequested as exc:
            raise optuna.TrialPruned() from exc
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("sampler objective must return a finite score")
        trial.set_user_attr("deposim_payload", payload)
        if settings.method == "cma_mae":
            measures = dict(payload.get("sampler_measures", {}))
            for name in dict(qd_context or {}).get("measure_names", ()):
                if name not in measures:
                    raise ValueError(f"CMA-MAE objective payload is missing measure {name!r}")
                trial.set_user_attr(str(name), float(measures[name]))
        return value

    def stop_on_plateau(study_obj: Any, frozen_trial: Any) -> None:
        nonlocal material_reference, last_material_trial, completed_in_run, attempted_in_run, termination
        attempted_in_run += 1
        if frozen_trial.state != optuna.trial.TrialState.COMPLETE:
            return
        completed_in_run += 1
        score = float(frozen_trial.value)
        if _material_improvement(score, material_reference, settings.relative_improvement):
            material_reference = score
            last_material_trial = completed_in_run
        if (
            completed_in_run >= settings.min_trials
            and completed_in_run - last_material_trial >= settings.patience
        ):
            termination = "score_plateau"
            study_obj.stop()

    try:
        study.optimize(wrapped, n_trials=budget, callbacks=[stop_on_plateau])
    except BaseException:
        _dispose_storage(owned_storage)
        raise
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        _dispose_storage(owned_storage)
        raise RuntimeError("all Optuna trials were pruned or failed")
    best_trial = study.best_trial
    payload = dict(best_trial.user_attrs.get("deposim_payload", {}))
    if not payload:
        _dispose_storage(owned_storage)
        raise RuntimeError("best Optuna trial does not contain fit payload")

    trace: list[dict[str, Any]] = []
    best_so_far = float("inf")
    for index, trial in enumerate(study.trials, start=1):
        value = float(trial.value) if trial.value is not None else float("nan")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = min(best_so_far, value)
        trace.append(
            {
                "seed": seed,
                "trial": index,
                "score": value,
                "best_score": best_so_far,
                "state": str(trial.state.name).lower(),
            }
        )
    result = SearchRun(
        seed=seed,
        method=settings.method,
        best_score=float(study.best_value),
        best_payload=payload,
        trial_count=attempted_in_run,
        converged=termination == "score_plateau",
        termination_reason=termination,
        trace=trace,
    )
    _dispose_storage(owned_storage)
    return result


__all__ = [
    "PruneRequested",
    "SearchRun",
    "SearchSettings",
    "parse_search_settings",
    "repetition_seeds",
    "run_search",
    "trial_budget",
]
