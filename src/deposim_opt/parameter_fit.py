"""Fit one reaction-role candidate with a sampler-neutral parameter workflow."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from deposim_schema import SimSpecV2
from deposim_sim.identifiability import compute_identifiability_diagnostics

from .fit_conditions import (
    ConditionSpec,
    evaluate_condition,
    extract_conditions,
    preflight_conditions,
    prepare_condition,
    validate_conditions,
)
from .parameter_space import (
    active_parameter_paths,
    compile_parameter_space,
    draw_parameter_sample,
    parameter_dimension,
)
from .samplers import PruneRequested, parse_search_settings, repetition_seeds, run_search


def _persistent_study_name(
    prefix: str,
    specs: list[SimSpecV2],
    settings: Mapping[str, Any],
) -> str:
    """Prevent resuming trials evaluated for different inputs or objectives."""

    digest = hashlib.sha256()
    payload = {"conditions": [asdict(spec) for spec in specs], "settings": settings}
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    for spec in specs:
        paths = [spec.inputs.fluent.file]
        if spec.measurement.enabled:
            paths.append(spec.measurement.file)
        for path in paths:
            file_hash = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(65536), b""):
                    file_hash.update(block)
            digest.update(file_hash.digest())
    return f"{prefix or 'role_fit'}_{digest.hexdigest()[:20]}"


def _analysis_block(parameter_fit: Any) -> dict[str, Any]:
    defaults = {
        "role_stability": {"enabled": True, "score_epsilon": 1.0e-6},
        "identifiability": {
            "enabled": False,
            "relative_step": 1.0e-2,
            "low_sensitivity_threshold": 1.0e-10,
            "correlation_threshold": 0.98,
        },
        "cache": {"enabled": True, "max_entries": 256},
        "preflight": {"enabled": True, "min_finite_ratio": 0.6},
    }
    raw = dict(getattr(parameter_fit, "analysis", {}) or {})
    return {key: {**value, **dict(raw.get(key, {}) or {})} for key, value in defaults.items()}


def _combine_weighted_components(
    *,
    component_rows: list[dict[str, float]],
    weights: list[float],
    prior_value: float,
) -> dict[str, float]:
    if not component_rows:
        return {
            "mse_nm2": float("nan"),
            "rmse_nm": float("nan"),
            "mae_nm": float("nan"),
            "max_abs_nm": float("nan"),
            "loss_data": 0.0,
            "loss_standardized": 0.0,
            "penalty_solver": 0.0,
            "penalty_prior": float(prior_value),
            "score_total": float(prior_value),
        }
    normalized = np.clip(np.asarray(weights, dtype=float), 0.0, np.inf)
    if float(np.sum(normalized)) <= 0.0:
        raise ValueError("active condition weights must contain a positive value")
    normalized /= float(np.sum(normalized))
    loss_scales = {bool(float(row.get("loss_standardized", 0.0))) for row in component_rows}
    if len(loss_scales) != 1:
        raise ValueError(
            "all active conditions must use the same loss standardization scale"
        )
    out = {
        "mse_nm2": 0.0,
        "mae_nm": 0.0,
        "max_abs_nm": 0.0,
        "loss_data": 0.0,
        "loss_standardized": float(loss_scales.pop()),
        "penalty_solver": 0.0,
    }
    for weight, row in zip(normalized, component_rows):
        out["mse_nm2"] += float(weight) * float(row.get("mse_nm2", 0.0))
        out["mae_nm"] += float(weight) * float(row.get("mae_nm", 0.0))
        out["max_abs_nm"] = max(out["max_abs_nm"], float(row.get("max_abs_nm", 0.0)))
        for name in ("loss_data", "penalty_solver"):
            out[name] += float(weight) * float(row.get(name, 0.0))
    out["penalty_prior"] = float(prior_value)
    out["rmse_nm"] = float(np.sqrt(max(out["mse_nm2"], 0.0)))
    out["score_total"] = sum(
        out[name]
        for name in (
            "loss_data",
            "penalty_solver",
            "penalty_prior",
        )
    )
    return out


def _cache_key(condition_name: str, params: Mapping[str, float]) -> tuple[Any, ...]:
    rounded = []
    for key, value in sorted(params.items()):
        numeric = float(value)
        rounded.append(
            (str(key), round(numeric, 12))
            if np.isfinite(numeric)
            else (str(key), str(numeric))
        )
    return str(condition_name), tuple(rounded)


def _extract_levels(parameter_fit: Any, condition_count: int) -> list[int]:
    fidelity = dict(getattr(parameter_fit, "fidelity", {}) or {})
    raw_levels = fidelity.get("levels", [condition_count])
    if isinstance(raw_levels, (int, float, str)):
        raw_levels = [raw_levels]
    levels: list[int] = []
    for item in list(raw_levels):
        try:
            value = int(item)
        except Exception as exc:
            raise ValueError(f"fidelity level must be int-like: {item!r}") from exc
        levels.append(max(1, min(value, condition_count)))
    levels = sorted(set(levels or [condition_count]))
    if levels[-1] != condition_count:
        levels.append(condition_count)
    return levels


def _evaluate_fidelity_levels(
    *,
    sample: Mapping[str, Any],
    levels: Sequence[int],
    conditions: Sequence[ConditionSpec],
    evaluate_subset: Any,
    step_hook: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    final_components: dict[str, float] = {}
    final_scores: dict[str, float] = {}
    for step, level in enumerate(levels):
        final_components, final_scores = evaluate_subset(sample, list(conditions[: int(level)]))
        if step_hook is not None:
            step_hook(step, final_components)
    return final_components, final_scores


def _fit_payload(
    sample: Mapping[str, Any],
    components: Mapping[str, float],
    condition_scores: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "components": {str(key): float(value) for key, value in components.items()},
        "condition_scores": {str(key): float(value) for key, value in condition_scores.items()},
        "condition_metrics": {
            str(key): {str(metric): float(value) for metric, value in dict(row).items()}
            for key, row in dict(sample.get("__condition_metrics__", {})).items()
        },
        "flat_params": {str(key): float(value) for key, value in sample["flat_params"].items()},
        "per_condition_params": {
            str(key): {str(path): float(value) for path, value in row.items()}
            for key, row in sample["per_condition"].items()
        },
    }


def fit_candidate_parameters(
    *,
    sim_spec: SimSpecV2,
    role_candidate: Any,
    order_candidate: dict[str, int],
    opt_spec: Any,
    conditions_override: Sequence[ConditionSpec] | None = None,
    analyze: bool = True,
) -> dict[str, Any]:
    """Fit one discrete role/order candidate and return comparable evidence."""

    role_has_i = role_candidate.I is not None
    role_has_b = role_candidate.B is not None
    parameter_fit = opt_spec.parameter_fit
    space = compile_parameter_space(
        list(parameter_fit.search_space),
        sim_spec=sim_spec,
        task=str(getattr(opt_spec, "task", "")),
        role_has_i=role_has_i,
        role_has_b=role_has_b,
    )
    conditions = list(conditions_override) if conditions_override is not None else extract_conditions(opt_spec)
    validate_conditions(conditions)
    train_conditions = sorted(
        [condition for condition in conditions if condition.split == "train" and condition.weight > 0],
        key=lambda condition: condition.name,
    )
    holdout_conditions = [
        condition for condition in conditions if condition.split == "holdout" and condition.weight > 0
    ]
    if holdout_conditions and any(bool(item.get("per_condition", False)) for item in space):
        raise ValueError(
            "holdout evaluation requires shared parameters; per_condition search items are not supported"
        )

    search_settings = parse_search_settings(parameter_fit)
    objective_cfg = dict(getattr(parameter_fit, "objective", {}) or {})
    penalties_cfg = dict(objective_cfg.get("penalties", {}) or {})
    lambda_prior = float(penalties_cfg.get("lambda_prior", 0.0))
    levels = _extract_levels(parameter_fit, len(train_conditions))
    condition_names = [condition.name for condition in train_conditions]
    condition_weights = {
        condition.name: max(float(condition.weight), 0.0) for condition in train_conditions
    }
    dimension = parameter_dimension(space, condition_names)

    analysis_cfg = _analysis_block(parameter_fit)
    preflight_cfg = analysis_cfg["preflight"]
    preflight_rows = (
        preflight_conditions(
            sim_spec=sim_spec,
            conditions=conditions,
            min_finite_ratio=float(preflight_cfg.get("min_finite_ratio", 0.6)),
        )
        if analyze and bool(preflight_cfg.get("enabled", True))
        else []
    )

    cache_cfg = analysis_cfg["cache"]
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_max_entries = max(int(cache_cfg.get("max_entries", 256)), 1)
    global_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    cache_stats = {
        "enabled": cache_enabled,
        "max_entries": cache_max_entries,
        "trial_hits": 0,
        "global_hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
    }

    def evaluate_subset(
        sample: Mapping[str, Any], active_conditions: list[ConditionSpec]
    ) -> tuple[dict[str, float], dict[str, float]]:
        rows: list[dict[str, float]] = []
        weights: list[float] = []
        per_condition_scores: dict[str, float] = {}
        per_condition_metrics: dict[str, dict[str, float]] = {}
        trial_cache = sample.setdefault("__trial_cache__", {})
        for condition in active_conditions:
            params = sample["per_condition"][condition.name]
            key = _cache_key(condition.name, params)
            if cache_enabled and key in trial_cache:
                cached = trial_cache[key]
                cache_stats["trial_hits"] += 1
            elif cache_enabled and key in global_cache:
                cached = global_cache[key]
                trial_cache[key] = cached
                global_cache.move_to_end(key)
                cache_stats["global_hits"] += 1
            else:
                trial_spec = prepare_condition(
                    base_spec=sim_spec,
                    role_candidate=role_candidate,
                    order_candidate=order_candidate,
                    condition=condition,
                    params=params,
                )
                components = evaluate_condition(trial_spec, objective_cfg)
                cached = {"components": components, "score": float(components["score_total"])}
                cache_stats["misses"] += 1
                if cache_enabled:
                    trial_cache[key] = cached
                    global_cache[key] = cached
                    global_cache.move_to_end(key)
                    cache_stats["stores"] += 1
                    while len(global_cache) > cache_max_entries:
                        global_cache.popitem(last=False)
                        cache_stats["evictions"] += 1
            components = dict(cached["components"])
            rows.append(components)
            weights.append(condition_weights[condition.name])
            per_condition_scores[condition.name] = float(cached["score"])
            per_condition_metrics[condition.name] = components
        prior_value = (
            lambda_prior * 0.5 * float(np.mean(np.asarray(sample["prior_terms"], dtype=float)))
            if sample["prior_terms"]
            else 0.0
        )
        sample["__condition_metrics__"] = per_condition_metrics
        return (
            _combine_weighted_components(
                component_rows=rows,
                weights=weights,
                prior_value=prior_value,
            ),
            per_condition_scores,
        )

    storage_cfg = search_settings.storage if conditions_override is None else {}
    search_runs = []
    for seed in repetition_seeds(search_settings):
        def random_objective(rng: np.random.Generator) -> tuple[float, dict[str, Any]]:
            sample = draw_parameter_sample(
                space=space,
                condition_names=condition_names,
                lambda_prior=lambda_prior,
                rng=rng,
            )
            components, scores = _evaluate_fidelity_levels(
                sample=sample,
                levels=levels,
                conditions=train_conditions,
                evaluate_subset=evaluate_subset,
            )
            return float(components["score_total"]), _fit_payload(sample, components, scores)

        def optuna_objective(trial: Any) -> tuple[float, dict[str, Any]]:
            sample = draw_parameter_sample(
                space=space,
                condition_names=condition_names,
                lambda_prior=lambda_prior,
                trial=trial,
            )

            def step_hook(step: int, components: Mapping[str, float]) -> None:
                trial.report(float(components["score_total"]), step=step)
                if trial.should_prune():
                    raise PruneRequested()

            components, scores = _evaluate_fidelity_levels(
                sample=sample,
                levels=levels,
                conditions=train_conditions,
                evaluate_subset=evaluate_subset,
                step_hook=step_hook,
            )
            return float(components["score_total"]), _fit_payload(sample, components, scores)

        study_kwargs: dict[str, Any] = {}
        storage_url = str(storage_cfg.get("url", "")).strip()
        if storage_url:
            fitted_inputs = [
                prepare_condition(
                    base_spec=sim_spec,
                    role_candidate=role_candidate,
                    order_candidate=order_candidate,
                    condition=condition,
                    params={},
                )
                for condition in train_conditions
            ]
            study_name = _persistent_study_name(
                str(storage_cfg.get("study_name", "")).strip(),
                fitted_inputs,
                {
                    "objective": objective_cfg,
                    "search_space": space,
                    "weights": condition_weights,
                    "levels": levels,
                    "method": search_settings.method,
                    "seed": seed,
                },
            )
            study_kwargs.update(
                storage=storage_url,
                study_name=study_name,
                load_if_exists=bool(storage_cfg.get("load_if_exists", False)),
            )
        search_runs.append(
            run_search(
                search_settings,
                seed=seed,
                dimension=dimension,
                random_objective=random_objective,
                optuna_objective=optuna_objective,
                study_kwargs=study_kwargs,
            )
        )

    best_run = min(search_runs, key=lambda run: run.best_score)
    best = best_run.best_payload
    best_params = dict(best["flat_params"])
    best_components = dict(best["components"])
    best_condition_scores = dict(best["condition_scores"])
    best_condition_metrics = dict(best["condition_metrics"])
    best_per_condition_params = dict(best["per_condition_params"])
    all_trace = [
        {"method": run.method, "repetition": repetition, **row}
        for repetition, run in enumerate(search_runs, start=1)
        for row in run.trace
    ]
    run_scores = np.asarray([run.best_score for run in search_runs], dtype=float)
    optimization = {
        "method": search_settings.method,
        "dimension": dimension,
        "repetitions": len(search_runs),
        "seeds": [run.seed for run in search_runs],
        "trial_count": int(sum(run.trial_count for run in search_runs)),
        "converged_repetitions": int(sum(run.converged for run in search_runs)),
        "all_repetitions_converged": bool(all(run.converged for run in search_runs)),
        "repeatability_assessed": bool(len(search_runs) >= 2),
        "termination_reasons": [run.termination_reason for run in search_runs],
        "best_score": float(np.min(run_scores)),
        "median_best_score": float(np.median(run_scores)),
        "best_score_range": (
            float(np.max(run_scores) - np.min(run_scores))
            if len(search_runs) >= 2
            else None
        ),
    }
    raw_loss_cfg = objective_cfg.get("loss", {"name": "mse"})
    if not isinstance(raw_loss_cfg, Mapping):
        raise ValueError("objective.loss must be a mapping with a named loss")
    loss_name = str(dict(raw_loss_cfg).get("name", "mse")).strip().lower()
    loss_standardized = bool(float(best_components.get("loss_standardized", 0.0)))
    loss_unit = "dimensionless"
    if not loss_standardized:
        loss_unit = "nm" if loss_name == "l1" else "nm^2"
    measurement_weight = float(dict(objective_cfg.get("weights", {}) or {}).get("measurement", 1.0))
    selection_baseline_scale = (
        measurement_weight if loss_name == "mse" and not loss_standardized else None
    )

    ident_cfg = analysis_cfg["identifiability"]
    parameter_paths = active_parameter_paths(space)
    ident_diag: dict[str, Any] = {
        "enabled": bool(ident_cfg.get("enabled", False)),
        "assessed": not parameter_paths,
        "warnings": [],
        "degeneracy_warning": False,
    }
    if analyze and bool(ident_cfg.get("enabled", False)) and parameter_paths and train_conditions:
        try:
            fitted_specs = [
                prepare_condition(
                    base_spec=sim_spec,
                    role_candidate=role_candidate,
                    order_candidate=order_candidate,
                    condition=condition,
                    params=best_per_condition_params[condition.name],
                )
                for condition in train_conditions
            ]
            ident_diag = compute_identifiability_diagnostics(
                fitted_specs[0],
                run_specs=fitted_specs,
                parameter_paths=parameter_paths,
                condition_weights=[condition.weight for condition in train_conditions],
                condition_names=[condition.name for condition in train_conditions],
                local_parameter_paths=[
                    str(item["name"]) for item in space if item.get("per_condition", False)
                ],
                parameter_bounds={
                    str(item["name"]): (float(item["low"]), float(item["high"]))
                    for item in space
                    if not item.get("per_condition", False)
                },
                relative_step=float(ident_cfg.get("relative_step", 1.0e-2)),
                low_sensitivity_threshold=float(ident_cfg.get("low_sensitivity_threshold", 1.0e-10)),
                correlation_threshold=float(ident_cfg.get("correlation_threshold", 0.98)),
            )
            ident_diag.update(enabled=True, assessed=True)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            ident_diag = {
                "enabled": True,
                "assessed": False,
                "warnings": [str(exc)],
                "degeneracy_warning": True,
                "error": str(exc),
            }

    holdout_scores: dict[str, float] = {}
    holdout_metrics: dict[str, dict[str, float]] = {}
    if holdout_conditions:
        shared_params = {
            str(item["name"]): float(best_params[str(item["name"])])
            for item in space
            if not bool(item.get("per_condition", False)) and str(item["name"]) in best_params
        }
        for condition in holdout_conditions:
            holdout_spec = prepare_condition(
                base_spec=sim_spec,
                role_candidate=role_candidate,
                order_candidate=order_candidate,
                condition=condition,
                params=shared_params,
            )
            components = evaluate_condition(holdout_spec, objective_cfg)
            if bool(float(components.get("loss_standardized", 0.0))) != loss_standardized:
                raise ValueError(
                    "train and holdout conditions must use the same loss standardization scale"
                )
            holdout_scores[condition.name] = float(components["score_total"])
            holdout_metrics[condition.name] = components

    return {
        "class_id": role_candidate.class_id,
        "roles": {"A": role_candidate.A, "I": role_candidate.I, "B": role_candidate.B},
        "quantity": "thickness",
        "unit": "nm",
        "effect_groups": role_candidate.effect_groups,
        "declared_effect_groups": role_candidate.effect_groups,
        "reduced_effect_groups": role_candidate.reduced_effect_groups,
        "effect_basis": "declared_state_model_roles",
        "orders": dict(order_candidate),
        "best_score": float(best_run.best_score),
        "best_params": best_params,
        "best_components": best_components,
        "condition_scores": best_condition_scores,
        "condition_metrics": best_condition_metrics,
        "holdout_scores": holdout_scores,
        "holdout_metrics": holdout_metrics,
        "condition_count": len(conditions),
        "train_condition_count": len(train_conditions),
        "holdout_condition_count": len(holdout_conditions),
        "fidelity_levels": list(levels),
        "search_space_count": dimension,
        "selection_metric": "loss_data",
        "selection_baseline_scale": selection_baseline_scale,
        "loss_definition": {
            "name": loss_name,
            "standardized": loss_standardized,
            "unit": loss_unit,
        },
        "optimization": optimization,
        "optimization_trace": all_trace,
        "cache_stats": dict(cache_stats),
        "fit_diagnostics": {
            "preflight": preflight_rows,
            "identifiability": ident_diag,
            "optimization": optimization,
        },
    }


__all__ = ["fit_candidate_parameters"]
