"""Continuous parameter fitting for one (roles, orders) candidate."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from deposim_schema import SimSpecV2
from deposim_sim.common.overrides import as_bool
from deposim_sim.identifiability import compute_identifiability_diagnostics

from .fit_conditions import (ConditionSpec, extract_conditions, validate_conditions,
    preflight_conditions, prepare_condition, evaluate_condition)

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for fit_optuna")


_ALLOWED_PARAMETER_GROUPS = {"surface_kinetics", "effective_transport", "measurement_or_interface"}
_ALLOWED_PARAMETER_STAGES = {"screening", "sobol", "calibration"}


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() else "_")
    clean = "".join(out).strip("_")
    return clean or "cond"


def _persistent_study_name(prefix: str, specs: list[SimSpecV2], settings: Mapping[str, Any]) -> str:
    """Do not resume trials evaluated for different roles, inputs, or objectives."""
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


def _hspace_value(item: Mapping[str, Any], *, trial: Any | None = None, rng: Any | None = None, name: str) -> float:
    low = float(item["low"])
    high = float(item["high"])
    if low > high:
        raise ValueError(f"search_space bounds invalid for {name}: low={low} > high={high}")
    kind = str(item.get("type", "loguniform")).strip().lower()

    if trial is not None:
        if kind in {"loguniform", "log"}:
            return float(trial.suggest_float(name, low, high, log=True))
        return float(trial.suggest_float(name, low, high))

    if rng is None:
        raise RuntimeError("rng is required when trial is not provided")
    if kind in {"loguniform", "log"}:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def _sample_search_space(
    search_space: list[dict[str, Any]],
    *,
    role_has_i: bool,
    role_has_b: bool,
    stage_filter: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in search_space:
        enabled = item.get("enabled", True)
        if not as_bool(enabled):
            continue
        stage = str(item.get("stage", "")).strip().lower()
        if stage_filter and stage and stage != stage_filter:
            continue
        cond = item.get("condition")
        if cond == "role_has_B" and not role_has_b:
            continue
        if cond == "role_has_no_B" and role_has_b:
            continue
        if cond == "role_has_I" and not role_has_i:
            continue
        out.append(dict(item))
    return out


def _validate_search_space_metadata(search_space: list[dict[str, Any]]) -> None:
    for item in search_space:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("search_space item requires non-empty name")
        group = str(item.get("group", "")).strip().lower()
        if group and group not in _ALLOWED_PARAMETER_GROUPS:
            raise ValueError(
                f"search_space item {name!r} has invalid group {group!r}; "
                f"expected one of {_ALLOWED_PARAMETER_GROUPS}"
            )
        stage = str(item.get("stage", "")).strip().lower()
        if stage and stage not in _ALLOWED_PARAMETER_STAGES:
            raise ValueError(
                f"search_space item {name!r} has invalid stage {stage!r}; "
                f"expected one of {_ALLOWED_PARAMETER_STAGES}"
            )
        symbol = item.get("symbol")
        if symbol is not None and not str(symbol).strip():
            raise ValueError(f"search_space item {name!r} has empty symbol")
        unit = item.get("unit")
        if unit is not None and not str(unit).strip():
            raise ValueError(f"search_space item {name!r} has empty unit")


def _search_space_stage_filter(opt_spec: Any) -> str | None:
    task = str(getattr(opt_spec, "task", "")).strip().lower()
    if task.startswith("fit"):
        return "calibration"
    return None


def _validate_transport_search_space(
    *,
    sim_spec: SimSpecV2,
    search_space: list[dict[str, Any]],
    role_has_b: bool,
) -> None:
    transport = dict(getattr(sim_spec.model.params, "transport", {}) or {})
    km_source = str(transport.get("km_source", "fit_scalar")).strip().lower()
    if km_source != "from_cfd_flux_sink":
        return

    names = {str(item.get("name", "")).strip() for item in search_space}
    forbidden = {"model.params.transport.km_A", "model.params.transport.km_B"}
    used_forbidden = sorted(name for name in names if name in forbidden)
    if used_forbidden:
        joined = ", ".join(used_forbidden)
        raise ValueError(
            "km_source=from_cfd_flux_sink forbids direct km optimization. "
            f"Remove: {joined}"
        )

    if "model.params.transport.gamma_km_A" not in names:
        raise ValueError(
            "km_source=from_cfd_flux_sink requires model.params.transport.gamma_km_A in opt.parameter_fit.search_space"
        )
    if role_has_b and "model.params.transport.gamma_km_B" not in names:
        raise ValueError(
            "km_source=from_cfd_flux_sink with role B requires model.params.transport.gamma_km_B in search_space"
        )


def _extract_levels(opt_spec: Any, condition_count: int) -> list[int]:
    fidelity = dict(getattr(opt_spec.parameter_fit, "fidelity", {}) or {})
    raw_levels = fidelity.get("levels", [condition_count])
    if isinstance(raw_levels, (int, float, str)):
        raw_levels = [raw_levels]

    levels: list[int] = []
    for item in list(raw_levels):
        try:
            val = int(item)
        except Exception as exc:
            raise ValueError(f"fidelity level must be int-like: {item!r}") from exc
        val = max(1, min(val, condition_count))
        levels.append(val)

    if not levels:
        levels = [condition_count]
    levels = sorted(set(levels))
    if levels[-1] != condition_count:
        levels.append(condition_count)
    return levels


def _resolve_sampler(name: str, seed: int) -> Any:
    if optuna is None:
        return None
    key = str(name).strip().lower()
    if key == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def _resolve_pruner(name: str) -> Any:
    if optuna is None:
        return None
    key = str(name).strip().lower()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()


def _analysis_block(opt_spec: Any) -> dict[str, Any]:
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
    raw = dict(getattr(opt_spec.parameter_fit, "analysis", {}) or {})
    out = dict(defaults)
    for key in defaults:
        row = dict(raw.get(key, {}) or {})
        merged = dict(defaults[key])
        merged.update(row)
        out[key] = merged
    return out


def _draw_sample(
    *,
    search_space: list[dict[str, Any]],
    condition_names: list[str],
    lambda_prior: float,
    trial: Any | None = None,
    rng: Any | None = None,
) -> dict[str, Any]:
    per_condition: dict[str, dict[str, float]] = {name: {} for name in condition_names}
    flat_params: dict[str, float] = {}
    prior_terms: list[float] = []

    for item in search_space:
        path = str(item["name"])
        per_cond = bool(item.get("per_condition", False))

        if not per_cond:
            val = _hspace_value(item, trial=trial, rng=rng, name=path)
            flat_params[path] = float(val)
            for cond in condition_names:
                per_condition[cond][path] = float(val)
            continue

        hier = dict(item.get("hierarchical", {}) or {})
        mode = str(hier.get("mode", "log_offset")).strip().lower()
        if mode != "log_offset":
            raise ValueError(f"unsupported hierarchical mode for {path}: {mode}")

        sigma = float(hier.get("sigma", 0.5))
        delta_low = float(hier.get("delta_low", -2.0))
        delta_high = float(hier.get("delta_high", 2.0))
        if delta_low > delta_high:
            raise ValueError(f"delta bounds invalid for {path}: [{delta_low}, {delta_high}]")

        base_key = f"{path}__base"
        base = float(_hspace_value(item, trial=trial, rng=rng, name=base_key))
        if base <= 0.0:
            raise ValueError(f"log_offset requires positive base parameter for {path}")
        flat_params[base_key] = base

        for cond in condition_names:
            safe = _safe_name(cond)
            delta_key = f"{path}__delta__{safe}"
            if trial is not None:
                delta = float(trial.suggest_float(delta_key, delta_low, delta_high))
            else:
                if rng is None:
                    raise RuntimeError("rng is required when trial is not provided")
                delta = float(rng.uniform(delta_low, delta_high))
            flat_params[delta_key] = delta
            per_condition[cond][path] = float(base * math.exp(delta))
            if sigma > 0.0 and lambda_prior > 0.0:
                prior_terms.append((delta / sigma) ** 2)

    return {
        "flat_params": flat_params,
        "per_condition": per_condition,
        "prior_terms": prior_terms,
    }


def _combine_weighted_components(
    *,
    component_rows: list[dict[str, float]],
    weights: list[float],
    prior_value: float,
    complexity_value: float,
) -> dict[str, float]:
    if not component_rows:
        return {
            "mse_nm2": float("nan"),
            "rmse_nm": float("nan"),
            "mae_nm": float("nan"),
            "max_abs_nm": float("nan"),
            "loss_data": 0.0,
            "penalty_solver": 0.0,
            "penalty_phys": 0.0,
            "penalty_prior": float(prior_value),
            "penalty_complexity": float(complexity_value),
            "score_total": float(prior_value + complexity_value),
        }

    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)
    w = w / float(np.sum(w))

    out = {
        "mse_nm2": 0.0,
        "mae_nm": 0.0,
        "max_abs_nm": 0.0,
        "loss_data": 0.0,
        "penalty_solver": 0.0,
        "penalty_phys": 0.0,
        "penalty_profile": 0.0,
    }
    for idx, row in enumerate(component_rows):
        out["mse_nm2"] += float(w[idx]) * float(row.get("mse_nm2", 0.0))
        out["mae_nm"] += float(w[idx]) * float(row.get("mae_nm", 0.0))
        out["max_abs_nm"] = max(out["max_abs_nm"], float(row.get("max_abs_nm", 0.0)))
        out["loss_data"] += float(w[idx]) * float(row["loss_data"])
        out["penalty_solver"] += float(w[idx]) * float(row["penalty_solver"])
        out["penalty_phys"] += float(w[idx]) * float(row["penalty_phys"])
        out["penalty_profile"] += float(w[idx]) * float(row.get("penalty_profile", 0.0))

    out["penalty_prior"] = float(prior_value)
    out["penalty_complexity"] = float(complexity_value)
    out["rmse_nm"] = float(np.sqrt(max(out["mse_nm2"], 0.0)))
    out["score_total"] = (
        out["loss_data"]
        + out["penalty_solver"]
        + out["penalty_phys"]
        + out["penalty_prior"]
        + out["penalty_profile"]
        + out["penalty_complexity"]
    )
    return out


def _cache_key(condition_name: str, params: Mapping[str, float]) -> tuple[Any, ...]:
    rounded = []
    for key, value in sorted(params.items()):
        val = float(value)
        if np.isfinite(val):
            rounded.append((str(key), round(val, 12)))
        else:
            rounded.append((str(key), str(val)))
    return (str(condition_name), tuple(rounded))


def _evaluate_fidelity_levels(
    *,
    sample: Mapping[str, Any],
    levels: Sequence[int],
    conditions: Sequence[ConditionSpec],
    evaluate_subset: Any,
    step_hook: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    final_components: dict[str, float] = {}
    final_cond_scores: dict[str, float] = {}
    for step_idx, level in enumerate(levels):
        subset = list(conditions[: int(level)])
        final_components, final_cond_scores = evaluate_subset(sample, subset)
        if step_hook is not None:
            step_hook(step_idx, final_components)
    return final_components, final_cond_scores


def fit_candidate_with_optuna(
    *,
    sim_spec: SimSpecV2,
    role_candidate: Any,
    order_candidate: dict[str, int],
    opt_spec: Any,
    conditions_override: Sequence[ConditionSpec] | None = None,
    analyze: bool = True,
) -> dict[str, Any]:
    """Fit one discrete candidate and return best score/params/components."""

    _require_numpy()
    role_has_i = role_candidate.I is not None
    role_has_b = role_candidate.B is not None
    stage_filter = _search_space_stage_filter(opt_spec)

    search_space = _sample_search_space(
        list(opt_spec.parameter_fit.search_space),
        role_has_i=role_has_i,
        role_has_b=role_has_b,
        stage_filter=stage_filter,
    )
    _validate_search_space_metadata(search_space)
    _validate_transport_search_space(sim_spec=sim_spec, search_space=search_space, role_has_b=role_has_b)
    conditions = list(conditions_override) if conditions_override is not None else extract_conditions(opt_spec)
    validate_conditions(conditions)
    train_conditions = sorted([cond for cond in conditions if cond.split == "train" and cond.weight > 0], key=lambda c: c.name)
    holdout_conditions = [cond for cond in conditions if cond.split == "holdout" and cond.weight > 0]
    if holdout_conditions and any(bool(item.get("per_condition", False)) for item in search_space):
        raise ValueError("holdout evaluation currently requires shared parameters; per_condition search items are not supported")

    n_trials = int(opt_spec.parameter_fit.n_trials_per_candidate)
    seed = int(opt_spec.parameter_fit.seed)

    objective_cfg = dict(getattr(opt_spec.parameter_fit, "objective", {}) or {})
    penalties_cfg = dict(objective_cfg.get("penalties", {}) or {})
    lambda_prior = float(penalties_cfg.get("lambda_prior", 0.0))
    lambda_complex_obj = float(penalties_cfg.get("lambda_complex", 0.0))
    lambda_role_obj = float(penalties_cfg.get("lambda_role", 0.0))
    class_complexity = dict(getattr(opt_spec.class_compare, "complexity_penalty", {}) or {})
    lambda_role_cfg = float(class_complexity.get("lambda_role", 0.0))
    lambda_complex = lambda_complex_obj + (lambda_role_obj if lambda_role_obj > 0.0 else lambda_role_cfg)

    levels = _extract_levels(opt_spec, len(train_conditions))
    condition_names = [cond.name for cond in train_conditions]
    condition_weight_map = {cond.name: max(float(cond.weight), 0.0) for cond in train_conditions}

    analysis_cfg = _analysis_block(opt_spec)
    preflight_cfg = dict(analysis_cfg.get("preflight", {}) or {})
    preflight_enabled = analyze and bool(preflight_cfg.get("enabled", True))
    preflight_rows: list[dict[str, Any]] = []
    if preflight_enabled:
        preflight_rows = preflight_conditions(
            sim_spec=sim_spec,
            conditions=conditions,
            min_finite_ratio=float(preflight_cfg.get("min_finite_ratio", 0.6)),
        )

    cache_cfg = dict(analysis_cfg.get("cache", {}) or {})
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

    def evaluate_subset(sample: Mapping[str, Any], active_conditions: list[ConditionSpec]) -> tuple[dict[str, float], dict[str, float]]:
        rows: list[dict[str, float]] = []
        weights: list[float] = []
        per_cond_total: dict[str, float] = {}
        per_cond_metrics: dict[str, dict[str, float]] = {}
        trial_cache = sample.setdefault("__trial_cache__", {})

        for cond in active_conditions:
            params_for_cond = sample["per_condition"][cond.name]
            key = _cache_key(cond.name, params_for_cond)

            cached = None
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
                    condition=cond,
                    params=params_for_cond,
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
            weights.append(condition_weight_map[cond.name])
            per_cond_total[cond.name] = float(cached["score"])
            per_cond_metrics[cond.name] = dict(components)

        prior_value = float(lambda_prior) * 0.5 * float(np.mean(np.asarray(sample["prior_terms"], dtype=float))) if sample["prior_terms"] else 0.0
        complexity_value = float(lambda_complex) * float(int(role_has_i) + int(role_has_b))
        merged = _combine_weighted_components(
            component_rows=rows,
            weights=weights,
            prior_value=prior_value,
            complexity_value=complexity_value,
        )
        sample["__condition_metrics__"] = per_cond_metrics
        return merged, per_cond_total

    best_score = float("inf")
    best_params: dict[str, float] = {}
    best_components: dict[str, float] = {}
    best_condition_scores: dict[str, float] = {}
    best_condition_metrics: dict[str, dict[str, float]] = {}
    best_per_condition_params: dict[str, dict[str, float]] = {}
    study_trial_count = 0

    engine_name = str(getattr(opt_spec.parameter_fit, "engine", "optuna")).strip().lower()
    pruner_name = str(getattr(opt_spec.parameter_fit, "pruner", "none")).strip().lower()
    # Each validation fold starts afresh; full-training trials must never leak
    # into a fold through a resumed Optuna study.
    storage_cfg = {} if conditions_override is not None else dict(getattr(opt_spec.parameter_fit, "storage", {}) or {})
    storage_url = str(storage_cfg.get("url", "")).strip()

    if optuna is None and pruner_name != "none":
        raise RuntimeError("Optuna pruner requested but optuna is unavailable")
    if optuna is None and storage_url:
        raise RuntimeError("Optuna storage requested but optuna is unavailable")

    use_optuna = optuna is not None and engine_name == "optuna"

    if use_optuna:
        sampler_name = str(getattr(opt_spec.parameter_fit, "sampler", "tpe"))
        sampler = _resolve_sampler(sampler_name, seed)
        pruner = _resolve_pruner(pruner_name)
        study_kwargs: dict[str, Any] = {
            "direction": "minimize",
            "sampler": sampler,
            "pruner": pruner,
        }
        if storage_url:
            fitted_inputs = [prepare_condition(
                base_spec=sim_spec, role_candidate=role_candidate,
                order_candidate=order_candidate, condition=cond, params={},
            ) for cond in train_conditions]
            study_name = _persistent_study_name(
                str(storage_cfg.get("study_name", "")).strip(), fitted_inputs,
                {"objective": objective_cfg, "search_space": search_space,
                 "weights": condition_weight_map, "levels": levels, "lambda_complex": lambda_complex},
            )
            study_kwargs["storage"] = storage_url
            study_kwargs["study_name"] = study_name
            study_kwargs["load_if_exists"] = bool(storage_cfg.get("load_if_exists", False))

        study = optuna.create_study(**study_kwargs)

        def objective(trial: Any) -> float:
            sample = _draw_sample(
                search_space=search_space,
                condition_names=condition_names,
                lambda_prior=lambda_prior,
                trial=trial,
            )

            def _step_hook(step_idx: int, components: Mapping[str, float]) -> None:
                trial.report(float(components["score_total"]), step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            final_components, final_cond_scores = _evaluate_fidelity_levels(
                sample=sample,
                levels=levels,
                conditions=train_conditions,
                evaluate_subset=evaluate_subset,
                step_hook=_step_hook,
            )

            trial.set_user_attr("components", dict(final_components))
            trial.set_user_attr("condition_scores", dict(final_cond_scores))
            trial.set_user_attr("condition_metrics", dict(sample.get("__condition_metrics__", {})))
            trial.set_user_attr("flat_params", dict(sample["flat_params"]))
            trial.set_user_attr(
                "per_condition_params",
                {k: {pk: float(pv) for pk, pv in v.items()} for k, v in sample["per_condition"].items()},
            )
            return float(final_components["score_total"])

        study.optimize(objective, n_trials=n_trials)
        if not study.best_trials:
            raise RuntimeError("All optuna trials were pruned or failed; no best trial is available")
        best_score = float(study.best_value)
        study_trial_count = len(study.trials)
        attrs = dict(study.best_trial.user_attrs)
        best_params = {str(k): float(v) for k, v in dict(attrs.get("flat_params", study.best_params)).items()}
        best_components = {str(k): float(v) for k, v in dict(attrs.get("components", {})).items()}
        best_condition_scores = {str(k): float(v) for k, v in dict(attrs.get("condition_scores", {})).items()}
        best_condition_metrics = {
            str(k): {str(mk): float(mv) for mk, mv in dict(v).items()}
            for k, v in dict(attrs.get("condition_metrics", {})).items()
            if isinstance(v, Mapping)
        }
        best_per_condition_params = {
            str(k): {str(pk): float(pv) for pk, pv in dict(v).items()}
            for k, v in dict(attrs.get("per_condition_params", {})).items()
            if isinstance(v, Mapping)
        }
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            sample = _draw_sample(
                search_space=search_space,
                condition_names=condition_names,
                lambda_prior=lambda_prior,
                rng=rng,
            )
            comps, cond_scores = _evaluate_fidelity_levels(
                sample=sample,
                levels=levels,
                conditions=train_conditions,
                evaluate_subset=evaluate_subset,
            )
            score = float(comps["score_total"])
            if score < best_score:
                best_score = score
                best_params = {str(k): float(v) for k, v in sample["flat_params"].items()}
                best_components = {str(k): float(v) for k, v in comps.items()}
                best_condition_scores = {str(k): float(v) for k, v in cond_scores.items()}
                best_condition_metrics = {
                    str(k): {str(mk): float(mv) for mk, mv in dict(v).items()}
                    for k, v in dict(sample.get("__condition_metrics__", {})).items()
                }
                best_per_condition_params = {
                    str(k): {str(pk): float(pv) for pk, pv in dict(v).items()}
                    for k, v in sample["per_condition"].items()
                }
        study_trial_count = int(n_trials)

    ident_cfg = dict(analysis_cfg.get("identifiability", {}) or {})
    parameter_paths = [str(item["name"]) for item in search_space if float(item["high"]) > float(item["low"])]
    ident_diag: dict[str, Any] = {
        "enabled": bool(ident_cfg.get("enabled", False)),
        "assessed": not parameter_paths,
        "warnings": [],
        "degeneracy_warning": False,
    }
    if analyze and bool(ident_cfg.get("enabled", False)):
        # Inspect every estimated parameter on all training observations. The old
        # max_paths cap silently hid dependence on parameters later in the list.
        if parameter_paths and train_conditions:
            try:
                fitted_specs = [
                    prepare_condition(
                        base_spec=sim_spec, role_candidate=role_candidate,
                        order_candidate=order_candidate, condition=cond,
                        params=best_per_condition_params[cond.name],
                    ) for cond in train_conditions
                ]
                ident_diag = compute_identifiability_diagnostics(
                    fitted_specs[0], run_specs=fitted_specs,
                    parameter_paths=parameter_paths,
                    condition_weights=[cond.weight for cond in train_conditions],
                    condition_names=[cond.name for cond in train_conditions],
                    local_parameter_paths=[str(item["name"]) for item in search_space if item.get("per_condition", False)],
                    parameter_bounds={str(item["name"]): (float(item["low"]), float(item["high"])) for item in search_space if not item.get("per_condition", False)},
                    relative_step=float(ident_cfg.get("relative_step", 1.0e-2)),
                    low_sensitivity_threshold=float(ident_cfg.get("low_sensitivity_threshold", 1.0e-10)),
                    correlation_threshold=float(ident_cfg.get("correlation_threshold", 0.98)),
                )
                ident_diag["enabled"] = True
                ident_diag["assessed"] = True
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                ident_diag = {"enabled": True, "warnings": [str(exc)], "degeneracy_warning": True, "error": str(exc)}

    holdout_scores: dict[str, float] = {}
    holdout_metrics: dict[str, dict[str, float]] = {}
    if holdout_conditions:
        shared_best_params = {
            str(item["name"]): float(best_params[str(item["name"])])
            for item in search_space
            if not bool(item.get("per_condition", False)) and str(item["name"]) in best_params
        }
        for cond in holdout_conditions:
            holdout_spec = prepare_condition(
                base_spec=sim_spec,
                role_candidate=role_candidate,
                order_candidate=order_candidate,
                condition=cond,
                params=shared_best_params,
            )
            holdout_components = evaluate_condition(holdout_spec, objective_cfg)
            holdout_scores[cond.name] = float(holdout_components["score_total"])
            holdout_metrics[cond.name] = dict(holdout_components)

    return {
        "class_id": role_candidate.class_id,
        "roles": {"A": role_candidate.A, "I": role_candidate.I, "B": role_candidate.B},
        "quantity": "thickness", "unit": "nm",
        "effect_groups": role_candidate.effect_groups,
        "declared_effect_groups": role_candidate.effect_groups,
        "reduced_effect_groups": role_candidate.reduced_effect_groups,
        "effect_basis": "declared_state_model_roles",
        "orders": dict(order_candidate),
        "best_score": float(best_score),
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
        "search_space_count": len(search_space),
        "study_trial_count": int(study_trial_count),
        "cache_stats": dict(cache_stats),
        "fit_diagnostics": {
            "preflight": preflight_rows,
            "identifiability": ident_diag,
        },
    }


__all__ = ["fit_candidate_with_optuna"]
