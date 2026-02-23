"""Continuous parameter fitting for one (roles, orders) candidate."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from deposim_schema import SimSpecV2
from deposim_sim.identifiability import compute_identifiability_diagnostics
from deposim_sim.pipeline import run_aib_from_spec

from .objective import evaluate_candidate_score

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


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    weight: float
    fluent_file: str | None
    measurement_file: str | None
    overrides: tuple[str, ...]
    keys: dict[str, Any]
    align: dict[str, Any]


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() else "_")
    clean = "".join(out).strip("_")
    return clean or "cond"


def _parse_override_value(raw: str) -> Any:
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [item.strip().strip("'\"") for item in body.split(",") if item.strip()]
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return raw


def _set_attr_path(root: Any, path: str, value: Any) -> None:
    cleaned = str(path)
    if cleaned.startswith("sim."):
        cleaned = cleaned[4:]
    parts = [p for p in cleaned.split(".") if p]
    if not parts:
        raise ValueError(f"invalid parameter path: {path!r}")

    cursor = root
    for key in parts[:-1]:
        if isinstance(cursor, dict):
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        else:
            cursor = getattr(cursor, key)

    if isinstance(cursor, dict):
        cursor[parts[-1]] = value
    else:
        setattr(cursor, parts[-1], value)


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
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in search_space:
        cond = item.get("condition")
        if cond == "role_has_B" and not role_has_b:
            continue
        if cond == "role_has_I" and not role_has_i:
            continue
        out.append(dict(item))
    return out


def _extract_conditions(opt_spec: Any) -> list[ConditionSpec]:
    measurement_cfg = dict(getattr(opt_spec, "measurement", {}) or {})
    default_keys = dict(measurement_cfg.get("keys", {"h": "h_nm", "xy": "xy"}))
    default_align = dict(measurement_cfg.get("align", {}))

    raw_conditions = measurement_cfg.get("conditions")
    if isinstance(raw_conditions, Sequence) and not isinstance(raw_conditions, (str, bytes)) and raw_conditions:
        conditions: list[ConditionSpec] = []
        for idx, row in enumerate(raw_conditions, start=1):
            item = dict(row or {})
            name = str(item.get("name", f"cond_{idx}"))
            conditions.append(
                ConditionSpec(
                    name=name,
                    weight=float(item.get("weight", 1.0)),
                    fluent_file=str(item.get("fluent_file", "")).strip() or None,
                    measurement_file=str(item.get("measurement_file", item.get("file", ""))).strip() or None,
                    overrides=tuple(str(x) for x in list(item.get("overrides", []))),
                    keys=dict(item.get("keys", default_keys)),
                    align=dict(item.get("align", default_align)),
                )
            )
        return conditions

    single = ConditionSpec(
        name="cond_1",
        weight=1.0,
        fluent_file=None,
        measurement_file=str(measurement_cfg.get("file", "")).strip() or None,
        overrides=tuple(),
        keys=default_keys,
        align=default_align,
    )
    return [single]


def _validate_conditions(conditions: list[ConditionSpec]) -> None:
    if not conditions:
        raise ValueError("at least one optimization condition is required")
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError("opt.measurement.conditions names must be unique")

    total = float(sum(max(c.weight, 0.0) for c in conditions))
    if total <= 0.0:
        raise ValueError("condition weights must contain at least one positive weight")

    for cond in conditions:
        if cond.fluent_file:
            path = Path(cond.fluent_file)
            if not path.exists():
                raise FileNotFoundError(f"condition fluent_file not found: {path}")
        if cond.measurement_file:
            path = Path(cond.measurement_file)
            if not path.exists():
                raise FileNotFoundError(f"condition measurement_file not found: {path}")


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
        "role_stability": {"enabled": True, "topk_window": 10, "score_epsilon": 1.0e-6},
        "identifiability": {
            "enabled": False,
            "max_paths": 3,
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


def _preflight_conditions(
    *,
    sim_spec: SimSpecV2,
    conditions: list[ConditionSpec],
    min_finite_ratio: float,
) -> list[dict[str, Any]]:
    min_ratio = float(min_finite_ratio)
    if min_ratio <= 0.0 or min_ratio > 1.0:
        raise ValueError(f"analysis.preflight.min_finite_ratio must be in (0,1], got {min_ratio}")

    fluent_keys = sim_spec.inputs.fluent.keys
    fluent_xy_key = str(getattr(fluent_keys, "xy", "xy"))
    fluent_cref_key = str(getattr(fluent_keys, "cref", "cref"))

    rows: list[dict[str, Any]] = []
    for cond in conditions:
        fluent_path = Path(str(cond.fluent_file or sim_spec.inputs.fluent.file))
        if not fluent_path.exists():
            raise FileNotFoundError(f"preflight fluent file not found: {fluent_path}")

        with np.load(fluent_path, allow_pickle=False) as data:
            if fluent_xy_key not in data.files or fluent_cref_key not in data.files:
                raise ValueError(f"preflight fluent file missing required keys: {fluent_path}")
            xy = np.asarray(data[fluent_xy_key], dtype=float)
            cref = np.asarray(data[fluent_cref_key], dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"preflight fluent xy must be shape [n,2], got {xy.shape} in {fluent_path}")
        n_pts = int(xy.shape[0])
        if cref.ndim == 2:
            cref_n_pts = int(cref.shape[0])
        elif cref.ndim == 3:
            cref_n_pts = int(cref.shape[1])
        else:
            raise ValueError(f"preflight cref must be 2D or 3D, got {cref.shape} in {fluent_path}")
        if cref_n_pts != n_pts:
            raise ValueError(f"preflight cref/xy point mismatch in {fluent_path}: {cref_n_pts} vs {n_pts}")

        cref_finite_ratio = float(np.mean(np.isfinite(cref))) if cref.size else 0.0
        if cref_finite_ratio < min_ratio:
            raise ValueError(
                f"preflight fluent finite ratio below threshold for {cond.name}: "
                f"{cref_finite_ratio:.4f} < {min_ratio:.4f}"
            )

        meas_finite_ratio = None
        measurement_path = Path(str(cond.measurement_file)) if cond.measurement_file else None
        if measurement_path is not None and measurement_path.exists():
            h_key = str(cond.keys.get("h", "h_nm"))
            xy_key = str(cond.keys.get("xy", "xy"))
            with np.load(measurement_path, allow_pickle=False) as data:
                if h_key not in data.files:
                    raise ValueError(f"preflight measurement file missing key {h_key!r}: {measurement_path}")
                h = np.asarray(data[h_key], dtype=float).reshape(-1)
                if xy_key in data.files:
                    xy_meas = np.asarray(data[xy_key], dtype=float)
                    if xy_meas.ndim != 2 or xy_meas.shape[1] != 2 or xy_meas.shape[0] != h.shape[0]:
                        raise ValueError(f"preflight measurement xy shape mismatch in {measurement_path}")

            meas_finite_ratio = float(np.mean(np.isfinite(h))) if h.size else 0.0
            if meas_finite_ratio < min_ratio:
                raise ValueError(
                    f"preflight measurement finite ratio below threshold for {cond.name}: "
                    f"{meas_finite_ratio:.4f} < {min_ratio:.4f}"
                )

        rows.append(
            {
                "condition": cond.name,
                "fluent_file": str(fluent_path),
                "measurement_file": str(measurement_path) if measurement_path else "",
                "n_points": n_pts,
                "fluent_finite_ratio": cref_finite_ratio,
                "measurement_finite_ratio": meas_finite_ratio,
            }
        )
    return rows


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


def _apply_candidate_to_spec(
    *,
    base_spec: SimSpecV2,
    role_candidate: Any,
    order_candidate: Mapping[str, int],
    condition: ConditionSpec,
    params: Mapping[str, float],
) -> SimSpecV2:
    trial_spec = deepcopy(base_spec)
    trial_spec.roles.A = role_candidate.A
    trial_spec.roles.I = role_candidate.I
    trial_spec.roles.B = role_candidate.B

    trial_spec.model.orders.adsorption_site_order = int(order_candidate["adsorption_site_order"])
    trial_spec.model.orders.reaction_site_order_A = int(order_candidate["reaction_site_order_A"])
    trial_spec.model.orders.reaction_site_order_star = int(order_candidate["reaction_site_order_star"])

    if condition.fluent_file:
        trial_spec.inputs.fluent.file = str(condition.fluent_file)

    meas_file = str(condition.measurement_file or "").strip()
    trial_spec.measurement.enabled = bool(meas_file)
    trial_spec.measurement.file = meas_file
    trial_spec.measurement.keys = dict(condition.keys)
    trial_spec.measurement.align = dict(condition.align)

    for key, value in params.items():
        _set_attr_path(trial_spec, key, float(value))

    for override in condition.overrides:
        text = str(override).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"condition override must use key=value format: {text!r}")
        key, raw = text.split("=", 1)
        _set_attr_path(trial_spec, key.strip(), _parse_override_value(raw))

    return trial_spec


def _combine_weighted_components(
    *,
    component_rows: list[dict[str, float]],
    weights: list[float],
    prior_value: float,
    complexity_value: float,
) -> dict[str, float]:
    if not component_rows:
        return {
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
        "loss_data": 0.0,
        "penalty_solver": 0.0,
        "penalty_phys": 0.0,
    }
    for idx, row in enumerate(component_rows):
        out["loss_data"] += float(w[idx]) * float(row["loss_data"])
        out["penalty_solver"] += float(w[idx]) * float(row["penalty_solver"])
        out["penalty_phys"] += float(w[idx]) * float(row["penalty_phys"])

    out["penalty_prior"] = float(prior_value)
    out["penalty_complexity"] = float(complexity_value)
    out["score_total"] = (
        out["loss_data"]
        + out["penalty_solver"]
        + out["penalty_phys"]
        + out["penalty_prior"]
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


def fit_candidate_with_optuna(
    *,
    sim_spec: SimSpecV2,
    role_candidate: Any,
    order_candidate: dict[str, int],
    opt_spec: Any,
) -> dict[str, Any]:
    """Fit one discrete candidate and return best score/params/components."""

    _require_numpy()
    role_has_i = role_candidate.I is not None
    role_has_b = role_candidate.B is not None

    search_space = _sample_search_space(
        list(opt_spec.parameter_fit.search_space),
        role_has_i=role_has_i,
        role_has_b=role_has_b,
    )
    conditions = _extract_conditions(opt_spec)
    _validate_conditions(conditions)

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

    levels = _extract_levels(opt_spec, len(conditions))
    condition_names = [cond.name for cond in conditions]
    condition_weight_map = {cond.name: max(float(cond.weight), 0.0) for cond in conditions}

    analysis_cfg = _analysis_block(opt_spec)
    preflight_cfg = dict(analysis_cfg.get("preflight", {}) or {})
    preflight_enabled = bool(preflight_cfg.get("enabled", True))
    preflight_rows: list[dict[str, Any]] = []
    if preflight_enabled:
        preflight_rows = _preflight_conditions(
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
                trial_spec = _apply_candidate_to_spec(
                    base_spec=sim_spec,
                    role_candidate=role_candidate,
                    order_candidate=order_candidate,
                    condition=cond,
                    params=params_for_cond,
                )
                result = run_aib_from_spec(trial_spec)
                components = evaluate_candidate_score(
                    residual_nm=result.fields.get("residual_nm"),
                    fields=result.fields,
                    diagnostics=result.diagnostics,
                    role_has_i=role_has_i,
                    role_has_b=role_has_b,
                    objective=objective_cfg,
                    lambda_complex=0.0,
                    prior_terms=None,
                )
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

        prior_value = float(lambda_prior) * 0.5 * float(np.mean(np.asarray(sample["prior_terms"], dtype=float))) if sample["prior_terms"] else 0.0
        complexity_value = float(lambda_complex) * float(int(role_has_i) + int(role_has_b))
        merged = _combine_weighted_components(
            component_rows=rows,
            weights=weights,
            prior_value=prior_value,
            complexity_value=complexity_value,
        )
        return merged, per_cond_total

    best_score = float("inf")
    best_params: dict[str, float] = {}
    best_components: dict[str, float] = {}
    best_condition_scores: dict[str, float] = {}
    best_per_condition_params: dict[str, dict[str, float]] = {}
    study_trial_count = 0

    engine_name = str(getattr(opt_spec.parameter_fit, "engine", "optuna")).strip().lower()
    pruner_name = str(getattr(opt_spec.parameter_fit, "pruner", "none")).strip().lower()
    storage_cfg = dict(getattr(opt_spec.parameter_fit, "storage", {}) or {})
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
            study_name = str(storage_cfg.get("study_name", "")).strip()
            if not study_name:
                study_name = (
                    f"aib_{role_candidate.class_id}_"
                    f"m{order_candidate['adsorption_site_order']}_"
                    f"pa{order_candidate['reaction_site_order_A']}_"
                    f"ps{order_candidate['reaction_site_order_star']}"
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

            final_components: dict[str, float] = {}
            final_cond_scores: dict[str, float] = {}
            for step_idx, level in enumerate(levels):
                subset = conditions[:level]
                final_components, final_cond_scores = evaluate_subset(sample, subset)
                trial.report(float(final_components["score_total"]), step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trial.set_user_attr("components", dict(final_components))
            trial.set_user_attr("condition_scores", dict(final_cond_scores))
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
            comps: dict[str, float] = {}
            cond_scores: dict[str, float] = {}
            for level in levels:
                subset = conditions[:level]
                comps, cond_scores = evaluate_subset(sample, subset)
            score = float(comps["score_total"])
            if score < best_score:
                best_score = score
                best_params = {str(k): float(v) for k, v in sample["flat_params"].items()}
                best_components = {str(k): float(v) for k, v in comps.items()}
                best_condition_scores = {str(k): float(v) for k, v in cond_scores.items()}
                best_per_condition_params = {
                    str(k): {str(pk): float(pv) for pk, pv in dict(v).items()}
                    for k, v in sample["per_condition"].items()
                }
        study_trial_count = int(n_trials)

    ident_cfg = dict(analysis_cfg.get("identifiability", {}) or {})
    ident_diag: dict[str, Any] = {
        "enabled": bool(ident_cfg.get("enabled", False)),
        "warnings": [],
        "degeneracy_warning": False,
    }
    if bool(ident_cfg.get("enabled", False)):
        parameter_paths: list[str] = []
        max_paths = max(int(ident_cfg.get("max_paths", 3)), 1)
        for item in search_space:
            path = str(item.get("name", "")).strip()
            if path and path not in parameter_paths:
                parameter_paths.append(path)
            if len(parameter_paths) >= max_paths:
                break

        if parameter_paths and conditions:
            try:
                base_cond = conditions[0]
                base_params = dict(best_per_condition_params.get(base_cond.name, {}))
                if not base_params:
                    for path in parameter_paths:
                        if path in best_params:
                            base_params[path] = float(best_params[path])
                trial_spec = _apply_candidate_to_spec(
                    base_spec=sim_spec,
                    role_candidate=role_candidate,
                    order_candidate=order_candidate,
                    condition=base_cond,
                    params=base_params,
                )
                ident_diag = compute_identifiability_diagnostics(
                    trial_spec,
                    parameter_paths=parameter_paths,
                    relative_step=float(ident_cfg.get("relative_step", 1.0e-2)),
                    low_sensitivity_threshold=float(ident_cfg.get("low_sensitivity_threshold", 1.0e-10)),
                    correlation_threshold=float(ident_cfg.get("correlation_threshold", 0.98)),
                )
                ident_diag["enabled"] = True
            except Exception as exc:
                ident_diag = {
                    "enabled": True,
                    "warnings": [f"Identifiability diagnostics failed: {exc}"],
                    "degeneracy_warning": True,
                    "error": str(exc),
                }

    return {
        "class_id": role_candidate.class_id,
        "roles": {"A": role_candidate.A, "I": role_candidate.I, "B": role_candidate.B},
        "orders": dict(order_candidate),
        "best_score": float(best_score),
        "best_params": best_params,
        "best_components": best_components,
        "condition_scores": best_condition_scores,
        "condition_count": len(conditions),
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
