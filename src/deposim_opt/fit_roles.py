"""Fit role candidates and repeat the fit with independent conditions withheld."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np
from deposim_sim.common.overrides import as_bool

from .enumerate_orders import enumerate_orders
from .enumerate_roles import RoleCandidate, class_id_from_roles, enumerate_roles
from .fit_conditions import extract_conditions
from .parameter_fit import fit_candidate_parameters
from .class_compare import rank_role_candidates


def condition_refits(conditions: Sequence[Any], fit: Callable, evaluate: Callable) -> list[dict[str, Any]]:
    """Fit only the remaining conditions, then evaluate the untouched condition.

    Models supply fitting and observation adapters. This function owns the fold
    boundary; it does not inspect a model, a coefficient, or a validation target.
    """
    if len(conditions) < 2:
        return []
    return [evaluate(fit([c for j, c in enumerate(conditions) if j != i]), held_out,
                     [c for j, c in enumerate(conditions) if j != i])
            for i, held_out in enumerate(conditions)]


def _training_mean(record: dict[str, Any], conditions: list[Any]) -> float:
    values = [record["condition_metrics"][c.name]["target_mean_nm"] for c in conditions]
    return float(np.average(values, weights=[c.weight for c in conditions]))


def fit_role_candidates(sim: Any, opt: Any) -> list[dict[str, Any]]:
    """Selection uses training conditions only; external holdouts never rank roles."""
    class_filter = list(opt.class_compare.classes) if opt.class_compare.enabled else None
    if opt.role_enumeration.enabled:
        roles = enumerate_roles(
            sim.inputs.fluent.species, roles_spec=opt.role_enumeration.roles,
            constraints=opt.role_enumeration.constraints, class_filter=class_filter,
        )
    else:
        roles = [RoleCandidate(A=sim.roles.A, I=sim.roles.I, B=sim.roles.B,
                               class_id=class_id_from_roles(I=sim.roles.I, B=sim.roles.B))]
        roles = [r for r in roles if class_filter is None or r.class_id in class_filter]
    candidates = [
        (role, order) for role in roles
        for order in enumerate_orders(
            list(opt.order_enumeration.candidates), has_b=role.B is not None,
            enforce_total_order_le=int(opt.order_enumeration.enforce_total_order_le),
        )
    ]
    if not candidates:
        raise ValueError("no role/order candidates are enabled")
    records = [fit_candidate_parameters(sim_spec=sim, role_candidate=r, order_candidate=o, opt_spec=opt)
               for r, o in candidates]
    conditions = extract_conditions(opt)
    train = sorted([c for c in conditions if c.split == "train" and c.weight > 0], key=lambda c: c.name)
    stability = dict(dict(getattr(opt.parameter_fit, "analysis", {}) or {}).get("role_stability", {}) or {})
    local_parameters = any(as_bool(item.get("enabled", True)) and as_bool(item.get("per_condition", False))
                           for item in opt.parameter_fit.search_space)
    measured = all("target_mean_nm" in record["condition_metrics"].get(c.name, {}) for record in records for c in train)
    can_refit = measured and len(train) > 1 and not local_parameters
    cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}

    def fitted(index: int, subset: list[Any]) -> dict[str, Any]:
        names = tuple(c.name for c in subset)
        key = (index, names)
        if key not in cache:
            role, order = candidates[index]
            cache[key] = fit_candidate_parameters(
                sim_spec=sim, role_candidate=role, order_candidate=order, opt_spec=opt,
                conditions_override=[c if c.name in names else replace(c, split="holdout") for c in train],
                analyze=False,
            )
        return cache[key]

    def evaluation(fold: dict[str, Any], held_out: Any, subset: list[Any]) -> dict[str, Any]:
        metrics = fold["holdout_metrics"][held_out.name]
        baseline = metrics["target_variance_nm2"] + (metrics["target_mean_nm"] - _training_mean(fold, subset))**2
        baseline_scale = fold.get("selection_baseline_scale")
        baseline_selection = (
            float(baseline) * float(baseline_scale)
            if baseline_scale is not None
            else float("nan")
        )
        return {**metrics, "condition": held_out.name, "weight": held_out.weight,
                "quantity": "thickness", "unit": "nm", "baseline_mse": baseline,
                "baseline_mse_nm2": baseline,
                "baseline_selection_score": baseline_selection,
                "refit_score": fold["best_score"],
                "optimization": fold.get("optimization", {}),
                "optimization_trace": fold.get("optimization_trace", []),
                "effect_groups": fold["effect_groups"], "roles": fold["roles"], "orders": fold["orders"]}

    def validation(index: int, subset: list[Any]) -> list[dict[str, Any]]:
        return condition_refits(subset, lambda remaining: fitted(index, remaining), evaluation)

    for index, record in enumerate(records):
        record["validation_conditions"] = validation(index, train) if can_refit else []
        record["selection_refits"] = []
        if measured and record["holdout_metrics"]:
            train_mean = _training_mean(record, train)
            for metrics in record["holdout_metrics"].values():
                metrics["baseline_mse_nm2"] = metrics["target_variance_nm2"] + (metrics["target_mean_nm"] - train_mean)**2
                metrics["baseline_mse"] = metrics["baseline_mse_nm2"]
    procedure_evaluation = []
    if can_refit and len(train) >= 3 and as_bool(stability.get("enabled", True)):
        for held_out in train:
            subset = [c for c in train if c.name != held_out.name]
            repeated = [{**fitted(index, subset), "candidate_index": index, "validation_conditions": validation(index, subset)}
                        for index, record in enumerate(records)]
            for row in rank_role_candidates(repeated):
                records[row["candidate_index"]]["selection_refits"].append({
                    "condition": held_out.name, "selected": row["score_tied_with_best"],
                    "selection_score": row["selection_score"],
                    "effect_groups": row["effect_groups"],
                    "roles": row["roles"],
                })
                if row["selected"]:
                    procedure_evaluation.append(evaluation(fitted(row["candidate_index"], subset), held_out, subset))
    ranked = rank_role_candidates(records)
    ranked[0]["evaluation_conditions"] = procedure_evaluation
    return ranked
