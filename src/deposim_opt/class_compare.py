"""Class-wise comparison utilities for A/AI/AB/AIB."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from typing import Any

import numpy as np

from .objective import metric_value


def effect_signature(row: dict[str, Any]) -> str:
    """Model-defined effect identity; a score tie never implies this identity."""
    groups = row.get("effect_groups")
    if groups is None:
        groups = {slot: [value] for slot, value in row.get("effective_roles", row.get("roles", {})).items()
                  if value is not None}
    return json.dumps(groups, ensure_ascii=True, sort_keys=True)


def _selection_score(row: dict[str, Any]) -> float:
    return float(row.get("selection_score", row.get("best_score", float("inf"))))


def _paired_comparison(row: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Describe refitted loss differences; this is not a significance test.

    Positive differences favor row. Crossing condition losses remain mixed;
    only floating-point roundoff is ignored. No CV standard-error cutoff is used.
    """
    reference = {f["condition"]: f for f in row.get("validation_conditions", [])}
    folds = other.get("validation_conditions", [])
    if not reference or {f["condition"] for f in folds} != set(reference):
        return {"status": "not_assessed", "conditions": []}
    conditions = []
    signs = {name: [] for name in ("mse", "mean_mse", "centered_mse")}
    for fold in folds:
        base = reference[fold["condition"]]
        values = {"condition": fold["condition"], "weight": fold["weight"]}
        for name in signs:
            a, b = metric_value(base, name), metric_value(fold, name)
            delta = b - a
            observation_scale = max(metric_value(base, "target_mean", 0.)**2,
                                    metric_value(base, "baseline_mse", 0.))
            floor = 32 * np.finfo(float).eps * max(abs(a), abs(b)) + (64 * np.finfo(float).eps)**2 * observation_scale
            signs[name].append(float("nan") if not np.isfinite(delta) else
                               1 if delta > floor else -1 if delta < -floor else 0)
            values[name + "_increase"] = delta
        conditions.append(values)
    def status(values):
        if len(values) < 2 or not np.isfinite(values).all():
            return "not_assessed"
        if min(values) >= 0 and max(values) > 0:
            return "consistent_benefit"
        if max(values) <= 0:
            return "no_benefit"
        return "mixed"
    return {"basis": "inner_condition_cv", "status": status(signs["mse"]),
            "mean_status": status(signs["mean_mse"]), "spatial_status": status(signs["centered_mse"]),
            "selection_error_increase": _selection_score(other) - _selection_score(row),
            "conditions": conditions}


def _role_evidence(row: dict[str, Any], ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assess term necessity separately from its assignment to raw species."""
    declared = row.get("declared_effect_groups", row.get("effect_groups", {}))
    evidence = []
    for group, species in row.get("effect_groups", {}).items():
        remaining = {key: value for key, value in declared.items() if key != group}
        reduced = [c for c in row["reduced_model_comparisons"] if c["effects"] == remaining]
        alternatives = []
        seen = set()
        for other in sorted(ranked, key=_selection_score):
            groups = other.get("declared_effect_groups", other.get("effect_groups", {}))
            signature = effect_signature({"effect_groups": groups})
            if (signature not in seen and set(groups) == set(declared) and groups[group] != species
                    and all(groups[key] == value for key, value in remaining.items())):
                alternatives.append({"effects": groups, **_paired_comparison(row, other)})
                seen.add(signature)
        necessity = reduced[0]["status"] if reduced else "not_assessed"
        assignment = ("distinguished" if alternatives and all(c["status"] == "consistent_benefit" for c in alternatives)
                      else "unresolved" if alternatives else "not_assessed")
        evidence.append({"effect": group, "species": species, "necessity": necessity,
                         "assignment": assignment, "alternatives": alternatives,
                         "basis": "inner_condition_cv"})
    return evidence


def rank_role_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank prediction risk; only numerical loss ties prefer fewer effects.

    The comparison uses refitted predictions, not the number of enumerated
    candidates. External holdouts are deliberately excluded from this function.
    """
    if not records:
        return []
    records = deepcopy(records)
    for row in records:
        row.setdefault("effect_groups", json.loads(effect_signature(row)))
        folds = row.get("validation_conditions", [])
        row["selection_basis"] = "condition_cv" if folds else "training"
        row["selection_score"] = float(np.average([metric_value(f, "mse") for f in folds], weights=[f["weight"] for f in folds])) if folds else float(row["best_score"])
        if not np.isfinite(row["selection_score"]):
            raise ValueError("candidate selection scores must be finite")
    ranked = sorted(records, key=_selection_score)
    best = ranked[0]
    reference = {f["condition"]: f for f in best.get("validation_conditions", [])}
    # Squared residuals of an exact fit can differ at roundoff even when the
    # predictions agree. Scale that floor in observation units, not in CV SE.
    observation_scale = max((max(metric_value(f, "baseline_mse", 0.0),
                                 metric_value(f, "target_mean", 0.0)**2)
                             for f in reference.values()), default=0.0)
    for row in ranked:
        folds = row.get("validation_conditions", [])
        if {f["condition"] for f in folds} != set(reference):
            raise ValueError("candidates must use the same validation conditions")
        if any((f.get("quantity"), f.get("unit")) !=
               (reference[f["condition"]].get("quantity"), reference[f["condition"]].get("unit")) for f in folds):
            raise ValueError("candidates must use the same observation quantity and unit")
        gap = _selection_score(row) - _selection_score(best)
        tolerance = (32 * np.finfo(float).eps * max(abs(_selection_score(best)), np.finfo(float).tiny)
                     + (64 * np.finfo(float).eps)**2 * observation_scale)
        row["score_tied_with_best"] = bool(abs(gap) <= tolerance)
        # Compatibility field: a numerical score tie, not statistical equivalence.
        row["equivalent_to_best"] = row["score_tied_with_best"]
        row["selection_gap"] = gap
        if folds:
            weights = np.asarray([f["weight"] for f in folds], dtype=float)
            if any(f["weight"] != reference[f["condition"]]["weight"] for f in folds):
                raise ValueError("candidates must use the same condition weights")
            weights /= weights.sum()
            baseline = float(np.average([metric_value(f, "baseline_mse") for f in folds], weights=weights))
            row["validation_skill"] = 1 - row["selection_score"] / baseline if baseline > 0 else 0.0
    comparable = [row for row in ranked if row["equivalent_to_best"]]
    chosen = min(comparable, key=lambda row: (row.get("active_effect_count", _complexity_count(row.get("roles", {}))), row.get("search_space_count", 0), _selection_score(row), effect_signature(row)))
    for row in ranked:
        row["selected"] = row is chosen
        row["same_effects_as_selected"] = effect_signature(row) == effect_signature(chosen)
        # Reduced structures have their own refits and CV scores. A boundary
        # coefficient on the full fit cannot substitute for that comparison.
        reductions = {effect_signature({"effect_groups": group}) for group in row.get("reduced_effect_groups", [])}
        reduced = [other for other in ranked if effect_signature({"effect_groups": other.get("declared_effect_groups", other.get("effect_groups", {}))}) in reductions]
        unique = {}
        for other in sorted(reduced, key=_selection_score):
            signature = effect_signature({"effect_groups": other.get("declared_effect_groups", other.get("effect_groups", {}))})
            unique.setdefault(signature, {"effects": json.loads(signature), **_paired_comparison(row, other)})
        row["reduced_model_comparisons"] = list(unique.values())
        row["role_evidence"] = _role_evidence(row, ranked)
    return [chosen, *[row for row in ranked if row is not chosen]]


def build_class_compare(records: list[dict[str, Any]], *, tie_epsilon: float = 1.0e-8) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[str(rec["class_id"])].append(rec)

    for class_id, items in by_class.items():
        best[class_id] = min(items, key=_selection_score)

    best_score = min((_selection_score(v) for v in best.values()), default=float("inf"))
    tied = lambda row: row.get("score_tied_with_best", abs(_selection_score(row) - best_score) <= float(tie_epsilon))
    tie_group_size = len({effect_signature(rec) for rec in records if tied(rec)})
    rows: list[dict[str, Any]] = []
    for class_id in sorted(best):
        row = dict(best[class_id])
        components = dict(row.pop("best_components", {}) or {})
        row.update({k: float(v) for k, v in components.items()})
        row["condition_scores"] = json.dumps(dict(row.get("condition_scores", {})), ensure_ascii=True, sort_keys=True)
        delta = _selection_score(row) - best_score
        row["delta_from_global_best"] = delta
        row["delta_from_best"] = delta
        row["tie_flag"] = int(tied(row))
        row["tie_group_size"] = int(tie_group_size)
        rows.append(row)
    return rows


def build_role_stability(
    records: list[dict[str, Any]], *, score_epsilon: float, topk_window: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize repeated selections, never winners of refitted training loss."""
    if not records:
        return [], {"warning": False, "tie_group_size": 0, "slot_species_counts": {}}
    slots = ("A", "I", "B", "AB")
    folds = sorted({f["condition"] for row in records for f in row.get("selection_refits", [])})
    support = []
    if folds:
        for name in folds:
            support.append({effect_signature({**row, **f}) for row in records
                            for f in row.get("selection_refits", [])
                            if f["condition"] == name and f["selected"]})
        basis = "repeated_condition_cv_selection"
    else:
        best = min(_selection_score(row) for row in records)
        tolerance = float(score_epsilon) * max(abs(best), np.finfo(float).tiny)
        support = [{effect_signature(row) for row in records if _selection_score(row) - best <= tolerance}]
        basis = "score_ties_only_not_selection_stability"
    counts = {slot: {} for slot in slots}
    all_signatures = set().union(*support)
    for winners in support:
        for winner in winners:
            groups = json.loads(winner)
            for slot in slots:
                value = groups.get(slot, [])
                species = (json.dumps(value, ensure_ascii=True) if slot == "AB" else value[0]) if value else "__NONE__"
                counts[slot][species] = counts[slot].get(species, 0.0) + 1.0 / len(winners)
    rows = [
        {"slot": slot, "species": species, "count": count,
         "frequency": count / len(support), "refit_count": len(folds), "basis": basis}
        for slot in slots for species, count in sorted(counts[slot].items())
    ]
    return rows, {"warning": len(all_signatures) > 1, "tie_group_size": len(all_signatures),
                  "slot_species_counts": counts, "refit_count": len(folds), "basis": basis,
                  "score_epsilon": score_epsilon}


def build_complexity_sensitivity(
    records: list[dict[str, Any]],
    *,
    multipliers: tuple[float, ...] = (0.0, 1.0, 10.0),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-rank candidates under simple complexity-penalty multipliers."""

    rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for multiplier in multipliers:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in records:
            score = float(row.get("best_score", float("inf")))
            penalty = float(dict(row.get("best_components", {}) or {}).get("penalty_complexity", 0.0))
            adjusted = score - penalty + float(multiplier) * penalty
            ranked.append((adjusted, row))
        ranked.sort(key=lambda item: item[0])
        for rank, (adjusted, row) in enumerate(ranked, start=1):
            roles = dict(row.get("roles", {}) or {})
            rows.append(
                {
                    "complexity_multiplier": float(multiplier),
                    "rank": rank,
                    "adjusted_score": float(adjusted),
                    "nominal_score": float(row.get("best_score", float("inf"))),
                    "class_id": row.get("class_id", ""),
                    "role_A": roles.get("A"),
                    "role_I": roles.get("I"),
                    "role_B": roles.get("B"),
                }
            )
        if ranked:
            winner_roles = dict(ranked[0][1].get("roles", {}) or {})
            winners.append(
                {
                    "complexity_multiplier": float(multiplier),
                    "class_id": ranked[0][1].get("class_id", ""),
                    "roles": winner_roles,
                    "adjusted_score": float(ranked[0][0]),
                }
            )

    winner_signatures = {
        (str(row["class_id"]), str(row["roles"].get("A")), str(row["roles"].get("I")), str(row["roles"].get("B")))
        for row in winners
    }
    diagnostics = {
        "warning": len(winner_signatures) > 1,
        "winner_count": len(winner_signatures),
        "multipliers": [float(v) for v in multipliers],
        "winners": winners,
    }
    return rows, diagnostics


def _complexity_count(roles: dict[str, Any]) -> int:
    return int(roles.get("I") is not None) + int(roles.get("B") is not None)


def build_condition_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One output adapter for all model paths and evaluation subjects."""
    rows = []
    for index, row in enumerate(records, start=1):
        groups = [
            ("train", "training_fit", [{"condition": k, **v} for k, v in row.get("condition_metrics", {}).items()]),
            ("holdout", "fixed_model_holdout", [{"condition": k, **v} for k, v in row.get("holdout_metrics", {}).items()]),
            ("condition_cv", "inner_selection", row.get("validation_conditions", [])),
            ("outer_condition_cv", "outer_selection_procedure", row.get("evaluation_conditions", [])),
        ]
        for split, scope, metrics in groups:
            for values in metrics:
                reductions = [{"effects": comparison["effects"], **difference}
                              for comparison in row.get("reduced_model_comparisons", [])
                              for difference in comparison.get("conditions", [])
                              if scope == "inner_selection" and difference["condition"] == values["condition"]]
                rows.append({"model_id": values.get("selected_model", row.get("model_id", "")),
                             "candidate_rank": row.get("adoption_rank", index) if scope != "outer_selection_procedure" else "",
                             "roles": row.get("roles", {}), "orders": row.get("orders", {}),
                             "quantity": row.get("quantity", "thickness"), "unit": row.get("unit", "nm"),
                             "split": split, "evaluation_scope": scope,
                             "effect_groups": row.get("effect_groups", {}),
                             "reduction_comparisons": reductions, **values})
    return rows


def assess_prediction(conditions: list[dict[str, Any]], *, scope: str,
                      application: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assess supplied evidence without selecting a model or inventing tolerances."""
    if not conditions:
        return {"scope": scope, "prediction_status": "not_validated",
                "spatial_status": "not_assessed", "application_status": "not_assessed",
                "condition_count": 0, "failed_conditions": []}
    mse = np.asarray([metric_value(m, "mse") for m in conditions])
    baseline = np.asarray([metric_value(m, "baseline_mse") for m in conditions])
    weights = np.asarray([m.get("weight", 1.0) for m in conditions], dtype=float)
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("evaluation conditions require positive finite weights")
    tested = np.isfinite(mse) & np.isfinite(baseline)
    failed = ~tested | (mse >= baseline)
    centered = np.asarray([m.get("centered_r2", float("nan")) for m in conditions], dtype=float)
    relative = np.asarray([m.get("relative_rmse", float("nan")) for m in conditions], dtype=float)
    application = dict(application or {})
    if not application:
        application_status = "not_specified"
    else:
        # Explicit condition scope and error tolerance are both user responsibilities.
        requested = {str(name) for name in application.get("conditions", [])}
        available = {str(m["condition"]) for m in conditions}
        tolerance = application.get("max_relative_rmse")
        if not requested or tolerance is None or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("application requires conditions and a nonnegative max_relative_rmse")
        if not requested.issubset(available):
            application_status = "scope_not_tested"
        else:
            subset = np.asarray([str(m["condition"]) in requested for m in conditions])
            passed = np.all(np.isfinite(relative[subset]) & (relative[subset] <= tolerance))
            if application.get("require_spatial", True):
                passed = passed and np.all(np.isfinite(centered[subset]) & (centered[subset] > 0))
            application_status = "meets_tolerance" if passed else "fails_tolerance"
    return {
        "scope": scope, "condition_count": len(conditions),
        "prediction_status": ("not_assessed" if not np.any(tested) else
                              "not_supported" if np.any(failed) else "improves_baseline"),
        "spatial_status": ("not_supported" if np.any(centered <= 0) else
                           "supported" if np.all(np.isfinite(centered)) else "not_assessed"),
        "application_status": application_status,
        "failed_conditions": [m["condition"] for m, bad in zip(conditions, failed) if bad],
        "mse": float(np.average(mse, weights=weights)),
        "baseline_mse": float(np.average(baseline, weights=weights)),
        "worst_relative_rmse": float(np.max(relative)) if np.all(np.isfinite(relative)) else float("nan"),
    }


def build_role_summary(
    records: list[dict[str, Any]], *, score_epsilon: float,
    role_stability_warning: bool, complexity_sensitivity_warning: bool = False,
    parameter_identifiability_warning: bool = False,
    application: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One decision path for empirical rates and physical thickness models.

    Inner scores rank candidates. Fixed-model holdouts and outer evaluations
    have different subjects and are never folded back into selection.
    """
    if not records:
        return []
    ranked = sorted(records, key=lambda row: (not row.get("selected", False), _selection_score(row)))
    best_score = _selection_score(ranked[0])
    rows = []
    for rank, row in enumerate(ranked, start=1):
        roles = dict(row.get("roles", {}) or {})
        holdout = [{"condition": name, **metrics} for name, metrics in row.get("holdout_metrics", {}).items()]
        inner = assess_prediction(row.get("validation_conditions", []), scope="inner_selection")
        fixed = assess_prediction(holdout, scope="fixed_model_holdout", application=application)
        procedure = assess_prediction(row.get("evaluation_conditions", []),
                                      scope="outer_selection_procedure", application=application)
        evaluation = procedure if procedure["condition_count"] else fixed
        evidence = evaluation if evaluation["condition_count"] else inner
        ident = dict(row.get("fit_diagnostics", {}).get("identifiability", {}))
        ambiguous = any(other.get("equivalent_to_best", False) and
                        effect_signature(other) != effect_signature(row) for other in ranked)
        symmetry = "AB" in row.get("effect_groups", {})
        role_evidence = row.get("role_evidence", [])
        necessity_supported = (len(role_evidence) == len(row.get("effect_groups", {}))
                               and all(e["necessity"] == "consistent_benefit" for e in role_evidence))
        assignment_supported = all(e["assignment"] == "distinguished" for e in role_evidence)
        unresolved = (ambiguous or symmetry or role_stability_warning or
                      parameter_identifiability_warning or ident.get("degeneracy_warning", False)
                      or not ident.get("assessed", False) or not necessity_supported or not assignment_supported)
        reasons = []
        if evidence["prediction_status"] == "not_supported":
            reasons.append("refitted or held-out prediction fails the training-only constant baseline in: " +
                           ", ".join(map(str, evidence["failed_conditions"])))
        if evidence["spatial_status"] == "not_supported":
            reasons.append("prediction does not explain within-condition spatial variation")
        if ambiguous:
            reasons.append("different effective roles have numerically tied validation scores")
        if symmetry:
            reasons.append("the AB product identifies a pair, not its A/B direction")
        if not necessity_supported:
            reasons.append("term removal has not shown consistent additional predictive benefit across conditions")
        if not assignment_supported:
            reasons.append("alternative raw-species assignments are not distinguished across conditions")
        if role_stability_warning:
            reasons.append("effective roles change across training-condition selections")
        if parameter_identifiability_warning or ident.get("degeneracy_warning", False):
            reasons.append("best-fit parameters are weakly identifiable or strongly correlated")
        if not ident.get("assessed", False):
            reasons.append("local parameter identifiability has not been assessed")
        if not evaluation["condition_count"]:
            reasons.append("independent prediction evaluation is unavailable; inner CV is selection evidence")
        elif application and fixed["application_status"] != "meets_tolerance":
            reasons.append("fixed-model application evidence: " + fixed["application_status"])
        if evaluation["application_status"] != "meets_tolerance":
            reasons.append("application scope/error tolerance: " + evaluation["application_status"])
        if complexity_sensitivity_warning and row.get("selection_basis", "training") == "training":
            reasons.append("training selection changes with the complexity penalty")
        if rank > 1:
            decision = "review" if row.get("equivalent_to_best", False) else "reject_lower_score"
            reason = ("numerically tied validation score; compare effective effects" if decision == "review"
                      else "higher selection error than the chosen candidate")
        elif evidence["prediction_status"] == "not_supported":
            decision, reason = "reject_prediction", "; ".join(reasons)
        elif (fixed["condition_count"] and fixed["application_status"] == "meets_tolerance"
              and not unresolved and fixed["prediction_status"] == "improves_baseline"):
            decision, reason = "adopt_candidate", "independent predictions meet the declared application criteria"
        else:
            decision, reason = "review", "; ".join(reasons) or "insufficient independent role support"
        rows.append({
            **{key: row[key] for key in (
                "model_id", "role_model_id", "class_id", "effective_roles", "effect_groups", "effect_basis",
                "effect_scopes", "response_structure",
                "inactive_roles", "role_symmetry", "regularization", "quantity", "unit",
                "best_score", "selection_score", "selection_basis", "validation_skill",
                "reduced_model_comparisons", "role_evidence",
            ) if key in row},
            "rank": rank, "decision": decision, "reason": reason,
            "role_A": roles.get("A"), "role_I": roles.get("I"), "role_B": roles.get("B"),
            "prediction_status": evidence["prediction_status"],
            "spatial_status": evidence["spatial_status"],
            "role_support": ("unresolved" if unresolved else "no_role_selected" if effect_signature(row) == "{}"
                             else "model_role_candidate" if row.get("effect_basis") == "declared_state_model_roles"
                             else "effective_role_candidate"),
            "application_status": evaluation["application_status"],
            "evaluation_scope": evidence["scope"],
            "fixed_model_assessment": fixed, "procedure_assessment": procedure,
            "role_complexity": len(json.loads(effect_signature(row))),
            "score_gap_from_best": _selection_score(row) - best_score,
            "next_best_gap": _selection_score(ranked[1]) - best_score if rank == 1 and len(ranked) > 1 else "",
        })
    return rows


__all__ = [
    "rank_role_candidates", "effect_signature", "build_class_compare",
    "build_complexity_sensitivity", "build_role_stability", "build_role_summary",
    "build_condition_scores", "assess_prediction",
]
