"""Class-wise comparison utilities for A/AI/AB/AIB."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any


def build_class_compare(records: list[dict[str, Any]], *, tie_epsilon: float = 1.0e-8) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[str(rec["class_id"])].append(rec)

    for class_id, items in by_class.items():
        best[class_id] = min(items, key=lambda row: float(row["best_score"]))

    best_score = min((float(v["best_score"]) for v in best.values()), default=float("inf"))
    tie_group_size = sum(
        1 for rec in records if abs(float(rec.get("best_score", float("inf"))) - best_score) <= float(tie_epsilon)
    )
    rows: list[dict[str, Any]] = []
    for class_id in sorted(best):
        row = dict(best[class_id])
        components = dict(row.pop("best_components", {}) or {})
        row.update({k: float(v) for k, v in components.items()})
        row["condition_scores"] = json.dumps(dict(row.get("condition_scores", {})), ensure_ascii=True, sort_keys=True)
        delta = float(row["best_score"]) - best_score
        row["delta_from_global_best"] = delta
        row["delta_from_best"] = delta
        row["tie_flag"] = int(abs(delta) <= float(tie_epsilon))
        row["tie_group_size"] = int(tie_group_size)
        rows.append(row)
    return rows


def build_role_stability(
    records: list[dict[str, Any]],
    *,
    topk_window: int,
    score_epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return [], {"warning": False, "tie_group_size": 0, "slot_species_counts": {}}

    ranked = sorted(records, key=lambda row: float(row.get("best_score", float("inf"))))
    best = float(ranked[0].get("best_score", float("inf")))
    topk = max(int(topk_window), 1)
    subset = ranked[: min(topk, len(ranked))]

    tie_group = [row for row in ranked if abs(float(row.get("best_score", float("inf"))) - best) <= float(score_epsilon)]
    slots = ("A", "I", "B")

    slot_species_counts: dict[str, dict[str, int]] = {slot: {} for slot in slots}
    rows: list[dict[str, Any]] = []
    for slot in slots:
        counts: dict[str, int] = {}
        for row in subset:
            roles = dict(row.get("roles", {}))
            value = roles.get(slot)
            key = "__NONE__" if value is None else str(value)
            counts[key] = counts.get(key, 0) + 1
        slot_species_counts[slot] = dict(counts)

        for species, count in sorted(counts.items()):
            gaps = []
            for row in ranked:
                roles = dict(row.get("roles", {}))
                value = roles.get(slot)
                key = "__NONE__" if value is None else str(value)
                if key == species:
                    gaps.append(float(row.get("best_score", float("inf"))) - best)
            best_gap = min(gaps) if gaps else float("inf")
            rows.append(
                {
                    "slot": slot,
                    "species": species,
                    "count": int(count),
                    "frequency": float(count) / float(len(subset)),
                    "best_gap": float(best_gap),
                    "topk_window": int(len(subset)),
                }
            )

    warning = False
    tie_slot_counts: dict[str, int] = {}
    for slot in slots:
        species_set = set()
        for row in tie_group:
            roles = dict(row.get("roles", {}))
            value = roles.get(slot)
            species_set.add("__NONE__" if value is None else str(value))
        tie_slot_counts[slot] = len(species_set)
        if len(species_set) > 1:
            warning = True

    diagnostics = {
        "warning": bool(warning),
        "tie_group_size": int(len(tie_group)),
        "slot_species_counts": slot_species_counts,
        "tie_slot_species_counts": tie_slot_counts,
        "score_epsilon": float(score_epsilon),
    }
    return rows, diagnostics


__all__ = ["build_class_compare", "build_role_stability"]
