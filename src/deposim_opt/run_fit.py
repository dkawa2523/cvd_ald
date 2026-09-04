"""Entry point for role/order enumeration and parameter fitting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deposim_schema import compose_and_save_opt_config, compose_opt_config
from deposim_sim.common.csv_io import write_rows_csv
from deposim_sim.common.report_html import write_artifact_list_report
from deposim_sim.common.run_artifacts import (
    build_provenance_metadata,
    build_manifest_and_summary,
    create_run_layout,
    finalize_run_outputs,
    standard_artifact_rows,
)
from deposim_sim.output_manifest import artifact_links

from .class_compare import build_class_compare, build_complexity_sensitivity, build_role_stability, build_role_summary, build_condition_scores
from .fit_roles import fit_role_candidates


def _json_default(value: Any) -> Any:
    """Convert NumPy diagnostics to plain JSON values without hiding bad types."""

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def run_fit(
    *,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = compose_opt_config(config_name, overrides=overrides)
    sim = spec.sim
    opt = spec.opt

    layout = create_run_layout(
        root_dir=Path(str(opt.output.get("root_dir", sim.output.root_dir))),
        project=str(opt.output.get("project", sim.output.project)),
        run_name=str(opt.output.get("run_name", "fit_aib")),
        with_inputs_dir=False,
    )
    run_id = layout.run_id
    run_dir = layout.run_dir
    compose_and_save_opt_config(run_dir / "config_resolved.yaml", config_name, overrides=overrides)
    input_paths: list[str] = [str(sim.inputs.fluent.file)]
    measurement_cfg = dict(getattr(opt, "measurement", {}) or {})
    raw_conditions = measurement_cfg.get("conditions")
    if isinstance(raw_conditions, list) and raw_conditions:
        input_paths = []
        for row in raw_conditions:
            item = dict(row or {})
            fluent_file = str(item.get("fluent_file", "")).strip() or str(sim.inputs.fluent.file)
            measurement_file = str(item.get("measurement_file", item.get("file", ""))).strip()
            input_paths.append(fluent_file)
            if measurement_file:
                input_paths.append(measurement_file)
    else:
        measurement_file = str(measurement_cfg.get("file", "")).strip()
        if measurement_file:
            input_paths.append(measurement_file)
    provenance = build_provenance_metadata(
        workflow_name="fit",
        config_payload=spec,
        input_paths=input_paths,
        extra_metadata={"task": str(opt.task)},
    )

    all_records = fit_role_candidates(sim, opt)
    objective_cfg = dict(opt.parameter_fit.objective or {})
    analysis_cfg = dict(getattr(opt.parameter_fit, "analysis", {}) or {})
    role_stability_cfg = dict(analysis_cfg.get("role_stability", {}) or {})
    tie_cfg = objective_cfg.get("tie", {})
    if isinstance(tie_cfg, dict):
        tie_eps = float(tie_cfg.get("abs_score_epsilon", tie_cfg.get("score_epsilon", 1.0e-8)))
    else:
        tie_eps = float(tie_cfg) if tie_cfg is not None else 1.0e-8
    class_rows = build_class_compare(all_records, tie_epsilon=tie_eps)
    topk_overall = max(int(opt.selection.get("topk_overall", len(all_records))), 0)
    topk_per_class = max(int(opt.selection.get("topk_per_class", len(all_records))), 0)
    ranking_rows = []
    role_ranking_rows = []
    for row in all_records:
        merged = dict(row)
        components = dict(merged.pop("best_components", {}) or {})
        merged.update({k: float(v) for k, v in components.items()})
        merged["condition_scores"] = json.dumps(dict(merged.get("condition_scores", {})), ensure_ascii=True, sort_keys=True)
        ranking_rows.append(merged)
    for rank, row in enumerate(ranking_rows, start=1):
        roles = dict(row.get("roles", {}) or {})
        role_ranking_rows.append(
            {
                "rank": rank,
                "class_id": row.get("class_id", ""),
                "role_A": roles.get("A"),
                "role_I": roles.get("I"),
                "role_B": roles.get("B"),
                "effect_groups": row.get("effect_groups", {}),
                "effect_basis": row.get("effect_basis", ""),
                "reduced_model_comparisons": row.get("reduced_model_comparisons", []),
                "role_evidence": row.get("role_evidence", []),
                "quantity": row.get("quantity", "thickness"), "unit": row.get("unit", "nm"),
                "best_score": row.get("best_score"),
                "selection_score": row.get("selection_score"),
                "selection_basis": row.get("selection_basis"),
                "validation_skill": row.get("validation_skill", ""),
                "score_total": row.get("score_total"),
                "loss_data": row.get("loss_data"),
                "rmse_nm": row.get("rmse_nm"),
                "mae_nm": row.get("mae_nm"),
                "max_abs_nm": row.get("max_abs_nm"),
                "penalty_profile": row.get("penalty_profile", 0.0),
                "penalty_phys": row.get("penalty_phys"),
                "penalty_solver": row.get("penalty_solver"),
                "penalty_complexity": row.get("penalty_complexity"),
                "condition_scores": row.get("condition_scores", "{}"),
            }
        )

    role_stability_rows, role_stability_diag = build_role_stability(
        all_records,
        score_epsilon=float(role_stability_cfg.get("score_epsilon", 1.0e-6)),
    )
    role_stability_enabled = bool(role_stability_cfg.get("enabled", True))
    role_stability_warning = bool(role_stability_diag.get("warning", False)) if role_stability_enabled else False
    # Preserve the former public field as a compatibility alias while exposing
    # the role-stability and parameter-identifiability meanings separately.
    role_identifiability_warning = role_stability_warning
    best_identifiability = {}
    if all_records:
        best_identifiability = dict(all_records[0].get("fit_diagnostics", {}) or {}).get("identifiability", {})
    parameter_identifiability_warning = bool(best_identifiability.get("degeneracy_warning", False))
    complexity_sensitivity_rows, complexity_sensitivity_diag = build_complexity_sensitivity(all_records)
    complexity_sensitivity_warning = bool(complexity_sensitivity_diag.get("warning", False)) and all_records[0].get("selection_basis") == "training"
    role_summary_rows = build_role_summary(
        ranking_rows,
        score_epsilon=float(role_stability_cfg.get("score_epsilon", 1.0e-6)),
        role_stability_warning=role_stability_warning,
        complexity_sensitivity_warning=complexity_sensitivity_warning,
        parameter_identifiability_warning=parameter_identifiability_warning,
        application=opt.selection.get("application"),
    )

    condition_score_rows = build_condition_scores(ranking_rows)

    class_ranks: dict[str, int] = {}
    topk_assignments: list[dict[str, Any]] = []
    for global_rank, row in enumerate(all_records, start=1):
        cid = str(row["class_id"])
        class_rank = class_ranks.get(cid, 0) + 1
        class_ranks[cid] = class_rank
        selected_overall = global_rank <= topk_overall if topk_overall > 0 else False
        selected_class = class_rank <= topk_per_class if topk_per_class > 0 else False
        if selected_overall or selected_class:
            topk_assignments.append(
                {
                    "rank_overall": global_rank,
                    "rank_in_class": class_rank,
                    "class_id": cid,
                    "selected_by_overall": int(selected_overall),
                    "selected_by_class": int(selected_class),
                    "selected": int(selected_overall or selected_class),
                    "best_score": float(row["best_score"]),
                    "roles": json.dumps(row.get("roles", {}), ensure_ascii=True, sort_keys=True),
                    "orders": json.dumps(row.get("orders", {}), ensure_ascii=True, sort_keys=True),
                    "best_params": json.dumps(row.get("best_params", {}), ensure_ascii=True, sort_keys=True),
                }
            )

    tables_dir = run_dir / "tables"
    outputs_dir = layout.outputs_dir
    write_rows_csv(tables_dir / "ranking.csv", ranking_rows)
    write_rows_csv(tables_dir / "role_summary.csv", role_summary_rows)
    write_rows_csv(tables_dir / "role_ranking.csv", role_ranking_rows)
    write_rows_csv(tables_dir / "condition_scores.csv", condition_score_rows)
    write_rows_csv(tables_dir / "class_compare.csv", class_rows)
    write_rows_csv(tables_dir / "topk_assignments.csv", topk_assignments)
    write_rows_csv(tables_dir / "role_stability.csv", role_stability_rows if role_stability_enabled else [])
    write_rows_csv(tables_dir / "complexity_sensitivity.csv", complexity_sensitivity_rows)

    cache_totals = {"trial_hits": 0, "global_hits": 0, "misses": 0, "stores": 0, "evictions": 0}
    for row in all_records:
        stats = dict(row.get("cache_stats", {}) or {})
        for key in cache_totals:
            cache_totals[key] += int(stats.get(key, 0))

    fit_diagnostics = {
        "role_stability_enabled": role_stability_enabled,
        "role_identifiability_warning": role_identifiability_warning,
        "role_stability_warning": role_stability_warning,
        "parameter_identifiability_warning": parameter_identifiability_warning,
        "role_stability": role_stability_diag,
        "complexity_sensitivity": complexity_sensitivity_diag,
        "cache_stats": cache_totals,
        "best_identifiability": best_identifiability,
    }
    (outputs_dir / "fit_diagnostics.json").write_text(
        json.dumps(fit_diagnostics, indent=2, default=_json_default),
        encoding="utf-8",
    )

    artifact_rows = standard_artifact_rows(
        include_report=True,
        extra_rows=[
            {"id": "ranking", "path": "tables/ranking.csv", "kind": "csv", "required": True},
            {"id": "role_summary", "path": "tables/role_summary.csv", "kind": "csv", "required": True},
            {"id": "role_ranking", "path": "tables/role_ranking.csv", "kind": "csv", "required": True},
            {"id": "condition_scores", "path": "tables/condition_scores.csv", "kind": "csv", "required": True},
            {"id": "class_compare", "path": "tables/class_compare.csv", "kind": "csv", "required": True},
            {"id": "topk_assignments", "path": "tables/topk_assignments.csv", "kind": "csv", "required": True},
            {"id": "role_stability", "path": "tables/role_stability.csv", "kind": "csv", "required": bool(role_stability_enabled)},
            {"id": "complexity_sensitivity", "path": "tables/complexity_sensitivity.csv", "kind": "csv", "required": True},
            {"id": "fit_diagnostics", "path": "outputs/fit_diagnostics.json", "kind": "json", "required": True},
        ],
    )
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    manifest, summary = build_manifest_and_summary(
        run_id=run_id,
        mode="fit",
        artifacts=artifact_rows,
        plots=[],
        metadata=provenance,
        timestamp_utc=timestamp_utc,
        summary_fields={
        "candidate_count": len(all_records),
        "class_count": len(class_rows),
        "topk_overall": topk_overall,
        "topk_per_class": topk_per_class,
        "ranking_count": len(ranking_rows),
        "role_summary_count": len(role_summary_rows),
        "role_ranking_count": len(role_ranking_rows),
        "condition_score_count": len(condition_score_rows),
        "topk_assignment_count": len(topk_assignments),
        "consistency": {
            "ranking_equals_candidates": len(ranking_rows) == len(all_records),
            "topk_not_exceed_candidates": len(topk_assignments) <= len(all_records),
        },
        "best_score": float(all_records[0]["best_score"]) if all_records else None,
        "selection_score": all_records[0].get("selection_score") if all_records else None,
        "selection_basis": all_records[0].get("selection_basis") if all_records else None,
        "decision": role_summary_rows[0]["decision"] if role_summary_rows else "review",
        "diagnostics_path": "outputs/fit_diagnostics.json",
        "role_identifiability_warning": role_identifiability_warning,
        "role_stability_warning": role_stability_warning,
        "parameter_identifiability_warning": parameter_identifiability_warning,
        "complexity_sensitivity_warning": complexity_sensitivity_warning,
        "cache_stats": cache_totals,
        **provenance,
        },
    )
    output_links = artifact_links(manifest)
    warning_msgs: list[str] = []
    if role_stability_warning:
        warning_msgs.append("different role assignments are supported across condition refits or training ties.")
    if complexity_sensitivity_warning:
        warning_msgs.append("the best role assignment changes when the complexity penalty is rescaled.")
    if parameter_identifiability_warning:
        warning_msgs.append("identifiability diagnostics detected high correlation / degeneracy.")
    note_lines = [
        f"Decision: {role_summary_rows[0]['decision']}. {role_summary_rows[0]['reason']}",
        f"Selected roles: {all_records[0]['roles']}",
        f"Selection basis: {all_records[0]['selection_basis']}; score: {all_records[0]['selection_score']:.6g}",
        "See condition_scores.csv for prediction error, mean bias, spatial shape, and training-only baselines.",
        (
            f"Cache stats: trial_hits={cache_totals['trial_hits']}, "
            f"global_hits={cache_totals['global_hits']}, misses={cache_totals['misses']}"
        )
    ]
    write_artifact_list_report(
        run_dir=run_dir,
        run_id=run_id,
        title=f"{sim.process.upper()} Role Fit Report",
        artifact_links=output_links,
        warnings=warning_msgs,
        notes=note_lines,
    )
    finalize_run_outputs(layout=layout, summary=summary, manifest=manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AIB role/order fit")
    parser.add_argument("--config-name", default="fit_cvd_steady_min")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = run_fit(config_name=args.config_name, overrides=args.overrides)
    print(f"[fit] wrote run artifacts to: {out['run_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
