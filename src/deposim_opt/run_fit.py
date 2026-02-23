"""Entry point for role/order enumeration and parameter fitting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deposim_schema import compose_and_save_opt_config, compose_opt_config
from deposim_sim.output_manifest import artifact_links, artifact_paths, build_manifest, write_manifest
from deposim_sim.results_index import next_run_dir, update_project_files

from .class_compare import build_class_compare, build_role_stability
from .enumerate_orders import enumerate_orders
from .enumerate_roles import RoleCandidate, class_id_from_roles, enumerate_roles
from .fit_optuna import fit_candidate_with_optuna


def _run_dir(sim: Any, opt: Any) -> tuple[Path, str, Path]:
    root_dir = Path(str(opt.output.get("root_dir", sim.output.root_dir)))
    project = str(opt.output.get("project", sim.output.project))
    run_name = str(opt.output.get("run_name", "fit_aib"))
    project_dir = root_dir / project
    project_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = next_run_dir(project_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return project_dir, run_id, run_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def run_fit(
    *,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = compose_opt_config(config_name, overrides=overrides)
    sim = spec.sim
    opt = spec.opt

    project_dir, run_id, run_dir = _run_dir(sim, opt)
    compose_and_save_opt_config(run_dir / "config_resolved.yaml", config_name, overrides=overrides)

    role_spec = opt.role_enumeration
    class_filter = list(opt.class_compare.classes) if bool(opt.class_compare.enabled) else None
    if bool(role_spec.enabled):
        all_roles = enumerate_roles(
            sim.inputs.fluent.species,
            roles_spec=role_spec.roles,
            constraints=role_spec.constraints,
            class_filter=class_filter,
        )
    else:
        cid = class_id_from_roles(I=sim.roles.I, B=sim.roles.B)
        if class_filter is not None and cid not in set(class_filter):
            all_roles = []
        else:
            all_roles = [RoleCandidate(A=sim.roles.A, I=sim.roles.I, B=sim.roles.B, class_id=cid)]
    all_records: list[dict[str, Any]] = []

    for role in all_roles:
        orders = enumerate_orders(
            list(opt.order_enumeration.candidates),
            has_b=role.B is not None,
            enforce_total_order_le=int(opt.order_enumeration.enforce_total_order_le),
        )
        for order in orders:
            rec = fit_candidate_with_optuna(
                sim_spec=sim,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt,
            )
            all_records.append(rec)

    all_records.sort(key=lambda row: float(row["best_score"]))
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
    for row in all_records:
        merged = dict(row)
        components = dict(merged.pop("best_components", {}) or {})
        merged.update({k: float(v) for k, v in components.items()})
        merged["condition_scores"] = json.dumps(dict(merged.get("condition_scores", {})), ensure_ascii=True, sort_keys=True)
        ranking_rows.append(merged)

    role_stability_rows, role_stability_diag = build_role_stability(
        all_records,
        topk_window=int(role_stability_cfg.get("topk_window", 10)),
        score_epsilon=float(role_stability_cfg.get("score_epsilon", 1.0e-6)),
    )
    role_stability_enabled = bool(role_stability_cfg.get("enabled", True))
    role_identifiability_warning = bool(role_stability_diag.get("warning", False)) if role_stability_enabled else False

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
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "ranking.csv", ranking_rows)
    _write_csv(tables_dir / "class_compare.csv", class_rows)
    _write_csv(tables_dir / "topk_assignments.csv", topk_assignments)
    _write_csv(tables_dir / "role_stability.csv", role_stability_rows if role_stability_enabled else [])

    cache_totals = {"trial_hits": 0, "global_hits": 0, "misses": 0, "stores": 0, "evictions": 0}
    for row in all_records:
        stats = dict(row.get("cache_stats", {}) or {})
        for key in cache_totals:
            cache_totals[key] += int(stats.get(key, 0))

    best_identifiability = {}
    if all_records:
        best_identifiability = dict(all_records[0].get("fit_diagnostics", {}) or {}).get("identifiability", {})

    fit_diagnostics = {
        "role_stability_enabled": role_stability_enabled,
        "role_identifiability_warning": role_identifiability_warning,
        "role_stability": role_stability_diag,
        "cache_stats": cache_totals,
        "best_identifiability": best_identifiability,
    }
    (outputs_dir / "fit_diagnostics.json").write_text(json.dumps(fit_diagnostics, indent=2), encoding="utf-8")

    artifact_rows = [
        {"id": "config", "path": "config_resolved.yaml", "kind": "yaml", "required": True},
        {"id": "summary", "path": "summary.json", "kind": "json", "required": True},
        {"id": "report", "path": "report.html", "kind": "html", "required": True},
        {"id": "manifest", "path": "outputs/manifest.json", "kind": "json", "required": True},
        {"id": "ranking", "path": "tables/ranking.csv", "kind": "csv", "required": True},
        {"id": "class_compare", "path": "tables/class_compare.csv", "kind": "csv", "required": True},
        {"id": "topk_assignments", "path": "tables/topk_assignments.csv", "kind": "csv", "required": True},
        {"id": "role_stability", "path": "tables/role_stability.csv", "kind": "csv", "required": bool(role_stability_enabled)},
        {"id": "fit_diagnostics", "path": "outputs/fit_diagnostics.json", "kind": "json", "required": True},
    ]
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        run_id=run_id,
        mode="fit",
        created_at_utc=timestamp_utc,
        artifacts=artifact_rows,
        plots=[],
        metadata={"task": str(opt.task)},
    )
    artifact_map = artifact_paths(manifest)

    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "mode": "fit",
        "candidate_count": len(all_records),
        "class_count": len(class_rows),
        "topk_overall": topk_overall,
        "topk_per_class": topk_per_class,
        "ranking_count": len(ranking_rows),
        "topk_assignment_count": len(topk_assignments),
        "consistency": {
            "ranking_equals_candidates": len(ranking_rows) == len(all_records),
            "topk_not_exceed_candidates": len(topk_assignments) <= len(all_records),
        },
        "best_score": float(all_records[0]["best_score"]) if all_records else None,
        "diagnostics_path": "outputs/fit_diagnostics.json",
        "role_identifiability_warning": role_identifiability_warning,
        "cache_stats": cache_totals,
        "manifest_path": "outputs/manifest.json",
        "artifact_paths": artifact_map,
    }
    output_links = artifact_links(manifest)
    output_items = "".join(f"<li><a href='{path}'>{path}</a></li>" for path in output_links)
    warning_html = ""
    if role_identifiability_warning:
        warning_html += "<p><strong>Warning:</strong> role identifiability is weak in near-best candidates.</p>"
    if bool(best_identifiability.get("degeneracy_warning", False)):
        warning_html += "<p><strong>Warning:</strong> identifiability diagnostics detected high correlation / degeneracy.</p>"
    cache_html = (
        f"<p>Cache stats: trial_hits={cache_totals['trial_hits']}, "
        f"global_hits={cache_totals['global_hits']}, misses={cache_totals['misses']}</p>"
    )
    report = (
        "<!doctype html><html><head><meta charset='utf-8'><title>AIB fit report</title></head><body>"
        f"<h1>AIB Fit Report: {run_id}</h1>"
        f"{warning_html}"
        f"{cache_html}"
        f"<ul>{output_items}</ul>"
        "</body></html>"
    )
    (run_dir / "report.html").write_text(report, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(run_dir, manifest)

    update_project_files(project_dir, summary)
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
