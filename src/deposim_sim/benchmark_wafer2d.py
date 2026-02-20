"""CVD wafer-2D trend benchmark runner (polar-first, synthetic + file inputs)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

from deposim_report.html_page import render_report_page
from deposim_report.physviz_report import write_physviz_report
from deposim_schema import compose_and_save_sim_config, compose_sim_config

from .domain import DomainGrid, build_domain_grid
from .input_builder import build_field_bundle
from .metrics import compute_kpi_metrics
from .models import mass_transfer
from .physics.cvd_steady import run_cvd_steady
from .physviz import (
    build_ald_phase_snapshots,
    build_cvd_pseudo_time_snapshots,
    compute_net_term_maps,
    compute_reaction_term_importance,
    compute_transport_term_maps,
)
from .results_index import next_run_dir, update_project_files
from .validation import validate_run_spec
from .zarr_output import save_array_store

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for wafer2d benchmark execution.")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    overrides: tuple[str, ...]
    file_pattern: str | None = None


def build_wafer2d_cases() -> list[BenchmarkCase]:
    """Return deterministic benchmark-case definitions."""
    return [
        BenchmarkCase(
            case_id="CASE-01_SYN_UNIFORM_RL",
            description="Synthetic uniform pattern in reaction-limited regime.",
            overrides=(
                "inputs.source_kind=synthetic",
                "inputs.synthetic_case=uniform",
                "inputs.c_ref_mol_m3=1.8",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.05",
                "model.kinetics_name=power_law",
                "model.kinetics_params.k0=0.2",
                "model.kinetics_params.order=1.0",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-02_SYN_UNIFORM_TL",
            description="Synthetic uniform pattern in transport-limited regime.",
            overrides=(
                "inputs.source_kind=synthetic",
                "inputs.synthetic_case=uniform",
                "inputs.c_ref_mol_m3=1.8",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.005",
                "model.kinetics_name=power_law",
                "model.kinetics_params.k0=3.0",
                "model.kinetics_params.order=1.0",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-03_SYN_RADIAL_GRAD",
            description="Synthetic radial gradient (center-rich) transfer check.",
            overrides=(
                "inputs.source_kind=synthetic",
                "inputs.synthetic_case=radial_gradient",
                "inputs.c_ref_mol_m3=1.8",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.02",
                "model.kinetics_name=power_law",
                "model.kinetics_params.k0=1.0",
                "model.kinetics_params.order=1.0",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-04_FILE_THETA_PATTERN",
            description="NPZ file input with theta modulation for transfer correlation.",
            overrides=(
                "inputs.source_kind=file",
                "inputs.io_loader_name=npz",
                "inputs.c_ref_mol_m3=1.8",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.02",
                "model.kinetics_name=power_law",
                "model.kinetics_params.k0=1.0",
                "model.kinetics_params.order=1.0",
            ),
            file_pattern="theta_pattern",
        ),
        BenchmarkCase(
            case_id="CASE-05_FILE_EDGE_DEPLETED",
            description="NPZ file input with edge-depleted radial pattern.",
            overrides=(
                "inputs.source_kind=file",
                "inputs.io_loader_name=npz",
                "inputs.c_ref_mol_m3=1.8",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.02",
                "model.kinetics_name=power_law",
                "model.kinetics_params.k0=1.0",
                "model.kinetics_params.order=1.0",
            ),
            file_pattern="edge_depleted",
        ),
        BenchmarkCase(
            case_id="CASE-06_SYN_SEEDED_LHHW_NET",
            description="Synthetic seeded nonuniformity with LHHW kinetics and dep/etch/loss net model.",
            overrides=(
                "inputs.source_kind=synthetic",
                "inputs.synthetic_case=seeded_perturbation",
                "random_seed=17",
                "inputs.c_ref_mol_m3=1.9",
                "inputs.temperature_k=735.0",
                "model.mass_transfer_name=rotating_disk",
                "inputs.omega_rad_s=120.0",
                "model.mass_transfer_params.diffusivity_m2_s=1.1e-4",
                "+model.mass_transfer_params.nu_m2_s=1.6e-5",
                "model.kinetics_name=lhhw_competition",
                "model.kinetics_params.k0=1.1",
                "+model.kinetics_params.ea_j_mol=14000.0",
                "+model.kinetics_params.numerator_orders.precursor=1.15",
                "+model.kinetics_params.denominator_coeffs.precursor=0.42",
                "+model.kinetics_params.denominator_orders.precursor=1.0",
                "+model.kinetics_params.denominator_power=1.0",
                "+model.kinetics_params.denominator_base=1.0",
                "model.net_name=dep_etch_loss",
                "+model.net_params.etch_fraction=0.10",
                "+model.net_params.loss_fraction=0.03",
            ),
        ),
        BenchmarkCase(
            case_id="CASE-07_FILE_COMPLEX_LHHW_NET",
            description="File-input mixed radial-theta pattern with LHHW kinetics and net etch/loss.",
            overrides=(
                "inputs.source_kind=file",
                "inputs.io_loader_name=npz",
                "inputs.c_ref_mol_m3=2.0",
                "inputs.temperature_k=745.0",
                "model.mass_transfer_name=stagnant_film",
                "model.mass_transfer_params.k_m_m_s=0.016",
                "model.kinetics_name=lhhw_competition",
                "model.kinetics_params.k0=1.3",
                "+model.kinetics_params.ea_j_mol=17000.0",
                "+model.kinetics_params.numerator_orders.precursor=1.25",
                "+model.kinetics_params.denominator_coeffs.precursor=0.50",
                "+model.kinetics_params.denominator_orders.precursor=1.1",
                "+model.kinetics_params.denominator_power=1.0",
                "+model.kinetics_params.denominator_base=1.0",
                "+model.kinetics_params.pattern_loading=0.9",
                "model.net_name=dep_etch_loss",
                "+model.net_params.etch_fraction=0.12",
                "+model.net_params.loss_fraction=0.05",
            ),
            file_pattern="theta_edge_mix",
        ),
    ]


def _file_pattern(name: str, grid: DomainGrid) -> np.ndarray:
    r_norm = np.asarray(grid.r_grid_mm, dtype=float) / max(float(grid.wafer_radius_mm), 1.0e-12)
    key = str(name).strip().lower()
    if key == "theta_pattern":
        if grid.theta_grid_rad is None:
            raise ValueError("theta_pattern requires a polar domain with theta_grid_rad.")
        return np.clip(1.0 + 0.25 * np.cos(3.0 * np.asarray(grid.theta_grid_rad, dtype=float)), 0.05, np.inf)
    if key == "edge_depleted":
        return np.clip(1.12 - 0.86 * (r_norm**2), 0.05, np.inf)
    if key == "theta_edge_mix":
        if grid.theta_grid_rad is None:
            raise ValueError("theta_edge_mix requires a polar domain with theta_grid_rad.")
        theta = np.asarray(grid.theta_grid_rad, dtype=float)
        radial = np.asarray(grid.r_grid_mm, dtype=float) / max(float(grid.wafer_radius_mm), 1.0e-12)
        return np.clip(1.0 + 0.30 * np.cos(2.0 * theta) * (0.25 + 0.75 * radial) - 0.35 * radial**2, 0.05, np.inf)
    raise ValueError(f"Unsupported file pattern name: {name!r}")


def write_case_input_npz(
    *,
    case: BenchmarkCase,
    grid: DomainGrid,
    output_dir: Path,
    c_ref_mol_m3: float,
    temperature_k: float,
    species: str,
) -> tuple[Path, np.ndarray]:
    """Create deterministic NPZ payload for file-input benchmark cases."""
    _require_numpy()
    if case.file_pattern is None:
        raise ValueError(f"case {case.case_id} does not declare file_pattern")
    pattern = _file_pattern(case.file_pattern, grid)
    c_ref = float(c_ref_mol_m3) * pattern
    payload_path = output_dir / f"{case.case_id.lower()}_inputs.npz"
    np.savez(
        payload_path,
        **{
            f"C_ref__{species}": c_ref,
            "T": np.full(grid.shape, float(temperature_k), dtype=float),
        },
    )
    return payload_path, c_ref


def _safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return float("nan")
    av = np.asarray(a[valid], dtype=float)
    bv = np.asarray(b[valid], dtype=float)
    if float(np.std(av)) <= 1.0e-14 or float(np.std(bv)) <= 1.0e-14:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _collect_np_arrays(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}__{key}" if prefix else str(key)
            _collect_np_arrays(child, value[key], out)
        return
    try:
        arr = np.asarray(value)
    except Exception:
        return
    if arr.dtype.kind in {"U", "S", "O"}:
        return
    out[prefix] = arr


def _resolve_nu_map(run_spec: Any, species: Sequence[str]) -> dict[str, float]:
    params = getattr(run_spec.model, "kinetics_params", {}) or {}
    if not isinstance(params, Mapping):
        params = {}
    nu_raw = params.get("nu", params.get("stoichiometry"))
    if isinstance(nu_raw, Mapping):
        out = {str(name): float(nu_raw.get(name, 1.0)) for name in species}
        return out
    if len(species) == 1:
        return {str(species[0]): 1.0}
    return {str(name): 1.0 for name in species}


def _categorize_physviz_plots(plot_paths: Sequence[str]) -> dict[str, list[str]]:
    sections = {
        "input_fields": [],
        "time_space_maps": [],
        "transport_terms": [],
        "reaction_terms": [],
        "net_terms": [],
    }
    for rel in plot_paths:
        name = Path(rel).name
        if name.startswith("physviz_input_"):
            sections["input_fields"].append(rel)
            continue
        if name.startswith("physviz_cvd_") or name.startswith("physviz_ald_"):
            sections["time_space_maps"].append(rel)
            continue
        if (
            name.startswith("physviz_transport_capacity_")
            or name.startswith("physviz_reaction_demand_")
            or name.startswith("physviz_depletion_ratio_")
            or name.startswith("physviz_utilization_")
        ):
            sections["transport_terms"].append(rel)
            continue
        if (
            name.startswith("physviz_reaction_sensitivity_")
            or name.startswith("physviz_reaction_ablation_")
            or name == "physviz_reaction_importance_rank.png"
        ):
            sections["reaction_terms"].append(rel)
            continue
        if name.startswith("physviz_net_"):
            sections["net_terms"].append(rel)
            continue
    return sections


def _weighted_net_scores(net_maps: Mapping[str, Any], grid: DomainGrid) -> list[dict[str, Any]]:
    weights = np.asarray(grid.area_weights_mm2, dtype=float)
    mask = np.asarray(grid.edge_mask, dtype=bool)
    out: list[dict[str, Any]] = []
    for name in ("etch_fraction_of_dep", "loss_fraction_of_dep"):
        if name not in net_maps:
            continue
        arr = np.asarray(net_maps[name], dtype=float)
        valid = np.isfinite(arr) & np.isfinite(weights) & mask
        if np.any(valid):
            score = float(np.sum(arr[valid] * weights[valid]) / max(np.sum(weights[valid]), _EPS))
        else:
            score = 0.0
        out.append(
            {
                "term_name": name,
                "importance_score": score,
                "sign": float(np.sign(score)),
                "spatial_hotspot_radius_mm": float("nan"),
                "notes": "weighted mean contribution",
            }
        )
    out.sort(key=lambda item: float(item["importance_score"]), reverse=True)
    return out


def evaluate_trend_assertions(case_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, bool | float]:
    """Evaluate trend-style benchmark assertions from per-case metrics."""
    required_cases = (
        "CASE-01_SYN_UNIFORM_RL",
        "CASE-02_SYN_UNIFORM_TL",
        "CASE-03_SYN_RADIAL_GRAD",
        "CASE-04_FILE_THETA_PATTERN",
        "CASE-05_FILE_EDGE_DEPLETED",
    )
    for case_id in required_cases:
        if case_id not in case_metrics:
            raise ValueError(f"missing benchmark metrics for case {case_id}")

    rl = case_metrics["CASE-01_SYN_UNIFORM_RL"]
    tl = case_metrics["CASE-02_SYN_UNIFORM_TL"]
    radial = case_metrics["CASE-03_SYN_RADIAL_GRAD"]
    theta = case_metrics["CASE-04_FILE_THETA_PATTERN"]

    delta_cs = float(rl["mean_cs_ratio"]) - float(tl["mean_cs_ratio"])
    delta_da = float(tl["mean_da_proxy"]) - float(rl["mean_da_proxy"])
    theta_corr = float(theta["theta_transfer_corr"])

    checks: dict[str, bool] = {
        "assert_regime_cs_ratio": float(rl["mean_cs_ratio"]) > float(tl["mean_cs_ratio"]),
        "assert_regime_da_proxy": float(rl["mean_da_proxy"]) < float(tl["mean_da_proxy"]),
        "assert_radial_trend": float(radial["center_mean"]) > float(radial["edge_mean"]),
        "assert_file_theta_transfer": theta_corr > 0.0,
        "assert_solver_health": all(float(m["root_failure_fraction"]) == 0.0 for m in case_metrics.values()),
    }
    return {
        "delta_cs_ratio_rl_minus_tl": delta_cs,
        "delta_da_proxy_tl_minus_rl": delta_da,
        "theta_transfer_corr_case4": theta_corr,
        **checks,
        "overall_passed": all(checks.values()),
    }


def _ranking_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>No ranking rows.</p>"
    body = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('term_name', '')))}</td>"
        f"<td>{float(row.get('importance_score', 0.0)):.8g}</td>"
        f"<td>{float(row.get('sign', 0.0)):.3g}</td>"
        f"<td>{float(row.get('spatial_hotspot_radius_mm', float('nan'))):.8g}</td>"
        f"<td>{escape(str(row.get('notes', '')))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><tr><th>term_name</th><th>importance_score</th><th>sign</th>"
        "<th>spatial_hotspot_radius_mm</th><th>notes</th></tr>"
        f"{body}</table>"
    )


def write_benchmark_report(
    *,
    run_dir: Path,
    run_id: str,
    summary: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    trend_assertions: Mapping[str, Any],
    physviz_context: Mapping[str, Any] | None = None,
) -> Path:
    """Write benchmark HTML report to run directory."""
    assertion_rows = "".join(
        (
            f"<tr><th>{name}</th><td>{value}</td></tr>"
            if not isinstance(value, bool)
            else f"<tr><th>{name}</th><td>{'PASS' if value else 'FAIL'}</td></tr>"
        )
        for name, value in trend_assertions.items()
    )
    case_table_rows = "".join(
        (
            f"<tr><td>{row['case_id']}</td><td>{row['input_source_kind']}</td>"
            f"<td>{float(row['mean_cs_ratio']):.8g}</td>"
            f"<td>{float(row['mean_da_proxy']):.8g}</td>"
            f"<td>{float(row['center_mean']):.8g}</td>"
            f"<td>{float(row['edge_mean']):.8g}</td>"
            f"<td>{float(row['root_failure_fraction']):.8g}</td></tr>"
        )
        for row in case_rows
    )
    sections = [
        "<h2>Summary</h2>"
        "<ul>"
        f"<li>case_count: {summary['case_count']}</li>"
        f"<li>domain_kind: {summary['domain_kind']}</li>"
        f"<li>grid_shape: {summary['grid_shape']}</li>"
        f"<li>overall_passed: {summary['trend_assertions']['overall_passed']}</li>"
        "</ul>",
        "<h2>Trend Assertions</h2>"
        f"<table>{assertion_rows}</table>",
        "<h2>Case Metrics</h2>"
        "<table>"
        "<tr><th>case_id</th><th>input</th><th>mean_cs_ratio</th><th>mean_da_proxy</th>"
        "<th>center_mean</th><th>edge_mean</th><th>root_failure_fraction</th></tr>"
        f"{case_table_rows}</table>",
        "<h2>Artifacts</h2>"
        "<ul>"
        f"<li><a href=\"{summary['artifact_paths']['benchmark_cases']}\">{summary['artifact_paths']['benchmark_cases']}</a></li>"
        "<li><a href=\"outputs/benchmark_case_metrics.json\">outputs/benchmark_case_metrics.json</a></li>"
        "<li><a href=\"summary.json\">summary.json</a></li>"
        "<li><a href=\"config_resolved.yaml\">config_resolved.yaml</a></li>"
        "</ul>",
    ]

    if isinstance(physviz_context, Mapping):
        section_maps = physviz_context.get("section_plots", {})
        if isinstance(section_maps, Mapping):
            for title, key in (
                ("Input Field Maps", "input_fields"),
                ("Time-Space Maps", "time_space_maps"),
                ("Transport Term Importance", "transport_terms"),
                ("Reaction Term Importance (Sensitivity + Ablation)", "reaction_terms"),
                ("Net Term Importance", "net_terms"),
            ):
                links = section_maps.get(key, [])
                items = "".join(f'<li><a href="{escape(str(path))}">{escape(str(path))}</a></li>' for path in links)
                if not items:
                    items = "<li>None</li>"
                sections.append(f"<h2>{title}</h2><ul>{items}</ul>")

        reaction_scores = physviz_context.get("reaction_scores", [])
        net_scores = physviz_context.get("net_scores", [])
        sections.append(
            "<h2>Ranking Tables</h2>"
            "<h3>Reaction</h3>"
            f"{_ranking_table(reaction_scores if isinstance(reaction_scores, list) else [])}"
            "<h3>Net</h3>"
            f"{_ranking_table(net_scores if isinstance(net_scores, list) else [])}"
        )

    style = (
        "body { font-family: sans-serif; margin: 1.2rem 1.8rem; }"
        "table { border-collapse: collapse; margin-bottom: 1rem; }"
        "th, td { border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; }"
    )
    page = render_report_page(
        title=f"Wafer2D Benchmark: {run_id}",
        heading=f"Wafer2D Benchmark: {run_id}",
        style=style,
        sections=sections,
    )
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def run_wafer2d_benchmark(
    config_name: str = "smoke",
    overrides: Sequence[str] | None = None,
    *,
    with_physviz: bool = False,
    physviz_fast: bool = False,
) -> dict[str, Any]:
    """Run polar-domain wafer 2D trend benchmark and persist artifacts."""
    _require_numpy()
    default_overrides = [
        "domain.kind=wafer_2d_polar",
        "domain.nr=32",
        "domain.ntheta=96",
        "compute.engine=numpy",
        "time.mode=cvd_steady",
        "output.run_dir_name=benchmark_wafer2d",
    ]
    base_overrides = [*default_overrides, *(list(overrides or []))]
    base_spec = compose_sim_config(config_name, overrides=base_overrides)
    if base_spec.domain.kind != "wafer_2d_polar":
        raise ValueError("wafer2d benchmark requires domain.kind='wafer_2d_polar'")
    if len(base_spec.reference_plane.species) != 1:
        raise ValueError("wafer2d benchmark currently supports exactly one species.")
    species = str(base_spec.reference_plane.species[0])

    project_dir = Path(base_spec.output.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = next_run_dir(project_dir, base_spec.output.run_dir_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    compose_and_save_sim_config(
        run_dir / base_spec.output.resolved_config_filename,
        config_name=config_name,
        overrides=base_overrides,
    )
    base_grid = build_domain_grid(base_spec.domain)
    cases = build_wafer2d_cases()

    case_rows: list[dict[str, Any]] = []
    case_metrics_by_id: dict[str, dict[str, Any]] = {}
    thickness_list: list[np.ndarray] = []
    da_proxy_list: list[np.ndarray] = []
    cs_ratio_list: list[np.ndarray] = []
    input_cref_list: list[np.ndarray] = []
    root_failures: list[float] = []
    case_transport: dict[str, list[np.ndarray]] = {}
    case_net: dict[str, list[np.ndarray]] = {}

    representative_case = "CASE-06_SYN_SEEDED_LHHW_NET"
    representative_payload: dict[str, Any] | None = None

    for case in cases:
        case_overrides = [*base_overrides, *list(case.overrides)]
        file_input_map: np.ndarray | None = None
        if case.file_pattern is not None:
            payload_path, file_input_map = write_case_input_npz(
                case=case,
                grid=base_grid,
                output_dir=inputs_dir,
                c_ref_mol_m3=float(base_spec.inputs.c_ref_mol_m3),
                temperature_k=float(base_spec.inputs.temperature_k),
                species=species,
            )
            case_overrides.append(f"inputs.field_path={payload_path.as_posix()}")

        case_spec = compose_sim_config(config_name, overrides=case_overrides)
        validate_run_spec(case_spec)
        grid = build_domain_grid(case_spec.domain)
        fields = build_field_bundle(case_spec, grid)
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=case_spec.model,
            process_time_s=case_spec.time.process_time_s,
            solver_config=case_spec.solver,
        )

        thickness = np.asarray(result.thickness, dtype=float)
        da_proxy = np.asarray(result.diagnostics["Da_proxy"], dtype=float)
        cs_ratio = np.asarray(result.diagnostics["Cs_over_Cref"][species], dtype=float)
        input_cref = np.asarray(fields.C_ref[species], dtype=float)
        if file_input_map is not None:
            input_cref = np.asarray(file_input_map, dtype=float)
        kpi = compute_kpi_metrics(
            thickness,
            grid,
            spec_min=case_spec.kpi.spec_min,
            spec_max=case_spec.kpi.spec_max,
            ring_count=case_spec.kpi.ring_count,
        )
        theta_corr = _safe_corr(thickness, input_cref, np.asarray(grid.edge_mask, dtype=bool))
        root_failure_fraction = float(result.diagnostics.get("root_failure_fraction", 0.0))

        row = {
            "case_id": case.case_id,
            "description": case.description,
            "input_source_kind": str(case_spec.inputs.source_kind),
            "mean_cs_ratio": float(np.mean(cs_ratio[np.asarray(grid.edge_mask, dtype=bool)])),
            "mean_da_proxy": float(np.mean(da_proxy[np.asarray(grid.edge_mask, dtype=bool)])),
            "center_mean": float(kpi["center_mean"]) if kpi["center_mean"] is not None else float("nan"),
            "edge_mean": float(kpi["edge_mean"]) if kpi["edge_mean"] is not None else float("nan"),
            "theta_transfer_corr": theta_corr,
            "root_failure_fraction": root_failure_fraction,
            "overrides": case_overrides,
        }
        case_rows.append(row)
        case_metrics_by_id[case.case_id] = row

        thickness_list.append(thickness)
        da_proxy_list.append(da_proxy)
        cs_ratio_list.append(cs_ratio)
        input_cref_list.append(input_cref)
        root_failures.append(root_failure_fraction)

        if with_physviz:
            nu_map = _resolve_nu_map(case_spec, tuple(fields.C_ref.keys()))
            km = mass_transfer.compute_km_from_model_config(
                case_spec.model,
                grid=grid,
                omega_rad_s=float(case_spec.inputs.omega_rad_s),
            )
            transport = compute_transport_term_maps(result, fields, km, nu_map)
            net = compute_net_term_maps(result)
            for key, arr in transport.items():
                case_transport.setdefault(key, []).append(np.asarray(arr, dtype=float))
            for key, arr in net.items():
                case_net.setdefault(key, []).append(np.asarray(arr, dtype=float))

            if case.case_id == representative_case:
                representative_payload = {
                    "case_id": case.case_id,
                    "overrides": case_overrides,
                    "grid": grid,
                    "fields": fields,
                    "result": result,
                    "transport": transport,
                    "net": net,
                }

    thickness_stack = np.stack(thickness_list, axis=0)
    da_stack = np.stack(da_proxy_list, axis=0)
    cs_ratio_stack = np.stack(cs_ratio_list, axis=0)
    input_cref_stack = np.stack(input_cref_list, axis=0)
    store_payload = save_array_store(
        base_path=outputs_dir / "benchmark_cases",
        arrays={
            "thickness": thickness_stack,
            "da_proxy": da_stack,
            "cs_over_cref": cs_ratio_stack,
            "input_cref": input_cref_stack,
            "root_failure_fraction": np.asarray(root_failures, dtype=float),
        },
        store=str(base_spec.output.array_store),
    )
    (outputs_dir / "benchmark_case_metrics.json").write_text(
        json.dumps(case_rows, indent=2),
        encoding="utf-8",
    )

    trend_assertions = evaluate_trend_assertions(case_metrics_by_id)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "benchmark_wafer2d",
        "domain_kind": str(base_spec.domain.kind),
        "grid_shape": list(thickness_stack.shape[1:]),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "array_store_requested": str(base_spec.output.array_store),
        "artifact_store": {"benchmark_cases": store_payload["store_used"]},
        "artifact_paths": {
            "benchmark_cases": Path(store_payload["path"]).relative_to(run_dir).as_posix(),
        },
        "trend_assertions": trend_assertions,
    }

    physviz_context: dict[str, Any] | None = None
    if with_physviz:
        if representative_payload is None:
            representative_payload = {
                "case_id": case_rows[0]["case_id"],
                "overrides": case_rows[0]["overrides"],
                "grid": base_grid,
                "fields": None,
                "result": None,
                "transport": {},
                "net": {},
            }
        rep_overrides = list(representative_payload["overrides"])
        if physviz_fast:
            rep_nr = min(int(base_spec.domain.nr), 16)
            rep_ntheta = min(int(base_spec.domain.ntheta), 32)
            rep_overrides.extend([f"domain.nr={rep_nr}", f"domain.ntheta={rep_ntheta}"])
        rep_spec = compose_sim_config(config_name, overrides=rep_overrides)
        validate_run_spec(rep_spec)
        rep_grid = build_domain_grid(rep_spec.domain)
        rep_fields = build_field_bundle(rep_spec, rep_grid)
        rep_result = run_cvd_steady(
            grid=rep_grid,
            fields=rep_fields,
            model_config=rep_spec.model,
            process_time_s=rep_spec.time.process_time_s,
            solver_config=rep_spec.solver,
        )
        rep_nu = _resolve_nu_map(rep_spec, tuple(rep_fields.C_ref.keys()))
        rep_km = mass_transfer.compute_km_from_model_config(
            rep_spec.model,
            grid=rep_grid,
            omega_rad_s=float(rep_spec.inputs.omega_rad_s),
        )
        rep_transport = compute_transport_term_maps(rep_result, rep_fields, rep_km, rep_nu)
        rep_net = compute_net_term_maps(rep_result)

        cvd_snap = build_cvd_pseudo_time_snapshots(
            rep_spec,
            [0.10, 0.25, 0.50, 0.75, 1.00],
            enable_input_time_variation=True,
            input_variation_amplitude=0.24,
        )
        ald_snap = None
        if str(rep_spec.time.mode).strip().lower() == "ald_cycle" or list(getattr(rep_spec.time, "phases", []) or []):
            ald_spec = compose_sim_config(config_name, overrides=rep_overrides + ["time.mode=ald_cycle"])
            try:
                ald_snap = build_ald_phase_snapshots(ald_spec)
            except Exception:
                ald_snap = None
        reaction_importance = compute_reaction_term_importance(rep_spec, mode="sensitivity+ablation")

        physviz_data = {
            "cvd_snapshots": cvd_snap,
            "ald_snapshots": ald_snap,
            "transport_maps": rep_transport,
            "reaction_importance": reaction_importance,
            "net_maps": rep_net,
        }
        physviz_plots = write_physviz_report(run_dir=run_dir, grid=rep_grid, physviz_data=physviz_data)
        section_plots = _categorize_physviz_plots(physviz_plots)

        physviz_arrays: dict[str, np.ndarray] = {}
        _collect_np_arrays("cvd_snapshots", cvd_snap, physviz_arrays)
        if ald_snap is not None:
            _collect_np_arrays("ald_snapshots", ald_snap, physviz_arrays)
        _collect_np_arrays("transport_maps", rep_transport, physviz_arrays)
        _collect_np_arrays("reaction_sensitivity_maps", reaction_importance.get("sensitivity_maps", {}), physviz_arrays)
        _collect_np_arrays("reaction_ablation_maps", reaction_importance.get("ablation_maps", {}), physviz_arrays)
        _collect_np_arrays("net_maps", rep_net, physviz_arrays)
        physviz_store = save_array_store(
            base_path=outputs_dir / "physviz_maps",
            arrays=physviz_arrays,
            store=str(base_spec.output.array_store),
        )

        case_terms_arrays: dict[str, np.ndarray] = {}
        for key, stack in sorted(case_transport.items()):
            case_terms_arrays[f"transport__{key}"] = np.stack(stack, axis=0)
        for key, stack in sorted(case_net.items()):
            case_terms_arrays[f"net__{key}"] = np.stack(stack, axis=0)
        case_terms_store = save_array_store(
            base_path=outputs_dir / "physviz_case_terms",
            arrays=case_terms_arrays,
            store=str(base_spec.output.array_store),
        )

        reaction_scores = reaction_importance.get("scores", [])
        if not isinstance(reaction_scores, list):
            reaction_scores = []
        net_scores = _weighted_net_scores(rep_net, rep_grid)
        reaction_json_path = outputs_dir / "physviz_reaction_importance.json"
        reaction_json_path.write_text(
            json.dumps(
                {
                    "scores": reaction_scores,
                    "failed_terms": reaction_importance.get("failed_terms", []),
                    "relative_step": reaction_importance.get("relative_step"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        summary["artifact_store"].update(
            {
                "physviz_maps": physviz_store["store_used"],
                "physviz_case_terms": case_terms_store["store_used"],
            }
        )
        summary["artifact_paths"].update(
            {
                "physviz_maps": Path(physviz_store["path"]).relative_to(run_dir).as_posix(),
                "physviz_case_terms": Path(case_terms_store["path"]).relative_to(run_dir).as_posix(),
                "physviz_reaction_importance": reaction_json_path.relative_to(run_dir).as_posix(),
            }
        )
        summary["physviz"] = {
            "enabled": True,
            "fast_mode": bool(physviz_fast),
            "representative_case_id": representative_payload["case_id"],
            "plot_count": len(physviz_plots),
            "section_plot_count": {key: len(paths) for key, paths in section_plots.items()},
        }
        physviz_context = {
            "section_plots": section_plots,
            "reaction_scores": reaction_scores,
            "net_scores": net_scores,
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_benchmark_report(
        run_dir=run_dir,
        run_id=run_id,
        summary=summary,
        case_rows=case_rows,
        trend_assertions=trend_assertions,
        physviz_context=physviz_context,
    )
    update_project_files(project_dir, summary)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "case_metrics": case_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CVD wafer2d trend benchmark.")
    parser.add_argument("--config-name", default="smoke")
    parser.add_argument("--with-physviz", action="store_true", help="Generate physical-interpretability visualization pack.")
    parser.add_argument("--physviz-fast", action="store_true", help="Use reduced grid for physviz-only analysis.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_wafer2d_benchmark(
        config_name=args.config_name,
        overrides=args.overrides,
        with_physviz=bool(args.with_physviz),
        physviz_fast=bool(args.physviz_fast),
    )
    run_dir = result["run_dir"]
    overall = bool(result["summary"]["trend_assertions"]["overall_passed"])
    print(f"[benchmark_wafer2d] wrote artifacts to: {run_dir}")
    print(f"[benchmark_wafer2d] overall_passed={overall}")
    return 0 if overall else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
