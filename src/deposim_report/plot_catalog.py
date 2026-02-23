"""Centralized static plot naming/style catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    filename: str
    source_key: str
    title: str
    cmap: str = "viridis"
    discrete: bool = False


RUN_REPORT_PRIMARY_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec("thickness_map", "thickness_map.png", "h_nm", "Thickness Map", "viridis"),
    PlotSpec("da_proxy_map", "da_proxy_map.png", "Da_proxy", "Da Proxy Map", "inferno"),
    PlotSpec("phi_B_map", "phi_B_map.png", "phi_B", "phi_B Map", "magma"),
    PlotSpec("f_I_map", "f_I_map.png", "f_I", "f_I Map", "viridis"),
)

RUN_REPORT_SOLVER_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec(
        "solver_health_map",
        "solver_health_map.png",
        "root_iteration_count|root_status_map",
        "Solver Health",
        "magma",
    ),
)

RUN_REPORT_COMPARISON_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec("measurement_map", "measurement_map.png", "measurement_thickness", "Measurement Thickness Map", "viridis"),
    PlotSpec(
        "comparison_error_map",
        "comparison_error_map.png",
        "comparison_error_nm",
        "Simulation - Measurement Error Map",
        "coolwarm",
    ),
)

RUN_REPORT_PROFILE_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec("radial_profile", "radial_profile.png", "h_nm", "Radial Profile", "line"),
)

RUN_REPORT_IDENTIFIABILITY_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec(
        "identifiability_correlation",
        "identifiability_correlation.png",
        "identifiability.correlation_matrix",
        "Identifiability Correlation",
        "coolwarm",
    ),
)

BENCHMARK_PHYSVIZ_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec("physviz_h_nm", "physviz_h_nm.png", "h_nm", "Representative h_nm", "viridis"),
    PlotSpec("physviz_phi_B", "physviz_phi_B.png", "phi_B", "Representative phi_B", "viridis"),
    PlotSpec("physviz_f_I", "physviz_f_I.png", "f_I", "Representative f_I", "viridis"),
)

DOE_KPI_MAPS: tuple[PlotSpec, ...] = (
    PlotSpec("kpi_nu_percent", "kpi_nu_percent.png", "nu_percent", "NU Percent", "line"),
    PlotSpec("kpi_center_edge_delta", "kpi_center_edge_delta.png", "center_edge_delta", "Center-Edge Delta", "line"),
)

DOE_ZREF_PLOT = PlotSpec("zref_sensitivity", "zref_sensitivity.png", "nu_percent_vs_z_ref", "z_ref Sensitivity", "line")


def sanitize_plot_token(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def run_report_species_spec(metric_key: str, species: str) -> PlotSpec:
    metric = str(metric_key)
    tag = sanitize_plot_token(species)
    if metric == "Cs_over_Cref":
        return PlotSpec(
            plot_id=f"cs_over_cref_{tag}",
            filename=f"cs_over_cref_{tag}.png",
            source_key=f"{metric}.{species}",
            title=f"cs_over_cref [{species}]",
            cmap="viridis",
        )
    if metric == "apparent_orders":
        return PlotSpec(
            plot_id=f"n_app_{tag}",
            filename=f"n_app_{tag}.png",
            source_key=f"{metric}.{species}",
            title=f"n_app [{species}]",
            cmap="coolwarm",
        )
    raise ValueError(f"unsupported run_report species metric: {metric_key}")


def benchmark_physviz_specs(*, fast: bool) -> list[PlotSpec]:
    specs = list(BENCHMARK_PHYSVIZ_MAPS)
    if fast:
        return specs[:2]
    return specs


def to_plot_record(spec: PlotSpec, *, rel_path: str) -> dict[str, Any]:
    return {
        "plot_id": spec.plot_id,
        "path": rel_path,
        "source_key": spec.source_key,
        "cmap": spec.cmap,
        "discrete": bool(spec.discrete),
    }


__all__ = [
    "PlotSpec",
    "RUN_REPORT_PRIMARY_MAPS",
    "RUN_REPORT_SOLVER_MAPS",
    "RUN_REPORT_COMPARISON_MAPS",
    "RUN_REPORT_PROFILE_PLOTS",
    "RUN_REPORT_IDENTIFIABILITY_MAPS",
    "BENCHMARK_PHYSVIZ_MAPS",
    "DOE_KPI_MAPS",
    "DOE_ZREF_PLOT",
    "sanitize_plot_token",
    "run_report_species_spec",
    "benchmark_physviz_specs",
    "to_plot_record",
]
