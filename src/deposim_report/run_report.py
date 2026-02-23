"""Plotting and HTML report generation for simulation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from deposim_sim.domain import DomainGrid, radial_profile
from deposim_sim.common.render_tri import render_unstructured_map
from deposim_sim.output_manifest import artifact_links

from .html_page import render_report_page
from .plot_catalog import (
    PlotSpec,
    RUN_REPORT_COMPARISON_MAPS,
    RUN_REPORT_IDENTIFIABILITY_MAPS,
    RUN_REPORT_PRIMARY_MAPS,
    RUN_REPORT_PROFILE_PLOTS,
    RUN_REPORT_SOLVER_MAPS,
    run_report_species_spec,
    to_plot_record,
)

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _map2d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return np.repeat(arr[:, None], 2, axis=1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got shape {arr.shape}")
    return arr


def _save_map(
    path: Path,
    *,
    grid: DomainGrid,
    value: Any,
    title: str,
    cmap: str = "viridis",
    xy_mm: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    discrete: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if grid.kind == "from_fluent_xy":
        if xy_mm is None:
            raise ValueError("xy_mm is required for from_fluent_xy map rendering")
        mesh = render_unstructured_map(
            ax,
            xy_mm=np.asarray(xy_mm, dtype=float),
            values=np.asarray(value, dtype=float),
            valid_mask=valid_mask,
            cmap=cmap,
            discrete=discrete,
        )
    else:
        mesh = ax.imshow(_map2d(value), origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_run_report(
    *,
    run_dir: Path,
    run_id: str,
    grid: DomainGrid,
    thickness: Any,
    diagnostics: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_links: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate standard plots and `report.html` for a run directory."""

    if np is None or plt is None:
        raise RuntimeError("NumPy and Matplotlib are required for report generation.")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    plot_records: list[dict[str, Any]] = []
    xy_mm_raw = diagnostics.get("xy_mm")
    xy_mm = None if xy_mm_raw is None else np.asarray(xy_mm_raw, dtype=float)
    valid_mask = np.asarray(grid.edge_mask, dtype=bool)

    def add_map(filename: str, data: Any, title: str, cmap: str = "viridis", *, discrete: bool = False) -> None:
        _save_map(
            plots_dir / filename,
            grid=grid,
            value=data,
            title=title,
            cmap=cmap,
            xy_mm=xy_mm,
            valid_mask=valid_mask,
            discrete=discrete,
        )
        rel = f"plots/{filename}"
        plot_paths.append(rel)
        plot_records.append(
            {
                "plot_id": filename.rsplit(".", 1)[0],
                "path": rel,
                "source_key": filename.rsplit(".", 1)[0],
                "cmap": cmap,
                "discrete": bool(discrete),
            }
        )

    def add_spec(spec: PlotSpec, data: Any) -> None:
        add_map(spec.filename, data, spec.title, cmap=spec.cmap, discrete=spec.discrete)
        plot_records[-1] = to_plot_record(spec, rel_path=f"plots/{spec.filename}")

    for spec in RUN_REPORT_PRIMARY_MAPS:
        if spec.source_key == "h_nm":
            value = thickness
        else:
            value = diagnostics.get(spec.source_key, np.zeros(grid.shape))
        add_spec(spec, value)

    comparison_rows: list[tuple[str, float]] = []
    meas = diagnostics.get("measurement_thickness")
    if meas is not None:
        meas_arr = np.asarray(meas, dtype=float)
        if meas_arr.shape == np.asarray(thickness, dtype=float).shape:
            err = np.asarray(thickness, dtype=float) - meas_arr
            for spec, value in zip(RUN_REPORT_COMPARISON_MAPS, (meas_arr, err)):
                add_spec(spec, value)
            finite = np.isfinite(err)
            if np.any(finite):
                mae = float(np.mean(np.abs(err[finite])))
                rmse = float(np.sqrt(np.mean(err[finite] ** 2)))
                comparison_rows.append(("comparison_mae", mae))
                comparison_rows.append(("comparison_rmse", rmse))

    identifiability_html = ""
    identifiability = diagnostics.get("identifiability")
    if isinstance(identifiability, Mapping):
        corr = identifiability.get("correlation_matrix")
        if corr is not None:
            add_spec(RUN_REPORT_IDENTIFIABILITY_MAPS[0], corr)
        norms = identifiability.get("sensitivity_norms", {})
        norm_rows = ""
        if isinstance(norms, Mapping):
            norm_rows = "".join(
                f"<tr><th>{escape(str(name))}</th><td>{float(value):.8g}</td></tr>"
                for name, value in sorted(norms.items())
            )
        warnings = identifiability.get("warnings", [])
        warning_items = ""
        if isinstance(warnings, Sequence):
            warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in warnings)
        if not warning_items:
            warning_items = "<li>None</li>"
        identifiability_html = (
            "<h2>Identifiability</h2>"
            "<p>Finite-difference sensitivity and pairwise correlation diagnostics.</p>"
            f"<table>{norm_rows}</table>"
            f"<ul>{warning_items}</ul>"
        )

    r_mm, profile = radial_profile(np.asarray(thickness, dtype=float), grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_mm, profile, lw=2.0)
    ax.set(xlabel="Radius [mm]", ylabel="Thickness", title="Radial Profile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "radial_profile.png", dpi=140)
    plt.close(fig)
    radial_spec = RUN_REPORT_PROFILE_PLOTS[0]
    plot_paths.append(f"plots/{radial_spec.filename}")
    plot_records.append(to_plot_record(radial_spec, rel_path=f"plots/{radial_spec.filename}"))

    for key in ("Cs_over_Cref", "apparent_orders"):
        values = diagnostics.get(key, {})
        if isinstance(values, Mapping):
            for species in sorted(values):
                spec = run_report_species_spec(key, species)
                add_spec(spec, values[species])

    solver_spec = RUN_REPORT_SOLVER_MAPS[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title, cmap, discrete in (
        (axes[0], diagnostics.get("root_iteration_count", np.zeros(grid.shape)), "Solver Iterations", "magma", False),
        (axes[1], diagnostics.get("root_status_map", np.zeros(grid.shape)), "Solver Status", "tab20", True),
    ):
        if grid.kind == "from_fluent_xy":
            if xy_mm is None:
                raise ValueError("xy_mm is required for from_fluent_xy map rendering")
            mesh = render_unstructured_map(
                ax,
                xy_mm=np.asarray(xy_mm, dtype=float),
                values=np.asarray(field, dtype=float),
                valid_mask=valid_mask,
                cmap=cmap,
                discrete=discrete,
            )
        else:
            mesh = ax.imshow(_map2d(field), origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / solver_spec.filename, dpi=140)
    plt.close(fig)
    plot_paths.append(f"plots/{solver_spec.filename}")
    plot_records.append(to_plot_record(solver_spec, rel_path=f"plots/{solver_spec.filename}"))

    rows = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(summary[k]))}</td></tr>" for k in sorted(summary))
    solver_warning_html = ""
    non_bracketed_total = int(diagnostics.get("non_bracketed_total", summary.get("non_bracketed_total", 0)))
    warn_threshold = float(diagnostics.get("solver_warning_non_bracketed_threshold", 0))
    if non_bracketed_total > warn_threshold:
        solver_warning_html = (
            "<div style='border:1px solid #d9534f;background:#fff1f1;padding:0.6rem;margin-bottom:0.8rem;'>"
            f"<strong>Solver Warning:</strong> non_bracketed_total={non_bracketed_total} exceeds threshold={warn_threshold:.0f}."
            "</div>"
        )
    comparison_html = ""
    if comparison_rows:
        comparison_html = "<h2>Comparison Metrics</h2><table>" + "".join(
            f"<tr><th>{escape(name)}</th><td>{value:.8g}</td></tr>" for name, value in comparison_rows
        ) + "</table>"

    benchmark_html = ""
    benchmark = summary.get("benchmark")
    if not isinstance(benchmark, Mapping):
        benchmark = diagnostics.get("benchmark")
    if isinstance(benchmark, Mapping):
        preferred = (
            "engine_requested",
            "engine_selected",
            "engine_execution_backend",
            "requested_engine",
            "engine_used",
            "repeats",
            "grid_cells",
            "best_timing_sec",
            "mean_timing_sec",
            "throughput_cells_per_s",
        )
        ordered = [key for key in preferred if key in benchmark]
        ordered.extend(key for key in sorted(benchmark) if key not in ordered)
        bench_rows = ""
        for key in ordered:
            value = benchmark[key]
            if isinstance(value, float):
                text = f"{value:.8g}"
            else:
                text = escape(str(value))
            bench_rows += f"<tr><th>{escape(str(key))}</th><td>{text}</td></tr>"
        benchmark_html = "<h2>Benchmark</h2><table>" + bench_rows + "</table>"
    resolved_output_links = list(output_links or [])
    if manifest is not None:
        resolved_output_links = artifact_links(manifest)
    output_items = "".join(f'<li><a href="{escape(p)}">{escape(p)}</a></li>' for p in resolved_output_links)
    plot_items = "".join(f'<li><a href="{escape(p)}">{escape(p)}</a></li>' for p in plot_paths)
    figures = "".join(f'<figure><img src="{escape(p)}" alt="{escape(p)}" /></figure>' for p in plot_paths)
    style = (
        "body { font-family: sans-serif; margin: 1.2rem 1.8rem; } "
        "table { border-collapse: collapse; } "
        "th, td { border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; } "
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 1rem; }"
    )
    html = render_report_page(
        title=f"deposim report: {run_id}",
        heading=f"Run Report: {run_id}",
        style=style,
        sections=[
            solver_warning_html,
            f"<h2>Summary</h2><table>{rows}</table>",
            comparison_html,
            identifiability_html,
            benchmark_html,
            f"<h2>Outputs</h2><ul>{output_items}</ul>",
            f"<h2>Plots</h2><ul>{plot_items}</ul>",
            figures,
        ],
    )
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    return plot_records


__all__ = ["render_unstructured_map", "write_run_report"]
