"""Plotting and HTML report generation for simulation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from deposim_sim.domain import DomainGrid, radial_profile

from .html_page import render_report_page

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


def _save_map(path: Path, value: Any, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
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
    output_links: Sequence[str],
) -> list[str]:
    """Generate standard plots and `report.html` for a run directory."""

    if np is None or plt is None:
        raise RuntimeError("NumPy and Matplotlib are required for report generation.")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    def add_map(filename: str, data: Any, title: str, cmap: str = "viridis") -> None:
        _save_map(plots_dir / filename, data, title, cmap)
        plot_paths.append(f"plots/{filename}")

    add_map("thickness_map.png", thickness, "Thickness Map")
    add_map("da_proxy_map.png", diagnostics.get("Da_proxy", np.zeros(grid.shape)), "Da Proxy Map", cmap="inferno")

    comparison_rows: list[tuple[str, float]] = []
    meas = diagnostics.get("measurement_thickness")
    if meas is not None:
        meas_arr = np.asarray(meas, dtype=float)
        if meas_arr.shape == np.asarray(thickness, dtype=float).shape:
            err = np.asarray(thickness, dtype=float) - meas_arr
            add_map("measurement_map.png", meas_arr, "Measurement Thickness Map")
            add_map("comparison_error_map.png", err, "Simulation - Measurement Error Map", cmap="coolwarm")
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
            add_map(
                "identifiability_correlation.png",
                corr,
                "Identifiability Correlation",
                cmap="coolwarm",
            )
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
    plot_paths.append("plots/radial_profile.png")

    for key, name, cmap in (
        ("Cs_over_Cref", "cs_over_cref", "viridis"),
        ("apparent_orders", "n_app", "coolwarm"),
    ):
        values = diagnostics.get(key, {})
        if isinstance(values, Mapping):
            for species in sorted(values):
                add_map(f"{name}_{species}.png", values[species], f"{name} [{species}]", cmap=cmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title, cmap in (
        (axes[0], diagnostics.get("root_iteration_count", np.zeros(grid.shape)), "Solver Iterations", "magma"),
        (axes[1], diagnostics.get("root_status_map", np.zeros(grid.shape)), "Solver Status", "tab20"),
    ):
        mesh = ax.imshow(_map2d(field), origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / "solver_health_map.png", dpi=140)
    plt.close(fig)
    plot_paths.append("plots/solver_health_map.png")

    rows = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(summary[k]))}</td></tr>" for k in sorted(summary))
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
    output_items = "".join(f'<li><a href="{escape(p)}">{escape(p)}</a></li>' for p in output_links)
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
    return plot_paths


__all__ = ["write_run_report"]
