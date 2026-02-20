"""Plot helpers for physical-interpretability benchmark outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deposim_sim.domain import DomainGrid

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


def _require_plot_deps() -> None:
    if np is None or plt is None:
        raise RuntimeError("NumPy and Matplotlib are required for physviz report generation.")


def _map2d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return np.repeat(arr[:, None], 2, axis=1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array for map plot, got shape {arr.shape}")
    return arr


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("center coordinates must be 1D with at least two points")
    edges = np.empty(arr.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
    edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return edges


def _draw_map(ax: Any, grid: DomainGrid, value: Any, *, cmap: str = "viridis") -> Any:
    data = _map2d(value)
    if grid.kind == "wafer_2d_polar" and grid.theta_edges_rad is not None and data.shape == grid.shape:
        r_edges = np.asarray(grid.r_edges_mm, dtype=float)
        theta_edges = np.asarray(grid.theta_edges_rad, dtype=float)
        rr, tt = np.meshgrid(r_edges, theta_edges, indexing="ij")
        xx = rr * np.cos(tt)
        yy = rr * np.sin(tt)
        masked = np.ma.array(data, mask=~np.asarray(grid.edge_mask, dtype=bool))
        mesh = ax.pcolormesh(xx, yy, masked, shading="auto", cmap=cmap)
        radius = float(grid.wafer_radius_mm)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    if grid.kind == "wafer_2d_xy" and grid.x_mm is not None and grid.y_mm is not None and data.shape == grid.shape:
        x_edges = _centers_to_edges(np.asarray(grid.x_mm, dtype=float))
        y_edges = _centers_to_edges(np.asarray(grid.y_mm, dtype=float))
        xx, yy = np.meshgrid(x_edges, y_edges, indexing="xy")
        masked = np.ma.array(data, mask=~np.asarray(grid.edge_mask, dtype=bool))
        mesh = ax.pcolormesh(xx, yy, masked, shading="auto", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return mesh

    mesh = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel("grid x-index")
    ax.set_ylabel("grid y-index")
    return mesh


def _save_map(path: Path, grid: DomainGrid, value: Any, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = _draw_map(ax, grid, value, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _sanitize(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def write_physviz_report(
    *,
    run_dir: Path,
    grid: DomainGrid,
    physviz_data: Mapping[str, Any],
) -> list[str]:
    """Write physviz PNG artifacts and return relative plot paths."""
    _require_plot_deps()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    def add(filename: str, value: Any, title: str, cmap: str = "viridis") -> None:
        _save_map(plots_dir / filename, grid, value, title, cmap)
        plot_paths.append(f"plots/{filename}")

    cvd_snap = physviz_data.get("cvd_snapshots")
    if isinstance(cvd_snap, Mapping):
        fractions = np.asarray(cvd_snap.get("time_fractions", []), dtype=float)
        thick = np.asarray(cvd_snap.get("thickness_snapshots", []), dtype=float)
        for idx, frac in enumerate(fractions):
            pct = int(round(float(frac) * 100.0))
            add(
                f"physviz_cvd_thickness_t{pct:02d}.png",
                thick[idx],
                f"CVD Thickness t/T={frac:.2f}",
            )
        delta = np.asarray(cvd_snap.get("delta_thickness_snapshots", []), dtype=float)
        if delta.size > 0:
            add(
                "physviz_cvd_delta_thickness_step.png",
                np.max(np.abs(delta), axis=0),
                "CVD Delta Thickness (max over steps)",
                cmap="coolwarm",
            )
        residual = np.asarray(cvd_snap.get("linearity_residual_max"), dtype=float)
        if residual.size > 0:
            add(
                "physviz_cvd_linearity_residual.png",
                residual,
                "CVD Linearity Residual (max abs)",
                cmap="coolwarm",
            )
        input_snapshots = cvd_snap.get("input_snapshots")
        if isinstance(input_snapshots, Mapping):
            for species, series in sorted(input_snapshots.items()):
                arr = np.asarray(series, dtype=float)
                if arr.ndim < 3 or arr.shape[0] != len(fractions):
                    continue
                tag = _sanitize(str(species))
                for idx, frac in enumerate(fractions):
                    pct = int(round(float(frac) * 100.0))
                    add(
                        f"physviz_input_cref_{tag}_t{pct:02d}.png",
                        arr[idx],
                        f"Input C_ref [{species}] t/T={frac:.2f}",
                    )
                delta_input = np.diff(arr, axis=0)
                if delta_input.size > 0:
                    add(
                        f"physviz_input_delta_cref_{tag}.png",
                        np.max(np.abs(delta_input), axis=0),
                        f"Input C_ref Delta [{species}] (max over steps)",
                        cmap="coolwarm",
                    )

    ald_snap = physviz_data.get("ald_snapshots")
    if isinstance(ald_snap, Mapping):
        phase_names = [str(name) for name in ald_snap.get("phase_names", [])]
        phase_thick = np.asarray(ald_snap.get("phase_thickness_snapshots", []), dtype=float)
        phase_cov = np.asarray(ald_snap.get("phase_coverage_snapshots", []), dtype=float)
        cumulative = np.asarray(ald_snap.get("cumulative_thickness_snapshots", []), dtype=float)
        for idx, name in enumerate(phase_names):
            add(
                f"physviz_ald_phase_thickness_p{idx+1:02d}.png",
                phase_thick[idx],
                f"ALD Phase Thickness: {name}",
            )
            add(
                f"physviz_ald_phase_coverage_p{idx+1:02d}.png",
                phase_cov[idx],
                f"ALD Phase Coverage: {name}",
                cmap="magma",
            )
        if cumulative.size > 0:
            add(
                "physviz_ald_cumulative_thickness.png",
                cumulative[-1],
                "ALD Cumulative Thickness",
            )

    transport = physviz_data.get("transport_maps")
    if isinstance(transport, Mapping):
        for key, title, cmap in (
            ("transport_capacity__", "Transport Capacity", "viridis"),
            ("reaction_demand__", "Reaction Demand", "magma"),
            ("depletion_ratio__", "Depletion Ratio", "inferno"),
            ("utilization__", "Utilization", "plasma"),
        ):
            for map_name, value in sorted(transport.items()):
                if not str(map_name).startswith(key):
                    continue
                species = str(map_name).split("__", 1)[1]
                filename = f"physviz_{key[:-2]}_{_sanitize(species)}.png"
                add(filename, value, f"{title} [{species}]", cmap=cmap)

    reaction = physviz_data.get("reaction_importance")
    if isinstance(reaction, Mapping):
        sens_maps = reaction.get("sensitivity_maps", {})
        if isinstance(sens_maps, Mapping):
            for term, value in sorted(sens_maps.items()):
                add(
                    f"physviz_reaction_sensitivity_{_sanitize(term)}.png",
                    value,
                    f"Reaction Sensitivity [{term}]",
                    cmap="coolwarm",
                )
        abla_maps = reaction.get("ablation_maps", {})
        if isinstance(abla_maps, Mapping):
            for term, value in sorted(abla_maps.items()):
                add(
                    f"physviz_reaction_ablation_{_sanitize(term)}.png",
                    value,
                    f"Reaction Ablation [{term}]",
                    cmap="coolwarm",
                )
        scores = reaction.get("scores", [])
        if isinstance(scores, list) and scores:
            labels = [str(item.get("term_name", "")) for item in scores]
            vals = np.asarray([float(item.get("importance_score", 0.0)) for item in scores], dtype=float)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(labels[::-1], vals[::-1], color="#1f77b4")
            ax.set_xlabel("importance_score")
            ax.set_title("Reaction Term Importance Ranking")
            fig.tight_layout()
            path = plots_dir / "physviz_reaction_importance_rank.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            plot_paths.append("plots/physviz_reaction_importance_rank.png")

    net_maps = physviz_data.get("net_maps")
    if isinstance(net_maps, Mapping):
        if "dep_rate" in net_maps:
            add("physviz_net_dep_rate.png", net_maps["dep_rate"], "Net Equation: Deposition Rate")
        if "etch_rate" in net_maps:
            add("physviz_net_etch_rate.png", net_maps["etch_rate"], "Net Equation: Etch Rate", cmap="magma")
        if "loss_rate" in net_maps:
            add("physviz_net_loss_rate.png", net_maps["loss_rate"], "Net Equation: Loss Rate", cmap="magma")

        rank_labels: list[str] = []
        rank_vals: list[float] = []
        weights = np.asarray(grid.area_weights_mm2, dtype=float)
        mask = np.asarray(grid.edge_mask, dtype=bool)
        for label, key in (
            ("etch_fraction_of_dep", "etch_fraction_of_dep"),
            ("loss_fraction_of_dep", "loss_fraction_of_dep"),
        ):
            if key not in net_maps:
                continue
            val = np.asarray(net_maps[key], dtype=float)
            valid = np.isfinite(val) & np.isfinite(weights) & mask
            if not np.any(valid):
                score = 0.0
            else:
                score = float(np.sum(val[valid] * weights[valid]) / max(np.sum(weights[valid]), 1.0e-12))
            rank_labels.append(label)
            rank_vals.append(score)
        if rank_labels:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(rank_labels, rank_vals, color="#2ca02c")
            ax.set_ylabel("weighted mean contribution")
            ax.set_title("Net Term Contribution Ranking")
            fig.tight_layout()
            path = plots_dir / "physviz_net_contribution_rank.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            plot_paths.append("plots/physviz_net_contribution_rank.png")

    return plot_paths


__all__ = ["write_physviz_report"]
