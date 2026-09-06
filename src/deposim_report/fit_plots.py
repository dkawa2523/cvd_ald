"""Simple scientific figures for optimizer and objective diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finish(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_fit_diagnostic_plots(
    *, run_dir: Path, records: list[dict[str, Any]], top_n: int = 8
) -> list[dict[str, Any]]:
    """Write convergence and loss-component figures from already fitted records."""

    if not records:
        return []
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rows: list[dict[str, Any]] = []

    trace = list(records[0].get("optimization_trace", []) or [])
    if trace:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        repetitions = sorted({int(row.get("repetition", 1)) for row in trace})
        all_best: list[float] = []
        for repetition in repetitions:
            rows = [row for row in trace if int(row.get("repetition", 1)) == repetition]
            x = np.asarray([float(row["trial"]) for row in rows], dtype=float)
            y = np.asarray([float(row["best_score"]) for row in rows], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.plot(x[finite], y[finite], linewidth=1.5, label=f"Seed {int(rows[0]['seed'])}")
                all_best.extend(y[finite].tolist())
        if all_best and min(all_best) > 0.0 and max(all_best) / min(all_best) >= 100.0:
            ax.set_yscale("log")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best loss")
        ax.set_title("Optimization convergence")
        ax.grid(alpha=0.25, linewidth=0.6)
        if len(repetitions) > 1:
            ax.legend(frameon=False)
        path = plot_dir / "optimization_convergence.png"
        _finish(fig, path)
        plot_rows.append(
            {
                "plot_id": "optimization_convergence",
                "path": "plots/optimization_convergence.png",
                "source_key": "optimization_trace.best_score",
            }
        )

    selected = records[: max(1, min(int(top_n), len(records)))]
    components = ("loss_data", "penalty_solver", "penalty_prior")
    values = {
        name: np.asarray(
            [float(dict(row.get("best_components", {}) or {}).get(name, 0.0)) for row in selected],
            dtype=float,
        )
        for name in components
    }
    if all(np.all(np.isfinite(value)) for value in values.values()):
        fig, ax = plt.subplots(figsize=(max(5.2, 0.58 * len(selected)), 3.5))
        x = np.arange(len(selected), dtype=float)
        bottom = np.zeros(len(selected), dtype=float)
        labels = {"loss_data": "Data", "penalty_solver": "Solver", "penalty_prior": "Prior"}
        colors = {"loss_data": "#4C78A8", "penalty_solver": "#F58518", "penalty_prior": "#54A24B"}
        for name in components:
            ax.bar(x, values[name], bottom=bottom, label=labels[name], color=colors[name], width=0.72)
            bottom += values[name]
        ax.set_xticks(x, [f"{index + 1}\n{row.get('class_id', '')}" for index, row in enumerate(selected)])
        ax.set_xlabel("Rank and role class")
        loss_unit = str(dict(selected[0].get("loss_definition", {}) or {}).get("unit", ""))
        ax.set_ylabel(f"Loss ({loss_unit})" if loss_unit else "Loss")
        ax.set_title("Fit loss by candidate")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if any(np.any(values[name] > 0.0) for name in ("penalty_solver", "penalty_prior")):
            ax.legend(frameon=False)
        path = plot_dir / "loss_components.png"
        _finish(fig, path)
        plot_rows.append(
            {
                "plot_id": "fit_loss_components",
                "path": "plots/loss_components.png",
                "source_key": "best_components",
            }
        )

    return plot_rows


__all__ = ["write_fit_diagnostic_plots"]
