"""Minimal synthetic assimilation loop on CPU."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from deposim_schema import compose_sim_config
from deposim_sim.domain import build_domain_grid
from deposim_sim.physics.cvd_steady import run_cvd_steady
from deposim_sim.synthetic_inputs import build_synthetic_field_bundle

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for assimilation.")


def _simulate_thickness(config_name: str, overrides: Sequence[str]) -> np.ndarray:
    run_spec = compose_sim_config(config_name, overrides=list(overrides))
    grid = build_domain_grid(run_spec.domain)
    fields = build_synthetic_field_bundle(run_spec, grid)
    result = run_cvd_steady(
        grid=grid,
        fields=fields,
        model_config=run_spec.model,
        process_time_s=run_spec.time.process_time_s,
        solver_config=run_spec.solver,
    )
    return np.asarray(result.thickness, dtype=float)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in sorted(summary.items()))
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>assimilation report</title></head><body>"
        "<h1>Assimilation Report</h1><table>"
        f"{rows}</table></body></html>"
    )
    path.write_text(html, encoding="utf-8")


def run_synthetic_assimilation(
    *,
    sim_config_name: str = "smoke",
    output_dir: str | Path = "results/assimilation",
    target_k0: float = 1.6,
    initial_k0: float = 0.7,
    max_iters: int = 12,
    sim_overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit `k0` on deterministic synthetic data and save outputs."""

    _require_numpy()
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    if target_k0 <= 0.0 or initial_k0 <= 0.0:
        raise ValueError("target_k0 and initial_k0 must be > 0")

    base_overrides = list(sim_overrides or ())
    target = _simulate_thickness(sim_config_name, [*base_overrides, f"model.kinetics_params.k0={target_k0}"])

    def loss_for(k0_value: float) -> float:
        pred = _simulate_thickness(sim_config_name, [*base_overrides, f"model.kinetics_params.k0={k0_value}"])
        return _rmse(pred, target)

    current = float(initial_k0)
    best_loss = loss_for(current)
    initial_loss = best_loss

    for _ in range(max_iters):
        candidates = [current * factor for factor in (0.6, 0.8, 1.0, 1.25, 1.6)]
        scored = [(loss_for(value), value) for value in candidates if value > 0.0]
        candidate_loss, candidate_value = min(scored, key=lambda item: item[0])
        if candidate_loss + 1.0e-14 >= best_loss:
            break
        best_loss = candidate_loss
        current = float(candidate_value)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted_params = {"k0": current}
    summary = {
        "sim_config_name": sim_config_name,
        "target_k0": float(target_k0),
        "initial_k0": float(initial_k0),
        "fitted_k0": float(current),
        "initial_loss": float(initial_loss),
        "final_loss": float(best_loss),
        "loss_reduction": float(initial_loss - best_loss),
    }
    (out_dir / "fitted_params.json").write_text(json.dumps(fitted_params, indent=2), encoding="utf-8")
    (out_dir / "assimilation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out_dir / "report.html", summary)
    return summary


__all__ = ["run_synthetic_assimilation"]
