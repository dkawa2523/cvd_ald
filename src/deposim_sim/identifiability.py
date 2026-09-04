"""Identifiability diagnostics using finite-difference sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .common.path_tools import get_attr_path, set_attr_path
from .pipeline import run_sim_from_spec
from .measurement_adapter import observation_residuals

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for identifiability diagnostics.")


def _pairwise_correlation(jacobian: np.ndarray) -> np.ndarray:
    n_params = int(jacobian.shape[0])
    corr = np.eye(n_params, dtype=float)
    for i in range(n_params):
        vi = jacobian[i]
        ni = float(np.linalg.norm(vi))
        for j in range(i + 1, n_params):
            vj = jacobian[j]
            nj = float(np.linalg.norm(vj))
            if ni <= 0.0 or nj <= 0.0:
                value = 0.0
            else:
                value = float(np.dot(vi, vj) / (ni * nj))
            value = float(np.clip(value, -1.0, 1.0))
            corr[i, j] = value
            corr[j, i] = value
    return corr


def analyze_sensitivities(
    jacobian: np.ndarray,
    parameter_paths: Sequence[str],
    *,
    low_sensitivity_threshold: float = 1.0e-10,
    correlation_threshold: float = 0.98,
) -> dict[str, Any]:
    """Diagnose scaled local directions, including dependencies of 3+ parameters."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != len(parameter_paths) or not np.all(np.isfinite(jacobian)):
        raise ValueError("finite sensitivity matrix must have one row per parameter")
    norms = dict(zip(parameter_paths, np.linalg.norm(jacobian, axis=1).tolist()))
    corr = _pairwise_correlation(jacobian)
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = max(float(singular[0]) * np.sqrt(np.finfo(float).eps), low_sensitivity_threshold) if singular.size else low_sensitivity_threshold
    rank = int(np.count_nonzero(singular > tolerance))
    low = [path for path, value in norms.items() if value <= low_sensitivity_threshold]
    pairs = [
        {"pair": [parameter_paths[i], parameter_paths[j]], "correlation": float(corr[i, j])}
        for i in range(len(parameter_paths)) for j in range(i + 1, len(parameter_paths))
        if abs(corr[i, j]) >= correlation_threshold
    ]
    if u.shape[1] < len(parameter_paths):
        u, _ = np.linalg.qr(u, mode="complete")
    combinations = [
        {path: float(value) for path, value in zip(parameter_paths, direction) if abs(value) > 1.0e-6}
        for direction in u[:, rank:].T
    ]
    warnings = []
    if low:
        warnings.append("Low sensitivity parameter(s): " + ", ".join(low))
    if rank < len(parameter_paths):
        warnings.append(f"Only {rank} of {len(parameter_paths)} local parameter directions are resolved.")
    if pairs:
        warnings.append("Parameters have nearly parallel observation sensitivities.")
    return {
        "parameter_paths": list(parameter_paths), "sensitivity_norms": norms,
        "correlation_matrix": corr, "correlation_threshold": correlation_threshold,
        "low_sensitivity_threshold": low_sensitivity_threshold,
        "high_correlation_pairs": pairs, "low_sensitivity_parameters": low,
        "singular_values": singular.tolist(), "effective_rank": rank,
        "parameter_count": len(parameter_paths), "weak_parameter_combinations": combinations,
        "degeneracy_warning": bool(warnings), "warnings": warnings,
    }


def compute_identifiability_diagnostics(
    run_spec: Any,
    *,
    parameter_paths: Sequence[str],
    relative_step: float = 1.0e-2,
    low_sensitivity_threshold: float = 1.0e-10,
    correlation_threshold: float = 0.98,
    run_specs: Sequence[Any] | None = None,
    condition_weights: Sequence[float] | None = None,
    condition_names: Sequence[str] | None = None,
    local_parameter_paths: Sequence[str] = (),
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Scale-aware local identifiability in the same observation space as fitting.

    Shared parameter directions span every condition. Condition-specific effective
    parameters get separate directions. No unobserved mesh point adds evidence.
    Without measurements this remains a simulator sensitivity exploration.
    """
    _require_numpy()
    if not parameter_paths or relative_step <= 0:
        raise ValueError("parameter_paths and a positive relative_step are required")
    specs = list(run_specs) if run_specs is not None else [run_spec]
    weights = np.asarray(condition_weights if condition_weights is not None else np.ones(len(specs)), dtype=float)
    if weights.shape != (len(specs),) or np.any(weights < 0) or not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        raise ValueError("condition weights must be finite, nonnegative, and have positive sum")
    weights /= np.sum(weights)
    names = list(condition_names) if condition_names is not None else [str(i + 1) for i in range(len(specs))]
    bounds = parameter_bounds or {}
    local = set(local_parameter_paths)
    blocks = []
    for spec, weight in zip(specs, weights):
        result = run_sim_from_spec(spec)
        observation = result.diagnostics.get("observation")
        measured = observation is not None
        values = observation_residuals(result) if measured else np.asarray(result.thickness).ravel()
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("identifiability requires finite observations")
        sigma_known = measured and observation.get("sigma_nm") is not None
        target = np.asarray(observation["target_nm"] if measured else result.thickness, dtype=float)
        scale = 1.0 if sigma_known else max(float(np.sqrt(np.mean(target**2))), float(np.sqrt(np.mean(values**2))), np.finfo(float).tiny)
        blocks.append((measured, values.size, np.sqrt(weight / values.size) / scale))
    directions = []
    labels = []
    for path in parameter_paths:
        scopes = [[i] for i in range(len(specs))] if path in local else [list(range(len(specs)))]
        for scope in scopes:
            label = f"{names[scope[0]]}:{path}" if path in local else path
            labels.append(label)
            pieces = []
            for i, spec in enumerate(specs):
                measured, count, output_scale = blocks[i]
                if i not in scope:
                    pieces.append(np.zeros(count))
                    continue
                base = float(get_attr_path(spec, path))
                low, high = bounds.get(path, (float("-inf"), float("inf")))
                parameter_scale = abs(base)
                if parameter_scale == 0:
                    parameter_scale = high - low if np.isfinite(low) and np.isfinite(high) else 1.0
                step = relative_step * parameter_scale
                plus_value, minus_value = min(base + step, high), max(base - step, low)
                if plus_value <= minus_value:
                    pieces.append(np.zeros(count))
                    continue
                plus_spec, minus_spec = deepcopy(spec), deepcopy(spec)
                set_attr_path(plus_spec, path, plus_value, create_missing_mappings=False)
                set_attr_path(minus_spec, path, minus_value, create_missing_mappings=False)
                plus_run, minus_run = run_sim_from_spec(plus_spec), run_sim_from_spec(minus_spec)
                plus = observation_residuals(plus_run) if measured else np.asarray(plus_run.thickness).ravel()
                minus = observation_residuals(minus_run) if measured else np.asarray(minus_run.thickness).ravel()
                if plus.size != count or minus.size != count:
                    raise ValueError("observation support changed during parameter perturbation")
                pieces.append((plus - minus) / (plus_value - minus_value) * parameter_scale * output_scale)
            directions.append(np.concatenate(pieces))
    out = analyze_sensitivities(
        np.stack(directions), labels, low_sensitivity_threshold=low_sensitivity_threshold,
        correlation_threshold=correlation_threshold,
    )
    out.update(condition_count=len(specs), observation_count=sum(b[1] for b in blocks),
               observation_space="measurement" if all(b[0] for b in blocks) else "simulation")
    return out


__all__ = ["compute_identifiability_diagnostics", "analyze_sensitivities"]
