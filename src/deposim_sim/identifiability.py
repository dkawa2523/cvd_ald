"""Identifiability diagnostics using finite-difference sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .common.nested_path import get_nested, set_nested
from .pipeline import run_aib_from_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for identifiability diagnostics.")


def _thickness_vector(run_spec: Any) -> np.ndarray:
    result = run_aib_from_spec(run_spec)
    return np.asarray(result.thickness, dtype=float).ravel()


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


def compute_identifiability_diagnostics(
    run_spec: Any,
    *,
    parameter_paths: Sequence[str],
    relative_step: float = 1.0e-2,
    low_sensitivity_threshold: float = 1.0e-10,
    correlation_threshold: float = 0.98,
) -> dict[str, Any]:
    """Compute finite-difference sensitivity and degeneracy diagnostics."""

    _require_numpy()
    if not parameter_paths:
        raise ValueError("parameter_paths must be non-empty")
    if relative_step <= 0.0:
        raise ValueError(f"relative_step must be > 0, got {relative_step}")

    vectors: list[np.ndarray] = []
    norms: dict[str, float] = {}

    for path in parameter_paths:
        base_value = float(get_nested(run_spec, path))
        step = relative_step * max(abs(base_value), 1.0)
        if step <= 0.0:
            step = relative_step

        plus_spec = deepcopy(run_spec)
        minus_spec = deepcopy(run_spec)
        set_nested(plus_spec, path, base_value + step)
        set_nested(minus_spec, path, base_value - step)

        plus = _thickness_vector(plus_spec)
        minus = _thickness_vector(minus_spec)
        sens_vec = (plus - minus) / (2.0 * step)
        vectors.append(sens_vec)
        norms[path] = float(np.linalg.norm(sens_vec))

    jacobian = np.stack(vectors, axis=0)
    corr = _pairwise_correlation(jacobian)

    low_sensitivity = [path for path, value in norms.items() if value <= low_sensitivity_threshold]
    high_correlation_pairs: list[dict[str, Any]] = []
    n_params = len(parameter_paths)
    for i in range(n_params):
        for j in range(i + 1, n_params):
            value = float(corr[i, j])
            if abs(value) >= correlation_threshold:
                high_correlation_pairs.append(
                    {"pair": [parameter_paths[i], parameter_paths[j]], "correlation": value}
                )

    warnings: list[str] = []
    if low_sensitivity:
        warnings.append(f"Low sensitivity parameter(s): {', '.join(low_sensitivity)}")
    if high_correlation_pairs:
        warnings.append("Potential parameter degeneracy detected from high sensitivity correlation.")

    return {
        "parameter_paths": list(parameter_paths),
        "sensitivity_norms": norms,
        "correlation_matrix": corr,
        "correlation_threshold": float(correlation_threshold),
        "low_sensitivity_threshold": float(low_sensitivity_threshold),
        "high_correlation_pairs": high_correlation_pairs,
        "low_sensitivity_parameters": low_sensitivity,
        "degeneracy_warning": bool(warnings),
        "warnings": warnings,
    }


__all__ = ["compute_identifiability_diagnostics"]
