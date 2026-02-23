"""Score decomposition utilities for AIB optimization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

_EPS = 1.0e-12


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for deposim_opt.objective")


def _mean_finite(values: Any, *, fallback: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(fallback)
    return float(np.mean(arr[finite]))


def huber_loss(residual_nm: np.ndarray, *, delta_nm: float) -> float:
    abs_r = np.abs(residual_nm)
    quad = np.minimum(abs_r, delta_nm)
    lin = abs_r - quad
    return float(np.mean(0.5 * quad**2 + delta_nm * lin))


def l1_loss(residual_nm: np.ndarray) -> float:
    return float(np.mean(np.abs(residual_nm)))


def data_loss(
    *,
    residual_nm: Any,
    loss_kind: str,
    huber_delta_nm: float,
    fallback_values_nm: Any | None = None,
) -> float:
    _require_numpy()
    residual = np.asarray(residual_nm, dtype=float)
    finite = np.isfinite(residual)
    if np.any(finite):
        data = residual[finite]
    elif fallback_values_nm is not None:
        fallback = np.asarray(fallback_values_nm, dtype=float)
        finite_fb = np.isfinite(fallback)
        data = fallback[finite_fb] if np.any(finite_fb) else np.zeros((1,), dtype=float)
    else:
        data = np.zeros((1,), dtype=float)

    kind = str(loss_kind).strip().lower()
    if kind == "l1":
        return l1_loss(data)
    return huber_loss(data, delta_nm=float(huber_delta_nm))


def solver_penalty(*, diagnostics: Mapping[str, Any], lambda_solver: float) -> float:
    if float(lambda_solver) <= 0.0:
        return 0.0

    non_bracket_map = diagnostics.get("root_non_bracket_count_map", diagnostics.get("root_status_map", 0.0))
    iter_map = diagnostics.get("root_iteration_count", 0.0)

    arr_non = np.asarray(non_bracket_map, dtype=float)
    arr_iter = np.asarray(iter_map, dtype=float)

    n_pts = max(int(arr_iter.size), 1)
    non_bracket_mean = float(np.nansum(np.clip(arr_non, 0.0, np.inf))) / float(n_pts)
    iter_mean = _mean_finite(arr_iter, fallback=0.0)
    iter_over = max(iter_mean - 4.0, 0.0) / 10.0
    return float(lambda_solver) * (non_bracket_mean + iter_over)


def physics_penalty(
    *,
    fields: Mapping[str, Any],
    role_has_i: bool,
    role_has_b: bool,
    lambda_phys: float,
    phi_b_min: float,
) -> float:
    if float(lambda_phys) <= 0.0:
        return 0.0

    penalty_raw = 0.0
    if role_has_i:
        penalty_raw += np.clip(_mean_finite(fields.get("f_I", 0.0), fallback=0.0), 0.0, np.inf)

    if role_has_b:
        phi_b = _mean_finite(fields.get("phi_B", 0.0), fallback=0.0)
        threshold = max(float(phi_b_min), _EPS)
        penalty_raw += max(threshold - phi_b, 0.0) / threshold

    return float(lambda_phys) * float(penalty_raw)


def prior_penalty(*, lambda_prior: float, prior_terms: Sequence[float] | None) -> float:
    terms = [float(v) for v in list(prior_terms or []) if np.isfinite(float(v))]
    if float(lambda_prior) <= 0.0 or not terms:
        return 0.0
    return float(lambda_prior) * 0.5 * float(np.mean(np.asarray(terms, dtype=float)))


def complexity_penalty(*, lambda_complex: float, role_has_i: bool, role_has_b: bool) -> float:
    if float(lambda_complex) <= 0.0:
        return 0.0
    return float(lambda_complex) * float(int(role_has_i) + int(role_has_b))


def evaluate_candidate_score(
    *,
    residual_nm: Any,
    fields: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    role_has_i: bool,
    role_has_b: bool,
    objective: Mapping[str, Any] | None,
    lambda_complex: float,
    prior_terms: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute decomposed candidate score components.

    Returned keys are stable and intended for ranking/report tables.
    """

    _require_numpy()
    objective_cfg = dict(objective or {})
    penalties = dict(objective_cfg.get("penalties", {}) or {})

    huber_delta = float(objective_cfg.get("huber_delta_nm", 10.0))
    loss_kind = str(objective_cfg.get("loss", "huber"))
    lambda_solver = float(penalties.get("lambda_solver", 0.0))
    lambda_phys = float(penalties.get("lambda_phys", 0.0))
    lambda_prior = float(penalties.get("lambda_prior", 0.0))
    phi_b_min = float(objective_cfg.get("phi_B_min", 0.05))

    loss_data = data_loss(
        residual_nm=residual_nm,
        loss_kind=loss_kind,
        huber_delta_nm=huber_delta,
        fallback_values_nm=fields.get("h_nm"),
    )
    pen_solver = solver_penalty(diagnostics=diagnostics, lambda_solver=lambda_solver)
    pen_phys = physics_penalty(
        fields=fields,
        role_has_i=role_has_i,
        role_has_b=role_has_b,
        lambda_phys=lambda_phys,
        phi_b_min=phi_b_min,
    )
    pen_prior = prior_penalty(lambda_prior=lambda_prior, prior_terms=prior_terms)
    pen_complex = complexity_penalty(
        lambda_complex=float(lambda_complex),
        role_has_i=role_has_i,
        role_has_b=role_has_b,
    )

    score_total = loss_data + pen_solver + pen_phys + pen_prior + pen_complex
    return {
        "loss_data": float(loss_data),
        "penalty_solver": float(pen_solver),
        "penalty_phys": float(pen_phys),
        "penalty_prior": float(pen_prior),
        "penalty_complexity": float(pen_complex),
        "score_total": float(score_total),
    }


__all__ = [
    "complexity_penalty",
    "data_loss",
    "evaluate_candidate_score",
    "huber_loss",
    "l1_loss",
    "physics_penalty",
    "prior_penalty",
    "solver_penalty",
]
