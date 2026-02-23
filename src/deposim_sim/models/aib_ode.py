"""Unified AIB-ODE core for A/AI/AB/AIB classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_EPS = 1.0e-30


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for deposim_sim.models.aib_ode")


@dataclass(frozen=True)
class StepResult:
    theta_next: np.ndarray
    h_next: np.ndarray
    diagnostics: dict[str, Any]


def _as_array(value: Any, shape: tuple[int, ...], *, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = np.full(shape, float(arr), dtype=float)
    else:
        out = np.broadcast_to(arr, shape).astype(float, copy=True)
    if nonnegative:
        out = np.clip(out, 0.0, np.inf)
    return out


def compute_theta_star(theta_a: np.ndarray, K_I: np.ndarray, cref_i: np.ndarray) -> np.ndarray:
    denom = 1.0 + K_I * cref_i
    return np.clip((1.0 - theta_a) / np.maximum(denom, _EPS), 0.0, 1.0)


def compute_cs_a(
    *,
    cref_a: np.ndarray,
    theta_a: np.ndarray,
    theta_star: np.ndarray,
    km_a: np.ndarray,
    k_ads: np.ndarray,
    k_des: np.ndarray,
    gamma_s: np.ndarray,
    m_ads: int,
) -> np.ndarray:
    num = km_a * cref_a + gamma_s * k_des * theta_a
    den = km_a + gamma_s * k_ads * np.power(np.clip(theta_star, 0.0, 1.0), float(m_ads))
    return np.clip(num / np.maximum(den, _EPS), 0.0, np.inf)


def compute_cs_b(
    *,
    cref_b: np.ndarray,
    theta_a: np.ndarray,
    theta_star: np.ndarray,
    km_b: np.ndarray,
    k_rxn: np.ndarray,
    gamma_s: np.ndarray,
    c_b_scale: np.ndarray,
    p_a: int,
    p_star: int,
) -> np.ndarray:
    base = gamma_s * k_rxn * np.power(np.clip(theta_a, 0.0, 1.0), float(p_a)) * np.power(
        np.clip(theta_star, 0.0, 1.0),
        float(p_star),
    ) / np.maximum(c_b_scale, _EPS)
    den = km_b + base
    return np.clip(km_b * cref_b / np.maximum(den, _EPS), 0.0, np.inf)


def compute_revent(
    *,
    theta_a: np.ndarray,
    theta_star: np.ndarray,
    k_rxn: np.ndarray,
    p_a: int,
    p_star: int,
    has_b: bool,
    cs_b: np.ndarray,
    c_b_scale: np.ndarray,
) -> np.ndarray:
    base = k_rxn * np.power(np.clip(theta_a, 0.0, 1.0), float(p_a)) * np.power(np.clip(theta_star, 0.0, 1.0), float(p_star))
    if not has_b:
        return np.clip(base, 0.0, np.inf)
    b_term = cs_b / np.maximum(c_b_scale, _EPS)
    return np.clip(base * b_term, 0.0, np.inf)


def _theta_rhs(
    theta_trial: np.ndarray,
    *,
    cref_a: np.ndarray,
    cref_i: np.ndarray,
    cref_b: np.ndarray,
    km_a: np.ndarray,
    km_b: np.ndarray,
    k_ads: np.ndarray,
    k_des: np.ndarray,
    k_rxn: np.ndarray,
    K_I: np.ndarray,
    gamma_s: np.ndarray,
    nu_a: np.ndarray,
    c_b_scale: np.ndarray,
    m_ads: int,
    p_a: int,
    p_star: int,
    has_b: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    theta_star = compute_theta_star(theta_trial, K_I, cref_i)
    cs_a = compute_cs_a(
        cref_a=cref_a,
        theta_a=theta_trial,
        theta_star=theta_star,
        km_a=km_a,
        k_ads=k_ads,
        k_des=k_des,
        gamma_s=gamma_s,
        m_ads=m_ads,
    )
    if has_b:
        cs_b = compute_cs_b(
            cref_b=cref_b,
            theta_a=theta_trial,
            theta_star=theta_star,
            km_b=km_b,
            k_rxn=k_rxn,
            gamma_s=gamma_s,
            c_b_scale=c_b_scale,
            p_a=p_a,
            p_star=p_star,
        )
    else:
        cs_b = np.full(theta_trial.shape, np.nan, dtype=float)

    r_event = compute_revent(
        theta_a=theta_trial,
        theta_star=theta_star,
        k_rxn=k_rxn,
        p_a=p_a,
        p_star=p_star,
        has_b=has_b,
        cs_b=np.nan_to_num(cs_b, nan=0.0),
        c_b_scale=c_b_scale,
    )

    rhs = k_ads * cs_a * np.power(np.clip(theta_star, 0.0, 1.0), float(m_ads)) - k_des * theta_trial - nu_a * r_event
    return rhs, {"theta_star": theta_star, "cs_a": cs_a, "cs_b": cs_b, "r_event": r_event}


def step_theta_implicit(
    *,
    theta_n: np.ndarray,
    h_n: np.ndarray,
    dt_s: float,
    cref_a: np.ndarray,
    cref_i: np.ndarray,
    cref_b: np.ndarray,
    km_a: np.ndarray,
    km_b: np.ndarray,
    k_ads: np.ndarray,
    k_des: np.ndarray,
    k_rxn: np.ndarray,
    K_I: np.ndarray,
    gamma_s: np.ndarray,
    nu_a: np.ndarray,
    alpha_h: np.ndarray,
    c_b_scale: np.ndarray,
    m_ads: int,
    p_a: int,
    p_star: int,
    has_b: bool,
    max_iter: int,
    theta_tol: float,
) -> StepResult:
    _require_numpy()
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")

    theta_n = np.clip(np.asarray(theta_n, dtype=float), 0.0, 1.0)
    h_n = np.asarray(h_n, dtype=float)
    shape = theta_n.shape

    lo = np.zeros(shape, dtype=float)
    hi = np.ones(shape, dtype=float)

    def g(theta_trial: np.ndarray) -> np.ndarray:
        rhs, _ = _theta_rhs(
            theta_trial,
            cref_a=cref_a,
            cref_i=cref_i,
            cref_b=cref_b,
            km_a=km_a,
            km_b=km_b,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            K_I=K_I,
            gamma_s=gamma_s,
            nu_a=nu_a,
            c_b_scale=c_b_scale,
            m_ads=m_ads,
            p_a=p_a,
            p_star=p_star,
            has_b=has_b,
        )
        return theta_trial - theta_n - dt_s * rhs

    g_lo = g(lo)
    g_hi = g(hi)

    bracket_ok = np.isfinite(g_lo) & np.isfinite(g_hi) & ((g_lo == 0.0) | (g_hi == 0.0) | (g_lo * g_hi <= 0.0))
    non_bracketed_count = int(np.sum(~bracket_ok))
    iter_count = np.zeros(shape, dtype=int)

    theta_next = np.empty(shape, dtype=float)
    if np.any(bracket_ok):
        lo_b = lo.copy()
        hi_b = hi.copy()
        active = bracket_ok.copy()
        for _ in range(max_iter):
            if not np.any(active):
                break
            mid = 0.5 * (lo_b + hi_b)
            g_mid = g(mid)
            move_left = (g_lo * g_mid) <= 0.0
            hi_b = np.where(move_left, mid, hi_b)
            lo_b = np.where(move_left, lo_b, mid)
            g_hi = np.where(move_left, g_mid, g_hi)
            g_lo = np.where(move_left, g_lo, g_mid)
            iter_count[active] += 1
            converged = active & (np.abs(hi_b - lo_b) <= theta_tol)
            active = active & ~converged
        theta_bisect = np.clip(0.5 * (lo_b + hi_b), 0.0, 1.0)
        theta_next[bracket_ok] = theta_bisect[bracket_ok]

    if non_bracketed_count > 0:
        rhs_n, _ = _theta_rhs(
            theta_n,
            cref_a=cref_a,
            cref_i=cref_i,
            cref_b=cref_b,
            km_a=km_a,
            km_b=km_b,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            K_I=K_I,
            gamma_s=gamma_s,
            nu_a=nu_a,
            c_b_scale=c_b_scale,
            m_ads=m_ads,
            p_a=p_a,
            p_star=p_star,
            has_b=has_b,
        )
        theta_explicit = np.clip(theta_n + dt_s * rhs_n, 0.0, 1.0)
        theta_next[~bracket_ok] = theta_explicit[~bracket_ok]

    rhs_next, raw = _theta_rhs(
        theta_next,
        cref_a=cref_a,
        cref_i=cref_i,
        cref_b=cref_b,
        km_a=km_a,
        km_b=km_b,
        k_ads=k_ads,
        k_des=k_des,
        k_rxn=k_rxn,
        K_I=K_I,
        gamma_s=gamma_s,
        nu_a=nu_a,
        c_b_scale=c_b_scale,
        m_ads=m_ads,
        p_a=p_a,
        p_star=p_star,
        has_b=has_b,
    )
    _ = rhs_next
    h_next = h_n + dt_s * alpha_h * gamma_s * raw["r_event"]

    return StepResult(
        theta_next=theta_next,
        h_next=h_next,
        diagnostics={
            "theta_star": raw["theta_star"],
            "CsA": raw["cs_a"],
            "CsB": raw["cs_b"],
            "r_event": raw["r_event"],
            "non_bracketed_count": non_bracketed_count,
            "bracket_ok_mask": bracket_ok,
            "iteration_count": iter_count,
            "fallback_mask": ~bracket_ok,
        },
    )


def compute_diagnostics(
    *,
    theta_a: np.ndarray,
    theta_star: np.ndarray,
    cs_a: np.ndarray,
    cs_b: np.ndarray,
    cref_a: np.ndarray,
    cref_b: np.ndarray,
    cref_i: np.ndarray,
    gamma_s: np.ndarray,
    k_rxn: np.ndarray,
    km_b: np.ndarray,
    c_b_scale: np.ndarray,
    p_a: int,
    p_star: int,
    K_I: np.ndarray,
    has_b: bool,
) -> dict[str, np.ndarray]:
    cref_a_safe = np.where(cref_a > _EPS, cref_a, np.nan)
    cs_a_ratio = cs_a / cref_a_safe

    if has_b:
        cref_b_safe = np.where(cref_b > _EPS, cref_b, np.nan)
        cs_b_ratio = cs_b / cref_b_safe
        phi_b = (
            gamma_s
            * k_rxn
            * np.power(np.clip(theta_a, 0.0, 1.0), float(p_a))
            * np.power(np.clip(theta_star, 0.0, 1.0), float(p_star))
            / np.maximum(c_b_scale, _EPS)
        ) / np.maximum(km_b, _EPS)
    else:
        cs_b_ratio = np.full(theta_a.shape, np.nan, dtype=float)
        phi_b = np.full(theta_a.shape, np.nan, dtype=float)

    f_i = 1.0 / np.maximum(1.0 + K_I * cref_i, _EPS)

    return {
        "theta_A": theta_a,
        "theta_star": theta_star,
        "CsA_over_CrefA": cs_a_ratio,
        "CsB_over_CrefB": cs_b_ratio,
        "phi_B": phi_b,
        "f_I": f_i,
    }


__all__ = [
    "StepResult",
    "compute_theta_star",
    "compute_cs_a",
    "compute_cs_b",
    "compute_revent",
    "step_theta_implicit",
    "compute_diagnostics",
]
