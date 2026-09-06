"""Mars-van Krevelen redox-reservoir state model.

The dimensionless state ``oxidized_fraction`` is the fraction of surface or
lattice redox capacity available to react with role A.  A consumes that
capacity while producing film; role B restores it.  Transport uses the same
independent film-coefficient closure as the other process models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1.0e-30


def _require_nonnegative_finite(name: str, value: np.ndarray) -> None:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True)
class MVKStepResult:
    oxidized_fraction: np.ndarray
    h_nm: np.ndarray
    reduction_rate: np.ndarray
    regeneration_rate: np.ndarray
    cs_a: np.ndarray
    cs_b: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MVKStateResult:
    time_s: np.ndarray
    oxidized_fraction: np.ndarray
    h_nm: np.ndarray
    reduction_rate: np.ndarray
    regeneration_rate: np.ndarray
    cs_a: np.ndarray
    cs_b: np.ndarray
    j_a_surface: np.ndarray
    j_b_surface: np.ndarray
    oxidized_fraction_history: np.ndarray
    h_nm_history: np.ndarray
    reduction_rate_history: np.ndarray
    regeneration_rate_history: np.ndarray
    cs_a_history: np.ndarray
    cs_b_history: np.ndarray
    j_a_surface_history: np.ndarray
    j_b_surface_history: np.ndarray
    diagnostics: dict[str, Any]


def _surface_concentration(
    reference: np.ndarray,
    km: np.ndarray,
    demand_velocity: np.ndarray,
) -> np.ndarray:
    """Solve ``km*(Cb-Cs)=demand_velocity*Cs`` for ``Cs``."""

    reference = np.clip(np.asarray(reference, dtype=float), 0.0, np.inf)
    km = np.clip(np.asarray(km, dtype=float), 0.0, np.inf)
    demand = np.clip(np.asarray(demand_velocity, dtype=float), 0.0, np.inf)
    inv_km = np.where(np.isinf(km), 0.0, 1.0 / np.maximum(km, _EPS))
    return reference / np.maximum(1.0 + demand * inv_km, _EPS)


def _rates(
    oxidized_fraction: np.ndarray,
    *,
    cref_a: np.ndarray,
    cref_b: np.ndarray,
    km_a: np.ndarray,
    km_b: np.ndarray,
    k_reduce: np.ndarray,
    k_regenerate: np.ndarray,
    gamma_s: np.ndarray,
    nu_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chi = np.clip(np.asarray(oxidized_fraction, dtype=float), 0.0, 1.0)
    reduced = 1.0 - chi
    cs_a = _surface_concentration(
        cref_a,
        km_a,
        gamma_s * k_reduce * chi,
    )
    cs_b = _surface_concentration(
        cref_b,
        km_b,
        gamma_s * nu_b * k_regenerate * reduced,
    )
    reduction_rate = k_reduce * cs_a * chi
    regeneration_rate = k_regenerate * cs_b * reduced
    return reduction_rate, regeneration_rate, cs_a, cs_b


def step_mvk_implicit(
    *,
    oxidized_fraction_n: np.ndarray,
    h_n_nm: np.ndarray,
    dt_s: float,
    cref_a: np.ndarray,
    cref_b: np.ndarray,
    km_a: np.ndarray,
    km_b: np.ndarray,
    k_reduce: np.ndarray,
    k_regenerate: np.ndarray,
    gamma_s: np.ndarray,
    nu_b: np.ndarray,
    alpha_h: np.ndarray,
    max_iter: int,
    state_tol: float,
) -> MVKStepResult:
    """Advance the redox state with bounded implicit Euler and bisection."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")
    chi_n = np.clip(np.asarray(oxidized_fraction_n, dtype=float), 0.0, 1.0)
    h_n = np.asarray(h_n_nm, dtype=float)
    lo = np.zeros_like(chi_n)
    hi = np.ones_like(chi_n)

    def residual(trial: np.ndarray) -> np.ndarray:
        reduction, regeneration, _cs_a, _cs_b = _rates(
            trial,
            cref_a=cref_a,
            cref_b=cref_b,
            km_a=km_a,
            km_b=km_b,
            k_reduce=k_reduce,
            k_regenerate=k_regenerate,
            gamma_s=gamma_s,
            nu_b=nu_b,
        )
        return trial - chi_n - dt_s * (regeneration - reduction)

    g_lo = residual(lo)
    g_hi = residual(hi)
    bracketed = np.isfinite(g_lo) & np.isfinite(g_hi) & (g_lo <= 0.0) & (g_hi >= 0.0)
    active = bracketed.copy()
    iteration_count = np.zeros_like(chi_n, dtype=int)
    for _ in range(max_iter):
        if not np.any(active):
            break
        mid = 0.5 * (lo + hi)
        g_mid = residual(mid)
        move_hi = active & (g_mid >= 0.0)
        move_lo = active & ~move_hi
        hi = np.where(move_hi, mid, hi)
        lo = np.where(move_lo, mid, lo)
        iteration_count[active] += 1
        active &= np.abs(hi - lo) > state_tol

    chi_next = np.clip(0.5 * (lo + hi), 0.0, 1.0)
    fallback_mask = ~bracketed
    if np.any(fallback_mask):
        reduction_n, regeneration_n, _cs_a_n, _cs_b_n = _rates(
            chi_n,
            cref_a=cref_a,
            cref_b=cref_b,
            km_a=km_a,
            km_b=km_b,
            k_reduce=k_reduce,
            k_regenerate=k_regenerate,
            gamma_s=gamma_s,
            nu_b=nu_b,
        )
        explicit = np.clip(
            chi_n + dt_s * (regeneration_n - reduction_n), 0.0, 1.0
        )
        chi_next = np.where(fallback_mask, explicit, chi_next)
    reduction, regeneration, cs_a, cs_b = _rates(
        chi_next,
        cref_a=cref_a,
        cref_b=cref_b,
        km_a=km_a,
        km_b=km_b,
        k_reduce=k_reduce,
        k_regenerate=k_regenerate,
        gamma_s=gamma_s,
        nu_b=nu_b,
    )
    h_next = h_n + dt_s * alpha_h * gamma_s * reduction
    return MVKStepResult(
        oxidized_fraction=chi_next,
        h_nm=h_next,
        reduction_rate=reduction,
        regeneration_rate=regeneration,
        cs_a=cs_a,
        cs_b=cs_b,
        diagnostics={
            "iteration_count": iteration_count,
            "unconverged_count": int(np.sum(active)),
            "non_bracketed_count": int(np.sum(fallback_mask)),
            "fallback_mask": fallback_mask,
            "redox_balance_rate": regeneration - reduction,
        },
    )


def run_mvk_state(
    *,
    c_a: np.ndarray,
    c_b: np.ndarray,
    km_provider: Any,
    time_s: np.ndarray,
    dt_max_s: float,
    oxidized_fraction0: np.ndarray,
    h0_nm: np.ndarray,
    k_reduce: np.ndarray,
    k_regenerate: np.ndarray,
    gamma_s: np.ndarray,
    nu_b: np.ndarray,
    alpha_h: np.ndarray,
    max_iter: int,
    state_tol: float,
) -> MVKStateResult:
    """Integrate A-driven reduction/growth and B-driven reservoir regeneration.

    Concentrations are ``[time, *space]`` in kmol/m3. ``k_reduce`` and
    ``k_regenerate`` are m3/(kmol s), ``gamma_s`` is kmol/m2, ``nu_b`` is
    dimensionless, and ``alpha_h`` is nm m2/kmol.
    """

    if dt_max_s <= 0.0:
        raise ValueError("dt_max_s must be > 0")
    if max_iter <= 0 or state_tol <= 0.0:
        raise ValueError("max_iter and state_tol must be > 0")
    for name, value in (
        ("k_reduce", k_reduce),
        ("k_regenerate", k_regenerate),
        ("gamma_s", gamma_s),
        ("nu_b", nu_b),
        ("alpha_h", alpha_h),
    ):
        _require_nonnegative_finite(name, value)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    c_a = np.asarray(c_a, dtype=float)
    c_b = np.asarray(c_b, dtype=float)
    if time.size < 2 or c_a.shape[0] < time.size - 1 or c_b.shape[0] < time.size - 1:
        raise ValueError("MvK execution requires concentration frames for every time interval")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be strictly increasing")

    chi = np.clip(np.asarray(oxidized_fraction0, dtype=float), 0.0, 1.0)
    h = np.asarray(h0_nm, dtype=float).copy()
    total_iterations = np.zeros_like(chi, dtype=int)
    unconverged_total = 0
    non_bracketed_total = 0
    fallback_count_map = np.zeros_like(chi, dtype=int)
    substep_count = 0
    final: MVKStepResult | None = None

    km_a_initial = np.asarray(km_provider.get_km("A", t_index=0), dtype=float)
    km_b_initial = np.asarray(km_provider.get_km("B", t_index=0), dtype=float)
    reduction_initial, regeneration_initial, cs_a_initial, cs_b_initial = _rates(
        chi,
        cref_a=c_a[0],
        cref_b=c_b[0],
        km_a=km_a_initial,
        km_b=km_b_initial,
        k_reduce=k_reduce,
        k_regenerate=k_regenerate,
        gamma_s=gamma_s,
        nu_b=nu_b,
    )
    chi_history = [chi.copy()]
    h_history = [h.copy()]
    reduction_history = [reduction_initial.copy()]
    regeneration_history = [regeneration_initial.copy()]
    cs_a_history = [cs_a_initial.copy()]
    cs_b_history = [cs_b_initial.copy()]
    j_a_history = [(gamma_s * reduction_initial).copy()]
    j_b_history = [(gamma_s * nu_b * regeneration_initial).copy()]

    for index, interval_s in enumerate(np.diff(time)):
        n_substeps = max(int(np.ceil(float(interval_s) / dt_max_s)), 1)
        dt_s = float(interval_s) / n_substeps
        km_a = np.asarray(km_provider.get_km("A", t_index=index), dtype=float)
        km_b = np.asarray(km_provider.get_km("B", t_index=index), dtype=float)
        for _ in range(n_substeps):
            final = step_mvk_implicit(
                oxidized_fraction_n=chi,
                h_n_nm=h,
                dt_s=dt_s,
                cref_a=c_a[index],
                cref_b=c_b[index],
                km_a=km_a,
                km_b=km_b,
                k_reduce=k_reduce,
                k_regenerate=k_regenerate,
                gamma_s=gamma_s,
                nu_b=nu_b,
                alpha_h=alpha_h,
                max_iter=max_iter,
                state_tol=state_tol,
            )
            chi = final.oxidized_fraction
            h = final.h_nm
            total_iterations += np.asarray(final.diagnostics["iteration_count"], dtype=int)
            unconverged_total += int(final.diagnostics["unconverged_count"])
            non_bracketed_total += int(final.diagnostics["non_bracketed_count"])
            fallback_count_map += np.asarray(
                final.diagnostics["fallback_mask"], dtype=bool
            ).astype(int)
            substep_count += 1
        chi_history.append(chi.copy())
        h_history.append(h.copy())
        reduction_history.append(final.reduction_rate.copy())
        regeneration_history.append(final.regeneration_rate.copy())
        cs_a_history.append(final.cs_a.copy())
        cs_b_history.append(final.cs_b.copy())
        j_a_history.append((gamma_s * final.reduction_rate).copy())
        j_b_history.append((gamma_s * nu_b * final.regeneration_rate).copy())

    assert final is not None
    j_a_surface = gamma_s * final.reduction_rate
    j_b_surface = gamma_s * nu_b * final.regeneration_rate
    relaxation_rate = k_reduce * final.cs_a + k_regenerate * final.cs_b
    return MVKStateResult(
        time_s=time.copy(),
        oxidized_fraction=chi,
        h_nm=h,
        reduction_rate=final.reduction_rate,
        regeneration_rate=final.regeneration_rate,
        cs_a=final.cs_a,
        cs_b=final.cs_b,
        j_a_surface=j_a_surface,
        j_b_surface=j_b_surface,
        oxidized_fraction_history=np.stack(chi_history, axis=0),
        h_nm_history=np.stack(h_history, axis=0),
        reduction_rate_history=np.stack(reduction_history, axis=0),
        regeneration_rate_history=np.stack(regeneration_history, axis=0),
        cs_a_history=np.stack(cs_a_history, axis=0),
        cs_b_history=np.stack(cs_b_history, axis=0),
        j_a_surface_history=np.stack(j_a_history, axis=0),
        j_b_surface_history=np.stack(j_b_history, axis=0),
        diagnostics={
            "substep_count": substep_count,
            "unconverged_total": unconverged_total,
            "non_bracketed_total": non_bracketed_total,
            "fallback_count_map": fallback_count_map,
            "iteration_count": total_iterations,
            "oxidized_fraction_min": float(np.min(chi)),
            "oxidized_fraction_max": float(np.max(chi)),
            "redox_balance_rate": final.regeneration_rate - final.reduction_rate,
            "relaxation_rate_s-1": relaxation_rate,
            "relaxation_time_s": 1.0 / np.maximum(relaxation_rate, _EPS),
        },
    )


__all__ = [
    "MVKStateResult",
    "MVKStepResult",
    "run_mvk_state",
    "step_mvk_implicit",
]
