"""Minimal ALD role-state assimilation kernel."""

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
        raise RuntimeError("NumPy is required for deposim_sim.models.ald_role_state")


@dataclass(frozen=True)
class ALDRoleStateResult:
    theta_a: np.ndarray
    theta_i: np.ndarray
    theta_free: np.ndarray
    h_nm: np.ndarray
    r_event: np.ndarray
    cs_a: np.ndarray
    cs_b: np.ndarray
    j_a_surface: np.ndarray
    j_b_surface: np.ndarray
    diagnostics: dict[str, Any]


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)


def _surface_sink_concentration(
    cref: np.ndarray,
    km: np.ndarray,
    sink_coefficient: np.ndarray,
    gamma_s: np.ndarray,
) -> np.ndarray:
    cref = np.clip(np.asarray(cref, dtype=float), 0.0, np.inf)
    km = np.clip(np.asarray(km, dtype=float), 0.0, np.inf)
    demand = np.clip(
        np.asarray(gamma_s, dtype=float)
        * np.asarray(sink_coefficient, dtype=float),
        0.0,
        np.inf,
    )
    inv_km = np.where(np.isinf(km), 0.0, 1.0 / np.maximum(km, _EPS))
    return cref / np.maximum(1.0 + demand * inv_km, _EPS)


def _surface_storage_concentration(
    cref: np.ndarray,
    km: np.ndarray,
    store_coefficient: np.ndarray,
    release_rate: np.ndarray,
    gamma_s: np.ndarray,
) -> np.ndarray:
    """Close ``km(Cb-Cs)=Gamma_s*(store*Cs-release)``."""
    cref = np.clip(np.asarray(cref, dtype=float), 0.0, np.inf)
    km = np.clip(np.asarray(km, dtype=float), 0.0, np.inf)
    inv_km = np.where(np.isinf(km), 0.0, 1.0 / np.maximum(km, _EPS))
    capacity = np.asarray(gamma_s, dtype=float)
    numerator = cref + capacity * np.asarray(release_rate, dtype=float) * inv_km
    denominator = 1.0 + capacity * np.asarray(store_coefficient, dtype=float) * inv_km
    return np.clip(numerator / np.maximum(denominator, _EPS), 0.0, np.inf)


def run_ald_role_state_transient(
    *,
    c_a: np.ndarray,
    c_i: np.ndarray,
    c_b: np.ndarray,
    km_provider: Any,
    time: np.ndarray,
    dt_max_s: float,
    theta_a0: np.ndarray,
    h0: np.ndarray,
    k_store_a: np.ndarray,
    k_release_a: np.ndarray,
    k_convert_a: np.ndarray,
    k_convert_ab: np.ndarray,
    k_store_i: np.ndarray,
    k_release_i: np.ndarray,
    alpha_h: np.ndarray,
    gamma_s: np.ndarray,
    nu_b: np.ndarray,
    has_b: bool,
    has_i: bool,
) -> ALDRoleStateResult:
    """Run the minimal ALD latent role-state model on transient role inputs."""

    _require_numpy()
    if dt_max_s <= 0.0:
        raise ValueError("dt_max_s must be > 0")

    time = np.asarray(time, dtype=float).reshape(-1)
    if time.shape[0] < 2:
        raise ValueError("ALD role-state transient execution requires at least two time points")

    theta_a = _clip01(theta_a0)
    theta_i = np.zeros_like(theta_a, dtype=float)
    h = np.asarray(h0, dtype=float).copy()
    r_event = np.zeros_like(theta_a, dtype=float)
    cs_a = np.zeros_like(theta_a, dtype=float)
    cs_b = np.full(theta_a.shape, np.nan, dtype=float)
    j_a_surface = np.zeros_like(theta_a, dtype=float)
    j_b_surface = np.full(theta_a.shape, np.nan, dtype=float)
    bounded_violation_count = 0
    state_projection_count = 0
    substep_count = 0

    for i in range(time.shape[0] - 1):
        seg = float(time[i + 1] - time[i])
        if seg <= 0.0:
            raise ValueError("transient time array must be strictly increasing")
        n_sub = max(int(np.ceil(seg / float(dt_max_s))), 1)
        dt = seg / float(n_sub)

        c_a_i = np.asarray(c_a[i], dtype=float)
        c_i_i = np.asarray(c_i[i], dtype=float)
        c_b_i = np.asarray(c_b[i], dtype=float)
        km_a_i = np.asarray(km_provider.get_km("A", t_index=i), dtype=float)
        km_b_i = np.asarray(km_provider.get_km("B", t_index=i), dtype=float) if has_b else np.zeros_like(km_a_i)

        for _ in range(n_sub):
            substep_count += 1
            theta_free = np.clip(1.0 - theta_a - theta_i, 0.0, 1.0)
            store_coefficient = k_store_a * theta_free
            release_rate = k_release_a * theta_a
            cs_a = _surface_storage_concentration(
                c_a_i, km_a_i, store_coefficient, release_rate, gamma_s
            )
            demand_b = nu_b * k_convert_ab * theta_a if has_b else np.zeros_like(theta_a)
            cs_b = (
                _surface_sink_concentration(c_b_i, km_b_i, demand_b, gamma_s)
                if has_b
                else np.full(theta_a.shape, np.nan, dtype=float)
            )
            cs_i = np.clip(c_i_i, 0.0, np.inf) if has_i else np.zeros_like(theta_a)

            convert = (
                k_convert_ab * np.nan_to_num(cs_b, nan=0.0) * theta_a
                if has_b
                else k_convert_a * theta_a
            )
            j_a_surface = gamma_s * (store_coefficient * cs_a - release_rate)
            j_b_surface = (
                gamma_s * nu_b * convert
                if has_b
                else np.full(theta_a.shape, np.nan, dtype=float)
            )
            dtheta_a = k_store_a * cs_a * theta_free - k_release_a * theta_a - convert
            dtheta_i = k_store_i * cs_i * theta_free - k_release_i * theta_i if has_i else np.zeros_like(theta_a)

            theta_a_raw = theta_a + dt * dtheta_a
            theta_i_raw = theta_i + dt * dtheta_i
            theta_a_violation = (theta_a_raw < -1.0e-10) | (theta_a_raw > 1.0 + 1.0e-10)
            theta_i_violation = (theta_i_raw < -1.0e-10) | (theta_i_raw > 1.0 + 1.0e-10)
            bounded_violation_count += int(np.sum(theta_a_violation))
            bounded_violation_count += int(np.sum(theta_i_violation))
            state_projection_count += int(np.sum(theta_a_violation | theta_i_violation))
            theta_a = _clip01(theta_a_raw)
            theta_i = _clip01(theta_i_raw)
            overflow = np.maximum(theta_a + theta_i - 1.0, 0.0)
            if np.any(overflow > 0.0):
                state_projection_count += int(np.sum(overflow > 0.0))
                total = np.maximum(theta_a + theta_i, _EPS)
                theta_a = theta_a - overflow * theta_a / total
                theta_i = theta_i - overflow * theta_i / total

            r_event = np.clip(convert, 0.0, np.inf)
            h = h + dt * alpha_h * r_event

    theta_free = np.clip(1.0 - theta_a - theta_i, 0.0, 1.0)
    diagnostics = {
        "bounded_violation_count": bounded_violation_count,
        "state_projection_count": state_projection_count,
        "substep_count": substep_count,
        "event_channel": "AB" if has_b else "A",
        "theta_A_min": float(np.nanmin(theta_a)),
        "theta_A_max": float(np.nanmax(theta_a)),
        "theta_I_min": float(np.nanmin(theta_i)),
        "theta_I_max": float(np.nanmax(theta_i)),
        "theta_free_min": float(np.nanmin(theta_free)),
        "theta_free_max": float(np.nanmax(theta_free)),
    }
    return ALDRoleStateResult(
        theta_a=theta_a,
        theta_i=theta_i,
        theta_free=theta_free,
        h_nm=h,
        r_event=r_event,
        cs_a=cs_a,
        cs_b=cs_b,
        j_a_surface=j_a_surface,
        j_b_surface=j_b_surface,
        diagnostics=diagnostics,
    )


__all__ = ["ALDRoleStateResult", "run_ald_role_state_transient"]
