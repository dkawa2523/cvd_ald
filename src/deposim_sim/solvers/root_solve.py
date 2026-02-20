"""Vectorized progress-variable root solver for transport-reaction coupling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
import warnings

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


STATUS_NON_MONOTONIC = 1 << 0
STATUS_FALLBACK_INTERVAL_SPLIT = 1 << 1
STATUS_NON_MONOTONIC_FAILURE = 1 << 2
STATUS_BRACKET_NOT_FOUND = 1 << 3
STATUS_MAX_ITER_REACHED = 1 << 4
STATUS_CS_CLIPPED = 1 << 5


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for deposim_sim.solvers.root_solve. "
            "Install numpy to evaluate progress-variable root solves."
        )


def _align(
    value: Any,
    shape: tuple[int, ...],
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = np.full(shape, float(arr), dtype=float)
    else:
        try:
            out = np.broadcast_to(arr, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} with shape {arr.shape} cannot broadcast to {shape}") from exc
    if positive and bool(np.any(out <= 0.0)):
        raise ValueError(f"{name} must be > 0 everywhere")
    if nonnegative and bool(np.any(out < 0.0)):
        raise ValueError(f"{name} must be >= 0 everywhere")
    return out


def _as_species_map(value: Mapping[str, Any] | Any, species: tuple[str, ...], field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for name in species:
            if name not in value:
                raise ValueError(f"{field} is missing species '{name}'")
            out[name] = value[name]
        return out
    return {name: value for name in species}


def _build_species_fields(
    c_ref: Mapping[str, Any],
    k_m: Mapping[str, Any] | Any,
    nu: Mapping[str, Any] | Any,
    state: Mapping[str, Any] | None,
    temperature: Any,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[int, ...]]:
    species = tuple(str(name) for name in c_ref)
    if not species:
        raise ValueError("c_ref must provide at least one species")

    shape: tuple[int, ...] = ()
    for value in c_ref.values():
        shape = np.broadcast_shapes(shape, np.asarray(value, dtype=float).shape)
    if temperature is not None:
        shape = np.broadcast_shapes(shape, np.asarray(temperature, dtype=float).shape)
    if isinstance(state, Mapping):
        for value in state.values():
            try:
                shape = np.broadcast_shapes(shape, np.asarray(value, dtype=float).shape)
            except Exception:
                continue

    km_map = _as_species_map(k_m, species, "k_m")
    nu_map = _as_species_map(nu, species, "nu")

    fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in species:
        fields[name] = (
            _align(c_ref[name], shape, f"c_ref[{name}]", nonnegative=True),
            _align(km_map[name], shape, f"k_m[{name}]", positive=True),
            _align(nu_map[name], shape, f"nu[{name}]"),
        )
    return fields, shape


def _compute_r_max(fields: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    shape = next(iter(fields.values()))[0].shape
    has_pos = np.zeros(shape, dtype=bool)
    r_max = np.full(shape, np.inf, dtype=float)

    for cref_arr, km_arr, nu_arr in fields.values():
        pos = nu_arr > 0.0
        has_pos |= pos
        bound = np.divide(
            km_arr * cref_arr,
            nu_arr,
            out=np.full(shape, np.inf, dtype=float),
            where=pos,
        )
        r_max = np.minimum(r_max, bound)

    if bool(np.any(~has_pos)):
        raise ValueError("nu must include at least one positive stoichiometric coefficient per grid point")
    return np.clip(r_max, 0.0, np.inf)


def _compute_cs(
    R: np.ndarray,
    fields: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    clip_nonnegative: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    Cs: dict[str, np.ndarray] = {}
    clipped = np.zeros(R.shape, dtype=bool)
    for name, (cref_arr, km_arr, nu_arr) in fields.items():
        value = cref_arr - (nu_arr / km_arr) * R
        if clip_nonnegative:
            below = value < 0.0
            clipped |= below
            value = np.where(below, 0.0, value)
        Cs[name] = value
    return Cs, clipped


def _eval_F(
    R: np.ndarray,
    *,
    fields: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_shape: tuple[int, ...],
    rate_fn: Callable[..., Any],
    state: Mapping[str, Any] | None,
    temperature: Any,
    rate_params: Mapping[str, Any] | None,
    rate_fn_kwargs: Mapping[str, Any] | None,
) -> np.ndarray:
    Cs_eval, _ = _compute_cs(R, fields, clip_nonnegative=True)
    kwargs: dict[str, Any] = {"Cs": Cs_eval, "state": state, "T": temperature, "params": rate_params}
    if isinstance(rate_fn_kwargs, Mapping):
        kwargs.update(rate_fn_kwargs)
    try:
        rate_raw = rate_fn(**kwargs)
    except TypeError as exc:
        raise TypeError(
            "rate_fn must accept keyword arguments compatible with (Cs, state, T, params)"
        ) from exc

    rate = np.asarray(rate_raw, dtype=float)
    if rate.ndim == 0:
        rate = np.full(out_shape, float(rate), dtype=float)
    else:
        try:
            rate = np.broadcast_to(rate, out_shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"rate_fn output with shape {rate.shape} cannot broadcast to {out_shape}") from exc
    return R - rate


def _sample_F(
    *,
    r_max: np.ndarray,
    sample_count: int,
    fields: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_shape: tuple[int, ...],
    rate_fn: Callable[..., Any],
    state: Mapping[str, Any] | None,
    temperature: Any,
    rate_params: Mapping[str, Any] | None,
    rate_fn_kwargs: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.linspace(0.0, 1.0, sample_count, dtype=float)
    samples_R = np.empty((sample_count, *out_shape), dtype=float)
    samples_F = np.empty((sample_count, *out_shape), dtype=float)
    for idx, alpha in enumerate(alphas):
        this_R = alpha * r_max
        samples_R[idx] = this_R
        samples_F[idx] = _eval_F(
            this_R,
            fields=fields,
            out_shape=out_shape,
            rate_fn=rate_fn,
            state=state,
            temperature=temperature,
            rate_params=rate_params,
            rate_fn_kwargs=rate_fn_kwargs,
        )
    return samples_R, samples_F


def solve_progress_R(
    *,
    c_ref: Mapping[str, Any],
    k_m: Mapping[str, Any] | Any,
    nu: Mapping[str, Any] | Any,
    rate_fn: Callable[..., Any],
    state: Mapping[str, Any] | None = None,
    T: Any = None,
    rate_params: Mapping[str, Any] | None = None,
    rate_fn_kwargs: Mapping[str, Any] | None = None,
    max_iter: int = 80,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-12,
    monotonicity_check: bool = True,
    monotonicity_samples: int = 9,
    monotonicity_tol: float = 1.0e-10,
    non_monotonic_mode: str = "warn_and_fail",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Solve the scalar root ``F(R)=0`` for a dominant reaction over broadcast grid points.

    Returns ``R, Cs, iteration_count, status_map``.
    """

    _require_numpy()
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if rtol <= 0.0:
        raise ValueError(f"rtol must be > 0, got {rtol}")
    if atol <= 0.0:
        raise ValueError(f"atol must be > 0, got {atol}")
    if monotonicity_samples < 3:
        raise ValueError(f"monotonicity_samples must be >= 3, got {monotonicity_samples}")

    mode = str(non_monotonic_mode).strip().lower()
    if mode not in {"warn_and_fail", "warn_and_split"}:
        raise ValueError(
            "non_monotonic_mode must be 'warn_and_fail' or 'warn_and_split' "
            f"(got {non_monotonic_mode!r})"
        )

    fields, out_shape = _build_species_fields(c_ref, k_m, nu, state, T)
    r_max = _compute_r_max(fields)
    lo = np.zeros(out_shape, dtype=float)
    hi = r_max.copy()
    status_map = np.zeros(out_shape, dtype=np.int32)
    iteration_count = np.zeros(out_shape, dtype=np.int32)

    samples_R: np.ndarray | None = None
    samples_F: np.ndarray | None = None
    non_monotonic = np.zeros(out_shape, dtype=bool)
    if monotonicity_check:
        samples_R, samples_F = _sample_F(
            r_max=r_max,
            sample_count=monotonicity_samples,
            fields=fields,
            out_shape=out_shape,
            rate_fn=rate_fn,
            state=state,
            temperature=T,
            rate_params=rate_params,
            rate_fn_kwargs=rate_fn_kwargs,
        )
        dF = np.diff(samples_F, axis=0)
        scale = np.maximum(np.nanmax(np.abs(samples_F), axis=0), 1.0)
        tol = monotonicity_tol * scale
        non_finite = ~np.all(np.isfinite(samples_F), axis=0)
        has_pos = np.any(dF > tol, axis=0)
        has_neg = np.any(dF < -tol, axis=0)
        non_monotonic = non_finite | (has_pos & has_neg)
        if bool(np.any(non_monotonic)):
            status_map[non_monotonic] |= STATUS_NON_MONOTONIC
            warnings.warn(
                "Potential non-monotonic F(R) detected; check status_map for affected points.",
                RuntimeWarning,
                stacklevel=2,
            )

    if samples_F is None:
        flo = _eval_F(
            lo,
            fields=fields,
            out_shape=out_shape,
            rate_fn=rate_fn,
            state=state,
            temperature=T,
            rate_params=rate_params,
            rate_fn_kwargs=rate_fn_kwargs,
        )
        fhi = _eval_F(
            hi,
            fields=fields,
            out_shape=out_shape,
            rate_fn=rate_fn,
            state=state,
            temperature=T,
            rate_params=rate_params,
            rate_fn_kwargs=rate_fn_kwargs,
        )
    else:
        flo = samples_F[0].copy()
        fhi = samples_F[-1].copy()

    excluded = np.zeros(out_shape, dtype=bool)
    if monotonicity_check and bool(np.any(non_monotonic)):
        if mode == "warn_and_fail":
            excluded = non_monotonic
            status_map[excluded] |= STATUS_NON_MONOTONIC_FAILURE
        else:
            assert samples_R is not None
            assert samples_F is not None
            split_mask = np.zeros(out_shape, dtype=bool)
            for idx in range(samples_R.shape[0] - 1):
                left_f = samples_F[idx]
                right_f = samples_F[idx + 1]
                sign_change = np.isfinite(left_f) & np.isfinite(right_f) & (
                    (left_f == 0.0) | (right_f == 0.0) | (left_f * right_f <= 0.0)
                )
                take = non_monotonic & ~split_mask & sign_change
                if not bool(np.any(take)):
                    continue
                lo[take] = samples_R[idx][take]
                hi[take] = samples_R[idx + 1][take]
                flo[take] = left_f[take]
                fhi[take] = right_f[take]
                split_mask[take] = True

            status_map[split_mask] |= STATUS_FALLBACK_INTERVAL_SPLIT
            excluded = non_monotonic & ~split_mask
            status_map[excluded] |= STATUS_NON_MONOTONIC_FAILURE

    bracket_ok = ~excluded & np.isfinite(flo) & np.isfinite(fhi) & (
        (flo == 0.0) | (fhi == 0.0) | (flo * fhi <= 0.0)
    )
    status_map[~excluded & ~bracket_ok] |= STATUS_BRACKET_NOT_FOUND

    R = np.zeros(out_shape, dtype=float)
    root_lo = bracket_ok & (flo == 0.0)
    root_hi = bracket_ok & (fhi == 0.0)
    R[root_lo] = lo[root_lo]
    R[root_hi] = hi[root_hi]

    active = bracket_ok & ~root_lo & ~root_hi
    for _ in range(max_iter):
        if not bool(np.any(active)):
            break
        mid = 0.5 * (lo + hi)
        fmid = _eval_F(
            mid,
            fields=fields,
            out_shape=out_shape,
            rate_fn=rate_fn,
            state=state,
            temperature=T,
            rate_params=rate_params,
            rate_fn_kwargs=rate_fn_kwargs,
        )
        left = active & (((flo <= 0.0) & (fmid >= 0.0)) | ((flo >= 0.0) & (fmid <= 0.0)))
        right = active & ~left
        hi[left] = mid[left]
        fhi[left] = fmid[left]
        lo[right] = mid[right]
        flo[right] = fmid[right]
        iteration_count[active] += 1

        f_ok = np.abs(fmid) <= (atol + rtol * np.abs(mid))
        width_ok = (hi - lo) <= (atol + rtol * np.maximum(np.abs(hi), 1.0))
        converged = active & (f_ok | width_ok)
        R[converged] = mid[converged]
        active[converged] = False

    if bool(np.any(active)):
        status_map[active] |= STATUS_MAX_ITER_REACHED
        R[active] = 0.5 * (lo[active] + hi[active])

    R = np.clip(R, 0.0, r_max)
    Cs, clipped = _compute_cs(R, fields, clip_nonnegative=True)
    status_map[clipped] |= STATUS_CS_CLIPPED
    return R, Cs, iteration_count, status_map


__all__ = [
    "STATUS_NON_MONOTONIC",
    "STATUS_FALLBACK_INTERVAL_SPLIT",
    "STATUS_NON_MONOTONIC_FAILURE",
    "STATUS_BRACKET_NOT_FOUND",
    "STATUS_MAX_ITER_REACHED",
    "STATUS_CS_CLIPPED",
    "solve_progress_R",
]
