"""IO helpers for AIB fluent/measurement payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .input_builder import FluentData, load_fluent_npz_v2

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MeasurementData:
    xy: np.ndarray
    h: np.ndarray
    sigma: np.ndarray | None = None


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for IO plugins.")


def _as_key(keys: Any, name: str, default: str) -> str:
    if keys is None:
        return default
    if isinstance(keys, Mapping):
        return str(keys.get(name, default))
    return str(getattr(keys, name, default))


def available_io_loaders() -> tuple[str, ...]:
    return ("csv", "npz")


def _load_csv_table(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
    if table.dtype.names is None:
        raise ValueError("CSV loader requires header row")
    if table.ndim == 0:
        table = np.asarray([table], dtype=table.dtype)
    return table


def _load_fluent_csv(
    path: Path,
    *,
    mode: str,
    species: list[str],
    keys: Any,
) -> FluentData:
    table = _load_csv_table(path)
    x_key = _as_key(keys, "x", "x")
    y_key = _as_key(keys, "y", "y")
    time_key = _as_key(keys, "time", "time")
    missing = [k for k in (x_key, y_key) if k not in table.dtype.names]
    if missing:
        raise ValueError(f"fluent csv missing required coordinate columns: {missing}")

    x = np.asarray(table[x_key], dtype=float).reshape(-1)
    y = np.asarray(table[y_key], dtype=float).reshape(-1)
    for name in species:
        if str(name) not in table.dtype.names:
            raise ValueError(f"fluent csv missing species column: {name!r}")

    if mode == "steady":
        xy = np.stack([x, y], axis=1)
        cref = np.stack([np.asarray(table[str(name)], dtype=float).reshape(-1) for name in species], axis=1)
        return FluentData(mode="steady", cref=cref, flux_sink=None, xy=xy, time=None, species=tuple(species))

    if time_key not in table.dtype.names:
        raise ValueError("fluent transient csv requires a time column")
    t = np.asarray(table[time_key], dtype=float).reshape(-1)
    times = np.unique(t)
    xy_ref: np.ndarray | None = None
    cref_series: list[np.ndarray] = []
    for time_value in times:
        mask = t == float(time_value)
        xy_now = np.stack([x[mask], y[mask]], axis=1)
        c_now = np.stack([np.asarray(table[str(name)], dtype=float).reshape(-1)[mask] for name in species], axis=1)
        if xy_ref is None:
            xy_ref = xy_now
        elif xy_now.shape != xy_ref.shape or not np.allclose(xy_now, xy_ref):
            raise ValueError("transient fluent csv must keep identical xy ordering for each time slice")
        cref_series.append(c_now)

    if xy_ref is None:
        raise ValueError("fluent csv has no rows")
    cref = np.stack(cref_series, axis=0)
    return FluentData(
        mode="transient",
        cref=cref,
        flux_sink=None,
        xy=xy_ref,
        time=np.asarray(times, dtype=float),
        species=tuple(species),
    )


def load_fluent_input(
    *,
    loader_name: str,
    path: str | Path,
    mode: str,
    species: list[str],
    keys: Any = None,
) -> FluentData:
    """Load Fluent inputs into AIB FluentData contract."""

    _require_numpy()
    loader = str(loader_name).strip().lower()
    resolved = Path(path)
    if loader == "npz":
        return load_fluent_npz_v2(path=resolved, mode=mode, keys=keys, species=species)
    if loader == "csv":
        return _load_fluent_csv(resolved, mode=mode, species=species, keys=keys)
    supported = ", ".join(available_io_loaders())
    raise ValueError(f"Unknown fluent loader {loader_name!r}. Supported: {{{supported}}}")


def _load_measurement_npz(path: Path, *, keys: Any) -> MeasurementData:
    with np.load(path, allow_pickle=False) as data:
        xy_key = _as_key(keys, "xy", "xy")
        h_key = _as_key(keys, "h", "h_nm")
        missing = [key for key in (xy_key, h_key) if key not in data.files]
        if missing:
            available = ", ".join(sorted(str(name) for name in data.files))
            raise ValueError(f"measurement npz missing required keys {missing} in {path}; available keys: [{available}]")
        xy = np.asarray(data[xy_key], dtype=float)
        h = np.asarray(data[h_key], dtype=float).reshape(-1)
        sigma_key = _as_key(keys, "sigma", "")
        sigma = np.asarray(data[sigma_key], dtype=float).reshape(-1) if sigma_key else None
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"measurement xy must be shape [n_pts,2], got {xy.shape}")
    if h.shape[0] != xy.shape[0]:
        raise ValueError("measurement h length must match xy rows")
    return MeasurementData(xy=xy, h=h, sigma=sigma)


def _load_measurement_csv(path: Path, *, keys: Any) -> MeasurementData:
    table = _load_csv_table(path)
    x_key = _as_key(keys, "x", "x")
    y_key = _as_key(keys, "y", "y")
    h_key = _as_key(keys, "h", "h_nm")
    missing = [k for k in (x_key, y_key, h_key) if k not in table.dtype.names]
    if missing:
        raise ValueError(f"measurement csv missing required columns: {missing}")
    xy = np.stack([np.asarray(table[x_key], dtype=float), np.asarray(table[y_key], dtype=float)], axis=1)
    h = np.asarray(table[h_key], dtype=float).reshape(-1)
    sigma_key = _as_key(keys, "sigma", "")
    sigma = np.asarray(table[sigma_key], dtype=float).reshape(-1) if sigma_key else None
    return MeasurementData(xy=xy, h=h, sigma=sigma)


def load_measurement_input(
    *,
    loader_name: str,
    path: str | Path,
    keys: Any = None,
) -> MeasurementData:
    """Load measurement inputs (xy/h) from NPZ or CSV."""

    _require_numpy()
    loader = str(loader_name).strip().lower()
    resolved = Path(path)
    if loader == "npz":
        return _load_measurement_npz(resolved, keys=keys)
    if loader == "csv":
        return _load_measurement_csv(resolved, keys=keys)
    supported = ", ".join(available_io_loaders())
    raise ValueError(f"Unknown measurement loader {loader_name!r}. Supported: {{{supported}}}")


def load_fluent_from_run_spec(run_spec: Any, path: str | Path | None = None) -> FluentData:
    sim = getattr(run_spec, "sim", run_spec)
    source = Path(path) if path is not None else Path(sim.inputs.fluent.file)
    explicit_loader = str(getattr(sim.inputs.fluent, "io_loader_name", "")).strip().lower()
    loader = explicit_loader or source.suffix.lstrip(".") or "npz"
    return load_fluent_input(
        loader_name=loader,
        path=source,
        mode=str(sim.inputs.fluent.mode),
        species=list(sim.inputs.fluent.species),
        keys=sim.inputs.fluent.keys,
    )


def load_measurement_from_run_spec(run_spec: Any, path: str | Path | None = None) -> MeasurementData:
    sim = getattr(run_spec, "sim", run_spec)
    source = Path(path) if path is not None else Path(sim.measurement.file)
    explicit_loader = str(getattr(sim.measurement, "io_loader_name", "")).strip().lower()
    loader = explicit_loader or source.suffix.lstrip(".") or "npz"
    return load_measurement_input(loader_name=loader, path=source, keys=sim.measurement.keys)


__all__ = [
    "MeasurementData",
    "available_io_loaders",
    "load_fluent_from_run_spec",
    "load_fluent_input",
    "load_measurement_from_run_spec",
    "load_measurement_input",
]
