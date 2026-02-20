"""IO plugin registry for CFD/measurement baseline loaders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


IOLoader = Callable[[Path], dict[str, Any]]
_IO_LOADERS: dict[str, IOLoader] = {}


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for IO plugins.")


def register_io_loader(name: str, loader: IOLoader, *, overwrite: bool = False) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("loader name must be non-empty")
    if key in _IO_LOADERS and not overwrite:
        raise ValueError(f"IO loader {key!r} is already registered")
    _IO_LOADERS[key] = loader


def available_io_loaders() -> tuple[str, ...]:
    return tuple(sorted(_IO_LOADERS))


def _load_npz(path: Path) -> dict[str, Any]:
    _require_numpy()
    with np.load(path, allow_pickle=False) as data:
        return {str(key): data[key] for key in data.files}


def _load_csv(path: Path) -> dict[str, Any]:
    _require_numpy()
    first_line = path.read_text(encoding="utf-8").splitlines()[0] if path.exists() else ""
    tokens = [part.strip() for part in first_line.split(",")] if first_line else []

    def _is_float(text: str) -> bool:
        try:
            float(text)
        except ValueError:
            return False
        return True

    has_header = bool(tokens) and not all(_is_float(token) for token in tokens)
    if has_header:
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
        if table.dtype.names:
            if table.ndim == 0:
                return {name: np.asarray(table[name], dtype=float) for name in table.dtype.names}
            return {name: np.asarray(table[name], dtype=float) for name in table.dtype.names}
    arr = np.loadtxt(path, delimiter=",")
    return {"array": np.asarray(arr, dtype=float)}


def resolve_io_loader(name: str) -> IOLoader:
    key = str(name).strip().lower()
    try:
        return _IO_LOADERS[key]
    except KeyError as exc:
        supported = ", ".join(available_io_loaders())
        raise ValueError(f"Unknown IO loader {name!r}. Supported: {{{supported}}}") from exc


def load_with_io_loader(name: str, path: str | Path) -> dict[str, Any]:
    loader = resolve_io_loader(name)
    return loader(Path(path))


def load_inputs_from_run_spec(run_spec: Any, path: str | Path) -> dict[str, Any]:
    loader_name = getattr(getattr(run_spec, "inputs", object()), "io_loader_name", "npz")
    return load_with_io_loader(str(loader_name), path)


register_io_loader("npz", _load_npz, overwrite=True)
register_io_loader("csv", _load_csv, overwrite=True)


__all__ = [
    "available_io_loaders",
    "load_inputs_from_run_spec",
    "load_with_io_loader",
    "register_io_loader",
    "resolve_io_loader",
]
