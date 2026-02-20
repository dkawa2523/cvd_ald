"""Optional array-store helper with NPZ/Zarr/HDF5 backends and fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for array store output.")


def is_zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


def is_h5py_available() -> bool:
    return importlib.util.find_spec("h5py") is not None


def save_array_store(
    *,
    base_path: str | Path,
    arrays: dict[str, Any],
    store: str = "npz",
) -> dict[str, str]:
    """Persist arrays using npz (default) or optional zarr/hdf5 backends."""

    _require_numpy()
    base = Path(base_path)
    key = str(store).strip().lower()
    if key == "npz":
        out = base.with_suffix(".npz")
        np.savez(out, **{name: np.asarray(value) for name, value in arrays.items()})
        return {"store_used": "npz", "path": str(out)}

    if key == "zarr":
        if is_zarr_available():
            import zarr  # type: ignore

            out = base.with_suffix(".zarr")
            root = zarr.open_group(str(out), mode="w")
            for name, value in arrays.items():
                arr = np.asarray(value)
                root.create_dataset(name, data=arr, shape=arr.shape, dtype=arr.dtype, overwrite=True)
            return {"store_used": "zarr", "path": str(out)}

        fallback = base.with_suffix(".npz")
        np.savez(fallback, **{name: np.asarray(value) for name, value in arrays.items()})
        return {"store_used": "npz_fallback", "path": str(fallback)}

    if key == "hdf5":
        if is_h5py_available():
            import h5py  # type: ignore

            out = base.with_suffix(".h5")
            with h5py.File(out, "w") as f:
                for name, value in arrays.items():
                    arr = np.asarray(value)
                    f.create_dataset(name, data=arr)
            return {"store_used": "hdf5", "path": str(out)}

        fallback = base.with_suffix(".npz")
        np.savez(fallback, **{name: np.asarray(value) for name, value in arrays.items()})
        return {"store_used": "npz_fallback", "path": str(fallback)}

    raise ValueError(f"Unsupported array store {store!r}; expected 'npz', 'zarr', or 'hdf5'.")


def load_array_store(path: str | Path) -> dict[str, Any]:
    """Load arrays from npz or zarr path into a plain mapping."""

    _require_numpy()
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".npz":
        with np.load(src, allow_pickle=False) as data:
            return {str(key): np.asarray(data[key]) for key in data.files}
    if suffix == ".zarr":
        if not is_zarr_available():
            raise RuntimeError("Cannot load .zarr artifact because zarr is not installed.")
        import zarr  # type: ignore

        root = zarr.open_group(str(src), mode="r")
        out: dict[str, Any] = {}
        for key in root.array_keys():
            out[str(key)] = np.asarray(root[key])
        return out
    if suffix in {".h5", ".hdf5"}:
        if not is_h5py_available():
            raise RuntimeError("Cannot load .h5/.hdf5 artifact because h5py is not installed.")
        import h5py  # type: ignore

        out: dict[str, Any] = {}
        with h5py.File(src, "r") as f:
            for key in f.keys():
                out[str(key)] = np.asarray(f[key])
        return out
    raise ValueError(f"Unsupported array store path: {path!r}")


__all__ = ["is_h5py_available", "is_zarr_available", "load_array_store", "save_array_store"]
