"""Compute engine selection helpers (NumPy baseline, optional JAX)."""

from __future__ import annotations

import importlib.util


def is_jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None


def available_engines() -> tuple[str, ...]:
    engines = ["numpy"]
    if is_jax_available():
        engines.append("jax")
    return tuple(engines)


def resolve_engine_selection(requested_engine: str) -> str:
    engine = str(requested_engine).strip().lower()
    if engine == "numpy":
        return "numpy"
    if engine == "jax":
        if not is_jax_available():
            raise RuntimeError(
                "compute.engine='jax' was requested but JAX is not available. "
                "Install optional extras (e.g., pip install 'deposim[jax]')."
            )
        return "jax"
    supported = ", ".join(("numpy", "jax"))
    raise ValueError(f"Unsupported compute engine {requested_engine!r}. Supported engines: {{{supported}}}")


def resolve_execution_backend(selected_engine: str) -> str:
    """Return the actual numerical backend used by the current implementation."""
    _ = selected_engine
    # Current solver/model kernels are NumPy-based even when JAX is selected for policy.
    return "numpy"


def build_engine_context(requested_engine: str) -> dict[str, object]:
    selected = resolve_engine_selection(requested_engine)
    execution_backend = resolve_execution_backend(selected)
    requested = str(requested_engine).strip().lower()
    return {
        "requested_engine": requested,
        "selected_engine": selected,
        "execution_backend": execution_backend,
        "available_engines": list(available_engines()),
    }


__all__ = [
    "available_engines",
    "build_engine_context",
    "is_jax_available",
    "resolve_engine_selection",
    "resolve_execution_backend",
]
