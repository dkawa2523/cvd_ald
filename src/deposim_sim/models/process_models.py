"""Small registry for role-based process model names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessModelInfo:
    name: str
    implementation: str
    processes: tuple[str, ...]
    time_modes: tuple[str, ...]
    description: str


_PROCESS_MODELS: dict[str, ProcessModelInfo] = {
    "aib_ode": ProcessModelInfo(
        name="aib_ode",
        implementation="aib_ode",
        processes=("cvd", "ald"),
        time_modes=("steady", "transient"),
        description="Compatibility role-based A/I/B ODE implementation.",
    ),
    "role_cvd_aib": ProcessModelInfo(
        name="role_cvd_aib",
        implementation="aib_ode",
        processes=("cvd",),
        time_modes=("steady", "transient"),
        description="CVD-facing alias for the current role-based A/I/B implementation.",
    ),
    "role_ald_compat": ProcessModelInfo(
        name="role_ald_compat",
        implementation="aib_ode",
        processes=("ald",),
        time_modes=("transient",),
        description="ALD-facing compatibility alias used before a dedicated ALD role-state model exists.",
    ),
    "role_ald_state": ProcessModelInfo(
        name="role_ald_state",
        implementation="ald_role_state",
        processes=("ald",),
        time_modes=("transient",),
        description="Minimal ALD latent role-state assimilation model.",
    ),
}


def available_process_models() -> tuple[str, ...]:
    return tuple(sorted(_PROCESS_MODELS))


def get_process_model_info(name: str) -> ProcessModelInfo:
    key = str(name).strip()
    try:
        return _PROCESS_MODELS[key]
    except KeyError as exc:
        supported = ", ".join(available_process_models())
        raise ValueError(f"Unknown process model '{key}'. Supported models: {{{supported}}}") from exc


def validate_process_model_choice(*, name: str, process: str, time_mode: str) -> ProcessModelInfo:
    info = get_process_model_info(name)
    if str(process) not in info.processes:
        raise ValueError(
            f"process model '{info.name}' does not support sim.process={process!r}; "
            f"supported processes are {info.processes}"
        )
    if str(time_mode) not in info.time_modes:
        raise ValueError(
            f"process model '{info.name}' does not support sim.time_mode={time_mode!r}; "
            f"supported time modes are {info.time_modes}"
        )
    return info


def canonical_process_implementation(name: str) -> str:
    return get_process_model_info(name).implementation


__all__ = [
    "ProcessModelInfo",
    "available_process_models",
    "canonical_process_implementation",
    "get_process_model_info",
    "validate_process_model_choice",
]
