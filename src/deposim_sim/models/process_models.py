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
    mechanism: str
    pathways: tuple[str, ...]
    state_variables: tuple[str, ...]
    required_roles: tuple[str, ...]
    quantity_units: tuple[tuple[str, str], ...]
    steady_observable_equivalence: str = ""


_PROCESS_MODELS: dict[str, ProcessModelInfo] = {
    "role_cvd_aib": ProcessModelInfo(
        name="role_cvd_aib",
        implementation="aib_ode",
        processes=("cvd",),
        time_modes=("steady", "transient"),
        description="Continuous CVD role model with coupled surface and transport balances.",
        mechanism="Langmuir-Rideal-type adsorbed-A conversion",
        pathways=("A", "AB"),
        state_variables=("theta_A",),
        required_roles=("A",),
        quantity_units=(
            ("concentration", "kmol/m^3"),
            ("theta_A", "1"),
            ("k_ads", "m^3/(kmol s)"),
            ("k_des", "1/s"),
            ("k_rxn", "1/s with dimensionless C_B/C_B_scale"),
            ("Gamma_s", "kmol/m^2"),
            ("alpha_h", "nm m^2/kmol"),
            ("time", "s"),
        ),
    ),
    "role_cvd_mvk": ProcessModelInfo(
        name="role_cvd_mvk",
        implementation="mvk_state",
        processes=("cvd",),
        time_modes=("steady", "transient"),
        description=(
            "Mars-van Krevelen redox-reservoir model with A reduction/growth "
            "and B regeneration pathways."
        ),
        mechanism="Mars-van Krevelen surface redox reservoir",
        pathways=("A_reduction_growth", "B_regeneration"),
        state_variables=("oxidized_fraction",),
        required_roles=("A", "B"),
        quantity_units=(
            ("concentration", "kmol/m^3"),
            ("oxidized_fraction", "1"),
            ("k_reduce", "m^3/(kmol s)"),
            ("k_regenerate", "m^3/(kmol s)"),
            ("Gamma_s", "kmol/m^2"),
            ("surface_flux", "kmol/(m^2 s)"),
            ("alpha_h", "nm m^2/kmol"),
            ("time", "s"),
        ),
        steady_observable_equivalence="aib_qss:AB:no_desorption",
    ),
    "role_ald_state": ProcessModelInfo(
        name="role_ald_state",
        implementation="ald_role_state",
        processes=("ald",),
        time_modes=("transient",),
        description="Minimal ALD latent role-state assimilation model.",
        mechanism="ALD storage-conversion role state",
        pathways=("A_storage", "A_or_AB_conversion"),
        state_variables=("theta_A", "theta_I"),
        required_roles=("A",),
        quantity_units=(
            ("concentration", "kmol/m^3"),
            ("theta_A", "1"),
            ("theta_I", "1"),
            ("k_store_A", "m^3/(kmol s)"),
            ("k_release_A", "1/s"),
            ("k_convert_A", "1/s"),
            ("k_convert_AB", "m^3/(kmol s)"),
            ("Gamma_s", "kmol/m^2"),
            ("surface_flux", "kmol/(m^2 s)"),
            ("alpha_h", "nm per unit coverage converted"),
            ("time", "s"),
            ("thickness", "nm"),
        ),
    ),
}


def available_process_models() -> tuple[str, ...]:
    return tuple(sorted(_PROCESS_MODELS))


def primary_process_models() -> tuple[ProcessModelInfo, ...]:
    """Return the public state-process model inventory."""

    return tuple(_PROCESS_MODELS.values())


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
    "primary_process_models",
    "validate_process_model_choice",
]
