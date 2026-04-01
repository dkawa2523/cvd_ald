"""Structured simulation/optimization configs for AIB-ODE workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

SIM_CONFIG_ROOT = Path("configs/sim")
OPT_CONFIG_ROOT = Path("configs/opt")
CONFIG_ROOTS: dict[str, Path] = {"sim": SIM_CONFIG_ROOT, "opt": OPT_CONFIG_ROOT}

_SIM_CONFIG_ALIASES: dict[str, str] = {
    "smoke": "cvd_steady_min",
    "example_cvd": "cvd_steady_min",
    "multiz": "cvd_steady_min",
    "ald_synthetic": "ald_transient_min",
}
_OPT_CONFIG_ALIASES: dict[str, str] = {
    "base": "fit_cvd_steady_min",
    "stub": "fit_cvd_steady_min",
}


_ALLOWED_TIME_MODES = {"steady", "transient"}
_ALLOWED_FLUENT_MODES = {"steady", "transient"}
_ALLOWED_PROCESS = {"cvd", "ald"}
_ALLOWED_ORDERS_M_ADS = {1, 2}
_ALLOWED_ORDERS_P_A = {1, 2}
_ALLOWED_ORDERS_P_STAR = {0, 1, 2}
_ALLOWED_OPT_ENGINES = {"optuna", "random"}
_ALLOWED_OPT_SAMPLERS = {"tpe", "cmaes", "random"}
_ALLOWED_OPT_PRUNERS = {"none", "median", "hyperband"}
_ALLOWED_DOMAIN_KINDS = {"from_fluent_xy", "wafer_2d_xy", "wafer_2d_polar", "wafer_1d_radial"}


def _ensure(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _to_abs(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


@dataclass
class FluentKeysSpec:
    cref: str = "cref"
    xy: str = "xy"
    time: str = "time"
    flux_sink: str = "flux_sink"


@dataclass
class FluentInputSpec:
    mode: str = "steady"
    file: str = ""
    keys: FluentKeysSpec = field(default_factory=FluentKeysSpec)
    species: list[str] = field(default_factory=lambda: ["s0"])

    def __post_init__(self) -> None:
        _ensure(self.mode in _ALLOWED_FLUENT_MODES, f"sim.inputs.fluent.mode must be one of {_ALLOWED_FLUENT_MODES}")
        _ensure(bool(self.file), "sim.inputs.fluent.file must be non-empty")
        _ensure(bool(self.species), "sim.inputs.fluent.species must be non-empty")
        _ensure(len(set(self.species)) == len(self.species), "sim.inputs.fluent.species must not contain duplicates")


@dataclass
class TemperatureSpec:
    mode: str = "scalar"
    value_K: float = 600.0

    def __post_init__(self) -> None:
        _ensure(self.mode in {"scalar", "field"}, "sim.inputs.temperature.mode must be scalar|field")
        _ensure(float(self.value_K) > 0.0, "sim.inputs.temperature.value_K must be > 0")


@dataclass
class InputsSpec:
    fluent: FluentInputSpec = field(default_factory=FluentInputSpec)
    temperature: TemperatureSpec = field(default_factory=TemperatureSpec)


@dataclass
class ReferencePlaneSpec:
    z_ref_mm: float = 1.0

    def __post_init__(self) -> None:
        _ensure(float(self.z_ref_mm) > 0.0, "sim.reference_plane.z_ref_mm must be > 0")


@dataclass
class DomainSpec:
    kind: str = "from_fluent_xy"
    xy_unit: str = "mm"
    wafer_radius_mm: float = 150.0
    nr: int = 64
    ntheta: int = 128
    nx: int = 128
    ny: int = 128
    edge_exclusion_mm: float = 0.0

    def __post_init__(self) -> None:
        _ensure(
            self.kind in _ALLOWED_DOMAIN_KINDS,
            f"sim.domain.kind must be one of {_ALLOWED_DOMAIN_KINDS}",
        )
        _ensure(self.xy_unit in {"mm", "m"}, "sim.domain.xy_unit must be mm|m")
        _ensure(float(self.wafer_radius_mm) > 0.0, "sim.domain.wafer_radius_mm must be > 0")
        _ensure(float(self.edge_exclusion_mm) >= 0.0, "sim.domain.edge_exclusion_mm must be >= 0")

        if self.kind in {"wafer_2d_xy", "wafer_2d_polar", "wafer_1d_radial"}:
            _ensure(int(self.nr) >= 2, "sim.domain.nr must be >= 2")
        if self.kind == "wafer_2d_polar":
            _ensure(int(self.ntheta) >= 2, "sim.domain.ntheta must be >= 2")
        if self.kind == "wafer_2d_xy":
            _ensure(int(self.nx) >= 2, "sim.domain.nx must be >= 2")
            _ensure(int(self.ny) >= 2, "sim.domain.ny must be >= 2")


@dataclass
class RoleSpec:
    mode: str = "fixed"
    A: str = "s0"
    I: str | None = None
    B: str | None = None

    def __post_init__(self) -> None:
        _ensure(self.mode == "fixed", "sim.roles.mode must be fixed")
        _ensure(bool(self.A), "sim.roles.A is required")
        role_values = [self.A, self.I, self.B]
        normalized = [v for v in role_values if v is not None]
        _ensure(len(set(normalized)) == len(normalized), "sim.roles A/I/B must be disjoint")


@dataclass
class AIBOrdersSpec:
    adsorption_site_order: int = 1
    reaction_site_order_A: int = 1
    reaction_site_order_star: int = 0
    enforce_total_order_le: int = 3

    def __post_init__(self) -> None:
        _ensure(self.adsorption_site_order in _ALLOWED_ORDERS_M_ADS, "adsorption_site_order must be 1|2")
        _ensure(self.reaction_site_order_A in _ALLOWED_ORDERS_P_A, "reaction_site_order_A must be 1|2")
        _ensure(self.reaction_site_order_star in _ALLOWED_ORDERS_P_STAR, "reaction_site_order_star must be 0|1|2")
        _ensure(self.enforce_total_order_le == 3, "enforce_total_order_le must be 3")


@dataclass
class AIBModelParamsSpec:
    transport: dict[str, Any] = field(
        default_factory=lambda: {
            "km_source": "fit_scalar",
            "km_A": {"mode": "constant", "value": 0.02},
            "km_B": {"mode": "constant", "value": 0.02},
            "Gamma_s": 1.0,
            "nu_A": 1.0,
            "gamma_km_A": 1.0,
            "gamma_km_B": 1.0,
            "from_cfd_flux_sink": {
                "eps_cref": 1.0e-12,
                "km_clip": [1.0e-8, 1.0e4],
                "flux_negative_policy": "error",
                "units_hint": "",
            },
        }
    )
    kinetics: dict[str, Any] = field(default_factory=lambda: {"k_ads": 1.0, "k_des": 0.1, "k_rxn": 0.01})
    inhibitor: dict[str, Any] = field(default_factory=lambda: {"K_I": 0.0})
    thickness: dict[str, Any] = field(default_factory=lambda: {"alpha_h": 1.0})
    scaling: dict[str, Any] = field(default_factory=lambda: {"C_B_scale": 1.0})


@dataclass
class AIBModelSpec:
    name: str = "aib_ode"
    orders: AIBOrdersSpec = field(default_factory=AIBOrdersSpec)
    params: AIBModelParamsSpec = field(default_factory=AIBModelParamsSpec)

    def __post_init__(self) -> None:
        _ensure(self.name == "aib_ode", "sim.model.name must be 'aib_ode'")


@dataclass
class TimeSolverSpec:
    name: str = "implicit_euler_bisect"
    max_iter: int = 60
    theta_tol: float = 1.0e-10

    def __post_init__(self) -> None:
        _ensure(self.name == "implicit_euler_bisect", "sim.time.solver.name must be implicit_euler_bisect")
        _ensure(self.max_iter >= 8, "sim.time.solver.max_iter must be >= 8")
        _ensure(self.theta_tol > 0.0, "sim.time.solver.theta_tol must be > 0")


@dataclass
class TimeSpec:
    t_proc_s: float = 30.0
    dt_s: float = 0.01
    solver: TimeSolverSpec = field(default_factory=TimeSolverSpec)

    def __post_init__(self) -> None:
        _ensure(self.t_proc_s > 0.0, "sim.time.t_proc_s must be > 0")
        _ensure(self.dt_s > 0.0, "sim.time.dt_s must be > 0")


@dataclass
class InitScalarSpec:
    mode: str = "scalar"
    value: float = 0.0

    def __post_init__(self) -> None:
        _ensure(self.mode == "scalar", "initial_conditions modes must be scalar")


@dataclass
class InitialConditionsSpec:
    theta_A: InitScalarSpec = field(default_factory=InitScalarSpec)
    h_nm: InitScalarSpec = field(default_factory=InitScalarSpec)


@dataclass
class MeasurementSpec:
    enabled: bool = False
    file: str = ""
    keys: dict[str, str] = field(default_factory=lambda: {"h": "h_nm", "xy": "xy"})
    align: dict[str, Any] = field(
        default_factory=lambda: {
            "enable": True,
            "shift_mm": [0.0, 0.0],
            "rotate_deg": 0.0,
            "mask_radius_mm": 150.0,
        }
    )


@dataclass
class OutputSpec:
    project: str = "demo"
    run_name: str = "cvd_steady_min"
    root_dir: str = "results"
    store: dict[str, Any] = field(default_factory=lambda: {"format": "npz"})
    save_fields: list[str] = field(
        default_factory=lambda: ["h_nm", "theta_A", "theta_star", "CsA_over_CrefA", "residual_nm"]
    )
    report: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "index_html": True})

    def __post_init__(self) -> None:
        _ensure(bool(self.project), "sim.output.project must be non-empty")
        _ensure(bool(self.run_name), "sim.output.run_name must be non-empty")
        _ensure(bool(self.root_dir), "sim.output.root_dir must be non-empty")


@dataclass
class SimSpecV2:
    process: str = "cvd"
    time_mode: str = "steady"
    reference_plane: ReferencePlaneSpec = field(default_factory=ReferencePlaneSpec)
    inputs: InputsSpec = field(default_factory=InputsSpec)
    domain: DomainSpec = field(default_factory=DomainSpec)
    roles: RoleSpec = field(default_factory=RoleSpec)
    model: AIBModelSpec = field(default_factory=AIBModelSpec)
    time: TimeSpec = field(default_factory=TimeSpec)
    initial_conditions: InitialConditionsSpec = field(default_factory=InitialConditionsSpec)
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        _ensure(self.process in _ALLOWED_PROCESS, "sim.process must be cvd|ald")
        _ensure(self.time_mode in _ALLOWED_TIME_MODES, "sim.time_mode must be steady|transient")
        _ensure(self.roles.A in self.inputs.fluent.species, "sim.roles.A must exist in sim.inputs.fluent.species")
        if self.roles.I is not None:
            _ensure(self.roles.I in self.inputs.fluent.species, "sim.roles.I must exist in species list")
        if self.roles.B is not None:
            _ensure(self.roles.B in self.inputs.fluent.species, "sim.roles.B must exist in species list")

        has_b = self.roles.B is not None
        total_order = self.model.orders.reaction_site_order_A + self.model.orders.reaction_site_order_star + (1 if has_b else 0)
        _ensure(
            total_order <= self.model.orders.enforce_total_order_le,
            "order constraint violated: p_A + p_* + m_B must be <= 3",
        )


@dataclass
class SimConfigV2:
    sim: SimSpecV2 = field(default_factory=SimSpecV2)


@dataclass
class ParameterFitSpec:
    engine: str = "optuna"
    sampler: str = "tpe"
    pruner: str = "none"
    seed: int = 123
    n_trials_per_candidate: int = 40
    storage: dict[str, Any] = field(
        default_factory=lambda: {
            "url": "",
            "study_name": "",
            "load_if_exists": False,
        }
    )
    fidelity: dict[str, Any] = field(default_factory=lambda: {"levels": [1]})
    objective: dict[str, Any] = field(
        default_factory=lambda: {
            "loss": "huber",
            "huber_delta_nm": 10.0,
            "phi_B_min": 0.05,
            "penalties": {
                "lambda_complex": 0.0,
                "lambda_role": 0.0,
                "lambda_solver": 0.0,
                "lambda_phys": 0.0,
                "lambda_prior": 0.0,
            },
            "tie": {"abs_score_epsilon": 1.0e-8},
        }
    )
    analysis: dict[str, Any] = field(
        default_factory=lambda: {
            "role_stability": {"enabled": True, "topk_window": 10, "score_epsilon": 1.0e-6},
            "identifiability": {
                "enabled": False,
                "max_paths": 3,
                "relative_step": 1.0e-2,
                "low_sensitivity_threshold": 1.0e-10,
                "correlation_threshold": 0.98,
            },
            "cache": {"enabled": True, "max_entries": 256},
            "preflight": {"enabled": True, "min_finite_ratio": 0.6},
        }
    )
    search_space: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        engine = str(self.engine).strip().lower()
        sampler = str(self.sampler).strip().lower()
        pruner = str(self.pruner).strip().lower()
        _ensure(engine in _ALLOWED_OPT_ENGINES, f"opt.parameter_fit.engine must be one of {_ALLOWED_OPT_ENGINES}")
        _ensure(sampler in _ALLOWED_OPT_SAMPLERS, f"opt.parameter_fit.sampler must be one of {_ALLOWED_OPT_SAMPLERS}")
        _ensure(pruner in _ALLOWED_OPT_PRUNERS, f"opt.parameter_fit.pruner must be one of {_ALLOWED_OPT_PRUNERS}")
        _ensure(int(self.n_trials_per_candidate) >= 1, "opt.parameter_fit.n_trials_per_candidate must be >= 1")


@dataclass
class RoleEnumerationSpec:
    enabled: bool = True
    species_source: str = "from_sim_input"
    constraints: dict[str, Any] = field(default_factory=lambda: {"disjoint": True, "allow_unused": True})
    roles: dict[str, Any] = field(
        default_factory=lambda: {
            "A": {"required": True, "candidates": "auto"},
            "I": {"required": False, "allow_none": True, "max_size": 1, "candidates": "auto"},
            "B": {"required": False, "allow_none": True, "max_size": 1, "candidates": "auto"},
        }
    )


@dataclass
class OrderEnumerationSpec:
    enabled: bool = True
    candidates: list[dict[str, int]] = field(default_factory=list)
    enforce_total_order_le: int = 3


@dataclass
class ClassCompareSpec:
    enabled: bool = True
    classes: list[str] = field(default_factory=lambda: ["A", "AI", "AB", "AIB"])
    complexity_penalty: dict[str, Any] = field(default_factory=lambda: {"lambda_role": 0.1})


@dataclass
class OptSpecV2:
    task: str = "fit_roles_and_params"
    measurement: dict[str, Any] = field(default_factory=dict)
    role_enumeration: RoleEnumerationSpec = field(default_factory=RoleEnumerationSpec)
    order_enumeration: OrderEnumerationSpec = field(default_factory=OrderEnumerationSpec)
    class_compare: ClassCompareSpec = field(default_factory=ClassCompareSpec)
    parameter_fit: ParameterFitSpec = field(default_factory=ParameterFitSpec)
    selection: dict[str, Any] = field(default_factory=lambda: {"topk_overall": 20, "topk_per_class": 10})
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptConfigV2:
    sim: SimSpecV2 = field(default_factory=SimSpecV2)
    opt: OptSpecV2 = field(default_factory=OptSpecV2)


# Backward-facing alias kept to avoid import failures in untouched modules.
RunSpec = SimSpecV2


def resolve_config_root(kind: str, project_root: str | Path | None = None) -> Path:
    if kind not in CONFIG_ROOTS:
        ordered = ", ".join(sorted(CONFIG_ROOTS))
        raise ValueError(f"kind must be one of {{{ordered}}}, got '{kind}'")
    base = Path(project_root) if project_root is not None else Path.cwd()
    return (base / CONFIG_ROOTS[kind]).resolve()


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_defaults_block(
    config_name: str,
    *,
    kind: str,
    visited: tuple[str, ...],
    project_root: str | Path | None,
) -> dict[str, Any]:
    token = f"{kind}:{config_name}"
    if token in visited:
        chain = " -> ".join((*visited, token))
        raise ValueError(f"Circular defaults reference detected: {chain}")

    root = resolve_config_root(kind, project_root=project_root)
    path = root / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg_obj = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg_obj, resolve=False)
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level config must be mapping: {path}")

    defaults = cfg.pop("defaults", []) or []
    if not isinstance(defaults, list):
        raise ValueError(f"defaults must be a list in {path}")

    composed: dict[str, Any] = {}
    used_self = False

    for entry in defaults:
        if entry == "_self_":
            composed = _deep_merge(composed, cfg)
            used_self = True
            continue

        include_kind = kind
        include_name: str
        if isinstance(entry, str):
            include_name = entry.rsplit("/", 1)[-1]
        elif isinstance(entry, Mapping):
            if len(entry) != 1:
                raise ValueError(f"defaults mapping entries must have one key in {path}")
            group, value = next(iter(entry.items()))
            include_name = str(value)
            group_norm = str(group).strip("/")
            if group_norm in {"sim", "opt"}:
                include_kind = group_norm
        else:
            raise ValueError(f"Unsupported defaults entry type {type(entry)!r} in {path}")

        child = _resolve_defaults_block(
            include_name,
            kind=include_kind,
            visited=(*visited, token),
            project_root=project_root,
        )
        composed = _deep_merge(composed, child)

    if not defaults or not used_self:
        composed = _deep_merge(composed, cfg)

    return composed


def _parse_override_value(raw: str) -> Any:
    try:
        parsed = OmegaConf.create({"v": raw})
        return OmegaConf.to_container(parsed, resolve=True)["v"]
    except Exception:
        text = raw.strip()
        if text.lower() in {"null", "none"}:
            return None
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        return raw


def _apply_overrides(config: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key_path, value_raw = item.split("=", 1)
        keys = [k for k in key_path.lstrip("+").split(".") if k]
        if not keys:
            raise ValueError(f"Invalid override key: {item}")
        cursor: dict[str, Any] = out
        for key in keys[:-1]:
            node = cursor.get(key)
            if not isinstance(node, dict):
                node = {}
                cursor[key] = node
            cursor = node
        cursor[keys[-1]] = _parse_override_value(value_raw)
    return out


def _resolve_config_alias(config_name: str, *, kind: str) -> str:
    if kind == "sim":
        aliases = _SIM_CONFIG_ALIASES
    elif kind == "opt":
        aliases = _OPT_CONFIG_ALIASES
    else:
        return str(config_name)

    name = str(config_name)
    resolved = aliases.get(name, name)
    if resolved != name:
        warnings.warn(
            f"Config name {name!r} is deprecated; using {resolved!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
    return resolved


def _build_sim_spec(data: Mapping[str, Any], *, project_root: Path) -> SimSpecV2:
    if "sim" in data and isinstance(data["sim"], Mapping):
        payload = dict(data["sim"])
    else:
        payload = dict(data)

    fluent_obj = payload.get("inputs", {}).get("fluent", {})
    file_raw = str(fluent_obj.get("file", "")).strip()
    if file_raw:
        payload.setdefault("inputs", {}).setdefault("fluent", {})["file"] = _to_abs(project_root, file_raw)

    return SimSpecV2(
        process=str(payload.get("process", "cvd")),
        time_mode=str(payload.get("time_mode", "steady")),
        reference_plane=ReferencePlaneSpec(**dict(payload.get("reference_plane", {}))),
        inputs=InputsSpec(
            fluent=FluentInputSpec(
                mode=str(payload.get("inputs", {}).get("fluent", {}).get("mode", "steady")),
                file=str(payload.get("inputs", {}).get("fluent", {}).get("file", "")),
                keys=FluentKeysSpec(**dict(payload.get("inputs", {}).get("fluent", {}).get("keys", {}))),
                species=list(payload.get("inputs", {}).get("fluent", {}).get("species", ["s0"])),
            ),
            temperature=TemperatureSpec(**dict(payload.get("inputs", {}).get("temperature", {}))),
        ),
        domain=DomainSpec(**dict(payload.get("domain", {}))),
        roles=RoleSpec(**dict(payload.get("roles", {}))),
        model=AIBModelSpec(
            name=str(payload.get("model", {}).get("name", "aib_ode")),
            orders=AIBOrdersSpec(**dict(payload.get("model", {}).get("orders", {}))),
            params=AIBModelParamsSpec(**dict(payload.get("model", {}).get("params", {}))),
        ),
        time=TimeSpec(
            t_proc_s=float(payload.get("time", {}).get("t_proc_s", 30.0)),
            dt_s=float(payload.get("time", {}).get("dt_s", 0.01)),
            solver=TimeSolverSpec(**dict(payload.get("time", {}).get("solver", {}))),
        ),
        initial_conditions=InitialConditionsSpec(
            theta_A=InitScalarSpec(**dict(payload.get("initial_conditions", {}).get("theta_A", {}))),
            h_nm=InitScalarSpec(**dict(payload.get("initial_conditions", {}).get("h_nm", {}))),
        ),
        measurement=MeasurementSpec(**dict(payload.get("measurement", {}))),
        output=OutputSpec(**dict(payload.get("output", {}))),
    )


def _build_opt_spec(data: Mapping[str, Any], *, project_root: Path) -> OptConfigV2:
    sim_spec = _build_sim_spec(data, project_root=project_root)
    opt_payload = dict(data.get("opt", {}))

    return OptConfigV2(
        sim=sim_spec,
        opt=OptSpecV2(
            task=str(opt_payload.get("task", "fit_roles_and_params")),
            measurement=dict(opt_payload.get("measurement", {})),
            role_enumeration=RoleEnumerationSpec(**dict(opt_payload.get("role_enumeration", {}))),
            order_enumeration=OrderEnumerationSpec(**dict(opt_payload.get("order_enumeration", {}))),
            class_compare=ClassCompareSpec(**dict(opt_payload.get("class_compare", {}))),
            parameter_fit=ParameterFitSpec(**dict(opt_payload.get("parameter_fit", {}))),
            selection=dict(opt_payload.get("selection", {"topk_overall": 20, "topk_per_class": 10})),
            output=dict(opt_payload.get("output", {})),
        ),
    )


def compose_sim_config(
    config_name: str = "cvd_steady_min",
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> SimSpecV2:
    selected_name = _resolve_config_alias(config_name, kind="sim")
    if config_dir is not None:
        root = Path(config_dir).resolve()
        data = OmegaConf.to_container(OmegaConf.load(root / f"{selected_name}.yaml"), resolve=False)
        if not isinstance(data, dict):
            raise ValueError("sim config must resolve to mapping")
    else:
        data = _resolve_defaults_block(selected_name, kind="sim", visited=(), project_root=project_root)

    merged = _apply_overrides(dict(data), overrides)
    base = Path(project_root).resolve() if project_root is not None else Path.cwd()
    return _build_sim_spec(merged, project_root=base)


def compose_opt_config(
    config_name: str,
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> OptConfigV2:
    selected_name = _resolve_config_alias(config_name, kind="opt")
    if config_dir is not None:
        root = Path(config_dir).resolve()
        data = OmegaConf.to_container(OmegaConf.load(root / f"{selected_name}.yaml"), resolve=False)
        if not isinstance(data, dict):
            raise ValueError("opt config must resolve to mapping")
    else:
        data = _resolve_defaults_block(selected_name, kind="opt", visited=(), project_root=project_root)

    merged = _apply_overrides(dict(data), overrides)
    base = Path(project_root).resolve() if project_root is not None else Path.cwd()
    return _build_opt_spec(merged, project_root=base)


def compose_and_save_sim_config(
    output_path: str | Path,
    config_name: str = "cvd_steady_min",
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> SimSpecV2:
    spec = compose_sim_config(
        config_name,
        overrides=overrides,
        config_dir=config_dir,
        project_root=project_root,
    )
    payload = {"sim": asdict(spec)}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(payload, resolve=True, sort_keys=False), encoding="utf-8")
    return spec


def compose_and_save_opt_config(
    output_path: str | Path,
    config_name: str,
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> OptConfigV2:
    spec = compose_opt_config(
        config_name,
        overrides=overrides,
        config_dir=config_dir,
        project_root=project_root,
    )
    payload = {"sim": asdict(spec.sim), "opt": asdict(spec.opt)}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(payload, resolve=True, sort_keys=False), encoding="utf-8")
    return spec


def register_sim_schema(config_name: str = "sim_schema") -> None:
    try:
        from hydra.core.config_store import ConfigStore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Hydra ConfigStore is unavailable") from exc
    ConfigStore.instance().store(name=config_name, node=SimConfigV2)


__all__ = [
    "CONFIG_ROOTS",
    "SIM_CONFIG_ROOT",
    "OPT_CONFIG_ROOT",
    "FluentKeysSpec",
    "FluentInputSpec",
    "TemperatureSpec",
    "InputsSpec",
    "ReferencePlaneSpec",
    "DomainSpec",
    "RoleSpec",
    "AIBOrdersSpec",
    "AIBModelParamsSpec",
    "AIBModelSpec",
    "TimeSolverSpec",
    "TimeSpec",
    "InitScalarSpec",
    "InitialConditionsSpec",
    "MeasurementSpec",
    "OutputSpec",
    "SimSpecV2",
    "SimConfigV2",
    "ParameterFitSpec",
    "RoleEnumerationSpec",
    "OrderEnumerationSpec",
    "ClassCompareSpec",
    "OptSpecV2",
    "OptConfigV2",
    "RunSpec",
    "resolve_config_root",
    "register_sim_schema",
    "compose_sim_config",
    "compose_opt_config",
    "compose_and_save_sim_config",
    "compose_and_save_opt_config",
]
