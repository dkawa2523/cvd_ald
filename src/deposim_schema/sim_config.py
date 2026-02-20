"""Structured simulation configs and Hydra composition helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SIM_CONFIG_ROOT = Path("configs/sim")
OPT_CONFIG_ROOT = Path("configs/opt")  # stub root for P0
CONFIG_ROOTS: dict[str, Path] = {"sim": SIM_CONFIG_ROOT, "opt": OPT_CONFIG_ROOT}

_DOMAIN_KINDS = {"wafer_2d_polar", "wafer_1d_radial", "wafer_2d_xy"}
_TIME_MODES = {"cvd_steady", "cvd_transient", "ald_cycle"}
_TIME_MODE_ALIASES = {
    "cvd_steady": "cvd_steady",
    "steady": "cvd_steady",
    "cvd_transient": "cvd_transient",
    "transient": "cvd_transient",
    "ald_cycle": "ald_cycle",
    "phases": "ald_cycle",
}
_ENGINES = {"numpy", "jax"}
_DEVICES = {"cpu", "gpu", "auto"}
_DTYPES = {"float32", "float64"}
_ARRAY_STORES = {"npz", "zarr", "hdf5"}


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _ensure_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _ensure_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        ordered = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{ordered}}}, got '{value}'")


def _normalize_time_mode(value: str) -> str:
    canonical = _TIME_MODE_ALIASES.get(value)
    if canonical is None:
        ordered = ", ".join(sorted(_TIME_MODE_ALIASES))
        raise ValueError(f"time.mode must be one of {{{ordered}}}, got '{value}'")
    if canonical not in _TIME_MODES:
        ordered = ", ".join(sorted(_TIME_MODES))
        raise ValueError(f"time.mode canonical value must be one of {{{ordered}}}, got '{canonical}'")
    return canonical


@dataclass
class DomainSpec:
    kind: str = "wafer_2d_polar"
    wafer_radius_mm: float = 150.0
    nr: int = 64
    ntheta: int = 180
    nx: int = 128
    ny: int = 128
    edge_exclusion_mm: float = 3.0

    def __post_init__(self) -> None:
        _ensure_choice("domain.kind", self.kind, _DOMAIN_KINDS)
        _ensure_positive("domain.wafer_radius_mm", self.wafer_radius_mm)
        _ensure_nonnegative("domain.edge_exclusion_mm", self.edge_exclusion_mm)
        if self.nr < 2:
            raise ValueError(f"domain.nr must be >= 2, got {self.nr}")
        if self.kind == "wafer_2d_polar" and self.ntheta < 2:
            raise ValueError(f"domain.ntheta must be >= 2 for polar domain, got {self.ntheta}")
        if self.kind == "wafer_1d_radial" and self.ntheta < 1:
            raise ValueError(f"domain.ntheta must be >= 1 for radial domain, got {self.ntheta}")
        if self.kind == "wafer_2d_xy":
            if self.nx < 2 or self.ny < 2:
                raise ValueError(
                    f"domain.nx and domain.ny must be >= 2 for xy domain, got ({self.nx}, {self.ny})"
                )


@dataclass
class ReferencePlaneSpec:
    z_ref_mm: float = 5.0
    z_ref_mm_list: list[float] = field(default_factory=list)
    species: list[str] = field(default_factory=lambda: ["precursor"])
    c_ref_unit: str = "mol_m3"

    def __post_init__(self) -> None:
        _ensure_positive("reference_plane.z_ref_mm", self.z_ref_mm)
        if not isinstance(self.z_ref_mm_list, list):
            raise ValueError("reference_plane.z_ref_mm_list must be a list of positive values")
        for idx, value in enumerate(self.z_ref_mm_list):
            if float(value) <= 0.0:
                raise ValueError(f"reference_plane.z_ref_mm_list[{idx}] must be > 0")
        if not self.species:
            raise ValueError("reference_plane.species must contain at least one species name")
        if len(set(self.species)) != len(self.species):
            raise ValueError("reference_plane.species must not contain duplicates")


@dataclass
class TimeSpec:
    mode: str = "cvd_steady"
    process_time_s: float = 60.0
    dt_s: float = 0.1
    ald_cycles: int = 1
    phases: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = _normalize_time_mode(self.mode)
        _ensure_positive("time.process_time_s", self.process_time_s)
        _ensure_positive("time.dt_s", self.dt_s)
        if self.ald_cycles < 1:
            raise ValueError(f"time.ald_cycles must be >= 1, got {self.ald_cycles}")
        if not isinstance(self.phases, list):
            raise ValueError("time.phases must be a list")
        for idx, phase in enumerate(self.phases):
            if not isinstance(phase, dict):
                raise ValueError(f"time.phases[{idx}] must be a mapping")
            duration = phase.get("duration_s")
            if duration is not None and float(duration) <= 0.0:
                raise ValueError(f"time.phases[{idx}].duration_s must be > 0 when specified")
            if "name" in phase and not str(phase["name"]).strip():
                raise ValueError(f"time.phases[{idx}].name must be non-empty when specified")


@dataclass
class InputsSpec:
    synthetic_case: str = "uniform"
    io_loader_name: str = "npz"
    source_kind: str = "synthetic"
    field_path: str = ""
    c_ref_mol_m3: float = 1.0
    temperature_k: float = 700.0
    pressure_pa: float = 1333.0
    omega_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.io_loader_name:
            raise ValueError("inputs.io_loader_name must be non-empty")
        source_kind = str(self.source_kind).strip().lower()
        if source_kind not in {"synthetic", "file"}:
            raise ValueError("inputs.source_kind must be one of {'synthetic', 'file'}")
        self.source_kind = source_kind
        if self.source_kind == "file" and not str(self.field_path).strip():
            raise ValueError("inputs.field_path must be non-empty when inputs.source_kind='file'")
        _ensure_positive("inputs.c_ref_mol_m3", self.c_ref_mol_m3)
        _ensure_positive("inputs.temperature_k", self.temperature_k)
        _ensure_positive("inputs.pressure_pa", self.pressure_pa)
        _ensure_nonnegative("inputs.omega_rad_s", self.omega_rad_s)


@dataclass
class DriversSpec:
    enable_time_driver: bool = False
    enable_spatial_driver: bool = False
    scalar_schedule: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_schedule, dict):
            raise ValueError("drivers.scalar_schedule must be a dict of scalar driver values")


@dataclass
class PluginCompatibilitySpec:
    requires: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    time_modes: list[str] = field(default_factory=lambda: ["cvd_steady", "cvd_transient", "ald_cycle"])
    governing_class: str = "generic"

    def __post_init__(self) -> None:
        if not self.governing_class:
            raise ValueError("governing_class must be non-empty")
        if not isinstance(self.requires, list) or not all(isinstance(x, str) and x for x in self.requires):
            raise ValueError("requires must be a list of non-empty strings")
        if not isinstance(self.excludes, list) or not all(isinstance(x, str) and x for x in self.excludes):
            raise ValueError("excludes must be a list of non-empty strings")
        if not isinstance(self.time_modes, list) or not self.time_modes:
            raise ValueError("time_modes must be a non-empty list of mode names")
        normalized = [_normalize_time_mode(mode) for mode in self.time_modes]
        self.time_modes = sorted(set(normalized))


@dataclass
class ModelSpec:
    mass_transfer_name: str = "stagnant_film"
    kinetics_name: str = "power_law"
    state_name: str = "none"
    net_name: str = "deposition_only"
    mass_transfer_params: dict[str, Any] = field(default_factory=lambda: {"k_m_m_s": 0.02})
    kinetics_params: dict[str, Any] = field(default_factory=lambda: {"k0": 1.0, "order": 1.0})
    net_params: dict[str, Any] = field(default_factory=dict)
    state_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mass_transfer_name:
            raise ValueError("model.mass_transfer_name must be non-empty")
        if not self.kinetics_name:
            raise ValueError("model.kinetics_name must be non-empty")
        if not self.net_name:
            raise ValueError("model.net_name must be non-empty")


@dataclass
class SolverSpec:
    root_solver_name: str = "bisection"
    max_iter: int = 80
    rtol: float = 1.0e-6
    atol: float = 1.0e-12
    monotonicity_check: bool = True

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(f"solver.max_iter must be >= 1, got {self.max_iter}")
        _ensure_positive("solver.rtol", self.rtol)
        _ensure_positive("solver.atol", self.atol)


@dataclass
class ComputeSpec:
    engine: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"
    batch_size: int = 0

    def __post_init__(self) -> None:
        _ensure_choice("compute.engine", self.engine, _ENGINES)
        _ensure_choice("compute.device", self.device, _DEVICES)
        _ensure_choice("compute.dtype", self.dtype, _DTYPES)
        if self.batch_size < 0:
            raise ValueError(f"compute.batch_size must be >= 0, got {self.batch_size}")


@dataclass
class OutputSpec:
    project_dir: str = "results"
    run_dir_name: str = "example_cvd"
    resolved_config_filename: str = "config_resolved.yaml"
    array_store: str = "npz"
    write_report: bool = True

    def __post_init__(self) -> None:
        if not self.project_dir:
            raise ValueError("output.project_dir must be non-empty")
        if not self.run_dir_name:
            raise ValueError("output.run_dir_name must be non-empty")
        if not self.resolved_config_filename:
            raise ValueError("output.resolved_config_filename must be non-empty")
        _ensure_choice("output.array_store", self.array_store, _ARRAY_STORES)


@dataclass
class MeasurementSpec:
    enabled: bool = False
    path: str = ""
    format: str = "npz"
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    edge_exclusion_mm: float = 0.0
    interpolation: str = "nearest"

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"measurement.scale must be > 0, got {self.scale}")
        _ensure_nonnegative("measurement.edge_exclusion_mm", self.edge_exclusion_mm)
        _ensure_choice("measurement.interpolation", self.interpolation, {"nearest", "bilinear"})


@dataclass
class KpiSpec:
    enabled: bool = True
    spec_min: float | None = None
    spec_max: float | None = None
    ring_count: int = 5

    def __post_init__(self) -> None:
        if self.ring_count < 1:
            raise ValueError(f"kpi.ring_count must be >= 1, got {self.ring_count}")
        if self.spec_min is not None and self.spec_max is not None and self.spec_min > self.spec_max:
            raise ValueError("kpi.spec_min must be <= kpi.spec_max")


@dataclass
class ValidatorSpec:
    enabled: bool = True
    strict: bool = True


@dataclass
class RunSpec:
    run_name: str = "example_cvd"
    random_seed: int = 0
    domain: DomainSpec = field(default_factory=DomainSpec)
    reference_plane: ReferencePlaneSpec = field(default_factory=ReferencePlaneSpec)
    time: TimeSpec = field(default_factory=TimeSpec)
    inputs: InputsSpec = field(default_factory=InputsSpec)
    drivers: DriversSpec = field(default_factory=DriversSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    kpi: KpiSpec = field(default_factory=KpiSpec)
    validator: ValidatorSpec = field(default_factory=ValidatorSpec)

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be >= 0, got {self.random_seed}")


def resolve_config_root(kind: str, project_root: str | Path | None = None) -> Path:
    """Return the configuration root for sim/opt."""
    if kind not in CONFIG_ROOTS:
        ordered = ", ".join(sorted(CONFIG_ROOTS))
        raise ValueError(f"kind must be one of {{{ordered}}}, got '{kind}'")
    base = Path(project_root) if project_root is not None else Path.cwd()
    return (base / CONFIG_ROOTS[kind]).resolve()


def register_sim_schema(config_name: str = "sim_schema") -> None:
    """Register RunSpec into Hydra ConfigStore."""
    try:
        from hydra.core.config_store import ConfigStore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Hydra ConfigStore is unavailable. Install hydra-core to register structured configs."
        ) from exc

    ConfigStore.instance().store(name=config_name, node=RunSpec)


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(line):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (idx == 0 or line[idx - 1].isspace()):
            return line[:idx].rstrip()
    return line.rstrip()


def _parse_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "~"}:
        return None
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _tokenize_yaml(text: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        clean = _strip_yaml_comment(stripped)
        if not clean:
            continue
        indent = len(raw) - len(stripped)
        tokens.append((indent, clean))
    return tokens


def _parse_yaml_list(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[list[Any], int]:
    parsed: list[Any] = []
    idx = start
    while idx < len(tokens) and tokens[idx][0] == indent and tokens[idx][1].startswith("- "):
        item_raw = tokens[idx][1][2:].strip()
        idx += 1
        if item_raw:
            parsed.append(_parse_yaml_scalar(item_raw))
            continue
        if idx >= len(tokens) or tokens[idx][0] <= indent:
            parsed.append(None)
            continue
        item, idx = _parse_yaml_block(tokens, idx, tokens[idx][0])
        parsed.append(item)
    return parsed, idx


def _parse_yaml_map(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[dict[str, Any], int]:
    parsed: dict[str, Any] = {}
    idx = start
    while idx < len(tokens):
        token_indent, token_text = tokens[idx]
        if token_indent != indent:
            break
        if token_text.startswith("- "):
            raise ValueError(f"Unexpected list item in mapping at token {idx}: {token_text!r}")
        if ":" not in token_text:
            raise ValueError(f"Expected key:value YAML pair at token {idx}: {token_text!r}")
        key, remainder = token_text.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        idx += 1
        if remainder:
            parsed[key] = _parse_yaml_scalar(remainder)
            continue
        if idx >= len(tokens) or tokens[idx][0] <= indent:
            parsed[key] = {}
            continue
        child, idx = _parse_yaml_block(tokens, idx, tokens[idx][0])
        parsed[key] = child
    return parsed, idx


def _parse_yaml_block(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(tokens):
        return {}, start
    if tokens[start][0] != indent:
        raise ValueError(f"Invalid YAML indentation near token {start}: expected {indent}")
    if tokens[start][1].startswith("- "):
        return _parse_yaml_list(tokens, start, indent)
    return _parse_yaml_map(tokens, start, indent)


def _parse_yaml_subset(path: Path) -> dict[str, Any]:
    tokens = _tokenize_yaml(path.read_text(encoding="utf-8"))
    if not tokens:
        return {}
    parsed, idx = _parse_yaml_block(tokens, 0, tokens[0][0])
    if idx != len(tokens):
        raise ValueError(f"YAML parser did not consume all tokens in {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Top-level YAML node must be a mapping in {path}")
    return parsed


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _compose_named_yaml_config(
    config_name: str,
    root: Path,
    stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if config_name in stack:
        chain = " -> ".join((*stack, config_name))
        raise ValueError(f"Circular defaults reference detected: {chain}")

    path = root / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _parse_yaml_subset(path)
    defaults = raw.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"'defaults' must be a list in {path}")

    composed: dict[str, Any] = {}
    used_self = False

    for entry in defaults:
        if entry == "_self_":
            composed = _deep_merge(composed, raw)
            used_self = True
            continue
        if not isinstance(entry, str):
            raise ValueError(
                f"Unsupported defaults entry type in {path}: {type(entry).__name__}. "
                "Only string entries are supported for fallback composition."
            )
        include_name = entry.rsplit("/", 1)[-1]
        composed = _deep_merge(composed, _compose_named_yaml_config(include_name, root, (*stack, config_name)))

    if not defaults or not used_self:
        composed = _deep_merge(composed, raw)

    return composed


def _apply_overrides(config: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value form, got: {override!r}")
        key_path, value_raw = override.split("=", 1)
        key_path = key_path.lstrip("+").strip()
        keys = [part.strip() for part in key_path.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"Invalid override key path: {override!r}")
        cursor: dict[str, Any] = updated
        for part in keys[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[keys[-1]] = _parse_yaml_scalar(value_raw)
    return updated


def _build_run_spec(config: dict[str, Any]) -> RunSpec:
    defaults = asdict(RunSpec())
    merged = _deep_merge(defaults, config)
    return RunSpec(
        run_name=merged["run_name"],
        random_seed=merged["random_seed"],
        domain=DomainSpec(**merged["domain"]),
        reference_plane=ReferencePlaneSpec(**merged["reference_plane"]),
        time=TimeSpec(**merged["time"]),
        inputs=InputsSpec(**merged["inputs"]),
        drivers=DriversSpec(**merged["drivers"]),
        model=ModelSpec(**merged["model"]),
        solver=SolverSpec(**merged["solver"]),
        compute=ComputeSpec(**merged["compute"]),
        output=OutputSpec(**merged["output"]),
        measurement=MeasurementSpec(**merged["measurement"]),
        kpi=KpiSpec(**merged["kpi"]),
        validator=ValidatorSpec(**merged["validator"]),
    )


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if value == "":
        return '""'
    text = str(value)
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/+-")
    if all(ch in safe_chars for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dump_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                if item:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_dump_yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {{}}")
                continue
            if isinstance(item, list):
                if item:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_dump_yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{pad}{key}: []")
                continue
            lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _compose_sim_hydra_or_fallback(
    config_name: str,
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[RunSpec, str]:
    root = Path(config_dir) if config_dir is not None else resolve_config_root("sim", project_root)
    if not root.exists():
        raise FileNotFoundError(f"Sim config root does not exist: {root}")

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception:
        composed = _compose_named_yaml_config(config_name, root)
        composed = _apply_overrides(composed, overrides)
        run_spec = _build_run_spec(composed)
        resolved_yaml = "\n".join(_dump_yaml_lines(asdict(run_spec))) + "\n"
        return run_spec, resolved_yaml

    with initialize_config_dir(version_base=None, config_dir=str(root)):
        yaml_cfg = compose(config_name=config_name, overrides=list(overrides or ()))

    merged = OmegaConf.merge(OmegaConf.structured(RunSpec()), yaml_cfg)
    run_spec = OmegaConf.to_object(merged)
    if not isinstance(run_spec, RunSpec):
        raise TypeError(f"Expected composed config to be RunSpec, got {type(run_spec)!r}")
    resolved_yaml = OmegaConf.to_yaml(merged, resolve=True, sort_keys=False)
    return run_spec, resolved_yaml


def compose_sim_config(
    config_name: str = "example_cvd",
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> RunSpec:
    """Compose YAML config with Hydra and return validated RunSpec."""
    run_spec, _ = _compose_sim_hydra_or_fallback(
        config_name,
        overrides=overrides,
        config_dir=config_dir,
        project_root=project_root,
    )
    return run_spec


def compose_and_save_sim_config(
    output_path: str | Path,
    config_name: str = "example_cvd",
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> RunSpec:
    """Compose sim config by Hydra and save resolved YAML."""
    run_spec, resolved_yaml = _compose_sim_hydra_or_fallback(
        config_name,
        overrides=overrides,
        config_dir=config_dir,
        project_root=project_root,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resolved_yaml, encoding="utf-8")
    return run_spec


__all__ = [
    "CONFIG_ROOTS",
    "SIM_CONFIG_ROOT",
    "OPT_CONFIG_ROOT",
    "DomainSpec",
    "ReferencePlaneSpec",
    "TimeSpec",
    "InputsSpec",
    "DriversSpec",
    "PluginCompatibilitySpec",
    "ModelSpec",
    "SolverSpec",
    "ComputeSpec",
    "OutputSpec",
    "MeasurementSpec",
    "KpiSpec",
    "ValidatorSpec",
    "RunSpec",
    "resolve_config_root",
    "register_sim_schema",
    "compose_sim_config",
    "compose_and_save_sim_config",
]
