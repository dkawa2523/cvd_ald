"""Compile model-aware fit configuration into a sampler-neutral search space."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from deposim_schema import SimSpecV2
from deposim_sim.common.overrides import as_bool


_ALLOWED_PARAMETER_GROUPS = {
    "surface_kinetics",
    "effective_transport",
    "measurement_or_interface",
}
_ALLOWED_PARAMETER_STAGES = {"screening", "sobol", "calibration"}


def _safe_name(text: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")
    return clean or "cond"


def _stage_filter(task: str) -> str | None:
    return "calibration" if str(task).strip().lower().startswith("fit") else None


def compile_parameter_space(
    raw_space: list[dict[str, Any]],
    *,
    sim_spec: SimSpecV2,
    task: str,
    role_has_i: bool,
    role_has_b: bool,
) -> list[dict[str, Any]]:
    """Filter conditional variables and validate their numerical metadata."""

    stage_filter = _stage_filter(task)
    compiled: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_space:
        item = dict(raw)
        if not as_bool(item.get("enabled", True)):
            continue
        stage = str(item.get("stage", "")).strip().lower()
        if stage_filter and stage and stage != stage_filter:
            continue
        condition = item.get("condition")
        if condition not in {None, "", "role_has_B", "role_has_no_B", "role_has_I"}:
            raise ValueError(f"unsupported search_space condition: {condition!r}")
        if condition == "role_has_B" and not role_has_b:
            continue
        if condition == "role_has_no_B" and role_has_b:
            continue
        if condition == "role_has_I" and not role_has_i:
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("search_space item requires non-empty name")
        if name in names:
            raise ValueError(f"duplicate search_space parameter: {name}")
        names.add(name)

        group = str(item.get("group", "")).strip().lower()
        if group and group not in _ALLOWED_PARAMETER_GROUPS:
            raise ValueError(
                f"search_space item {name!r} has invalid group {group!r}; "
                f"expected one of {sorted(_ALLOWED_PARAMETER_GROUPS)}"
            )
        if stage and stage not in _ALLOWED_PARAMETER_STAGES:
            raise ValueError(
                f"search_space item {name!r} has invalid stage {stage!r}; "
                f"expected one of {sorted(_ALLOWED_PARAMETER_STAGES)}"
            )
        for metadata_name in ("symbol", "unit"):
            value = item.get(metadata_name)
            if value is not None and not str(value).strip():
                raise ValueError(f"search_space item {name!r} has empty {metadata_name}")

        low = float(item["low"])
        high = float(item["high"])
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"search_space bounds invalid for {name}: [{low}, {high}]")
        kind = str(item.get("type", "loguniform")).strip().lower()
        if kind not in {"linear", "uniform", "log", "loguniform"}:
            raise ValueError(f"unsupported search_space type for {name}: {kind!r}")
        if kind in {"log", "loguniform"} and low <= 0.0:
            raise ValueError(f"log search requires a positive lower bound for {name}")
        item["per_condition"] = as_bool(item.get("per_condition", False))
        if item["per_condition"]:
            hierarchical = dict(item.get("hierarchical", {}) or {})
            mode = str(hierarchical.get("mode", "log_offset")).strip().lower()
            if mode != "log_offset":
                raise ValueError(f"unsupported hierarchical mode for {name}: {mode}")
            if low <= 0.0:
                raise ValueError(f"log_offset requires a positive base bound for {name}")
            sigma = float(hierarchical.get("sigma", 0.5))
            delta_low = float(hierarchical.get("delta_low", -2.0))
            delta_high = float(hierarchical.get("delta_high", 2.0))
            if not all(np.isfinite(value) for value in (sigma, delta_low, delta_high)):
                raise ValueError(f"hierarchical values must be finite for {name}")
            if sigma < 0.0 or delta_low > delta_high:
                raise ValueError(f"hierarchical bounds or sigma are invalid for {name}")
        compiled.append(item)

    _validate_transport_space(sim_spec, compiled, role_has_b=role_has_b)
    return compiled


def _validate_transport_space(
    sim_spec: SimSpecV2,
    space: list[dict[str, Any]],
    *,
    role_has_b: bool,
) -> None:
    transport = dict(getattr(sim_spec.model.params, "transport", {}) or {})
    if str(transport.get("km_source", "fit_scalar")).strip().lower() != "from_cfd_flux_sink":
        return
    names = {str(item["name"]) for item in space}
    forbidden = sorted(names & {"model.params.transport.km_A", "model.params.transport.km_B"})
    if forbidden:
        raise ValueError(
            "km_source=from_cfd_flux_sink forbids direct km optimization. Remove: "
            + ", ".join(forbidden)
        )
    if "model.params.transport.gamma_km_A" not in names:
        raise ValueError(
            "km_source=from_cfd_flux_sink requires model.params.transport.gamma_km_A in search_space"
        )
    if role_has_b and "model.params.transport.gamma_km_B" not in names:
        raise ValueError(
            "km_source=from_cfd_flux_sink with role B requires "
            "model.params.transport.gamma_km_B in search_space"
        )


def parameter_dimension(space: list[dict[str, Any]], condition_names: list[str]) -> int:
    dimension = 0
    for item in space:
        if float(item["high"]) > float(item["low"]):
            dimension += 1
        if item.get("per_condition", False):
            hierarchical = dict(item.get("hierarchical", {}) or {})
            if float(hierarchical.get("delta_high", 2.0)) > float(
                hierarchical.get("delta_low", -2.0)
            ):
                dimension += len(condition_names)
    return dimension


def active_parameter_paths(space: list[dict[str, Any]]) -> list[str]:
    active = []
    for item in space:
        hierarchical = dict(item.get("hierarchical", {}) or {})
        varying_offset = bool(item.get("per_condition", False)) and float(
            hierarchical.get("delta_high", 2.0)
        ) > float(hierarchical.get("delta_low", -2.0))
        if float(item["high"]) > float(item["low"]) or varying_offset:
            active.append(str(item["name"]))
    return active


def _sample_value(
    item: Mapping[str, Any],
    *,
    name: str,
    trial: Any | None,
    rng: Any | None,
) -> float:
    low = float(item["low"])
    high = float(item["high"])
    kind = str(item.get("type", "loguniform")).strip().lower()
    if low == high:
        return low
    if trial is not None:
        return float(trial.suggest_float(name, low, high, log=kind in {"log", "loguniform"}))
    if rng is None:
        raise RuntimeError("rng is required when trial is not provided")
    if kind in {"log", "loguniform"}:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def draw_parameter_sample(
    *,
    space: list[dict[str, Any]],
    condition_names: list[str],
    lambda_prior: float,
    trial: Any | None = None,
    rng: Any | None = None,
) -> dict[str, Any]:
    """Draw one shared or hierarchical sample from an already compiled space."""

    per_condition: dict[str, dict[str, float]] = {name: {} for name in condition_names}
    flat_params: dict[str, float] = {}
    prior_terms: list[float] = []
    for item in space:
        path = str(item["name"])
        if not bool(item.get("per_condition", False)):
            value = _sample_value(item, trial=trial, rng=rng, name=path)
            flat_params[path] = value
            for condition in condition_names:
                per_condition[condition][path] = value
            continue

        hierarchical = dict(item.get("hierarchical", {}) or {})
        mode = str(hierarchical.get("mode", "log_offset")).strip().lower()
        if mode != "log_offset":
            raise ValueError(f"unsupported hierarchical mode for {path}: {mode}")
        sigma = float(hierarchical.get("sigma", 0.5))
        delta_low = float(hierarchical.get("delta_low", -2.0))
        delta_high = float(hierarchical.get("delta_high", 2.0))
        if delta_low > delta_high:
            raise ValueError(f"delta bounds invalid for {path}: [{delta_low}, {delta_high}]")

        base_key = f"{path}__base"
        base = _sample_value(item, trial=trial, rng=rng, name=base_key)
        if base <= 0.0:
            raise ValueError(f"log_offset requires positive base parameter for {path}")
        flat_params[base_key] = base
        for condition in condition_names:
            delta_key = f"{path}__delta__{_safe_name(condition)}"
            if trial is not None:
                delta = float(trial.suggest_float(delta_key, delta_low, delta_high))
            else:
                if rng is None:
                    raise RuntimeError("rng is required when trial is not provided")
                delta = float(rng.uniform(delta_low, delta_high))
            flat_params[delta_key] = delta
            per_condition[condition][path] = float(base * math.exp(delta))
            if sigma > 0.0 and lambda_prior > 0.0:
                prior_terms.append((delta / sigma) ** 2)

    return {
        "flat_params": flat_params,
        "per_condition": per_condition,
        "prior_terms": prior_terms,
    }


__all__ = [
    "active_parameter_paths",
    "compile_parameter_space",
    "draw_parameter_sample",
    "parameter_dimension",
]
