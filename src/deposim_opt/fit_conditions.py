"""Condition inputs, simulator preparation, and observation scoring for fitting."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from deposim_schema import SimSpecV2
from deposim_sim.common.literals import parse_literal_value
from deposim_sim.common.path_tools import set_attr_path
from deposim_sim.io_plugins import load_fluent_input, load_measurement_input
from deposim_sim.pipeline import run_sim_from_spec
from .objective import evaluate_candidate_score

@dataclass(frozen=True)
class ConditionSpec:
    name: str
    split: str
    weight: float
    fluent_file: str | None
    measurement_file: str | None
    overrides: tuple[str, ...]
    keys: dict[str, Any]
    align: dict[str, Any]
    quantity: str | None = None
    sigma: float | None = None
    xy_unit: str | None = None


def extract_conditions(opt_spec: Any) -> list[ConditionSpec]:
    measurement_cfg = dict(getattr(opt_spec, "measurement", {}) or {})
    default_keys = dict(measurement_cfg.get("keys", {"h": "h_nm", "xy": "xy"}))
    default_align = dict(measurement_cfg.get("align", {}))

    raw_conditions = measurement_cfg.get("conditions")
    if isinstance(raw_conditions, Sequence) and not isinstance(raw_conditions, (str, bytes)) and raw_conditions:
        conditions: list[ConditionSpec] = []
        for idx, row in enumerate(raw_conditions, start=1):
            item = dict(row or {})
            name = str(item.get("name", f"cond_{idx}"))
            conditions.append(
                ConditionSpec(
                    name=name,
                    split=str(item.get("split", "train")).strip().lower(),
                    weight=float(item.get("weight", 1.0)),
                    fluent_file=str(item.get("fluent_file", "")).strip() or None,
                    measurement_file=str(item.get("measurement_file", item.get("file", ""))).strip() or None,
                    overrides=tuple(str(x) for x in list(item.get("overrides", []))),
                    keys=dict(item.get("keys", default_keys)),
                    align=dict(item.get("align", default_align)),
                    quantity=item.get("quantity", measurement_cfg.get("quantity")),
                    sigma=item.get("sigma", measurement_cfg.get("sigma")),
                    xy_unit=item.get("xy_unit", measurement_cfg.get("xy_unit")),
                )
            )
        return conditions

    single = ConditionSpec(
        name="cond_1",
        split="train",
        weight=1.0,
        fluent_file=None,
        measurement_file=str(measurement_cfg.get("file", "")).strip() or None,
        overrides=tuple(),
        keys=default_keys,
        align=default_align,
        quantity=measurement_cfg.get("quantity"),
        sigma=measurement_cfg.get("sigma"),
        xy_unit=measurement_cfg.get("xy_unit"),
    )
    return [single]


def validate_conditions(conditions: list[ConditionSpec]) -> None:
    if not conditions:
        raise ValueError("at least one optimization condition is required")
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError("opt.measurement.conditions names must be unique")

    invalid_splits = sorted({c.split for c in conditions if c.split not in {"train", "holdout"}})
    if invalid_splits:
        raise ValueError(f"condition split must be train|holdout, got {invalid_splits}")

    train_conditions = [c for c in conditions if c.split == "train"]
    if not train_conditions:
        raise ValueError("at least one train condition is required")
    total = float(sum(max(c.weight, 0.0) for c in train_conditions))
    if total <= 0.0:
        raise ValueError("train condition weights must contain at least one positive weight")

    for cond in conditions:
        if not np.isfinite(cond.weight) or cond.weight < 0.0:
            raise ValueError("condition weights must be finite and nonnegative")
        if cond.fluent_file:
            path = Path(cond.fluent_file)
            if not path.exists():
                raise FileNotFoundError(f"condition fluent_file not found: {path}")
        if cond.measurement_file:
            path = Path(cond.measurement_file)
            if not path.exists():
                raise FileNotFoundError(f"condition measurement_file not found: {path}")


def preflight_conditions(
    *,
    sim_spec: SimSpecV2,
    conditions: list[ConditionSpec],
    min_finite_ratio: float,
) -> list[dict[str, Any]]:
    min_ratio = float(min_finite_ratio)
    if min_ratio <= 0.0 or min_ratio > 1.0:
        raise ValueError(f"analysis.preflight.min_finite_ratio must be in (0,1], got {min_ratio}")

    rows: list[dict[str, Any]] = []
    for cond in conditions:
        fluent_path = Path(str(cond.fluent_file or sim_spec.inputs.fluent.file))
        if not fluent_path.exists():
            raise FileNotFoundError(f"preflight fluent file not found: {fluent_path}")

        fluent_loader = (
            str(getattr(sim_spec.inputs.fluent, "io_loader_name", "")).strip().lower()
            or fluent_path.suffix.lstrip(".")
            or "npz"
        )
        fluent = load_fluent_input(
            loader_name=fluent_loader,
            path=fluent_path,
            mode=str(sim_spec.inputs.fluent.mode),
            species=list(sim_spec.inputs.fluent.species),
            keys=sim_spec.inputs.fluent.keys,
        )
        xy = np.asarray(fluent.xy, dtype=float)
        cref = np.asarray(fluent.cref, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"preflight fluent xy must be shape [n,2], got {xy.shape} in {fluent_path}")
        n_pts = int(xy.shape[0])
        if cref.ndim == 2:
            cref_n_pts = int(cref.shape[0])
        elif cref.ndim == 3:
            cref_n_pts = int(cref.shape[1])
        else:
            raise ValueError(f"preflight cref must be 2D or 3D, got {cref.shape} in {fluent_path}")
        if cref_n_pts != n_pts:
            raise ValueError(f"preflight cref/xy point mismatch in {fluent_path}: {cref_n_pts} vs {n_pts}")

        cref_finite_ratio = float(np.mean(np.isfinite(cref))) if cref.size else 0.0
        if cref_finite_ratio < min_ratio:
            raise ValueError(
                f"preflight fluent finite ratio below threshold for {cond.name}: "
                f"{cref_finite_ratio:.4f} < {min_ratio:.4f}"
            )

        meas_finite_ratio = None
        measurement_path = Path(str(cond.measurement_file)) if cond.measurement_file else None
        if measurement_path is not None and measurement_path.exists():
            measurement_loader = (
                str(getattr(sim_spec.measurement, "io_loader_name", "")).strip().lower()
                or measurement_path.suffix.lstrip(".")
                or "npz"
            )
            measurement = load_measurement_input(
                loader_name=measurement_loader,
                path=measurement_path,
                keys=cond.keys,
            )
            h = np.asarray(measurement.h, dtype=float).reshape(-1)
            xy_meas = np.asarray(measurement.xy, dtype=float)
            if xy_meas.ndim != 2 or xy_meas.shape[1] != 2 or xy_meas.shape[0] != h.shape[0]:
                raise ValueError(f"preflight measurement xy shape mismatch in {measurement_path}")

            meas_finite_ratio = float(np.mean(np.isfinite(h))) if h.size else 0.0
            if meas_finite_ratio < min_ratio:
                raise ValueError(
                    f"preflight measurement finite ratio below threshold for {cond.name}: "
                    f"{meas_finite_ratio:.4f} < {min_ratio:.4f}"
                )

        rows.append(
            {
                "condition": cond.name,
                "split": cond.split,
                "fluent_file": str(fluent_path),
                "measurement_file": str(measurement_path) if measurement_path else "",
                "n_points": n_pts,
                "fluent_finite_ratio": cref_finite_ratio,
                "measurement_finite_ratio": meas_finite_ratio,
            }
        )
    return rows


def prepare_condition(
    *,
    base_spec: SimSpecV2,
    role_candidate: Any,
    order_candidate: Mapping[str, int],
    condition: ConditionSpec,
    params: Mapping[str, float],
) -> SimSpecV2:
    trial_spec = deepcopy(base_spec)
    trial_spec.roles.A = role_candidate.A
    trial_spec.roles.I = role_candidate.I
    trial_spec.roles.B = role_candidate.B

    trial_spec.model.orders.adsorption_site_order = int(order_candidate["adsorption_site_order"])
    trial_spec.model.orders.reaction_site_order_A = int(order_candidate["reaction_site_order_A"])
    trial_spec.model.orders.reaction_site_order_star = int(order_candidate["reaction_site_order_star"])

    if condition.fluent_file:
        trial_spec.inputs.fluent.file = str(condition.fluent_file)

    meas_file = str(condition.measurement_file or "").strip()
    trial_spec.measurement.enabled = bool(meas_file)
    trial_spec.measurement.file = meas_file
    trial_spec.measurement.keys = dict(condition.keys)
    trial_spec.measurement.align = dict(condition.align)
    if condition.quantity is not None:
        trial_spec.measurement.quantity = condition.quantity
    if condition.sigma is not None:
        trial_spec.measurement.sigma = condition.sigma
    if condition.xy_unit is not None:
        trial_spec.measurement.xy_unit = condition.xy_unit

    for override in condition.overrides:
        text = str(override).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"condition override must use key=value format: {text!r}")
        key, raw = text.split("=", 1)
        set_attr_path(
            trial_spec,
            key.strip(),
            parse_literal_value(raw),
            strip_sim_prefix=True,
        )

    for key, value in params.items():
        set_attr_path(trial_spec, key, float(value), strip_sim_prefix=True)

    return trial_spec


def evaluate_condition(spec: SimSpecV2, objective: Mapping[str, Any]) -> dict[str, float]:
    """One simulator/measurement path for train, held-out, and refit scoring."""
    result = run_sim_from_spec(spec)
    components = evaluate_candidate_score(
        residual_nm=result.fields.get("residual_nm"), fields=result.fields,
        diagnostics=result.diagnostics, objective=objective,
    )
    alignment = result.diagnostics.get("observation") or result.diagnostics.get("measurement_alignment", {})
    components.update(
        alignment_mean_distance_mm=float(alignment.get("mean_distance_mm", float("nan"))),
        alignment_max_distance_mm=float(alignment.get("max_distance_mm", float("nan"))),
        alignment_rejected_count=float(alignment.get("distance_rejected_count", 0)),
    )
    return components
