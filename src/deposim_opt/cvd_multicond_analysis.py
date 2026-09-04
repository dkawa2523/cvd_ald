"""Transfer evaluation for multiple CVD concentration/rate maps.

Multiple process conditions identify a transferable response and one condition
remains completely held out for prediction.  The production path compares
quasi-steady site-balance reductions with observable lumped parameters.  The
previous total-concentration/composition power response remains available as an
explicit compatibility model.

The fitted quantities are effective transfer coefficients.  They are not
elementary kinetic constants or proof of chemical causality.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .class_compare import rank_role_candidates, build_role_summary, build_role_stability, effect_signature, build_condition_scores
from .fit_roles import condition_refits
from .objective import prediction_metrics
from .cvd_surface_kinetics import (
    SurfaceKineticCandidate,
    SurfaceKineticFit,
    enumerate_surface_kinetic_candidates,
    fit_surface_kinetic,
    predict_surface_kinetic,
    surface_formula,
)

from .cvd_spatial_analysis import (
    RoleResponseCandidate,
    _EPS,
    _align_validation,
    _angular_groups,
    _coordinate_matrix,
    _fit_nonnegative_effects,
    _json_safe,
    _metrics,
    _radial_groups,
    _read_numeric_csv,
    _safe_corr,
    _sha256_file,
    _write_json,
    _write_rows,
    enumerate_role_response_candidates,
)

# Fixed before evaluation; selected jointly with the role by training-condition CV.
# The loss is mean squared log-rate error; elasticities have unit prior scale.
REGULARIZATION_GRID = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
RESPONSE_STRUCTURES = ("shared", "within_between")
RESPONSE_MODELS = ("surface_qss", "empirical_power")


@dataclass(frozen=True)
class ConditionCase:
    case_id: int
    condition_path: Path
    validation_path: Path
    xyz: np.ndarray
    species: tuple[str, ...]
    concentrations: dict[str, np.ndarray]
    mole_fractions: dict[str, np.ndarray]
    density: np.ndarray
    total_concentration: np.ndarray
    rate: np.ndarray
    quality: dict[str, Any]


@dataclass(frozen=True)
class CombinedCases:
    case_ids: tuple[int, ...]
    xyz: np.ndarray
    condition_id: np.ndarray
    species: tuple[str, ...]
    concentrations: dict[str, np.ndarray]
    species_fractions: dict[str, np.ndarray]
    total_concentration: np.ndarray
    rate: np.ndarray


@dataclass(frozen=True)
class TransferFit:
    candidate: RoleResponseCandidate
    coefficients: np.ndarray
    prediction: np.ndarray
    design: np.ndarray
    effect_names: tuple[str, ...]
    reference_total_concentration: float
    reference_species_fractions: dict[str, float]
    active_effects: tuple[bool, ...]
    regularization: float = 0.0
    response_structure: str = "shared"

    @property
    def reference_rate_nm_s(self) -> float:
        return float(math.exp(float(self.coefficients[0])))

    @property
    def common_order(self) -> float:
        """Between-condition total order; also the within order in shared fits."""
        return float(self.coefficients[1])

    @property
    def coefficient_blocks(self) -> dict[str, np.ndarray]:
        count = len(_effect_names(self.candidate))
        if self.response_structure == "shared":
            return {"shared": self.coefficients[1:]}
        return {"between": self.coefficients[1:1 + count], "within": self.coefficients[1 + count:]}

    @property
    def within_order(self) -> float:
        return float(list(self.coefficient_blocks.values())[-1][0])

    @property
    def coefficient_terms(self) -> list[tuple[str, str]]:
        """Scope and effect identity, independent of user-facing species names."""
        return [(scope, name) for scope in self.coefficient_blocks for name in _effect_names(self.candidate)]

    @property
    def effect_scopes(self) -> dict[str, list[str]]:
        return {group: [scope for scope, values in self.coefficient_blocks.items() if values[i + 1] > 0]
                for i, group in enumerate(self.candidate.effect_groups)}

    @property
    def effective_roles(self) -> dict[str, str | None]:
        roles = {"A": None, "I": None, "B": None}
        for group in self.effect_groups:
            for slot in (("A", "B") if group == "AB" else (group,)):
                roles[slot] = getattr(self.candidate, slot)
        return roles

    @property
    def effect_groups(self) -> dict[str, list[str]]:
        return {group: species for group, species in self.candidate.effect_groups.items()
                if self.effect_scopes[group]}


def _positive_min_step(values: np.ndarray) -> float | None:
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size < 2:
        return None
    differences = np.diff(unique)
    positive = differences[differences > 0.0]
    return float(np.min(positive)) if positive.size else None


def _load_case(case_id: int, condition_path: Path, validation_path: Path) -> ConditionCase:
    condition_headers, condition = _read_numeric_csv(condition_path)
    validation_headers, validation = _read_numeric_csv(validation_path)
    if "dr_nm_per_sec" not in validation:
        raise ValueError(f"{validation_path} must contain dr_nm_per_sec")
    concentration_columns = [name for name in condition_headers if name.startswith("concentration_")]
    if not concentration_columns:
        raise ValueError(f"No concentration_* columns found in {condition_path}")
    species = tuple(name.removeprefix("concentration_") for name in concentration_columns)
    concentrations = {
        name: np.asarray(condition[f"concentration_{name}"], dtype=float) for name in species
    }
    if any(np.any(~np.isfinite(values)) or np.any(values <= 0.0) for values in concentrations.values()):
        raise ValueError(f"All concentrations must be positive and finite in {condition_path}")
    condition_xyz = _coordinate_matrix(condition)
    validation_xyz = _coordinate_matrix(validation)
    rate, alignment = _align_validation(
        condition_xyz,
        validation_xyz,
        validation["dr_nm_per_sec"],
        coordinate_decimals=6,
    )
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
        raise ValueError(f"All deposition rates must be positive and finite in {validation_path}")

    mole_fractions: dict[str, np.ndarray] = {}
    missing_molef: list[str] = []
    for name in species:
        column = f"molef_{name}"
        if column in condition:
            mole_fractions[name] = np.asarray(condition[column], dtype=float)
        else:
            missing_molef.append(column)
    total = np.sum(np.column_stack([concentrations[name] for name in species]), axis=1)
    molef_sum = (
        np.sum(np.column_stack([mole_fractions[name] for name in species]), axis=1)
        if not missing_molef
        else np.full(rate.shape, np.nan)
    )
    species_total_estimates = (
        np.column_stack(
            [concentrations[name] / np.maximum(mole_fractions[name], _EPS) for name in species]
        )
        if not missing_molef
        else np.empty((rate.size, 0), dtype=float)
    )
    relative_consistency = (
        np.abs(species_total_estimates - total[:, None]) / np.maximum(total[:, None], _EPS)
        if species_total_estimates.size
        else np.empty((rate.size, 0), dtype=float)
    )
    density = (
        np.asarray(condition["density"], dtype=float)
        if "density" in condition
        else np.full(rate.shape, np.nan)
    )
    nonfinite_count = int(
        sum(
            values.size - np.count_nonzero(np.isfinite(values))
            for values in [*condition.values(), *validation.values()]
        )
    )
    precision = {
        "rate_unique_count": int(np.unique(rate).size),
        "rate_min_positive_step_nm_s": _positive_min_step(rate),
        "species": {
            name: {
                "unique_count": int(np.unique(concentrations[name]).size),
                "min_positive_step_kmol_m3": _positive_min_step(concentrations[name]),
                "relative_range_vs_median": float(
                    np.ptp(concentrations[name]) / max(float(np.median(concentrations[name])), _EPS)
                ),
            }
            for name in species
        },
    }
    quality = {
        "case_id": int(case_id),
        "condition_columns": condition_headers,
        "validation_columns": validation_headers,
        "row_count": int(rate.size),
        "condition_column_count": len(condition_headers),
        "validation_column_count": len(validation_headers),
        "coordinate_alignment": alignment,
        "nonfinite_value_count": nonfinite_count,
        "rate_min_nm_s": float(np.min(rate)),
        "rate_median_nm_s": float(np.median(rate)),
        "rate_mean_nm_s": float(np.mean(rate)),
        "rate_max_nm_s": float(np.max(rate)),
        "total_concentration_min_kmol_m3": float(np.min(total)),
        "total_concentration_median_kmol_m3": float(np.median(total)),
        "total_concentration_max_kmol_m3": float(np.max(total)),
        "density_min_kg_m3": float(np.nanmin(density)),
        "density_median_kg_m3": float(np.nanmedian(density)),
        "density_max_kg_m3": float(np.nanmax(density)),
        "missing_mole_fraction_columns": missing_molef,
        "mole_fraction_sum_min": float(np.nanmin(molef_sum)),
        "mole_fraction_sum_max": float(np.nanmax(molef_sum)),
        "mole_fraction_sum_max_abs_error_from_one": float(np.nanmax(np.abs(molef_sum - 1.0))),
        "concentration_mole_fraction_max_relative_inconsistency": (
            float(np.max(relative_consistency)) if relative_consistency.size else None
        ),
        "derived_mixture_molar_mass_min_kg_kmol": float(np.nanmin(density / total)),
        "derived_mixture_molar_mass_max_kg_kmol": float(np.nanmax(density / total)),
        "precision": precision,
    }
    return ConditionCase(
        case_id=int(case_id),
        condition_path=Path(condition_path),
        validation_path=Path(validation_path),
        xyz=condition_xyz,
        species=species,
        concentrations=concentrations,
        mole_fractions=mole_fractions,
        density=density,
        total_concentration=total,
        rate=np.asarray(rate, dtype=float),
        quality=quality,
    )


def _grid_alignment(cases: Iterable[ConditionCase], decimals: int = 6) -> dict[str, Any]:
    case_list = list(cases)
    base = case_list[0]
    base_keys = [tuple(row) for row in np.round(base.xyz, decimals=decimals)]
    result: dict[str, Any] = {"reference_case": base.case_id, "rounding_decimals": decimals, "pairs": []}
    for other in case_list[1:]:
        other_keys = [tuple(row) for row in np.round(other.xyz, decimals=decimals)]
        same_set = set(base_keys) == set(other_keys)
        lookup = {key: index for index, key in enumerate(other_keys)}
        if not same_set:
            raise ValueError(
                f"Spatial grids differ beyond {decimals} decimals: condition {base.case_id} vs {other.case_id}"
            )
        order = np.asarray([lookup[key] for key in base_keys], dtype=int)
        max_difference = float(np.max(np.abs(base.xyz - other.xyz[order])))
        result["pairs"].append(
            {
                "case_1": base.case_id,
                "case_2": other.case_id,
                "same_grid_after_rounding": True,
                "max_abs_coordinate_difference": max_difference,
            }
        )
    return result


def _combine_cases(cases: Iterable[ConditionCase]) -> CombinedCases:
    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one condition is required")
    species = case_list[0].species
    if any(case.species != species for case in case_list[1:]):
        raise ValueError("All conditions must contain the same concentration species in the same order")
    concentrations = {
        name: np.concatenate([case.concentrations[name] for case in case_list]) for name in species
    }
    condition_id = np.concatenate(
        [np.full(case.rate.size, case.case_id, dtype=int) for case in case_list]
    )
    total_concentration = np.concatenate([case.total_concentration for case in case_list])
    species_fractions = {
        name: concentrations[name] / np.maximum(total_concentration, _EPS) for name in species
    }
    return CombinedCases(
        case_ids=tuple(case.case_id for case in case_list),
        xyz=np.vstack([case.xyz for case in case_list]),
        condition_id=condition_id,
        species=species,
        concentrations=concentrations,
        species_fractions=species_fractions,
        total_concentration=total_concentration,
        rate=np.concatenate([case.rate for case in case_list]),
    )


def _effect_names(candidate: RoleResponseCandidate) -> tuple[str, ...]:
    names = ["common_total_order"]
    if candidate.A is not None and candidate.B is None:
        names.append(f"A:{candidate.A}")
    elif candidate.A is not None and candidate.B is not None:
        names.append(f"AB:{candidate.A}*{candidate.B}")
    if candidate.I is not None:
        names.append(f"I:{candidate.I}")
    return tuple(names)


def _transfer_design(
    candidate: RoleResponseCandidate,
    data: CombinedCases,
    indices: np.ndarray,
    *,
    reference_total_concentration: float,
    reference_species_fractions: dict[str, float],
    response_structure: str = "shared",
) -> np.ndarray:
    # Compute input-only map means on the full supplied Fluent map. Prediction
    # batches, spatial holdouts and bootstrap resampling must not redefine them.
    idx = np.arange(data.rate.size) if response_structure == "within_between" else np.asarray(indices, dtype=int)
    columns: list[np.ndarray] = [
        np.ones(idx.size, dtype=float),
        np.log(data.total_concentration[idx] / reference_total_concentration),
    ]
    if candidate.A is not None and candidate.B is None:
        columns.append(
            np.log(
                data.species_fractions[candidate.A][idx]
                / reference_species_fractions[candidate.A]
            )
        )
    elif candidate.A is not None and candidate.B is not None:
        columns.append(
            np.log(
                (data.species_fractions[candidate.A][idx] / reference_species_fractions[candidate.A])
                * (data.species_fractions[candidate.B][idx] / reference_species_fractions[candidate.B])
            )
        )
    if candidate.I is not None:
        columns.append(
            -np.log(
                data.species_fractions[candidate.I][idx]
                / reference_species_fractions[candidate.I]
            )
        )
    design = np.column_stack(columns)
    if response_structure == "shared":
        return design
    if response_structure != "within_between":
        raise ValueError(f"Unknown response structure: {response_structure}")
    features = design[:, 1:]
    means = np.empty_like(features)
    for label in np.unique(data.condition_id):
        mask = data.condition_id == label
        means[mask] = features[mask].mean(axis=0)
    return np.column_stack([design[:, 0], means, features - means])[indices]


def _transfer_penalty(candidate: RoleResponseCandidate, response_structure: str) -> np.ndarray:
    count = len(_effect_names(candidate))
    if response_structure == "shared":
        return np.eye(count + 1)[1:]
    # lambda * ((||between||² + ||within||²)/2 + ||between-within||²).
    # On the shared subspace this is exactly the original ridge penalty.
    eye = np.eye(count)
    effects = np.vstack([np.eye(2 * count) / np.sqrt(2), np.column_stack([eye, -eye])])
    return np.column_stack([np.zeros(effects.shape[0]), effects])


def _condition_mean_rate(data: CombinedCases, indices: np.ndarray | None = None) -> float:
    indices = np.arange(data.rate.size) if indices is None else indices
    labels, rates = data.condition_id[indices], data.rate[indices]
    return float(np.mean([np.mean(rates[labels == label]) for label in np.unique(labels)]))


def _fit_transfer(
    candidate: RoleResponseCandidate,
    data: CombinedCases,
    train_indices: np.ndarray,
    predict_indices: np.ndarray | None = None,
    *,
    reference_total_concentration: float | None = None,
    reference_species_fractions: dict[str, float] | None = None,
    regularization: float = 0.0,
    response_structure: str = "shared",
) -> TransferFit:
    train_idx = np.asarray(train_indices, dtype=int)
    pred_idx = train_idx if predict_indices is None else np.asarray(predict_indices, dtype=int)
    reference = (
        float(np.median(data.total_concentration[train_idx]))
        if reference_total_concentration is None
        else float(reference_total_concentration)
    )
    if reference <= 0.0:
        raise ValueError("Positive total-concentration reference is required")
    fraction_references = (
        {
            name: float(np.median(data.species_fractions[name][train_idx]))
            for name in data.species
        }
        if reference_species_fractions is None
        else {name: float(reference_species_fractions[name]) for name in data.species}
    )
    if any(value <= 0.0 for value in fraction_references.values()):
        raise ValueError("Positive species-fraction references are required")
    design_train = _transfer_design(
        candidate,
        data,
        train_idx,
        reference_total_concentration=reference,
        reference_species_fractions=fraction_references,
        response_structure=response_structure,
    )
    labels, counts = np.unique(data.condition_id[train_idx], return_counts=True)
    weights = np.asarray([1.0 / counts[np.searchsorted(labels, label)]
                          for label in data.condition_id[train_idx]])
    coefficients, active = _fit_nonnegative_effects(
        design_train, np.log(data.rate[train_idx]),
        weights=weights, regularization=regularization,
        penalty_matrix=_transfer_penalty(candidate, response_structure),
    )
    design_prediction = _transfer_design(
        candidate,
        data,
        pred_idx,
        reference_total_concentration=reference,
        reference_species_fractions=fraction_references,
        response_structure=response_structure,
    )
    return TransferFit(
        candidate=candidate,
        coefficients=coefficients,
        prediction=np.exp(design_prediction @ coefficients),
        design=design_prediction,
        effect_names=(_effect_names(candidate) if response_structure == "shared" else
                      tuple(f"{scope}:{term}" for scope in ("between", "within") for term in _effect_names(candidate))),
        reference_total_concentration=reference,
        reference_species_fractions=fraction_references,
        active_effects=active,
        regularization=regularization,
        response_structure=response_structure,
    )


def _predict_transfer(fit: TransferFit, data: CombinedCases) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(data.rate.size, dtype=int)
    design = _transfer_design(
        fit.candidate,
        data,
        indices,
        reference_total_concentration=fit.reference_total_concentration,
        reference_species_fractions=fit.reference_species_fractions,
        response_structure=fit.response_structure,
    )
    return np.exp(design @ fit.coefficients), design


def _predict_response(
    fit: TransferFit | SurfaceKineticFit, data: CombinedCases,
) -> tuple[np.ndarray, np.ndarray | dict[str, np.ndarray]]:
    if isinstance(fit, SurfaceKineticFit):
        return predict_surface_kinetic(fit, data)
    return _predict_transfer(fit, data)


def _blocked_predictions(
    candidate: RoleResponseCandidate,
    data: CombinedCases,
    groups: np.ndarray,
    *,
    reference_total_concentration: float,
    reference_species_fractions: dict[str, float],
    regularization: float = 0.0,
    response_structure: str = "shared",
) -> np.ndarray:
    prediction = np.full(data.rate.shape, np.nan, dtype=float)
    all_indices = np.arange(data.rate.size, dtype=int)
    for group in np.unique(groups):
        validation_idx = all_indices[groups == group]
        train_idx = all_indices[groups != group]
        fit = _fit_transfer(
            candidate,
            data,
            train_idx,
            validation_idx,
            reference_total_concentration=reference_total_concentration,
            reference_species_fractions=reference_species_fractions,
            regularization=regularization,
            response_structure=response_structure,
        )
        prediction[validation_idx] = fit.prediction
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Blocked CV failed for {candidate.model_id}")
    return prediction


def _condition_holdout_predictions(
    candidate: RoleResponseCandidate, data: CombinedCases, *, regularization: float = 0.0,
    response_structure: str = "shared",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Evaluate a fixed structure/strength on identical condition folds."""
    prediction = np.full(data.rate.shape, np.nan, dtype=float)
    all_indices = np.arange(data.rate.size, dtype=int)

    def fit(remaining):
        idx = all_indices[np.isin(data.condition_id, remaining)]
        return _fit_transfer(candidate, data, idx, all_indices, regularization=regularization,
                             response_structure=response_structure)

    def evaluate(fitted, held_out, remaining):
        valid = data.condition_id == held_out
        train_idx = all_indices[np.isin(data.condition_id, remaining)]
        prediction[valid] = fitted.prediction[valid]
        metrics = _transfer_metrics(data.rate[valid], fitted.prediction[valid],
                                    _condition_mean_rate(data, train_idx))
        return {**metrics, "condition": int(held_out), "weight": 1.0,
                "quantity": "deposition_rate", "unit": "nm/s",
                "refit_score": float(np.mean((fitted.prediction[train_idx] - data.rate[train_idx])**2)),
                "effect_groups": fitted.effect_groups, "effective_roles": fitted.effective_roles,
                "effect_scopes": fitted.effect_scopes, "response_structure": response_structure,
                "regularization": regularization, "common_total_order": fitted.common_order,
                "within_total_order": fitted.within_order,
                "max_effect_coefficient": float(np.max(fitted.coefficients[1:]))}

    folds = condition_refits(data.case_ids, fit, evaluate)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Condition-holdout CV failed for {candidate.model_id}")
    return prediction, folds


def _eligibility_reasons(candidate: RoleResponseCandidate, design: np.ndarray) -> list[str]:
    """Reject unresolved design directions, independent of species names/units."""
    if candidate.model_id == "baseline":
        return []
    scale = np.linalg.norm(design, axis=0)
    if np.any(scale == 0):
        return ["a role effect has no independent variation in the training inputs"]
    singular = np.linalg.svd(design / scale, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * np.sqrt(np.finfo(float).eps)))
    return [] if rank == design.shape[1] else [f"only {rank} of {design.shape[1]} model directions are resolved"]


def _ranking_for_training(
    train_cases: list[ConditionCase],
    *, response_structure: str = "shared",
) -> tuple[list[dict[str, Any]], dict[str, TransferFit], CombinedCases, np.ndarray, np.ndarray]:
    if response_structure not in (*RESPONSE_STRUCTURES, "select"):
        raise ValueError("response_structure must be shared, within_between or select")
    structures = RESPONSE_STRUCTURES if response_structure == "select" else (response_structure,)
    data = _combine_cases(train_cases)
    candidates = enumerate_role_response_candidates(data.species, include_reductions=True)
    indices = np.arange(data.rate.size, dtype=int)
    reference = float(np.median(data.total_concentration))
    fraction_references = {
        name: float(np.median(data.species_fractions[name])) for name in data.species
    }
    angular_groups = _angular_groups(data.xyz[:, :2])
    radial_groups = _radial_groups(data.xyz[:, :2])
    fits: dict[str, TransferFit] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        # Joint role/strength selection: these are inner CV scores, not an
        # unbiased evaluation of the selected model. _split_evaluation supplies
        # the untouched outer condition for that purpose.
        strengths = []
        for structure in structures:
            for strength in REGULARIZATION_GRID:
                try:
                    # An unresolved, unregularized direction can overflow on a
                    # withheld condition. Reject that setting, without clipping
                    # its predictions or aborting other valid fits.
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        prediction, folds = _condition_holdout_predictions(candidate, data, regularization=strength,
                                                                           response_structure=structure)
                        risk = float(np.mean([fold["mse"] for fold in folds]))
                    strengths.append((risk, strength, prediction, folds, structure, ""))
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    strengths.append((float("inf"), strength, None, [], structure, str(error)))
        if not any(np.isfinite(item[0]) for item in strengths):
            raise RuntimeError(f"No numerically valid condition-CV fit for {candidate.model_id}")
        best_risk = min(item[0] for item in strengths)
        scale = float(np.mean(data.rate**2))
        floor = 32 * np.finfo(float).eps * abs(best_risk) + (64 * np.finfo(float).eps)**2 * scale
        tied = [item for item in strengths if item[0] <= best_risk + floor]
        _, strength, condition_prediction, condition_rows, structure, _ = min(
            tied, key=lambda item: (item[4] != "shared", -item[1], item[0]))
        fit = _fit_transfer(
            candidate,
            data,
            indices,
            reference_total_concentration=reference,
            reference_species_fractions=fraction_references,
            regularization=strength,
            response_structure=structure,
        )
        fits[candidate.model_id] = fit
        angular_prediction = _blocked_predictions(
            candidate,
            data,
            angular_groups,
            reference_total_concentration=reference,
            reference_species_fractions=fraction_references,
            regularization=strength,
            response_structure=structure,
        )
        radial_prediction = _blocked_predictions(
            candidate,
            data,
            radial_groups,
            reference_total_concentration=reference,
            reference_species_fractions=fraction_references,
            regularization=strength,
            response_structure=structure,
        )
        in_sample = _metrics(data.rate, fit.prediction)
        angular = _metrics(data.rate, angular_prediction)
        radial = _metrics(data.rate, radial_prediction)
        condition = _metrics(data.rate, condition_prediction)
        condition_number = float(np.linalg.cond(fit.design))
        reasons = _eligibility_reasons(candidate, fit.design)
        rows.append(
            {
                "model_id": "common_mode_power" if candidate.model_id == "baseline" else f"common_mode+{candidate.model_id}",
                "role_model_id": candidate.model_id,
                "roles": {"A": candidate.A, "I": candidate.I, "B": candidate.B},
                "quantity": "deposition_rate", "unit": "nm/s",
                "effect_groups": fit.effect_groups,
                "effect_scopes": fit.effect_scopes, "response_structure": structure,
                "declared_effect_groups": candidate.effect_groups,
                "reduced_effect_groups": [r.effect_groups for r in candidate.reductions()],
                "effect_basis": "fitted_nonzero_terms",
                "best_score": float(np.mean([
                    np.mean((np.log(data.rate[data.condition_id == label]) -
                             np.log(fit.prediction[data.condition_id == label]))**2)
                    for label in data.case_ids
                ]) + strength * np.sum((_transfer_penalty(candidate, structure) @ fit.coefficients)**2)),
                "search_space_count": fit.design.shape[1],
                "validation_conditions": condition_rows,
                "class_id": candidate.class_id,
                "role_A": candidate.A or "",
                "role_I": candidate.I or "",
                "role_B": candidate.B or "",
                "role_effect_count": len(candidate.effect_groups),
                "active_effect_count": int(sum(fit.active_effects)),
                "effective_roles": fit.effective_roles,
                "inactive_roles": ";".join(slot for slot in ("A", "I", "B")
                                           if getattr(candidate, slot) is not None and fit.effective_roles[slot] is None),
                "role_symmetry": "A/B exchange" if "AB" in fit.effect_groups else "",
                "fit_diagnostics": {"identifiability": {"assessed": True, "degeneracy_warning": bool(reasons)}},
                "regularization": strength,
                "regularization_scores": [{"strength": item[1], "response_structure": item[4],
                                           "condition_mse": item[0], "numerical_failure": item[5]} for item in strengths],
                "common_total_order": fit.common_order,
                "within_total_order": fit.within_order,
                "reference_rate_nm_s": fit.reference_rate_nm_s,
                "in_sample_rmse_nm_s": in_sample["rmse_nm_s"],
                "in_sample_r2": in_sample["r2"],
                "angular_cv_rmse_nm_s": angular["rmse_nm_s"],
                "angular_cv_r2": angular["r2"],
                "radial_cv_rmse_nm_s": radial["rmse_nm_s"],
                "radial_cv_r2": radial["r2"],
                "blocked_cv_rmse_nm_s": max(angular["rmse_nm_s"], radial["rmse_nm_s"]),
                "condition_cv_rmse_nm_s": float(np.sqrt(np.mean([fold["mse"] for fold in condition_rows]))),
                "condition_cv_r2": condition["r2"],
                "condition_cv_worst_relative_rmse": max(
                    float(row["relative_rmse"]) for row in condition_rows
                ),
                "condition_cv_worst_case": int(
                    max(condition_rows, key=lambda row: float(row["relative_rmse"]))["condition"]
                ),
                "design_condition_number": condition_number,
                "eligible_for_adoption": bool(np.isfinite(condition_prediction).all()),
                "ineligibility_reasons": "",
                "design_identifiable": not reasons,
                "design_information": "; ".join(reasons),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["condition_cv_rmse_nm_s"]),
            int(row["role_effect_count"]),
            str(row["model_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["numerical_rank"] = rank
    eligible = rank_role_candidates([row for row in rows if bool(row["eligible_for_adoption"])])
    for rank, row in enumerate(eligible, start=1):
        row["adoption_rank"] = rank
    by_id = {row["model_id"]: row for row in eligible}
    rows = [by_id.get(row["model_id"], row) for row in rows]
    for row in rows:
        row.setdefault("adoption_rank", "")
    baseline_condition_cv = float(
        next(row for row in rows if row["role_model_id"] == "baseline")[
            "condition_cv_rmse_nm_s"
        ]
    )
    for row in rows:
        row["condition_cv_improvement_vs_baseline"] = float(
            (baseline_condition_cv - float(row["condition_cv_rmse_nm_s"]))
            / max(baseline_condition_cv, _EPS)
        )
    return rows, fits, data, angular_groups, radial_groups


def _surface_condition_holdout_predictions(
    candidate: SurfaceKineticCandidate,
    data: CombinedCases,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Condition CV for a fixed physical reduction with fold-local references."""
    prediction = np.full(data.rate.shape, np.nan, dtype=float)
    all_indices = np.arange(data.rate.size, dtype=int)
    rows: list[dict[str, Any]] = []
    for held_out in data.case_ids:
        train_idx = all_indices[data.condition_id != held_out]
        valid_idx = all_indices[data.condition_id == held_out]
        fitted = fit_surface_kinetic(candidate, data, train_idx)
        prediction[valid_idx] = fitted.prediction[valid_idx]
        metrics = _transfer_metrics(
            data.rate[valid_idx], fitted.prediction[valid_idx],
            _condition_mean_rate(data, train_idx),
        )
        rows.append({
            **metrics,
            "condition": int(held_out),
            "weight": 1.0,
            "quantity": "deposition_rate",
            "unit": "nm/s",
            "refit_score": fitted.objective_mse,
            "effect_groups": fitted.effect_groups,
            "effective_roles": fitted.effective_roles,
            "effect_scopes": fitted.effect_scopes,
            "response_structure": fitted.response_structure,
            "regularization": 0.0,
            "common_total_order": float("nan"),
            "within_total_order": float("nan"),
            "max_effect_coefficient": max(fitted.shape_parameters.values(), default=0.0),
            "boundary_parameters": list(fitted.boundary_parameters),
        })
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Condition-holdout CV failed for {candidate.model_id}")
    return prediction, rows


def _surface_blocked_predictions(
    candidate: SurfaceKineticCandidate,
    data: CombinedCases,
    groups: np.ndarray,
    reference_concentrations: dict[str, float],
) -> np.ndarray:
    prediction = np.full(data.rate.shape, np.nan, dtype=float)
    all_indices = np.arange(data.rate.size, dtype=int)
    for held_out in np.unique(groups):
        train_idx = all_indices[groups != held_out]
        valid_idx = all_indices[groups == held_out]
        fitted = fit_surface_kinetic(
            candidate,
            data,
            train_idx,
            reference_concentrations=reference_concentrations,
        )
        prediction[valid_idx] = fitted.prediction[valid_idx]
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Blocked CV failed for {candidate.model_id}")
    return prediction


def _surface_ranking_for_training(
    train_cases: list[ConditionCase],
) -> tuple[list[dict[str, Any]], dict[str, SurfaceKineticFit], CombinedCases, np.ndarray, np.ndarray]:
    """Rank site-balance reductions using raw-rate condition prediction."""
    data = _combine_cases(train_cases)
    indices = np.arange(data.rate.size, dtype=int)
    candidates = enumerate_surface_kinetic_candidates(data.species, include_boundaries=True)
    angular_groups = _angular_groups(data.xyz[:, :2])
    radial_groups = _radial_groups(data.xyz[:, :2])
    fits: dict[str, SurfaceKineticFit] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        condition_prediction, condition_rows = _surface_condition_holdout_predictions(candidate, data)
        fit = fit_surface_kinetic(candidate, data, indices)
        fits[candidate.model_id] = fit
        in_sample = _metrics(data.rate, fit.prediction)
        condition = _metrics(data.rate, condition_prediction)
        reasons = _eligibility_reasons(candidate, fit.design)
        if fit.boundary_parameters:
            reasons.append(
                "shape optimum reached numerical search boundary: "
                + ", ".join(fit.boundary_parameters)
            )
        rows.append({
            "model_id": candidate.model_id,
            "role_model_id": candidate.model_id,
            "model_family": "surface_qss",
            "kinetic_limit": candidate.kinetic_limit,
            "roles": {"A": candidate.A, "I": candidate.I, "B": candidate.B},
            "quantity": "deposition_rate",
            "unit": "nm/s",
            "effect_groups": fit.effect_groups,
            "effect_scopes": fit.effect_scopes,
            "response_structure": fit.response_structure,
            "declared_effect_groups": candidate.effect_groups,
            "reduced_effect_groups": [
                reduction.effect_groups
                for reduction in candidate.reductions()
                if reduction.effect_groups != candidate.effect_groups
            ],
            "reduced_model_ids": [reduction.model_id for reduction in candidate.reductions()],
            "effect_basis": "declared_state_model_roles",
            "best_score": fit.objective_mse,
            "search_space_count": fit.design.shape[1],
            "validation_conditions": condition_rows,
            "class_id": candidate.class_id,
            "role_A": candidate.A or "",
            "role_I": candidate.I or "",
            "role_B": candidate.B or "",
            "role_effect_count": len(candidate.effect_groups),
            "active_effect_count": len(candidate.effect_groups),
            "effective_roles": fit.effective_roles,
            "inactive_roles": "",
            "role_symmetry": candidate.role_symmetry,
            "fit_diagnostics": {
                "identifiability": {
                    "assessed": True,
                    "degeneracy_warning": bool(reasons),
                    "boundary_parameters": list(fit.boundary_parameters),
                }
            },
            "regularization": 0.0,
            "regularization_scores": [],
            "common_total_order": float("nan"),
            "within_total_order": float("nan"),
            "reference_rate_nm_s": fit.reference_rate_nm_s,
            "reference_concentrations_kmol_m3": fit.reference_concentrations,
            "observable_parameters": fit.shape_parameters,
            "formula": surface_formula(candidate),
            "in_sample_rmse_nm_s": in_sample["rmse_nm_s"],
            "in_sample_r2": in_sample["r2"],
            "angular_cv_rmse_nm_s": float("nan"),
            "angular_cv_r2": float("nan"),
            "radial_cv_rmse_nm_s": float("nan"),
            "radial_cv_r2": float("nan"),
            "blocked_cv_rmse_nm_s": float("nan"),
            "condition_cv_rmse_nm_s": float(np.sqrt(np.mean([fold["mse"] for fold in condition_rows]))),
            "condition_cv_r2": condition["r2"],
            "condition_cv_worst_relative_rmse": max(float(row["relative_rmse"]) for row in condition_rows),
            "condition_cv_worst_case": int(max(condition_rows, key=lambda row: float(row["relative_rmse"]))["condition"]),
            "design_condition_number": float(np.linalg.cond(fit.design)),
            "eligible_for_adoption": bool(np.isfinite(condition_prediction).all()),
            "ineligibility_reasons": "",
            "design_identifiable": not reasons,
            "design_information": "; ".join(reasons),
        })
    rows.sort(key=lambda row: (
        float(row["condition_cv_rmse_nm_s"]),
        int(row["role_effect_count"]),
        int(row["search_space_count"]),
        str(row["model_id"]),
    ))
    for rank, row in enumerate(rows, start=1):
        row["numerical_rank"] = rank
    eligible = rank_role_candidates([row for row in rows if bool(row["eligible_for_adoption"])])
    for rank, row in enumerate(eligible, start=1):
        row["adoption_rank"] = rank
    by_id = {row["model_id"]: row for row in eligible}
    rows = [by_id.get(row["model_id"], row) for row in rows]
    for row in rows:
        row.setdefault("adoption_rank", "")

    selected = next(row for row in rows if row.get("adoption_rank") == 1)
    selected_fit = fits[str(selected["role_model_id"])]
    angular_prediction = _surface_blocked_predictions(
        selected_fit.candidate, data, angular_groups, selected_fit.reference_concentrations
    )
    radial_prediction = _surface_blocked_predictions(
        selected_fit.candidate, data, radial_groups, selected_fit.reference_concentrations
    )
    angular = _metrics(data.rate, angular_prediction)
    radial = _metrics(data.rate, radial_prediction)
    selected.update({
        "angular_cv_rmse_nm_s": angular["rmse_nm_s"],
        "angular_cv_r2": angular["r2"],
        "radial_cv_rmse_nm_s": radial["rmse_nm_s"],
        "radial_cv_r2": radial["r2"],
        "blocked_cv_rmse_nm_s": max(angular["rmse_nm_s"], radial["rmse_nm_s"]),
    })

    baseline_cv = float(next(row for row in rows if row["class_id"] == "baseline")["condition_cv_rmse_nm_s"])
    for row in rows:
        row["condition_cv_improvement_vs_baseline"] = (
            baseline_cv - float(row["condition_cv_rmse_nm_s"])
        ) / max(baseline_cv, _EPS)
    return rows, fits, data, angular_groups, radial_groups


def _select_model(ranking: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    selected = next(row for row in ranking if row.get("adoption_rank") == 1)
    equivalent = [row["role_model_id"] for row in ranking if row.get("equivalent_to_best", False)]
    if selected.get("model_family") == "surface_qss":
        reason = ("Minimum condition-refit raw-rate prediction error among quasi-steady site-balance "
                  "reductions; numerical ties prefer fewer observable effects and parameters.")
    else:
        reason = "Minimum condition-refit prediction error, jointly selecting role, response structure and regularization; numerical ties prefer shared response and fewer active effects."
    if len(equivalent) > 1:
        reason += " Numerical loss ties: " + ", ".join(equivalent)
    return selected, ranking[0], reason


def _transfer_metrics(target: np.ndarray, prediction: np.ndarray, train_mean: float) -> dict[str, Any]:
    """Rate adapter over the same metrics used by the thickness fitter."""
    metrics = prediction_metrics(target, prediction, baseline=train_mean)
    constant_rmse = float(np.sqrt(metrics["baseline_mse"]))
    target_range = float(np.ptp(target))
    prediction_range = float(np.ptp(prediction))
    return {
        **metrics, **_metrics(target, prediction),
        "relative_rmse_vs_test_mean": metrics["relative_rmse"],
        "relative_bias_vs_test_mean": float(metrics["mean_bias"] / max(abs(np.mean(target)), _EPS)),
        "centered_spatial_rmse_nm_s": metrics["centered_rmse"],
        "centered_spatial_r2": metrics["centered_r2"],
        "constant_train_mean_rmse_nm_s": constant_rmse,
        "rmse_improvement_vs_constant_train_mean": float((constant_rmse - metrics["rmse"]) / max(constant_rmse, _EPS)),
        "target_range_nm_s": target_range, "prediction_range_nm_s": prediction_range,
        "range_capture_fraction": float(prediction_range / max(target_range, _EPS)),
    }


def _extrapolation_summary(train: CombinedCases, test: CombinedCases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = {"total_concentration": (train.total_concentration, test.total_concentration)}
    fields.update(
        {
            f"concentration_{name}": (train.concentrations[name], test.concentrations[name])
            for name in train.species
        }
    )
    for field, (train_values, test_values) in fields.items():
        train_min = float(np.min(train_values))
        train_max = float(np.max(train_values))
        outside = (test_values < train_min) | (test_values > train_max)
        rows.append(
            {
                "field": field,
                "train_min": train_min,
                "train_max": train_max,
                "test_min": float(np.min(test_values)),
                "test_max": float(np.max(test_values)),
                "test_outside_train_range_fraction": float(np.mean(outside)),
            }
        )
    return rows


def _bootstrap_coefficients(
    fit: TransferFit | SurfaceKineticFit,
    data: CombinedCases,
    angular_groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(angular_groups)
    rows: list[np.ndarray] = []
    for _ in range(samples):
        selected = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(angular_groups == group) for group in selected])
        if isinstance(fit, SurfaceKineticFit):
            sample_fit = fit_surface_kinetic(
                fit.candidate,
                data,
                indices,
                reference_concentrations=fit.reference_concentrations,
                initial_fit=fit,
                local_only=True,
            )
        else:
            sample_fit = _fit_transfer(
                fit.candidate,
                data,
                indices,
                reference_total_concentration=fit.reference_total_concentration,
                reference_species_fractions=fit.reference_species_fractions,
                regularization=fit.regularization,
                response_structure=fit.response_structure,
            )
        rows.append(sample_fit.coefficients)
    return np.vstack(rows)


def _coefficient_rows(
    fit: TransferFit | SurfaceKineticFit, bootstrap: np.ndarray,
) -> list[dict[str, Any]]:
    if isinstance(fit, SurfaceKineticFit):
        rows = []
        for index, name in enumerate(fit.parameter_names):
            samples = bootstrap[:, index]
            role = "rate_scale" if name == "rate_scale_nm_s" else "observable_shape"
            rows.append({
                "model_id": fit.candidate.model_id,
                "term": name,
                "response_scope": "observable_reduction",
                "response_structure": fit.response_structure,
                "role": role,
                "value": float(fit.coefficients[index]),
                "unit": "nm/s" if name == "rate_scale_nm_s" else "dimensionless",
                "bootstrap_p05": float(np.quantile(samples, 0.05)),
                "bootstrap_p50": float(np.quantile(samples, 0.50)),
                "bootstrap_p95": float(np.quantile(samples, 0.95)),
                "bootstrap_zero_fraction": float(np.mean(samples <= 1.0e-14)),
                "kinetic_limit": fit.candidate.kinetic_limit,
                "elementary_constant": False,
            })
        return rows
    rows: list[dict[str, Any]] = []
    terms = [("reference", "log_reference_rate"), *fit.coefficient_terms]
    role_labels = ["common_condition_scale", *fit.candidate.effect_groups]
    for index, (scope, term) in enumerate(terms):
        samples = bootstrap[:, index]
        if term == "log_reference_rate":
            value = fit.reference_rate_nm_s
            transformed = np.exp(samples)
            role = "baseline"
            unit = "nm/s"
            reported_term = "reference_rate"
        elif term == "common_total_order":
            value = float(fit.coefficients[index])
            transformed = samples
            role = "common_condition_scale"
            unit = "dimensionless power order"
            reported_term = term
        else:
            value = float(fit.coefficients[index])
            transformed = samples
            role = role_labels[(index - 1) % len(role_labels)]
            unit = "dimensionless species-fraction elasticity"
            reported_term = term
        rows.append(
            {
                "model_id": "common_mode_power" if fit.candidate.model_id == "baseline" else f"common_mode+{fit.candidate.model_id}",
                "term": f"{scope}:{reported_term}" if scope in {"between", "within"} else reported_term,
                "response_scope": scope, "response_structure": fit.response_structure,
                "role": role,
                "value": value,
                "unit": unit,
                "bootstrap_p05": float(np.quantile(transformed, 0.05)),
                "bootstrap_p50": float(np.quantile(transformed, 0.50)),
                "bootstrap_p95": float(np.quantile(transformed, 0.95)),
                "bootstrap_zero_fraction": (
                    0.0 if term == "log_reference_rate" else float(np.mean(samples <= 1.0e-14))
                ),
            }
        )
    return rows


def _alternative_predictors(
    train: CombinedCases,
    test: CombinedCases,
) -> list[dict[str, Any]]:
    predictors: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("total_concentration", train.total_concentration, test.total_concentration)
    ]
    predictors.extend(
        (name, train.concentrations[name], test.concentrations[name]) for name in train.species
    )
    rows: list[dict[str, Any]] = []
    for name, train_values, test_values in predictors:
        reference = float(np.median(train_values))
        train_design = np.column_stack(
            [np.ones(train.rate.size, dtype=float), np.log(train_values / reference)]
        )
        coefficients, _ = _fit_nonnegative_effects(train_design, np.log(train.rate))
        train_prediction = np.exp(train_design @ coefficients)
        test_design = np.column_stack(
            [np.ones(test.rate.size, dtype=float), np.log(test_values / reference)]
        )
        test_prediction = np.exp(test_design @ coefficients)
        test_metrics = _transfer_metrics(test.rate, test_prediction, _condition_mean_rate(train))
        rows.append(
            {
                "predictor": name,
                "reference_concentration_kmol_m3": reference,
                "power_order": float(coefficients[1]),
                "train_rmse_nm_s": _metrics(train.rate, train_prediction)["rmse_nm_s"],
                "test_rmse_nm_s": test_metrics["rmse_nm_s"],
                "test_relative_rmse": test_metrics["relative_rmse_vs_test_mean"],
                "test_bias_nm_s": test_metrics["bias_nm_s"],
                "test_spatial_correlation": test_metrics["spatial_correlation"],
                "test_mean_prediction_nm_s": float(np.mean(test_prediction)),
            }
        )
    rows.sort(key=lambda row: float(row["test_rmse_nm_s"]))
    for rank, row in enumerate(rows, start=1):
        row["test_rank_diagnostic_only"] = rank
    return rows


def _candidate_test_diagnostics(
    train: CombinedCases,
    test: CombinedCases,
    ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every training-selected candidate on the untouched test map.

    These rows are diagnostic only.  Their test ranks must never feed back into
    model selection, but they make clear whether a simpler role assignment
    transfers as well as the selected training candidate.
    """
    surface = any(row.get("model_family") == "surface_qss" for row in ranking)
    available = (
        enumerate_surface_kinetic_candidates(train.species, include_boundaries=True)
        if surface
        else enumerate_role_response_candidates(train.species, include_reductions=True)
    )
    candidate_lookup = {candidate.model_id: candidate for candidate in available}
    indices = np.arange(train.rate.size, dtype=int)
    rows: list[dict[str, Any]] = []
    for training_row in ranking:
        candidate = candidate_lookup[str(training_row["role_model_id"])]
        if surface:
            fit = fit_surface_kinetic(candidate, train, indices)
        else:
            fit = _fit_transfer(candidate, train, indices, regularization=training_row["regularization"],
                                response_structure=training_row["response_structure"])
        prediction, _ = _predict_response(fit, test)
        metrics = _transfer_metrics(test.rate, prediction, _condition_mean_rate(train))
        rows.append(
            {
                "model_id": training_row["model_id"],
                "training_numerical_rank": training_row["numerical_rank"],
                "training_adoption_rank": training_row["adoption_rank"],
                "training_eligible_for_adoption": training_row["eligible_for_adoption"],
                "regularization": fit.regularization,
                "response_structure": fit.response_structure,
                "test_rmse_nm_s": metrics["rmse_nm_s"],
                "test_relative_rmse": metrics["relative_rmse_vs_test_mean"],
                "test_bias_nm_s": metrics["bias_nm_s"],
                "test_centered_spatial_r2": metrics["centered_spatial_r2"],
                "test_spatial_correlation": metrics["spatial_correlation"],
                "test_prediction_range_nm_s": metrics["prediction_range_nm_s"],
                "test_range_capture_fraction": metrics["range_capture_fraction"],
                "used_for_model_selection": False,
            }
        )
    rows.sort(key=lambda row: (float(row["test_rmse_nm_s"]), str(row["model_id"])))
    for rank, row in enumerate(rows, start=1):
        row["test_rank_diagnostic_only"] = rank
    return rows


def _split_evaluation(
    train_cases: list[ConditionCase],
    test_case: ConditionCase,
    *, response_structure: str = "shared", response_model: str = "empirical_power",
) -> tuple[dict[str, Any], list[dict[str, Any]], TransferFit | SurfaceKineticFit, np.ndarray, CombinedCases, CombinedCases]:
    if response_model not in RESPONSE_MODELS:
        raise ValueError(f"response_model must be one of {RESPONSE_MODELS}")
    if response_model == "surface_qss":
        ranking, fits, train, _, _ = _surface_ranking_for_training(train_cases)
    else:
        ranking, fits, train, _, _ = _ranking_for_training(train_cases, response_structure=response_structure)
    selected_row, unrestricted_row, selection_reason = _select_model(ranking)
    selected_fit = fits[str(selected_row["role_model_id"])]
    test = _combine_cases([test_case])
    test_prediction, _ = _predict_response(selected_fit, test)
    test_metrics = _transfer_metrics(test.rate, test_prediction, _condition_mean_rate(train))
    unrestricted_fit = fits[str(unrestricted_row["role_model_id"])]
    unrestricted_prediction, _ = _predict_response(unrestricted_fit, test)
    unrestricted_metrics = _transfer_metrics(
        test.rate,
        unrestricted_prediction,
        _condition_mean_rate(train),
    )
    result = {
        "train_cases": "+".join(str(case.case_id) for case in train_cases),
        "test_case": test_case.case_id,
        "selected_model": selected_row["model_id"],
        "selected_role_model_id": selected_row["role_model_id"],
        "selected_role_A": selected_fit.candidate.A or "",
        "selected_role_I": selected_fit.candidate.I or "",
        "selected_role_B": selected_fit.candidate.B or "",
        "selected_role_terms": "|".join(selected_fit.effect_names),
        "response_model": response_model,
        "effective_roles": selected_fit.effective_roles,
        "effect_groups": selected_fit.effect_groups,
        "effect_scopes": selected_fit.effect_scopes, "response_structure": selected_fit.response_structure,
        "inactive_roles": selected_row["inactive_roles"],
        "role_symmetry": selected_row["role_symmetry"],
        "regularization": selected_fit.regularization,
        "selection_reason": selection_reason,
        "unrestricted_numerical_winner": unrestricted_row["model_id"],
        "unrestricted_winner_eligible": bool(unrestricted_row["eligible_for_adoption"]),
        "unrestricted_winner_rejection_reason": unrestricted_row["ineligibility_reasons"],
        "train_blocked_cv_rmse_nm_s": float(selected_row["blocked_cv_rmse_nm_s"]),
        "train_condition_cv_rmse_nm_s": float(selected_row["condition_cv_rmse_nm_s"]),
        "train_condition_cv_improvement_vs_baseline": float(
            selected_row["condition_cv_improvement_vs_baseline"]
        ),
        "train_condition_cv_worst_relative_rmse": float(
            selected_row["condition_cv_worst_relative_rmse"]
        ),
        "train_condition_cv_worst_case": int(selected_row["condition_cv_worst_case"]),
        "common_total_order": selected_fit.common_order,
        "within_total_order": selected_fit.within_order,
        "reference_total_concentration_kmol_m3": selected_fit.reference_total_concentration,
        "reference_rate_nm_s": selected_fit.reference_rate_nm_s,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "unrestricted_test_rmse_nm_s": unrestricted_metrics["rmse_nm_s"],
        "unrestricted_test_relative_rmse": unrestricted_metrics["relative_rmse_vs_test_mean"],
        "unrestricted_test_centered_spatial_r2": unrestricted_metrics[
            "centered_spatial_r2"
        ],
        "unrestricted_test_spatial_correlation": unrestricted_metrics[
            "spatial_correlation"
        ],
        "unrestricted_test_prediction_range_nm_s": unrestricted_metrics[
            "prediction_range_nm_s"
        ],
        "unrestricted_test_range_capture_fraction": unrestricted_metrics[
            "range_capture_fraction"
        ],
    }
    for effect_index, effect_name in enumerate(selected_fit.effect_names, start=1):
        safe_name = effect_name.replace(":", "_").replace("*", "_").replace("|", "_")
        result[f"selected_coefficient_{safe_name}"] = float(
            selected_fit.coefficients[effect_index]
        )
    return result, ranking, selected_fit, test_prediction, train, test


def _condition_quality_rows(cases: Iterable[ConditionCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        precision = case.quality["precision"]
        row: dict[str, Any] = {
            "condition": case.case_id,
            "condition_columns": "|".join(case.quality["condition_columns"]),
            "validation_columns": "|".join(case.quality["validation_columns"]),
            "rows": case.quality["row_count"],
            "rate_mean_nm_s": case.quality["rate_mean_nm_s"],
            "rate_median_nm_s": case.quality["rate_median_nm_s"],
            "rate_unique_count": precision["rate_unique_count"],
            "rate_min_step_nm_s": precision["rate_min_positive_step_nm_s"],
            "total_concentration_median_kmol_m3": case.quality[
                "total_concentration_median_kmol_m3"
            ],
            "mole_fraction_sum_max_abs_error": case.quality[
                "mole_fraction_sum_max_abs_error_from_one"
            ],
            "coordinate_exact_match_with_validation": case.quality["coordinate_alignment"][
                "coordinate_exact_match"
            ],
            "coordinate_tolerance_match_with_validation": case.quality["coordinate_alignment"][
                "coordinate_tolerance_match"
            ],
            "coordinate_max_abs_difference": case.quality["coordinate_alignment"][
                "coordinate_max_abs_difference"
            ],
        }
        for name in case.species:
            row[f"{name}_unique_count"] = precision["species"][name]["unique_count"]
            row[f"{name}_relative_range"] = precision["species"][name][
                "relative_range_vs_median"
            ]
        rows.append(row)
    return rows


def _scaling_rows(cases: list[ConditionCase]) -> list[dict[str, Any]]:
    base = cases[0]
    rows: list[dict[str, Any]] = []
    for case in cases[1:]:
        for name in base.species:
            ratios = case.concentrations[name] / base.concentrations[name]
            rows.append(
                {
                    "reference_condition": base.case_id,
                    "condition": case.case_id,
                    "field": f"concentration_{name}",
                    "median_ratio": float(np.median(ratios)),
                    "ratio_std": float(np.std(ratios)),
                }
            )
        total_ratios = case.total_concentration / base.total_concentration
        rows.append(
            {
                "reference_condition": base.case_id,
                "condition": case.case_id,
                "field": "total_concentration",
                "median_ratio": float(np.median(total_ratios)),
                "ratio_std": float(np.std(total_ratios)),
            }
        )
    return rows


def _correlation_rows(cases: Iterable[ConditionCase]) -> list[dict[str, Any]]:
    combined = _combine_cases(cases)
    matrix = np.column_stack([combined.concentrations[name] for name in combined.species])
    correlation = np.corrcoef(matrix, rowvar=False)
    rows: list[dict[str, Any]] = []
    for left in range(len(combined.species)):
        for right in range(left + 1, len(combined.species)):
            rows.append(
                {
                    "species_1": combined.species[left],
                    "species_2": combined.species[right],
                    "pooled_pearson_correlation": float(correlation[left, right]),
                }
            )
    return rows


def _plot_results(
    output_dir: Path,
    cases: list[ConditionCase],
    primary_fit: TransferFit | SurfaceKineticFit,
    primary_train: CombinedCases,
    primary_test: CombinedCases,
    primary_prediction: np.ndarray,
    ranking: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for case in cases:
        color = "#2563eb" if case.case_id in primary_train.case_ids else "#d97706"
        combined_case = _combine_cases([case])
        case_prediction, _ = _predict_response(primary_fit, combined_case)
        mean_total = float(np.mean(case.total_concentration))
        mean_measured = float(np.mean(case.rate))
        mean_prediction = float(np.mean(case_prediction))
        axis.plot(
            [mean_total, mean_total],
            [mean_measured, mean_prediction],
            color=color,
            alpha=0.45,
            linewidth=1.2,
        )
        axis.scatter(
            mean_total,
            mean_measured,
            s=80,
            marker="o",
            color=color,
            edgecolor="#111827",
            linewidth=0.6,
            label="measured identification" if case.case_id == primary_train.case_ids[0] else (
                "measured held-out" if case.case_id == primary_test.case_ids[0] else None
            ),
        )
        axis.scatter(
            mean_total,
            mean_prediction,
            s=85,
            marker="x",
            color=color,
            linewidth=1.8,
            label="model prediction" if case.case_id == cases[0].case_id else None,
        )
        axis.annotate(
            f"condition {case.case_id}",
            (mean_total, mean_measured),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.set_xlabel("Mean total concentration [kmol/m³]")
    axis.set_ylabel("Mean deposition rate [nm/s]")
    axis.set_title("Condition-mean transfer of the selected role model")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False)
    axis.grid(alpha=0.25)
    path = plot_dir / "condition_mean_transfer.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    axis.scatter(primary_test.rate, primary_prediction, s=55, color="#2563eb", edgecolor="#1f2937", linewidth=0.6)
    low = min(float(np.min(primary_test.rate)), float(np.min(primary_prediction)))
    high = max(float(np.max(primary_test.rate)), float(np.max(primary_prediction)))
    axis.plot([low, high], [low, high], linestyle="--", color="#334155", linewidth=1.5)
    axis.set_xlabel("Measured deposition rate [nm/s]")
    axis.set_ylabel("Held-out prediction [nm/s]")
    axis.set_title(
        f"Condition {primary_test.case_ids[0]}: measured versus no-refit prediction"
    )
    axis.grid(alpha=0.25)
    path = plot_dir / "test_measured_vs_predicted.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    xy = primary_test.xyz[:, :2]
    rate_low = min(float(np.min(primary_test.rate)), float(np.min(primary_prediction)))
    rate_high = max(float(np.max(primary_test.rate)), float(np.max(primary_prediction)))
    measured_plot = axes[0].scatter(xy[:, 0], xy[:, 1], c=primary_test.rate, cmap="viridis", vmin=rate_low, vmax=rate_high, s=65)
    predicted_plot = axes[1].scatter(xy[:, 0], xy[:, 1], c=primary_prediction, cmap="viridis", vmin=rate_low, vmax=rate_high, s=65)
    residual = primary_prediction - primary_test.rate
    residual_limit = max(float(np.max(np.abs(residual))), _EPS)
    residual_plot = axes[2].scatter(xy[:, 0], xy[:, 1], c=residual, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit, s=65)
    axes[0].set_title("Measured rate [nm/s]")
    axes[1].set_title("Held-out prediction [nm/s]")
    axes[2].set_title("Residual: predicted - measured [nm/s]")
    for axis_item in axes:
        axis_item.set_xlabel("x [source coordinate unit]")
        axis_item.set_ylabel("y [source coordinate unit]")
        axis_item.set_aspect("equal", adjustable="box")
    figure.colorbar(measured_plot, ax=axes[:2], shrink=0.86)
    figure.colorbar(residual_plot, ax=axes[2], shrink=0.86)
    path = plot_dir / "test_spatial_maps.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    top = ranking[:8]
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    labels = [str(row["model_id"]) for row in reversed(top)]
    values = [float(row["blocked_cv_rmse_nm_s"]) for row in reversed(top)]
    colors = ["#2563eb" if bool(row["eligible_for_adoption"]) else "#cbd5e1" for row in reversed(top)]
    axis.barh(labels, values, color=colors, edgecolor="#475569", linewidth=0.5)
    axis.set_xlabel("Conservative blocked-CV RMSE [nm/s]")
    axis.set_title("Training-only candidate ranking; grey candidates fail adoption gates")
    axis.grid(axis="x", alpha=0.25)
    path = plot_dir / "training_candidate_ranking.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)
    return outputs


def _write_notebook(
    output_dir: Path,
    train_case_ids: tuple[int, ...],
    test_case_id: int,
) -> Path:
    notebook_path = output_dir / "cvd_multicond_transfer_analysis.ipynb"
    evaluation = json.loads((output_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    validity = evaluation["validity"]
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CVD multi-condition transfer analysis\n",
                "\n",
                "## tl;dr\n",
                f"Conditions {list(train_case_ids)} identify the response model; condition {test_case_id} is held out without refitting. "
                f"Selected model: {evaluation['primary_split']['selected_model']}. "
                f"Response structure: {evaluation['primary_split']['response_structure']}. "
                f"Decision: {validity['decision']}; role support: {validity['species_role_assessment']}. "
                "Outer condition folds assess the selection procedure; each fold fits its own model.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Context & Methods\n",
                "\n",
                "Log-rate uses total-concentration and optional species-fraction role terms. "
                "Training-condition CV selects roles, shared or within/between responses, and regularization. "
                "Within/between responses use full Fluent input-map means and deviations; their coefficient differences are regularized. "
                "Term removal and alternative assignments are compared on the same conditions; crossing losses leave role support unresolved. "
                "Numerical loss ties prefer fewer effects and parameters. Angular/radial blocked CV and design rank are diagnostics.\n",
                "\n",
                "### Key Assumptions\n",
                "- Response coefficients transfer across conditions; there are no measured-rate corrections for an unseen condition.\n",
                "- The test condition is not used for fitting or model selection.\n",
                "- Coefficients are effective transfer responses, not elementary kinetics.\n",
                "- Wall concentration is set to zero only as a driving-force proxy; absolute wall flux is not calculated.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import csv, json\n",
                "from pathlib import Path\n",
                "output = Path('.')\n",
                "with (output / 'analysis_summary.json').open(encoding='utf-8') as handle:\n",
                "    summary = json.load(handle)\n",
                "with (output / 'condition_quality.csv').open(encoding='utf-8') as handle:\n",
                "    quality = list(csv.DictReader(handle))\n",
                "print('train/test:', summary['primary_split']['train_cases'], '->', summary['primary_split']['test_case'])\n",
                "print('condition rows:', [(row['condition'], row['rows'], row['rate_unique_count']) for row in quality])\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "primary = summary['primary_split']\n",
                "print('selected model:', primary['selected_model'])\n",
                "print('common order:', primary['common_total_order'])\n",
                "print('test RMSE [nm/s]:', primary['test_rmse_nm_s'])\n",
                "print('test relative RMSE:', primary['test_relative_rmse_vs_test_mean'])\n",
                "print('test spatial R2:', primary['test_centered_spatial_r2'])\n",
                "print('species-role assessment:', summary['validity']['species_role_assessment'])\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Figures\n",
                "\n",
                "![Condition transfer](plots/condition_mean_transfer.png)\n",
                "\n",
                "![Held-out fit](plots/test_measured_vs_predicted.png)\n",
                "\n",
                "![Held-out maps](plots/test_spatial_maps.png)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Takeaways\n",
                "\n",
                f"- Fixed-model holdout: {validity['fixed_model_assessment']['prediction_status']}; "
                f"spatial shape: {validity['fixed_model_assessment']['spatial_status']}.\n",
                f"- Outer selection procedure: {validity['procedure_assessment']['prediction_status']}. "
                f"Application criteria: {validity['application_status']}.\n",
                "- Raw species are candidate inputs. An unresolved steady AB response does not determine its A/B direction.\n",
                f"- Decision evidence: {validity['reason']}.\n",
            ],
        },
    ]
    execution_globals: dict[str, Any] = {"__name__": "__notebook__"}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        buffer = StringIO()
        previous_cwd = Path.cwd()
        try:
            import os

            os.chdir(output_dir)
            with redirect_stdout(buffer):
                exec(compile(source, str(notebook_path), "exec"), execution_globals)
        finally:
            os.chdir(previous_cwd)
        text = buffer.getvalue()
        cell["outputs"] = (
            [{"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}]
            if text
            else []
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    parsed = json.loads(notebook_path.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4 or not isinstance(parsed.get("cells"), list):
        raise RuntimeError("Generated notebook failed structural validation")
    return notebook_path


def _fit_formula(fit: TransferFit | SurfaceKineticFit) -> str:
    if isinstance(fit, SurfaceKineticFit):
        return surface_formula(fit.candidate)
    if fit.response_structure == "within_between":
        return ("log(rate) = log(reference_rate) + mean_map(x) @ beta_between + "
                "(x - mean_map(x)) @ beta_within; x = " + str(list(_effect_names(fit.candidate))) +
                "; x uses log(total/reference), log(A fraction/reference) or log(AB fraction product/reference), "
                "and -log(I fraction/reference). Map means use Fluent inputs only.")
    formula = (
        "rate = reference_rate * (total_concentration / reference_total_concentration) "
        "** common_total_order"
    )
    candidate = fit.candidate
    if candidate.A is not None and candidate.B is None:
        formula += (
            f" * (fraction_{candidate.A} / reference_fraction_{candidate.A}) "
            f"** elasticity_A_{candidate.A}"
        )
    elif candidate.A is not None and candidate.B is not None:
        formula += (
            f" * ((fraction_{candidate.A} * fraction_{candidate.B}) / "
            f"(reference_fraction_{candidate.A} * reference_fraction_{candidate.B})) "
            f"** elasticity_AB_{candidate.A}_{candidate.B}"
        )
    if candidate.I is not None:
        formula += (
            f" * (fraction_{candidate.I} / reference_fraction_{candidate.I}) "
            f"** (-elasticity_I_{candidate.I})"
        )
    return formula


def _write_markdown_report(
    output_dir: Path, summary: dict[str, Any], coefficient_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> None:
    """Render measured results; interpretation is computed before presentation."""
    primary, validity = summary["primary_split"], summary["validity"]
    response_line = (
        "Response model: `surface_qss`; parameters are observable dimensionless groups of the quasi-steady site balance."
        if primary.get("response_model") == "surface_qss"
        else (f"Response structure: `{primary['response_structure']}`; between/within total orders: "
              f"{primary['common_total_order']:.6g} / {primary['within_total_order']:.6g}.")
    )
    lines = [
        "# CVD multi-condition role evaluation", "",
        f"Training conditions: {primary['train_cases']}; untouched test: {primary['test_case']}.",
        f"Selected candidate: `{primary['selected_model']}`. Decision: `{validity['decision']}`.",
        response_line,
        f"Role evidence: `{validity['species_role_assessment']}`.",
        f"Spatial prediction: `{validity['spatial_map_assessment']}`.", "",
        "Selection uses condition refits in deposition-rate units; numerical loss ties prefer fewer effects and parameters.",
        "The external test never selects coefficients, roles, or thresholds.", "",
        f"Decision evidence: {validity['evaluation_scope']}. {validity['reason']}",
        "Outer condition refits evaluate the selection procedure, with a separately fitted model in each fold.", "",
        "## Prediction", "",
        f"Test RMSE: {primary['test_rmse_nm_s']:.6g} nm/s; centered R2: {primary['test_centered_spatial_r2']:.6g}.",
        f"Test spatial correlation: {primary['test_spatial_correlation']:.6g}; predicted/observed range: {primary['test_range_capture_fraction']:.6g}.",
        f"Condition-CV RMSE: {primary['train_condition_cv_rmse_nm_s']:.6g} nm/s.", "",
        "## Coefficients", "", f"`{summary['selected_model']['formula']}`", "",
        "|Term|Value|Conditional spatial bootstrap 5-95%|", "|---|---:|---|",
    ]
    for row in coefficient_rows:
        lines.append(f"|{row['term']}|{row['value']:.6g}|{row['bootstrap_p05']:.6g} - {row['bootstrap_p95']:.6g}|")
    lines += ["", "Intervals condition on the selected model and supplied conditions; they do not include model-selection uncertainty.", "",
              "## Condition refits", "", "|Held-out condition|Selected candidate|Response structure|Relative RMSE|Centered R2|", "|---|---|---|---:|---:|"]
    for row in split_rows:
        model_label = str(row['selected_model']).replace("|", "\\|")
        lines.append(f"|{row['test_case']}|{model_label}|{row['response_structure']}|{row['test_relative_rmse_vs_test_mean']:.4%}|{row['test_centered_spatial_r2']:.4g}|")
    lines += ["", "## Interpretation", "",
              "Raw species are candidate inputs, not established chemical identities. Indistinguishable assignments remain unresolved.",
              "Wall-zero concentration differences are driving-force proxies, not absolute wall fluxes.",
              "Measurement uncertainty and independent process conditions are needed to assess practical identifiability.", "",
              "See role_summary.csv, role_ranking.csv, role_stability.csv, and condition_scores.csv for decisions and evidence."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_cvd_multicond_case(
    *,
    data_dir: Path,
    response_structure: str = "shared",
    response_model: str = "surface_qss",
    train_case_ids: tuple[int, ...] = (1, 2, 4, 5),
    test_case_id: int = 3,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 123,
    application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_case_ids = tuple(sorted({*train_case_ids, int(test_case_id)}))
    if len(train_case_ids) < 2:
        raise ValueError("At least two identification conditions are required")
    if len(set(train_case_ids)) != len(train_case_ids) or test_case_id in train_case_ids:
        raise ValueError("Identification condition IDs must be distinct and exclude the test condition")
    cases = [
        _load_case(
            case_id,
            data_dir / f"condition_{case_id}.csv",
            data_dir / f"validation_{case_id}.csv",
        )
        for case_id in all_case_ids
    ]
    case_lookup = {case.case_id: case for case in cases}
    grid_alignment = _grid_alignment(cases)
    train_cases = [case_lookup[case_id] for case_id in train_case_ids]
    test_case = case_lookup[test_case_id]

    primary, ranking, selected_fit, test_prediction, train, test = _split_evaluation(
        train_cases,
        test_case,
        response_structure=response_structure,
        response_model=response_model,
    )
    selected_row = next(row for row in ranking if row["model_id"] == primary["selected_model"])
    unrestricted_row = next(
        row for row in ranking if row["model_id"] == primary["unrestricted_numerical_winner"]
    )
    angular_groups = _angular_groups(train.xyz[:, :2])
    bootstrap = _bootstrap_coefficients(
        selected_fit,
        train,
        angular_groups,
        samples=max(int(bootstrap_samples), 100),
        seed=seed,
    )
    coefficient_rows = _coefficient_rows(selected_fit, bootstrap)

    test_prediction, test_diagnostics = _predict_response(selected_fit, test)
    test_prediction_rows: list[dict[str, Any]] = []
    for index in range(test.rate.size):
        if isinstance(selected_fit, SurfaceKineticFit):
            common_log_contribution = float("nan")
            common_rate_multiplier = float("nan")
        else:
            test_design = test_diagnostics
            assert isinstance(test_design, np.ndarray)
            common_log_contribution = float(sum(
                test_design[index, i] * selected_fit.coefficients[i]
                for i, (_, term) in enumerate(selected_fit.coefficient_terms, start=1)
                if term == "common_total_order"
            ))
            common_rate_multiplier = float(math.exp(common_log_contribution))
        row: dict[str, Any] = {
            "condition": test_case_id,
            "response_structure": selected_fit.response_structure,
            "x": float(test.xyz[index, 0]),
            "y": float(test.xyz[index, 1]),
            "z": float(test.xyz[index, 2]),
            "measured_rate_nm_s": float(test.rate[index]),
            "predicted_rate_nm_s": float(test_prediction[index]),
            "residual_pred_minus_measured_nm_s": float(test_prediction[index] - test.rate[index]),
            "total_concentration_kmol_m3": float(test.total_concentration[index]),
            "common_log_contribution": common_log_contribution,
            "common_rate_multiplier": common_rate_multiplier,
            "common_rate_delta_from_reference_nm_s": float(
                selected_fit.reference_rate_nm_s * (common_rate_multiplier - 1.0)
            ),
            "predicted_rate_delta_from_reference_nm_s": float(
                test_prediction[index] - selected_fit.reference_rate_nm_s
            ),
            "kinetic_limit": getattr(selected_fit.candidate, "kinetic_limit", "empirical"),
        }
        for name in test.species:
            row[f"concentration_{name}_kmol_m3"] = float(test.concentrations[name][index])
            row[f"assumed_wall_concentration_{name}_kmol_m3"] = 0.0
            row[f"wall_zero_driving_concentration_{name}_kmol_m3"] = float(
                test.concentrations[name][index]
            )
            row[f"fraction_{name}"] = float(test.species_fractions[name][index])
            row[f"reference_fraction_{name}"] = float(
                selected_fit.reference_species_fractions[name]
            )
            if isinstance(selected_fit, SurfaceKineticFit):
                row[f"reference_concentration_{name}_kmol_m3"] = float(
                    selected_fit.reference_concentrations[name]
                )
        if isinstance(selected_fit, SurfaceKineticFit):
            assert isinstance(test_diagnostics, dict)
            for name, values in test_diagnostics.items():
                row[name] = float(values[index])
        else:
            assert isinstance(test_diagnostics, np.ndarray)
            for effect_index, effect_name in enumerate(selected_fit.effect_names, start=1):
                log_contribution = float(
                    test_diagnostics[index, effect_index]
                    * selected_fit.coefficients[effect_index]
                )
                row[f"log_contribution_{effect_name}"] = log_contribution
                row[f"multiplier_{effect_name}"] = float(math.exp(log_contribution))
        test_prediction_rows.append(row)

    split_rows: list[dict[str, Any]] = []
    for row in ranking:
        row["selection_refits"] = []
    selected_row["holdout_metrics"] = {test_case_id: _transfer_metrics(test.rate, test_prediction, _condition_mean_rate(train))}
    for held_out in all_case_ids:
        split_train = [case for case in cases if case.case_id != held_out]
        split_result, split_ranking, _, _, _, _ = _split_evaluation(split_train, case_lookup[held_out],
                                                                  response_structure=response_structure,
                                                                  response_model=response_model)
        split_rows.append(split_result)
        refits = {r["role_model_id"]: r for r in split_ranking}
        for row in ranking:
            refit = refits[row["role_model_id"]]
            row["selection_refits"].append({
                "condition": held_out, "selected": refit["score_tied_with_best"],
                "selection_score": refit["selection_score"],
                "effect_groups": refit["effect_groups"], "effective_roles": refit["effective_roles"],
                "regularization": refit["regularization"],
                "response_structure": refit["response_structure"], "effect_scopes": refit["effect_scopes"],
            })
    selected_row["evaluation_conditions"] = [
        {**{key[5:]: value for key, value in row.items() if key.startswith("test_")},
         "condition": row["test_case"], "weight": 1.0,
         "selected_model": row["selected_model"], "effect_groups": row["effect_groups"],
         "roles": {slot: row[f"selected_role_{slot}"] or None for slot in ("A", "I", "B")}}
        for row in split_rows
    ]
    role_stability_rows, stability = build_role_stability(ranking, score_epsilon=0.0)
    split_role_rows = [
        {
            "held_out_condition": row["test_case"],
            "identification_conditions": row["train_cases"],
            "selected_model": row["selected_model"],
            "selected_role_model_id": row["selected_role_model_id"],
            "selected_role_A": row["selected_role_A"],
            "selected_role_I": row["selected_role_I"],
            "selected_role_B": row["selected_role_B"],
            "effective_roles": row["effective_roles"],
            "inactive_roles": row["inactive_roles"],
            "regularization": row["regularization"],
            "response_structure": row["response_structure"], "effect_scopes": row["effect_scopes"],
            "basis": "outer_condition_cv_selection",
            "common_total_order": row["common_total_order"],
            "within_total_order": row["within_total_order"],
            "selected_role_A_coefficient": row.get(
                f"selected_coefficient_A_{row['selected_role_A']}",
                row.get(f"selected_coefficient_between_A_{row['selected_role_A']}", "")
            ),
            "test_relative_rmse": row["test_relative_rmse_vs_test_mean"],
            "test_spatial_correlation": row["test_spatial_correlation"],
            "test_range_capture_fraction": row["test_range_capture_fraction"],
        }
        for row in split_rows
    ]

    alternative_rows = _alternative_predictors(train, test)
    candidate_test_rows = _candidate_test_diagnostics(train, test, ranking)
    extrapolation_rows = _extrapolation_summary(train, test)
    quality_rows = _condition_quality_rows(cases)
    scaling_rows = _scaling_rows(cases)
    correlation_rows = _correlation_rows(cases)

    all_concentrations = np.vstack(
        [np.column_stack([case.concentrations[name] for name in case.species]) for case in cases]
    )
    pooled_corr = np.corrcoef(all_concentrations, rowvar=False)
    max_pooled_correlation = float(
        np.max(np.abs(pooled_corr - np.eye(len(cases[0].species))))
    )
    test_outside_fraction = max(
        float(row["test_outside_train_range_fraction"]) for row in extrapolation_rows
    )
    order_values = [
        float(row["common_total_order"])
        for row in split_rows
        if np.isfinite(float(row["common_total_order"]))
    ]
    quantized_rate_cases = [
        int(row["condition"])
        for row in quality_rows
        if int(row["rate_unique_count"]) < int(row["rows"])
    ]
    low_species_precision = [
        {
            "condition": case.case_id,
            "species": name,
            "unique_count": case.quality["precision"]["species"][name]["unique_count"],
        }
        for case in cases
        for name in case.species
        if int(case.quality["precision"]["species"][name]["unique_count"]) < 5
    ]
    primary_role_id = str(primary["selected_role_model_id"])
    selected_role_a = str(primary.get("selected_role_A", ""))
    loco_role_consistency = float(
        np.mean(
            [effect_signature(row) == effect_signature(selected_row) for row in split_rows]
        )
    )
    loco_role_a_coefficients = [
        float(row["selected_role_A_coefficient"])
        for row in split_role_rows
        if str(row.get("selected_role_A", "")) == selected_role_a
        and row.get("selected_role_A_coefficient", "") != ""
    ]
    role_summary_rows = build_role_summary(
        ranking, score_epsilon=0.0, role_stability_warning=stability["warning"],
        parameter_identifiability_warning=not selected_row["design_identifiable"],
        application=application,
    )
    assessment = role_summary_rows[0]
    role_ambiguous = assessment["role_support"] == "unresolved"
    cv_supported = selected_row.get("validation_skill", 0.0) > 0.0
    test_supported = primary["test_rmse_improvement_vs_constant_train_mean"] > 0.0
    spatial_supported = primary["test_centered_spatial_r2"] > 0.0
    decision = assessment["decision"]
    validity = {
        "overall_assessment": "needs_revision" if decision == "reject_prediction" else "share_with_caveats",
        "condition_mean_transfer_assessment": ("improves_constant_baseline" if
            float(np.mean(test_prediction - test.rate))**2 < (float(np.mean(test.rate)) - _condition_mean_rate(train))**2
            else "not_supported"),
        "spatial_map_assessment": "improves_centered_constant_baseline" if spatial_supported else "not_supported",
        "species_role_assessment": assessment["role_support"],
        "species_role_adoption": decision,
        "composition_role_cross_condition_validation": "unresolved" if role_ambiguous else "see_condition_refits",
        "condition_holdout_cv_assessment": "improves_constant_baseline" if cv_supported else "not_supported",
        "elementary_kinetics_validated": False,
        "test_was_refit": False,
        "test_relative_rmse": primary["test_relative_rmse_vs_test_mean"],
        "test_spatial_r2": primary["test_centered_spatial_r2"],
        "test_range_capture_fraction": primary["test_range_capture_fraction"],
        "diagnostic_role_candidate_assessment": "test_metrics_are_diagnostic_only",
        "diagnostic_role_candidate_test_spatial_r2": primary[
            "unrestricted_test_centered_spatial_r2"
        ],
        "diagnostic_role_candidate_range_capture_fraction": primary[
            "unrestricted_test_range_capture_fraction"
        ],
        "max_pooled_species_correlation": max_pooled_correlation,
        "test_outside_train_range_fraction_max": test_outside_fraction,
        "loco_common_order_min": min(order_values) if order_values else None,
        "loco_common_order_max": max(order_values) if order_values else None,
        "loco_selected_role_consistency_fraction": loco_role_consistency,
        "loco_role_A_coefficient_min": (
            min(loco_role_a_coefficients) if loco_role_a_coefficients else None
        ),
        "loco_role_A_coefficient_max": (
            max(loco_role_a_coefficients) if loco_role_a_coefficients else None
        ),
        "quantized_rate_cases": quantized_rate_cases,
        "low_species_precision": low_species_precision,
        "decision": decision,
        "role_ambiguous": role_ambiguous,
        "prediction_status": assessment["prediction_status"],
        "application_status": assessment["application_status"],
        "evaluation_scope": assessment["evaluation_scope"],
        "fixed_model_assessment": assessment["fixed_model_assessment"],
        "procedure_assessment": assessment["procedure_assessment"],
        "reason": assessment["reason"],
        "effective_roles": selected_fit.effective_roles,
        "inactive_roles": selected_row["inactive_roles"],
        "role_symmetry": selected_row["role_symmetry"],
    }

    condition_mean_rows: list[dict[str, Any]] = []
    for case in cases:
        combined = _combine_cases([case])
        prediction, _ = _predict_response(selected_fit, combined)
        condition_mean_row: dict[str, Any] = {
            "condition": case.case_id,
            "split": "identification" if case.case_id in train_case_ids else "held-out test",
            "mean_total_concentration_kmol_m3": float(np.mean(case.total_concentration)),
            "mean_measured_rate_nm_s": float(np.mean(case.rate)),
            "mean_predicted_rate_nm_s": float(np.mean(prediction)),
            "mean_bias_nm_s": float(np.mean(prediction - case.rate)),
        }
        for name in combined.species:
            condition_mean_row[f"mean_fraction_{name}"] = float(
                np.mean(combined.species_fractions[name])
            )
        condition_mean_rows.append(condition_mean_row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "estimation": {"response_model": response_model,
                       "loss": ("condition-balanced squared raw-rate error" if response_model == "surface_qss"
                                else "condition-balanced squared log-rate error"),
                       "penalty": ("none; exact site-balance reductions and condition CV control complexity"
                                   if response_model == "surface_qss" else
                                   "shared: lambda*||beta||^2; within_between: lambda*((||between||^2+||within||^2)/2+||between-within||^2); intercept unpenalized"),
                       "regularization_grid": [] if response_model == "surface_qss" else list(REGULARIZATION_GRID),
                       "response_structure_policy": "site_balance_qss" if response_model == "surface_qss" else response_structure,
                       "response_structures": (["surface_qss"] if response_model == "surface_qss" else
                                               list(RESPONSE_STRUCTURES) if response_structure == "select" else [response_structure]),
                       "map_centering": ("not used; absolute concentrations are normalized by identification-data references"
                                         if response_model == "surface_qss" else
                                         "full supplied Fluent input map, independent of measured rates and prediction batch"),
                       "selection": ("role/reduction inner condition CV MSE in rate units"
                                     if response_model == "surface_qss" else
                                     "joint role/response-structure/regularization inner condition CV MSE in rate units")},
        "analysis_type": (
            f"{len(train_case_ids)}-condition identification plus one-condition no-refit test"
        ),
        "surface_state_assumption": (
            "quasi-steady site balance with observable lumped parameters; no fitted condition-specific offsets"
            if response_model == "surface_qss" else
            "effective response coefficients transfer across conditions; no fitted condition-specific offsets"
        ),
        "primary_split": primary,
        "selected_model": {
            "model_id": primary["selected_model"],
            "formula": _fit_formula(selected_fit),
            "reference_rate_nm_s": selected_fit.reference_rate_nm_s,
            "reference_total_concentration_kmol_m3": selected_fit.reference_total_concentration,
            "reference_species_fractions": selected_fit.reference_species_fractions,
            "common_total_order": selected_fit.common_order,
            "within_total_order": selected_fit.within_order,
            "response_structure": selected_fit.response_structure,
            "effect_scopes": selected_fit.effect_scopes,
            "species_role_terms": [name for name, (_, term) in zip(selected_fit.effect_names, selected_fit.coefficient_terms)
                                   if term != "common_total_order"],
            "selection_reason": primary["selection_reason"],
            "regularization": selected_fit.regularization,
            "effective_roles": selected_fit.effective_roles,
            "kinetic_limit": getattr(selected_fit.candidate, "kinetic_limit", "empirical"),
            "observable_parameters": (
                selected_fit.shape_parameters if isinstance(selected_fit, SurfaceKineticFit) else {}
            ),
            "reference_concentrations_kmol_m3": (
                selected_fit.reference_concentrations if isinstance(selected_fit, SurfaceKineticFit) else {}
            ),
        },
        "transport_proxy_assumption": {
            "wall_species_concentration_kmol_m3": 0.0,
            "driving_concentration_definition": "bulk_concentration_minus_wall_concentration",
            "driving_concentration_equals_supplied_concentration": True,
            "absolute_wall_molar_flux_calculated": False,
            "missing_for_absolute_flux": [
                "species diffusivity or mass-transfer coefficient",
                "wall-normal distance or concentration gradient",
            ],
        },
        "unrestricted_numerical_winner": unrestricted_row,
        "validity": validity,
        "grid_alignment": grid_alignment,
        "data_quality": [case.quality for case in cases],
        "missing_information": [
            "independent feasible composition perturbations to distinguish competing role assignments",
            ("independent concentration perturbations that traverse low-coverage and saturation regimes"
             if response_model == "surface_qss" else
             "another independent total-concentration level at nearly fixed composition to validate the common total order"),
            "species diffusivity or mass-transfer coefficient and wall-normal distance for absolute wall flux",
            "chemical identities, molar masses, feed/byproduct roles, and stoichiometry",
            "coordinate unit, wafer temperature map, and pressure",
            "surface/site density and adsorption/desorption or sticking information",
            "measurement uncertainty and replicate deposition maps",
        ],
        "interpretation_limits": [
            ("Roles and exact kinetic reductions are chosen by inner condition CV; only outer predictions evaluate the selected procedure."
             if response_model == "surface_qss" else
             "Regularization and roles are jointly chosen by inner condition CV; only outer predictions evaluate the selected procedure."),
            "A numerical score tie is not statistical or practical equivalence. A no-inhibitor steady AB response cannot identify A/B direction.",
            ("Observable dimensionless groups are not separate elementary rate constants or a surface relaxation time."
             if response_model == "surface_qss" else
             "The common order describes the supplied condition scaling and is not an elementary reaction order."),
            "A species excluded by the adoption gate is not proven inert.",
            "Negative test R2 can coexist with low relative RMSE because within-map variation is much smaller than the absolute condition-level rate.",
            "See test_extrapolation.csv for the supplied test condition; a reduced physical form does not establish the true mechanism out of domain.",
            "Bootstrap intervals condition on the same identification conditions and do not include between-condition or model-form uncertainty.",
        ],
        "sources": [
            {
                "case_id": case.case_id,
                "condition_path": str(case.condition_path),
                "validation_path": str(case.validation_path),
                "condition_sha256": _sha256_file(case.condition_path),
                "validation_sha256": _sha256_file(case.validation_path),
            }
            for case in cases
        ],
    }


    _write_rows(output_dir / "condition_quality.csv", quality_rows)
    _write_rows(output_dir / "concentration_scaling.csv", scaling_rows)
    _write_rows(output_dir / "pooled_concentration_correlations.csv", correlation_rows)
    _write_rows(output_dir / "model_ranking.csv", ranking)
    _write_rows(output_dir / "role_ranking.csv", ranking)
    _write_rows(output_dir / "role_summary.csv", role_summary_rows)
    _write_rows(output_dir / "coefficients.csv", coefficient_rows)
    _write_rows(output_dir / "test_predictions.csv", test_prediction_rows)
    _write_rows(output_dir / "split_sensitivity.csv", split_rows)
    _write_rows(output_dir / "role_stability.csv", role_stability_rows)
    _write_rows(output_dir / "condition_scores.csv", build_condition_scores(ranking))
    _write_rows(output_dir / "alternative_predictors.csv", alternative_rows)
    _write_rows(output_dir / "candidate_test_diagnostics.csv", candidate_test_rows)
    _write_rows(output_dir / "test_extrapolation.csv", extrapolation_rows)
    _write_rows(output_dir / "condition_means.csv", condition_mean_rows)
    _write_json(output_dir / "analysis_summary.json", summary)
    plot_paths = _plot_results(
        output_dir,
        cases,
        selected_fit,
        train,
        test,
        test_prediction,
        ranking,
    )

    _write_markdown_report(
        output_dir,
        summary,
        coefficient_rows,
        split_rows,
    )
    notebook_path = _write_notebook(output_dir, tuple(train_case_ids), int(test_case_id))

    source_files = [
        str(path)
        for case in cases
        for path in (case.condition_path, case.validation_path)
    ]
    source_metadata = {
        "label": "CVD Fluent concentration maps and deposition-rate maps for conditions 1–3",
        "files": source_files,
        "filters": [
            f"Identification conditions: {list(train_case_ids)}",
            f"Held-out no-refit test condition: {test_case_id}",
            f"{sum(case.rate.size for case in cases)} matched observations across {len(cases)} conditions",
        ],
    }
    source_metadata["label"] = (
        f"CVD Fluent concentration maps and deposition-rate maps for conditions {list(all_case_ids)}"
    )
    report_snapshot = {
        "title": f"CVD role evaluation: {validity['decision']}",
        "generatedAt": summary["generated_at"],
        "status": "authored",
        "filters": [],
        "queries": {
            "primary_summary": {
                "rows": [primary],
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Held-out test RMSE",
                            "definition": (
                                f"RMSE on condition {test_case_id} after fitting and selecting only on "
                                f"conditions {list(train_case_ids)}; no test refit."
                            ),
                            "componentIds": ["report-summary", "test-quality"],
                            "sourceLineage": [{"files": source_files}],
                        },
                        {
                            "label": ("Observable surface-response parameters" if response_model == "surface_qss"
                                      else "Common total-concentration order"),
                            "definition": (
                                "Dimensionless groups of the quasi-steady site balance; they are not separate elementary constants."
                                if response_model == "surface_qss" else
                                "Effective power response to the sum of supplied concentration_* fields; not an elementary reaction order."
                            ),
                            "componentIds": ["coefficient-table", "report-summary"],
                            "sourceLineage": [{"files": source_files}],
                        },
                    ],
                },
            },
            "condition_means": {
                "rows": condition_mean_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Condition mean deposition rate",
                            "definition": "Arithmetic mean across the matched observations in each condition.",
                            "componentIds": ["condition-transfer-chart", "condition-mean-table"],
                            "sourceLineage": [{"files": source_files}],
                        }
                    ],
                },
            },
            "condition_quality": {"rows": quality_rows, "source": source_metadata},
            "model_ranking": {
                "rows": ranking,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Candidate selection",
                            "definition": (
                                ("Training-condition CV compares independently refitted role assignments and exact kinetic reductions. "
                                 if response_model == "surface_qss" else
                                 "Training-condition CV compares independently refitted structures and regularization strengths. ")
                                + "Numerical score ties prefer fewer effects; application adoption requires independent evidence and declared tolerances."
                            ),
                            "componentIds": ["role-ranking-table"],
                            "sourceLineage": [{"files": source_files}],
                        }
                    ],
                },
            },
            "coefficients": {"rows": coefficient_rows, "source": source_metadata},
            "test_predictions": {
                "rows": test_prediction_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "No-refit prediction",
                            "definition": (
                                f"Condition {test_case_id} prediction from coefficients selected and fitted "
                                f"only on conditions {list(train_case_ids)}."
                            ),
                            "componentIds": ["test-fit-chart"],
                            "sourceLineage": [{"files": source_files}],
                        }
                    ],
                },
            },
            "split_sensitivity": {"rows": split_rows, "source": source_metadata},
            "role_stability": {"rows": role_stability_rows, "source": source_metadata},
            "alternative_predictors": {"rows": alternative_rows, "source": source_metadata},
            "candidate_test_diagnostics": {
                "rows": candidate_test_rows,
                "source": source_metadata,
            },
            "test_extrapolation": {"rows": extrapolation_rows, "source": source_metadata},
        },
    }
    report_snapshot["title"] = (
        f"CVD composition-role transfer: {len(train_case_ids)}-condition fit and held-out test"
    )
    _write_json(output_dir / "report_snapshot.json", report_snapshot)

    artifact_paths = [
        output_dir / "condition_quality.csv",
        output_dir / "concentration_scaling.csv",
        output_dir / "pooled_concentration_correlations.csv",
        output_dir / "model_ranking.csv",
        output_dir / "role_ranking.csv",
        output_dir / "role_summary.csv",
        output_dir / "coefficients.csv",
        output_dir / "test_predictions.csv",
        output_dir / "split_sensitivity.csv",
        output_dir / "role_stability.csv",
        output_dir / "condition_scores.csv",
        output_dir / "alternative_predictors.csv",
        output_dir / "candidate_test_diagnostics.csv",
        output_dir / "test_extrapolation.csv",
        output_dir / "condition_means.csv",
        output_dir / "analysis_summary.json",
        output_dir / "report.md",
        output_dir / "report_snapshot.json",
        notebook_path,
        *plot_paths,
    ]
    manifest = {
        "generated_at": summary["generated_at"],
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return _json_safe(summary)


__all__ = ["ConditionCase", "TransferFit", "analyze_cvd_multicond_case"]
