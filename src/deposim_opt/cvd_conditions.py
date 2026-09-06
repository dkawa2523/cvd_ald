"""Load and align multi-condition CVD inputs for role-model fitting.

This module owns file discovery, source-column interpretation, coordinate
alignment, and assembly of condition data into the model-neutral role fields.
It does not enumerate equations, fit candidates, or make adoption decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cvd_analysis_io import align_validation, coordinate_matrix, read_numeric_csv
from .role_fields import RoleFieldSet
from .spatial_validation import EPS


def condition_paths(
    data_dir: Path,
    case_ids: Iterable[int],
    conditions_file: Path | None,
) -> dict[int, tuple[Path, Path]]:
    """Resolve condition and validation files from an optional JSON manifest."""

    requested = tuple(int(case_id) for case_id in case_ids)
    if conditions_file is None:
        return {
            case_id: (
                data_dir / f"condition_{case_id}.csv",
                data_dir / f"validation_{case_id}.csv",
            )
            for case_id in requested
        }

    manifest_path = Path(conditions_file)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("conditions", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Condition manifest must be a list or contain a 'conditions' list")
    resolved: dict[int, tuple[Path, Path]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each condition manifest entry must be an object")
        case_id = int(entry["id"])
        if case_id in resolved:
            raise ValueError(f"Duplicate condition id in manifest: {case_id}")
        condition_path = Path(str(entry["condition"]))
        validation_path = Path(str(entry["validation"]))
        if not condition_path.is_absolute():
            condition_path = manifest_path.parent / condition_path
        if not validation_path.is_absolute():
            validation_path = manifest_path.parent / validation_path
        resolved[case_id] = (condition_path, validation_path)
    missing = sorted(set(requested) - set(resolved))
    if missing:
        raise ValueError(f"Condition manifest is missing requested ids: {missing}")
    return {case_id: resolved[case_id] for case_id in requested}


@dataclass(frozen=True)
class ConditionCase:
    """One aligned Fluent condition and its measured film-rate map."""

    case_id: int
    condition_path: Path
    validation_path: Path
    xyz: np.ndarray
    species: tuple[str, ...]
    bulk_concentrations: dict[str, np.ndarray]
    surface_concentrations: dict[str, np.ndarray]
    transport_capacity_flux: dict[str, np.ndarray]
    realized_reactive_flux: dict[str, np.ndarray]
    mole_fractions: dict[str, np.ndarray]
    density: np.ndarray
    total_concentration: np.ndarray
    rate: np.ndarray
    rate_sigma: np.ndarray | None
    quality: dict[str, Any]


def _positive_min_step(values: np.ndarray) -> float | None:
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size < 2:
        return None
    differences = np.diff(unique)
    positive = differences[differences > 0.0]
    return float(np.min(positive)) if positive.size else None


def _optional_species_fields(
    table: dict[str, np.ndarray],
    species: tuple[str, ...],
    *,
    prefix: str,
    source: Path,
    nonnegative: bool,
) -> dict[str, np.ndarray]:
    columns = {name: f"{prefix}{name}" for name in species}
    present = {name: column in table for name, column in columns.items()}
    if not any(present.values()):
        return {}
    missing = [columns[name] for name, found in present.items() if not found]
    if missing:
        raise ValueError(
            f"{source} supplies only part of {prefix} fields; missing: {missing}"
        )
    fields = {
        name: np.asarray(table[column], dtype=float) for name, column in columns.items()
    }
    invalid = any(
        np.any(~np.isfinite(values))
        or (nonnegative and np.any(values < 0.0))
        for values in fields.values()
    )
    if invalid:
        qualifier = "nonnegative and " if nonnegative else ""
        raise ValueError(f"All {prefix} fields must be {qualifier}finite in {source}")
    return fields


def load_case(case_id: int, condition_path: Path, validation_path: Path) -> ConditionCase:
    """Load and align one condition without applying a reaction model."""

    condition_headers, condition = read_numeric_csv(condition_path)
    validation_headers, validation = read_numeric_csv(validation_path)
    if "dr_nm_per_sec" not in validation:
        raise ValueError(f"{validation_path} must contain dr_nm_per_sec")
    concentration_columns = [
        name for name in condition_headers if name.startswith("concentration_")
    ]
    if not concentration_columns:
        raise ValueError(f"No concentration_* columns found in {condition_path}")
    species = tuple(name.removeprefix("concentration_") for name in concentration_columns)
    bulk_concentrations = {
        name: np.asarray(condition[f"concentration_{name}"], dtype=float)
        for name in species
    }
    if any(
        np.any(~np.isfinite(values)) or np.any(values <= 0.0)
        for values in bulk_concentrations.values()
    ):
        raise ValueError(f"All concentrations must be positive and finite in {condition_path}")
    surface_concentrations = _optional_species_fields(
        condition,
        species,
        prefix="surface_concentration_",
        source=condition_path,
        nonnegative=True,
    )
    transport_capacity_flux = _optional_species_fields(
        condition,
        species,
        prefix="transport_capacity_flux_",
        source=condition_path,
        nonnegative=True,
    )
    realized_reactive_flux = _optional_species_fields(
        condition,
        species,
        prefix="realized_reactive_flux_",
        source=condition_path,
        nonnegative=True,
    )
    condition_xyz = coordinate_matrix(condition)
    validation_xyz = coordinate_matrix(validation)
    rate, alignment = align_validation(
        condition_xyz,
        validation_xyz,
        validation["dr_nm_per_sec"],
        coordinate_decimals=6,
    )
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
        raise ValueError(f"All deposition rates must be positive and finite in {validation_path}")
    rate_sigma = None
    if "sigma_nm_per_sec" in validation:
        rate_sigma, _ = align_validation(
            condition_xyz,
            validation_xyz,
            validation["sigma_nm_per_sec"],
            coordinate_decimals=6,
        )
        if np.any(~np.isfinite(rate_sigma)) or np.any(rate_sigma <= 0.0):
            raise ValueError(
                f"All sigma_nm_per_sec values must be positive and finite in {validation_path}"
            )

    mole_fractions: dict[str, np.ndarray] = {}
    missing_molef: list[str] = []
    for name in species:
        column = f"molef_{name}"
        if column in condition:
            mole_fractions[name] = np.asarray(condition[column], dtype=float)
        else:
            missing_molef.append(column)
    total = np.sum(
        np.column_stack([bulk_concentrations[name] for name in species]), axis=1
    )
    molef_sum = (
        np.sum(np.column_stack([mole_fractions[name] for name in species]), axis=1)
        if not missing_molef
        else np.full(rate.shape, np.nan)
    )
    species_total_estimates = (
        np.column_stack(
            [
                bulk_concentrations[name]
                / np.maximum(mole_fractions[name], EPS)
                for name in species
            ]
        )
        if not missing_molef
        else np.empty((rate.size, 0), dtype=float)
    )
    relative_consistency = (
        np.abs(species_total_estimates - total[:, None])
        / np.maximum(total[:, None], EPS)
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
                "unique_count": int(np.unique(bulk_concentrations[name]).size),
                "min_positive_step_kmol_m3": _positive_min_step(
                    bulk_concentrations[name]
                ),
                "relative_range_vs_median": float(
                    np.ptp(bulk_concentrations[name])
                    / max(float(np.median(bulk_concentrations[name])), EPS)
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
        "mole_fraction_sum_max_abs_error_from_one": float(
            np.nanmax(np.abs(molef_sum - 1.0))
        ),
        "concentration_mole_fraction_max_relative_inconsistency": (
            float(np.max(relative_consistency)) if relative_consistency.size else None
        ),
        "derived_mixture_molar_mass_min_kg_kmol": float(np.nanmin(density / total)),
        "derived_mixture_molar_mass_max_kg_kmol": float(np.nanmax(density / total)),
        "precision": precision,
        "input_capabilities": [
            "bulk_concentration",
            *(["surface_concentration"] if surface_concentrations else []),
            *(["transport_capacity_flux"] if transport_capacity_flux else []),
            *(["realized_reactive_flux"] if realized_reactive_flux else []),
            *(["uncertainty"] if rate_sigma is not None else []),
        ],
    }
    return ConditionCase(
        case_id=int(case_id),
        condition_path=Path(condition_path),
        validation_path=Path(validation_path),
        xyz=condition_xyz,
        species=species,
        bulk_concentrations=bulk_concentrations,
        surface_concentrations=surface_concentrations,
        transport_capacity_flux=transport_capacity_flux,
        realized_reactive_flux=realized_reactive_flux,
        mole_fractions=mole_fractions,
        density=density,
        total_concentration=total,
        rate=np.asarray(rate, dtype=float),
        rate_sigma=rate_sigma,
        quality=quality,
    )


def grid_alignment(
    cases: Iterable[ConditionCase], decimals: int = 6
) -> dict[str, Any]:
    """Verify that conditions share the same spatial grid."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one condition is required")
    base = case_list[0]
    base_keys = [tuple(row) for row in np.round(base.xyz, decimals=decimals)]
    result: dict[str, Any] = {
        "reference_case": base.case_id,
        "rounding_decimals": decimals,
        "pairs": [],
    }
    for other in case_list[1:]:
        other_keys = [tuple(row) for row in np.round(other.xyz, decimals=decimals)]
        same_set = set(base_keys) == set(other_keys)
        lookup = {key: index for index, key in enumerate(other_keys)}
        if not same_set:
            raise ValueError(
                f"Spatial grids differ beyond {decimals} decimals: "
                f"condition {base.case_id} vs {other.case_id}"
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


def combine_cases(cases: Iterable[ConditionCase]) -> RoleFieldSet:
    """Assemble aligned conditions into model-neutral role fields."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one condition is required")
    species = case_list[0].species
    if any(case.species != species for case in case_list[1:]):
        raise ValueError(
            "All conditions must contain the same concentration species in the same order"
        )
    bulk_concentrations = {
        name: np.concatenate([case.bulk_concentrations[name] for case in case_list])
        for name in species
    }

    def combine_optional(attribute: str) -> dict[str, np.ndarray]:
        mappings = [getattr(case, attribute) for case in case_list]
        if not any(mappings):
            return {}
        if not all(set(mapping) == set(species) for mapping in mappings):
            raise ValueError(
                f"{attribute} must be available for every species and condition in one fit"
            )
        return {
            name: np.concatenate([mapping[name] for mapping in mappings])
            for name in species
        }

    condition_id = np.concatenate(
        [np.full(case.rate.size, case.case_id, dtype=int) for case in case_list]
    )
    total_concentration = np.concatenate(
        [case.total_concentration for case in case_list]
    )
    species_fractions = {
        name: bulk_concentrations[name] / np.maximum(total_concentration, EPS)
        for name in species
    }
    sigma_available = [case.rate_sigma is not None for case in case_list]
    if any(sigma_available) and not all(sigma_available):
        raise ValueError(
            "rate uncertainty must be available for every condition in one fit"
        )
    return RoleFieldSet(
        case_ids=tuple(case.case_id for case in case_list),
        xyz=np.vstack([case.xyz for case in case_list]),
        condition_id=condition_id,
        species=species,
        bulk_concentrations=bulk_concentrations,
        species_fractions=species_fractions,
        total_concentration=total_concentration,
        rate=np.concatenate([case.rate for case in case_list]),
        surface_concentrations=combine_optional("surface_concentrations"),
        transport_capacity_flux=combine_optional("transport_capacity_flux"),
        realized_reactive_flux=combine_optional("realized_reactive_flux"),
        rate_sigma=(
            np.concatenate(
                [np.asarray(case.rate_sigma, dtype=float) for case in case_list]
            )
            if all(sigma_available)
            else None
        ),
    )


__all__ = [
    "ConditionCase",
    "combine_cases",
    "condition_paths",
    "grid_alignment",
    "load_case",
]
