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

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .class_compare import rank_role_candidates, build_role_summary, build_role_stability, effect_signature, build_condition_scores
from .fit_roles import condition_refits
from .metrics import prediction_metrics
from deposim_sim.models.aib_reductions import (
    AIB_QSS,
    SurfaceKineticCandidate,
    available_surface_model_families,
    candidate_evidence_requirements,
    candidate_physical_question,
    default_surface_model_families,
    enumerate_surface_kinetic_candidates,
    get_surface_model_family,
    reduction_removed_effects,
    surface_formula,
)
from deposim_sim.models.process_models import get_process_model_info
from .surface_fit import (
    SurfaceKineticFit,
    SurfaceOptimizationSettings,
    fit_surface_kinetic,
    parameter_loss_slice_rows,
    parameter_sensitivity_rows,
    parameter_design_diagnostics,
    predict_surface_kinetic,
    role_input_sensitivity_rows,
    role_response_curve_rows,
)
from .role_fields import RoleFieldSet, condition_contrast_summary
from .evidence_requirements import (
    build_capability_requirements,
    required_measurements_for,
)
from .cvd_multicond_report import (
    _fit_formula,
    _write_markdown_report,
    _write_notebook,
    plot_multicond_results,
)
from .cvd_conditions import (
    ConditionCase,
    combine_cases as _combine_cases,
    condition_paths as _condition_paths,
    grid_alignment as _grid_alignment,
    load_case as _load_case,
)
from .cvd_analysis_io import (
    json_safe as _json_safe,
    sha256_file as _sha256_file,
    write_json as _write_json,
    write_rows as _write_rows,
)
from .empirical_response import (
    RoleResponseCandidate,
    enumerate_role_response_candidates,
    fit_nonnegative_effects as _fit_nonnegative_effects,
)
from .spatial_validation import (
    EPS as _EPS,
    angular_groups as _angular_groups,
    radial_groups as _radial_groups,
    rate_metrics as _metrics,
)
from .spatial_response import (
    SPATIAL_RESPONSE_MODES,
    apply_spatial_response,
    fit_spatial_response,
    spatial_coefficient_rows,
)

# Fixed before evaluation; selected jointly with the role by training-condition CV.
# The loss is mean squared log-rate error; elasticities have unit prior scale.
REGULARIZATION_GRID = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
RESPONSE_STRUCTURES = ("shared", "within_between")
RESPONSE_MODELS = ("surface_compare", "empirical_power")


def _surface_families(
    response_model: str, requested: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Resolve an optional family subset for the physical comparison path."""
    if response_model == "surface_compare":
        available = available_surface_model_families()
        names = tuple(str(name) for name in requested) if requested else ()
        if "all" in names:
            if names != ("all",):
                raise ValueError("Use model family 'all' by itself")
            selected = available
        else:
            selected = tuple(
                dict.fromkeys(names or default_surface_model_families())
            )
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(
                f"Unknown surface model families {unknown}; available: {list(available)}"
            )
        if not selected:
            raise ValueError("At least one surface model family is required")
        return selected
    if requested:
        raise ValueError("model_families applies only to response_model='surface_compare'")
    return ()


def _uses_surface_response(response_model: str) -> bool:
    return response_model == "surface_compare"


def _candidate_role_species(candidate: SurfaceKineticCandidate) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(species)
            for species in (candidate.A, candidate.I, candidate.B)
            if species is not None
        )
    )


def _annotate_surface_candidates(
    rows: list[dict[str, Any]], data: RoleFieldSet
) -> None:
    """Attach input-only contrast evidence and fit-derived physical conclusions."""

    for row in rows:
        candidate = row.pop("_candidate")
        family = get_surface_model_family(candidate.family)
        contrast = condition_contrast_summary(
            data,
            _candidate_role_species(candidate),
            transport_mode=candidate.transport_mode,
        )
        row.update(
            {
                "computable": True,
                "applicability_status": (
                    "production" if family.enabled_by_default else "exploratory"
                ),
                "physical_question": candidate_physical_question(candidate),
                "required_evidence": list(candidate_evidence_requirements(candidate)),
                "contrast_status": contrast["status"],
                "contrast_rank": contrast["rank"],
                "contrast_species_count": contrast["species_count"],
                "contrast_condition_number": contrast["condition_number"],
                "contrast_max_abs_correlation": contrast["max_abs_correlation"],
                "contrast_log10_span": contrast["log10_span"],
                "contrast_confounded_species": contrast["confounded_species"],
            }
        )


def _finalize_surface_evidence(rows: list[dict[str, Any]]) -> None:
    """Summarize what each fitted equation supports without using the holdout."""

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["equation_family"]), []).append(row)
    for family_rows in by_family.values():
        for rank, row in enumerate(
            sorted(family_rows, key=lambda item: float(item["selection_score"])),
            start=1,
        ):
            row["family_rank"] = rank

    for row in rows:
        comparisons = row.get("reduced_model_comparisons", [])
        deltas = [
            float(item["selection_error_increase"])
            for item in comparisons
            if np.isfinite(float(item.get("selection_error_increase", float("nan"))))
        ]
        row["reduction_delta_mse"] = min(deltas) if deltas else float("nan")
        claims: list[str] = []
        missing: list[str] = []
        if float(row.get("validation_skill", 0.0)) > 0.0:
            claims.append("improves the conditionwise constant-rate baseline")
        if row.get("contrast_status") == "limited":
            species = row.get("contrast_confounded_species", [])
            detail = ", ".join(species) if species else "assigned species"
            missing.append(f"independent between-condition variation for {detail}")
        if (
            row.get("contrast_status") == "limited"
            or row.get("applicability_status") == "exploratory"
        ):
            missing.extend(str(item) for item in row.get("required_evidence", []))
        evidence = row.get("role_evidence", [])
        for item in evidence:
            effect = str(item["effect"])
            if item.get("necessity") == "consistent_benefit":
                claims.append(f"{effect} effect improves its exact reduction")
            elif item.get("necessity") in {"mixed", "no_benefit"}:
                missing.append(f"consistent parent-versus-reduction benefit for {effect}")
            if item.get("assignment") == "distinguished":
                claims.append(f"{effect} raw-species assignment is separated in inner CV")
            elif item.get("assignment") == "unresolved":
                missing.append(f"independent perturbations separating the {effect} assignment")
        role_effects = set(row.get("effect_groups", {}))
        for comparison in comparisons:
            for effect in comparison.get("removed_effects", []):
                if effect in role_effects:
                    continue
                if comparison.get("status") == "consistent_benefit":
                    claims.append(f"{effect} improves its exact reduction")
                elif comparison.get("status") in {"mixed", "no_benefit"}:
                    missing.append(
                        f"consistent parent-versus-reduction benefit for {effect}"
                    )
        if row.get("role_symmetry"):
            missing.append(str(row["role_symmetry"]))
        boundary = row.get("fit_diagnostics", {}).get("identifiability", {}).get(
            "boundary_parameters", []
        )
        if boundary:
            missing.append("interior parameter support for: " + ", ".join(boundary))
        row["supported_claims"] = list(dict.fromkeys(claims))
        row["missing_evidence"] = list(dict.fromkeys(missing))
        row["distinguishable"] = bool(
            row.get("effect_groups")
            and row.get("contrast_status") == "sufficient"
            and evidence
            and all(item.get("necessity") == "consistent_benefit" for item in evidence)
            and all(item.get("assignment") == "distinguished" for item in evidence)
            and not row.get("role_symmetry")
        )


def _equation_family_assessments(
    ranking: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    families: tuple[str, ...],
    available_inputs: Iterable[str],
) -> list[dict[str, Any]]:
    """Build one concise, training-ranked assessment per requested equation family."""

    available = set(str(name) for name in available_inputs)
    global_best = min(
        (float(row["condition_cv_rmse_nm_s"]) for row in ranking),
        default=float("nan"),
    )
    selected_counts = {
        family: sum(
            str(row.get("selected_equation_family")) == family for row in split_rows
        )
        for family in families
    }
    assessments: list[dict[str, Any]] = []
    for name in families:
        family = get_surface_model_family(name)
        candidates = [
            row
            for row in ranking
            if row.get("equation_family") == name
            and row.get("class_id") != "baseline"
        ]
        if not set(family.required_inputs).issubset(available) or not candidates:
            missing_inputs = sorted(set(family.required_inputs) - available)
            assessments.append(
                {
                    "equation_family": name,
                    "computable": False,
                    "applicability_status": "not_applicable",
                    "physical_question": family.physical_question,
                    "mechanism": family.mechanism,
                    "pathways": list(family.pathways),
                    "state_variables": list(family.state_variables),
                    "best_model_id": "",
                    "condition_cv_rmse_nm_s": float("nan"),
                    "relative_rmse_gap_to_best": float("nan"),
                    "outer_selection_count": 0,
                    "outer_selection_frequency": 0.0,
                    "contrast_status": "not_assessed",
                    "distinguishable": False,
                    "supported_claims": [],
                    "missing_evidence": [
                        *(f"required input: {item}" for item in missing_inputs),
                        *family.evidence_requirements,
                    ],
                }
            )
            continue
        best = min(candidates, key=lambda row: float(row["selection_score"]))
        count = selected_counts[name]
        missing = list(best.get("missing_evidence", []))
        if split_rows and count != len(split_rows):
            missing.append("stable equation-family selection across outer condition folds")
        assessments.append(
            {
                "equation_family": name,
                "computable": True,
                "applicability_status": (
                    "production" if family.enabled_by_default else "exploratory"
                ),
                "physical_question": family.physical_question,
                "mechanism": family.mechanism,
                "pathways": list(family.pathways),
                "state_variables": list(family.state_variables),
                "best_model_id": best["role_model_id"],
                "condition_cv_rmse_nm_s": best["condition_cv_rmse_nm_s"],
                "relative_rmse_gap_to_best": (
                    float(best["condition_cv_rmse_nm_s"]) / global_best - 1.0
                    if global_best > 0.0
                    else float("nan")
                ),
                "outer_selection_count": count,
                "outer_selection_frequency": (
                    count / len(split_rows) if split_rows else 0.0
                ),
                "contrast_status": best.get("contrast_status", "not_assessed"),
                "distinguishable": bool(best.get("distinguishable", False)),
                "supported_claims": best.get("supported_claims", []),
                "missing_evidence": list(dict.fromkeys(missing)),
            }
        )
    return assessments


def _reaction_mechanism_assessments(
    ranking: list[dict[str, Any]],
    equation_assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Separate fitted response equations from their mechanism interpretations."""

    rows: list[dict[str, Any]] = []
    for equation in equation_assessments:
        rows.append(
            {
                "mechanism_id": equation["equation_family"],
                "problem_layer": "steady_surface_response",
                "mechanism": equation.get("mechanism", ""),
                "pathways": equation.get("pathways", []),
                "state_variables": equation.get("state_variables", []),
                "evaluation_status": (
                    "fitted" if equation.get("computable") else "not_applicable"
                ),
                "steady_representation": equation.get("best_model_id", ""),
                "condition_cv_rmse_nm_s": equation.get(
                    "condition_cv_rmse_nm_s", float("nan")
                ),
                "distinguishable": bool(equation.get("distinguishable", False)),
                "supported_claims": equation.get("supported_claims", []),
                "missing_evidence": equation.get("missing_evidence", []),
            }
        )

    mvk = get_process_model_info("role_cvd_mvk")
    equivalents = [
        row
        for row in ranking
        if row.get("equation_family") == AIB_QSS
        and row.get("class_id") == "AB"
        and row.get("reduction_id") == "no_desorption"
    ]
    representative = (
        min(equivalents, key=lambda row: float(row["selection_score"]))
        if equivalents
        else None
    )
    rows.append(
        {
            "mechanism_id": "mars_van_krevelen",
            "problem_layer": "dynamic_surface_state",
            "mechanism": mvk.mechanism,
            "pathways": list(mvk.pathways),
            "state_variables": list(mvk.state_variables),
            "evaluation_status": (
                "steady_observable_equivalent"
                if representative is not None
                else "not_evaluated"
            ),
            "steady_representation": (
                representative["role_model_id"] if representative is not None else ""
            ),
            "condition_cv_rmse_nm_s": (
                representative["condition_cv_rmse_nm_s"]
                if representative is not None
                else float("nan")
            ),
            "distinguishable": False,
            "supported_claims": (
                [
                    "steady two-reactant turnover is represented by the exact "
                    "aib_qss AB no-desorption response"
                ]
                if representative is not None
                else []
            ),
            "missing_evidence": [
                "time-resolved A/B switching or pulse response separating reservoir memory",
                "surface or lattice oxidation-state observation for oxidized_fraction",
                "independent regeneration conditions that identify k_regenerate",
            ],
            "steady_observable_equivalence": mvk.steady_observable_equivalence,
        }
    )
    return rows


def _workflow_layers(
    equation_assessments: list[dict[str, Any]],
    *,
    selected_reaction_input: dict[str, str],
    spatial_response_mode: str,
) -> list[dict[str, Any]]:
    """Describe the finite model scope without mixing physical responsibilities."""

    return [
        {
            "layer": "observation_baseline",
            "responsibility": "test whether any supplied field is needed",
            "models": ["constant_rate", "empirical_power_compatibility"],
            "execution_scope": "constant-rate baseline evaluated in the current analysis",
            "output_quantity": "film deposition rate",
            "output_unit": "nm/s",
        },
        {
            "layer": "steady_surface_response",
            "responsibility": "compare observable reaction forms and exact reductions",
            "models": [row["equation_family"] for row in equation_assessments],
            "execution_scope": "enumerated and optimized in the current steady-map analysis",
            "input_quantity": (
                "normalized local driver u_j=X_j/X_j,ref; "
                f"X is {selected_reaction_input['quantity']}"
            ),
            "input_location": selected_reaction_input["location"],
            "input_unit": "1",
            "shape_parameter_unit": "1",
            "rate_scale_unit": "nm/s",
        },
        {
            "layer": "dynamic_surface_state",
            "responsibility": "represent memory that steady maps cannot identify",
            "models": ["role_cvd_aib", "role_cvd_mvk", "role_ald_state"],
            "execution_scope": (
                "registered process models; not dynamically fitted because the current "
                "analysis input has no time axis"
            ),
            "state_unit": "1",
            "mvk_kinetic_coefficient_unit": "m^3/(kmol s)",
            "time_unit": "s",
        },
        {
            "layer": "transport_closure",
            "responsibility": "map reference-plane concentration to surface concentration",
            "models": [
                "bulk_as_surface_approximation",
                "direct_surface",
                "fit_scalar",
                "from_cfd_flux_sink",
            ],
            "supporting_models": [
                "stagnant_film",
                "rotating_disk",
                "bosanquet_diffusivity",
            ],
            "execution_scope": (
                f"current analysis mode is {selected_reaction_input['mode']}; direct_surface, "
                "fit_scalar, and from_cfd_flux_sink are simulation-pipeline inputs; "
                "supporting_models are registered km calculators and are not "
                "automatically dispatched"
            ),
            "concentration_unit": "kmol/m^3",
            "km_unit": "m/s",
            "flux_unit": "kmol/(m^2 s)",
        },
        {
            "layer": "spatial_residual_response",
            "responsibility": (
                "model transferable residual map shape after the chemical model is frozen"
            ),
            "models": ["none", "radial_quadratic", "radial_quartic"],
            "execution_scope": (
                f"current mode is {spatial_response_mode}; it does not participate in "
                "reaction-family or anonymous-role selection"
            ),
            "mean_rate_policy": "preserve each condition's chemical prediction mean",
            "temperature_policy": "no temperature spatial term; wafer temperature is uniform",
        },
        {
            "layer": "net_film_balance",
            "responsibility": "compose deposition, etch, and loss with one sign convention",
            "models": ["deposition_only", "dep_etch_loss"],
            "execution_scope": (
                "registered rate-composition utilities; separate from reaction-role selection"
            ),
            "input_unit": "nm/s",
            "output_unit": "nm/s",
        },
        {
            "layer": "selection_and_validation",
            "responsibility": "separate numerical fit, role evidence, and mechanism evidence",
            "models": ["inner_condition_cv", "exact_reduction", "outer_condition_cv", "fixed_holdout"],
            "execution_scope": "all four stages applied in the current analysis",
        },
    ]


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
    data: RoleFieldSet,
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


def _condition_mean_rate(data: RoleFieldSet, indices: np.ndarray | None = None) -> float:
    indices = np.arange(data.rate.size) if indices is None else indices
    labels, rates = data.condition_id[indices], data.rate[indices]
    return float(np.mean([np.mean(rates[labels == label]) for label in np.unique(labels)]))


def _fit_transfer(
    candidate: RoleResponseCandidate,
    data: RoleFieldSet,
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


def _predict_transfer(fit: TransferFit, data: RoleFieldSet) -> tuple[np.ndarray, np.ndarray]:
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
    fit: TransferFit | SurfaceKineticFit, data: RoleFieldSet,
) -> tuple[np.ndarray, np.ndarray | dict[str, np.ndarray]]:
    if isinstance(fit, SurfaceKineticFit):
        return predict_surface_kinetic(fit, data)
    return _predict_transfer(fit, data)


def _blocked_predictions(
    candidate: RoleResponseCandidate,
    data: RoleFieldSet,
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
    candidate: RoleResponseCandidate, data: RoleFieldSet, *, regularization: float = 0.0,
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
    """Return structural and practical parameter-resolution warnings."""
    if candidate.model_id == "baseline":
        return []
    return list(parameter_design_diagnostics(design)["reasons"])


def _ranking_for_training(
    train_cases: list[ConditionCase],
    *, response_structure: str = "shared",
) -> tuple[list[dict[str, Any]], dict[str, TransferFit], RoleFieldSet, np.ndarray, np.ndarray]:
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
        # no-refit evaluation of the selected model. _split_evaluation supplies
        # the fixed outer condition for that purpose.
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
        design_diagnostics = parameter_design_diagnostics(fit.design)
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
                "fit_diagnostics": {
                    "identifiability": {
                        "assessed": True,
                        "degeneracy_warning": bool(reasons),
                        **design_diagnostics,
                    }
                },
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
                "design_condition_number": design_diagnostics["condition_number"],
                "design_full_rank": design_diagnostics["full_rank"],
                "design_max_abs_parameter_correlation": design_diagnostics[
                    "max_abs_parameter_correlation"
                ],
                "parameter_identifiability_status": (
                    "sufficient" if candidate.model_id == "baseline" else design_diagnostics["status"]
                ),
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
    data: RoleFieldSet,
    optimization: SurfaceOptimizationSettings,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Condition CV for a fixed physical reduction with fold-local references."""
    prediction = np.full(data.rate.shape, np.nan, dtype=float)
    all_indices = np.arange(data.rate.size, dtype=int)

    def fit(remaining: tuple[int, ...]) -> SurfaceKineticFit:
        train_idx = all_indices[np.isin(data.condition_id, remaining)]
        return fit_surface_kinetic(
            candidate, data, train_idx, optimization=optimization
        )

    def evaluate(
        fitted: SurfaceKineticFit, held_out: int, remaining: tuple[int, ...]
    ) -> dict[str, Any]:
        train_idx = all_indices[np.isin(data.condition_id, remaining)]
        valid_idx = all_indices[data.condition_id == held_out]
        prediction[valid_idx] = fitted.prediction[valid_idx]
        metrics = _transfer_metrics(
            data.rate[valid_idx], fitted.prediction[valid_idx],
            _condition_mean_rate(data, train_idx),
        )
        return {
            **metrics,
            "condition": int(held_out),
            "weight": 1.0,
            "quantity": "deposition_rate",
            "unit": "nm/s",
            "refit_score": fitted.objective_value,
            "effect_groups": fitted.effect_groups,
            "effective_roles": fitted.effective_roles,
            "effect_scopes": fitted.effect_scopes,
            "response_structure": fitted.response_structure,
            "regularization": 0.0,
            "common_total_order": float("nan"),
            "within_total_order": float("nan"),
            "max_effect_coefficient": max(fitted.shape_parameters.values(), default=0.0),
            "boundary_parameters": list(fitted.boundary_parameters),
        }

    rows = condition_refits(data.case_ids, fit, evaluate)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Condition-holdout CV failed for {candidate.model_id}")
    return prediction, rows


def _surface_blocked_predictions(
    candidate: SurfaceKineticCandidate,
    data: RoleFieldSet,
    groups: np.ndarray,
    reference_concentrations: dict[str, float],
    optimization: SurfaceOptimizationSettings,
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
            optimization=optimization,
        )
        prediction[valid_idx] = fitted.prediction[valid_idx]
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Blocked CV failed for {candidate.model_id}")
    return prediction


def _surface_ranking_for_training(
    train_cases: list[ConditionCase],
    *,
    families: tuple[str, ...] = (AIB_QSS,),
    candidate_id: str | None = None,
    optimization: SurfaceOptimizationSettings | None = None,
    reaction_input_mode: str = "bulk_as_surface",
    record_optimization_history: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, SurfaceKineticFit], RoleFieldSet, np.ndarray, np.ndarray]:
    """Rank site-balance reductions using one declared whole-wafer loss."""
    optimization = optimization or SurfaceOptimizationSettings()
    data = _combine_cases(train_cases)
    indices = np.arange(data.rate.size, dtype=int)
    candidates = enumerate_surface_kinetic_candidates(
        data.species,
        include_boundaries=True,
        families=families,
        available_inputs=data.available_inputs(),
        transport_modes=(reaction_input_mode,),
    )
    if candidate_id is not None:
        candidates = [candidate for candidate in candidates if candidate.model_id == candidate_id]
        if not candidates:
            raise ValueError(
                f"Candidate {candidate_id!r} is not applicable to the supplied fields and model families"
            )
    angular_groups = _angular_groups(data.xyz[:, :2])
    radial_groups = _radial_groups(data.xyz[:, :2])
    fits: dict[str, SurfaceKineticFit] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        condition_prediction, condition_rows = _surface_condition_holdout_predictions(
            candidate, data, optimization
        )
        fit = fit_surface_kinetic(
            candidate,
            data,
            indices,
            optimization=optimization,
            record_optimization_history=record_optimization_history,
        )
        fits[candidate.model_id] = fit
        in_sample = _metrics(data.rate, fit.prediction)
        condition = _metrics(data.rate, condition_prediction)
        design_diagnostics = parameter_design_diagnostics(fit.design)
        reasons = _eligibility_reasons(candidate, fit.design)
        if fit.boundary_parameters:
            reasons.append(
                "shape optimum reached numerical search boundary: "
                + ", ".join(fit.boundary_parameters)
            )
        rows.append({
            "_candidate": candidate,
            "model_id": candidate.model_id,
            "role_model_id": candidate.model_id,
            "model_family": candidate.model_family,
            "equation_family": (
                "observation_baseline"
                if candidate.class_id in {"baseline", "total_power"}
                else candidate.family
            ),
            "reduction_id": candidate.reduction_id,
            "transport_mode": candidate.transport_mode,
            "reaction_input": data.reaction_input_metadata(candidate.transport_mode),
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
            "reduced_model_effects": {
                reduction.model_id: list(
                    reduction_removed_effects(candidate, reduction)
                )
                for reduction in candidate.reductions()
            },
            "effect_basis": "declared_state_model_roles",
            "best_score": fit.objective_value,
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
                    **design_diagnostics,
                }
            },
            "regularization": 0.0,
            "regularization_scores": [],
            "common_total_order": float("nan"),
            "within_total_order": float("nan"),
            "reference_rate_nm_s": fit.reference_rate_nm_s,
            "reference_reaction_inputs": fit.reference_concentrations,
            "reaction_input_unit": data.reaction_input_metadata(
                candidate.transport_mode
            )["unit"],
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
            "design_condition_number": design_diagnostics["condition_number"],
            "design_full_rank": design_diagnostics["full_rank"],
            "design_max_abs_parameter_correlation": design_diagnostics[
                "max_abs_parameter_correlation"
            ],
            "parameter_identifiability_status": design_diagnostics["status"],
            "eligible_for_adoption": bool(np.isfinite(condition_prediction).all()),
            "ineligibility_reasons": "",
            "design_identifiable": not reasons,
            "design_information": "; ".join(reasons),
            "optimizer_method": fit.optimizer_method,
            "optimizer_trial_count": fit.optimizer_trial_count,
            "loss_name": fit.loss_name,
            "_optimization_history": fit.optimization_history,
        })
    _annotate_surface_candidates(rows, data)
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
    _finalize_surface_evidence(rows)

    selected = next(row for row in rows if row.get("adoption_rank") == 1)
    selected_fit = fits[str(selected["role_model_id"])]
    angular_prediction = _surface_blocked_predictions(
        selected_fit.candidate,
        data,
        angular_groups,
        selected_fit.reference_concentrations,
        optimization,
    )
    radial_prediction = _surface_blocked_predictions(
        selected_fit.candidate,
        data,
        radial_groups,
        selected_fit.reference_concentrations,
        optimization,
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

    baseline_row = next(
        (row for row in rows if row["class_id"] == "baseline"), None
    )
    baseline_cv = (
        float(baseline_row["condition_cv_rmse_nm_s"])
        if baseline_row is not None
        else float("nan")
    )
    for row in rows:
        row["condition_cv_improvement_vs_baseline"] = (
            baseline_cv - float(row["condition_cv_rmse_nm_s"])
        ) / max(baseline_cv, _EPS)
    return rows, fits, data, angular_groups, radial_groups


def _select_model(ranking: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    selected = next(row for row in ranking if row.get("adoption_rank") == 1)
    equivalent = [row["role_model_id"] for row in ranking if row.get("equivalent_to_best", False)]
    if str(selected.get("model_family", "")).startswith("surface_"):
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


def _extrapolation_summary(train: RoleFieldSet, test: RoleFieldSet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = {"total_concentration": (train.total_concentration, test.total_concentration)}
    fields.update(
        {
            f"concentration_{name}": (train.bulk_concentrations[name], test.bulk_concentrations[name])
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


def _selection_structure(selection: dict[str, Any]) -> dict[str, Any]:
    """Return the discrete choices that define one refittable model structure."""

    return {
        "role_model_id": str(selection["selected_role_model_id"]),
        "equation_family": str(selection["selected_equation_family"]),
        "transport_mode": str(selection.get("transport_mode", "empirical")),
        "response_structure": str(selection["response_structure"]),
        "regularization": float(selection["regularization"]),
    }


def _model_structure_uncertainty(
    train: RoleFieldSet,
    test: RoleFieldSet,
    selections: Iterable[dict[str, Any]],
    *,
    response_model: str,
    selected_prediction: np.ndarray,
    model_families: tuple[str, ...] | None = None,
    candidate_id: str | None = None,
    surface_optimization: SurfaceOptimizationSettings | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refit structures selected across outer folds on one identification set.

    The resulting range is a selection-sensitivity envelope, not a confidence
    interval. Numerically identical predictions are collapsed so exact model
    symmetries cannot inflate the reported number of alternatives.
    """

    surface = _uses_surface_response(response_model)
    if surface:
        candidates: Iterable[SurfaceKineticCandidate | RoleResponseCandidate] = (
            enumerate_surface_kinetic_candidates(
                train.species,
                include_boundaries=True,
                families=_surface_families(response_model, model_families),
                available_inputs=train.available_inputs(),
                transport_modes=train.available_transport_modes(),
            )
        )
    else:
        candidates = enumerate_role_response_candidates(
            train.species, include_reductions=True
        )
    candidate_lookup = {candidate.model_id: candidate for candidate in candidates}
    if candidate_id is not None:
        candidate_lookup = {
            key: value for key, value in candidate_lookup.items() if key == candidate_id
        }
    indices = np.arange(train.rate.size, dtype=int)

    structures: dict[str, tuple[dict[str, Any], int]] = {}
    for selection in selections:
        structure = _selection_structure(selection)
        key = json.dumps(structure, ensure_ascii=True, sort_keys=True)
        previous = structures.get(key)
        structures[key] = (structure, 1 if previous is None else previous[1] + 1)

    members: list[dict[str, Any]] = []
    equality_scale = max(float(np.max(np.abs(test.rate))), _EPS)
    equality_atol = 128.0 * np.finfo(float).eps * equality_scale
    for structure, selection_count in structures.values():
        candidate = candidate_lookup.get(structure["role_model_id"])
        if candidate is None:
            raise RuntimeError(
                f"Selected structure is not available in the current candidate registry: {structure}"
            )
        if surface:
            assert isinstance(candidate, SurfaceKineticCandidate)
            fitted: TransferFit | SurfaceKineticFit = fit_surface_kinetic(
                candidate,
                train,
                indices,
                optimization=surface_optimization,
            )
        else:
            assert isinstance(candidate, RoleResponseCandidate)
            fitted = _fit_transfer(
                candidate,
                train,
                indices,
                regularization=structure["regularization"],
                response_structure=structure["response_structure"],
            )
        prediction, _ = _predict_response(fitted, test)
        equivalent = next(
            (
                member
                for member in members
                if np.allclose(
                    member["prediction"], prediction, rtol=128.0 * np.finfo(float).eps,
                    atol=equality_atol,
                )
            ),
            None,
        )
        if equivalent is None:
            members.append(
                {
                    "prediction": prediction,
                    "structures": [structure],
                    "selection_count": selection_count,
                }
            )
        else:
            equivalent["structures"].append(structure)
            equivalent["selection_count"] += selection_count

    matrix = np.column_stack([member["prediction"] for member in members])
    lower = np.min(matrix, axis=1)
    upper = np.max(matrix, axis=1)
    mean = np.mean(matrix, axis=1)
    width = upper - lower
    rows = [
        {
            "condition": int(test.condition_id[index]),
            "x": float(test.xyz[index, 0]),
            "y": float(test.xyz[index, 1]),
            "z": float(test.xyz[index, 2]),
            "measured_rate_nm_s": float(test.rate[index]),
            "selected_model_prediction_nm_s": float(selected_prediction[index]),
            "structure_mean_prediction_nm_s": float(mean[index]),
            "structure_min_prediction_nm_s": float(lower[index]),
            "structure_max_prediction_nm_s": float(upper[index]),
            "structure_envelope_width_nm_s": float(width[index]),
            "distinct_prediction_count": len(members),
        }
        for index in range(test.rate.size)
    ]
    return rows, {
        "interpretation": (
            "Prediction range after refitting the distinct model structures selected "
            "across outer condition folds on the primary identification conditions; "
            "this is selection sensitivity, not a confidence interval."
        ),
        "selection_count": int(sum(member["selection_count"] for member in members)),
        "distinct_structure_count": len(structures),
        "distinct_prediction_count": len(members),
        "members": [
            {
                "selection_count": int(member["selection_count"]),
                "equivalent_structures": member["structures"],
            }
            for member in members
        ],
        "mean_envelope_width_nm_s": float(np.mean(width)),
        "max_envelope_width_nm_s": float(np.max(width)),
        "mean_envelope_width_relative_to_test_mean": float(
            np.mean(width) / max(abs(float(np.mean(test.rate))), _EPS)
        ),
    }


def _bootstrap_coefficients(
    fit: TransferFit | SurfaceKineticFit,
    data: RoleFieldSet,
    angular_groups: np.ndarray,
    *,
    samples: int,
    seed: int,
    surface_optimization: SurfaceOptimizationSettings | None = None,
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
                optimization=surface_optimization,
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
                "reduction_id": fit.candidate.reduction_id,
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


def _split_evaluation(
    train_cases: list[ConditionCase],
    test_case: ConditionCase,
    *,
    response_structure: str = "shared",
    response_model: str = "empirical_power",
    model_families: tuple[str, ...] | None = None,
    candidate_id: str | None = None,
    surface_optimization: SurfaceOptimizationSettings | None = None,
    reaction_input_mode: str = "bulk_as_surface",
    record_optimization_history: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], TransferFit | SurfaceKineticFit, np.ndarray, RoleFieldSet, RoleFieldSet]:
    if response_model not in RESPONSE_MODELS:
        raise ValueError(f"response_model must be one of {RESPONSE_MODELS}")
    families = _surface_families(response_model, model_families)
    if families:
        ranking, fits, train, _, _ = _surface_ranking_for_training(
            train_cases,
            families=families,
            candidate_id=candidate_id,
            optimization=surface_optimization,
            reaction_input_mode=reaction_input_mode,
            record_optimization_history=record_optimization_history,
        )
    elif candidate_id is not None:
        raise ValueError("candidate_id applies only to response_model='surface_compare'")
    else:
        if reaction_input_mode != "bulk_as_surface":
            raise ValueError(
                "response_model='empirical_power' is defined only for bulk_concentration"
            )
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
    reaction_input_metadata = (
        train.reaction_input_metadata(selected_fit.candidate.transport_mode)
        if isinstance(selected_fit, SurfaceKineticFit)
        else {
            "mode": "bulk_as_surface",
            "quantity": "concentration",
            "location": "reference_plane",
            "unit": "kmol/m^3",
            "interpretation": "empirical bulk-concentration response",
        }
    )
    reference_input_total = (
        selected_fit.reference_input_total
        if isinstance(selected_fit, SurfaceKineticFit)
        else selected_fit.reference_total_concentration
    )
    reference_input_shares = (
        selected_fit.reference_input_shares
        if isinstance(selected_fit, SurfaceKineticFit)
        else selected_fit.reference_species_fractions
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
        "selected_equation_family": getattr(
            selected_fit.candidate, "family", selected_fit.response_structure
        ),
        "selected_applicability_status": selected_row.get(
            "applicability_status", "production"
        ),
        "selected_contrast_status": selected_row.get(
            "contrast_status", "not_assessed"
        ),
        "selected_distinguishable": bool(
            selected_row.get("distinguishable", False)
        ),
        "selected_supported_claims": selected_row.get("supported_claims", []),
        "selected_missing_evidence": selected_row.get("missing_evidence", []),
        "transport_mode": getattr(selected_fit.candidate, "transport_mode", "empirical"),
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
        "reaction_input_quantity": reaction_input_metadata["quantity"],
        "reaction_input_location": reaction_input_metadata["location"],
        "reaction_input_unit": reaction_input_metadata["unit"],
        "reference_reaction_input_total": reference_input_total,
        "reference_reaction_input_shares": reference_input_shares,
        "reference_total_concentration_kmol_m3": (
            reference_input_total
            if reaction_input_metadata["quantity"] == "concentration"
            else None
        ),
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
            ratios = case.bulk_concentrations[name] / base.bulk_concentrations[name]
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
    matrix = np.column_stack([combined.bulk_concentrations[name] for name in combined.species])
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


def _optimization_history_rows(
    ranking: list[dict[str, Any]],
    family_assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return convergence records for the best assignment in each equation family."""

    by_id = {str(row["model_id"]): row for row in ranking}
    rows: list[dict[str, Any]] = []
    for family in family_assessments:
        model_id = str(family.get("best_model_id", ""))
        candidate = by_id.get(model_id)
        if candidate is None:
            continue
        loss_name = str(candidate.get("loss_name", "mse"))
        for item in candidate.get("_optimization_history", ()):
            best_score = float(item["best_score"])
            best_error = (
                math.sqrt(max(best_score, 0.0))
                if loss_name
                in {"mse", "wafer_normalized_mse", "symmetric_normalized_mse"}
                else best_score
            )
            rows.append(
                {
                    "equation_family": family["equation_family"],
                    "model_id": model_id,
                    "role_A": candidate.get("role_A", ""),
                    "role_I": candidate.get("role_I", ""),
                    "role_B": candidate.get("role_B", ""),
                    "optimizer": candidate.get("optimizer_method", ""),
                    "loss": loss_name,
                    "trial": int(item["trial"]),
                    "objective": float(item["score"]),
                    "best_objective": best_score,
                    "best_error": best_error,
                    "best_error_name": (
                        "training_rmse_nm_s"
                        if loss_name == "mse"
                        else "normalized_error"
                    ),
                }
            )
    for candidate in ranking:
        candidate.pop("_optimization_history", None)
    return rows


def _best_family_role_rows(
    ranking: list[dict[str, Any]],
    family_assessments: list[dict[str, Any]],
    reaction_input_metadata: dict[str, str],
) -> list[dict[str, Any]]:
    """Describe which raw species fills each role in each best family fit."""

    by_id = {str(row["model_id"]): row for row in ranking}
    labels = {"A": "surface reactant", "I": "inhibitor", "B": "co-reactant"}
    rows: list[dict[str, Any]] = []
    for family in family_assessments:
        model_id = str(family.get("best_model_id", ""))
        candidate = by_id.get(model_id)
        if candidate is None:
            continue
        for role in ("A", "I", "B"):
            rows.append(
                {
                    "equation_family": family["equation_family"],
                    "model_id": model_id,
                    "role": role,
                    "reaction_role": labels[role],
                    "species": candidate.get(f"role_{role}", "") or "",
                    "reaction_input_quantity": reaction_input_metadata["quantity"],
                    "reaction_input_location": reaction_input_metadata["location"],
                    "reaction_input_unit": reaction_input_metadata["unit"],
                    "condition_cv_rmse_nm_s": candidate["condition_cv_rmse_nm_s"],
                }
            )
    return rows


def _condition_mean_input_correlation_rows(
    data: RoleFieldSet,
    reaction_input_mode: str,
) -> list[dict[str, Any]]:
    """Correlate log condition means of the selected reaction input."""

    inputs = data.reaction_inputs_for(reaction_input_mode)
    conditions = np.unique(data.condition_id)
    matrix = np.asarray(
        [
            [
                float(np.mean(np.asarray(inputs[species])[data.condition_id == condition]))
                for species in data.species
            ]
            for condition in conditions
        ],
        dtype=float,
    )
    matrix = np.log(np.maximum(matrix, np.finfo(float).tiny))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix, rowvar=False)
    metadata = data.reaction_input_metadata(reaction_input_mode)
    return [
        {
            "species_1": first,
            "species_2": second,
            "pearson_correlation": float(correlation[i, j]),
            "condition_count": int(conditions.size),
            "reaction_input_quantity": metadata["quantity"],
            "reaction_input_location": metadata["location"],
        }
        for i, first in enumerate(data.species)
        for j, second in enumerate(data.species)
    ]


def _reaction_state_summary_rows(
    prediction_rows: list[dict[str, Any]],
    fit: SurfaceKineticFit | None,
) -> list[dict[str, Any]]:
    """Summarize fitted site populations and pathway fractions on the holdout wafer."""

    if fit is None:
        return []
    family = get_surface_model_family(fit.candidate.family)
    site_fields = ["theta_free", *family.state_variables]
    if fit.candidate.I is None and "theta_I" in site_fields:
        site_fields.remove("theta_I")
    if fit.candidate.B is None and "theta_B" in site_fields:
        site_fields.remove("theta_B")
    field_labels = {
        "theta_free": "vacant sites",
        "theta_A": "adsorbed A",
        "theta_B": "adsorbed B",
        "theta_I": "sites blocked by I",
        "A": "A-only pathway",
        "AB": "A + B pathway",
    }
    fields = [
        (field, "site_fraction", field_labels[field]) for field in site_fields
    ]
    fields.extend(
        (f"path_{pathway}_fraction", "pathway_fraction", field_labels[pathway])
        for pathway in family.pathways
    )
    rows: list[dict[str, Any]] = []
    for field, quantity, label in fields:
        if not prediction_rows or field not in prediction_rows[0]:
            continue
        values = np.asarray([float(row[field]) for row in prediction_rows], dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        rows.append(
            {
                "quantity": quantity,
                "component": label,
                "mean_fraction": float(np.mean(values)),
                "minimum_fraction": float(np.min(values)),
                "maximum_fraction": float(np.max(values)),
            }
        )
    return rows


def _role_importance_and_stability_rows(
    sensitivity_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    *,
    held_out_rmse_nm_s: float,
) -> list[dict[str, Any]]:
    """Join prediction sensitivity with raw-species selection frequency."""

    rows: list[dict[str, Any]] = []
    denominator = max(float(held_out_rmse_nm_s), _EPS)
    for sensitivity in sensitivity_rows:
        role = str(sensitivity["role"])
        species = str(sensitivity["species"])
        selections = [str(row.get(f"selected_role_{role}") or "none") for row in split_rows]
        counts = {name: selections.count(name) for name in sorted(set(selections))}
        rms_change = float(sensitivity["rms_prediction_change_nm_s"])
        rows.append(
            {
                "role": role,
                "species": species,
                "selection_frequency": counts.get(species, 0) / max(len(selections), 1),
                "selection_count": counts.get(species, 0),
                "refit_count": len(selections),
                "alternative_assignments": json.dumps(counts, sort_keys=True),
                "rms_prediction_change_nm_s": rms_change,
                "held_out_rmse_nm_s": float(held_out_rmse_nm_s),
                "prediction_change_to_rmse_ratio": rms_change / denominator,
            }
        )
    return rows


def _best_family_diagnostic_rows(
    ranking: list[dict[str, Any]],
    family_assessments: list[dict[str, Any]],
    train: RoleFieldSet,
    test: RoleFieldSet,
    selected_fit: SurfaceKineticFit,
    selected_prediction: np.ndarray,
    optimization: SurfaceOptimizationSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refit the best assignment in each family for prediction and state comparison."""

    families = tuple(
        str(row["equation_family"])
        for row in family_assessments
        if row.get("best_model_id")
    )
    candidates = enumerate_surface_kinetic_candidates(
        train.species,
        include_boundaries=True,
        families=families,
        available_inputs=train.available_inputs(),
        transport_modes=(selected_fit.candidate.transport_mode,),
    )
    candidate_by_id = {candidate.model_id: candidate for candidate in candidates}
    ranking_by_id = {str(row["model_id"]): row for row in ranking}
    indices = np.arange(train.rate.size, dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    selected_rmse = float(_metrics(test.rate, selected_prediction)["rmse_nm_s"])
    for assessment in family_assessments:
        model_id = str(assessment.get("best_model_id", ""))
        candidate = candidate_by_id.get(model_id)
        ranking_row = ranking_by_id.get(model_id)
        if candidate is None or ranking_row is None:
            continue
        fit = (
            selected_fit
            if model_id == selected_fit.candidate.model_id
            else fit_surface_kinetic(
                candidate,
                train,
                indices,
                optimization=optimization,
            )
        )
        prediction, state = predict_surface_kinetic(fit, test)
        metrics = _metrics(test.rate, prediction)
        difference = np.asarray(prediction) - np.asarray(selected_prediction)
        prediction_rows.append(
            {
                "equation_family": candidate.family,
                "model_id": model_id,
                "role_A": candidate.A or "",
                "role_I": candidate.I or "",
                "role_B": candidate.B or "",
                "condition_cv_rmse_nm_s": float(
                    ranking_row["condition_cv_rmse_nm_s"]
                ),
                "held_out_rmse_nm_s": float(metrics["rmse_nm_s"]),
                "rms_difference_from_selected_nm_s": float(
                    np.sqrt(np.mean(np.square(difference)))
                ),
                "rms_difference_to_selected_rmse_ratio": float(
                    np.sqrt(np.mean(np.square(difference)))
                    / max(selected_rmse, _EPS)
                ),
                "maximum_difference_from_selected_nm_s": float(
                    np.max(np.abs(difference))
                ),
            }
        )
        family = get_surface_model_family(candidate.family)
        state_fields = ["theta_free", *family.state_variables]
        if candidate.I is None and "theta_I" in state_fields:
            state_fields.remove("theta_I")
        if candidate.B is None and "theta_B" in state_fields:
            state_fields.remove("theta_B")
        state_fields.extend(f"path_{pathway}_fraction" for pathway in family.pathways)
        for field in state_fields:
            values = np.asarray(state.get(field, []), dtype=float)
            if values.size == 0 or not np.all(np.isfinite(values)):
                continue
            state_rows.append(
                {
                    "equation_family": candidate.family,
                    "model_id": model_id,
                    "quantity": (
                        "pathway_fraction"
                        if field.startswith("path_")
                        else "site_fraction"
                    ),
                    "component": field,
                    "mean_fraction": float(np.mean(values)),
                    "minimum_fraction": float(np.min(values)),
                    "maximum_fraction": float(np.max(values)),
                }
            )
    return prediction_rows, state_rows


def analyze_cvd_multicond_case(
    *,
    data_dir: Path,
    response_structure: str = "shared",
    response_model: str = "surface_compare",
    train_case_ids: tuple[int, ...] = (1, 2, 4, 5),
    test_case_id: int = 3,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 123,
    application: dict[str, Any] | None = None,
    conditions_file: Path | None = None,
    model_families: tuple[str, ...] | None = None,
    candidate_id: str | None = None,
    surface_loss: str = "mse",
    surface_sampler: str = "pattern",
    surface_trials: int = 256,
    surface_sampler_options: Mapping[str, Any] | None = None,
    edge_uncertainty_ratio: float = 1.0,
    radial_uncertainty_power: float = 2.0,
    reaction_input: str = "bulk_concentration",
    spatial_response: str = "none",
    wafer_temperature_k: float | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_response = str(spatial_response).strip().lower()
    if spatial_response not in SPATIAL_RESPONSE_MODES:
        raise ValueError(
            f"spatial_response must be one of {SPATIAL_RESPONSE_MODES}"
        )
    if wafer_temperature_k is not None and (
        not np.isfinite(wafer_temperature_k) or wafer_temperature_k <= 0.0
    ):
        raise ValueError("wafer_temperature_k must be positive when supplied")
    all_case_ids = tuple(sorted({*train_case_ids, int(test_case_id)}))
    if len(train_case_ids) < 2:
        raise ValueError("At least two identification conditions are required")
    if len(set(train_case_ids)) != len(train_case_ids) or test_case_id in train_case_ids:
        raise ValueError("Identification condition IDs must be distinct and exclude the test condition")
    paths = _condition_paths(data_dir, all_case_ids, conditions_file)
    cases = [_load_case(case_id, *paths[case_id]) for case_id in all_case_ids]
    case_lookup = {case.case_id: case for case in cases}
    all_fields = _combine_cases(cases)
    reaction_input_mode = all_fields.resolve_reaction_input_mode(reaction_input)
    grid_alignment = _grid_alignment(cases)
    train_cases = [case_lookup[case_id] for case_id in train_case_ids]
    test_case = case_lookup[test_case_id]
    surface_optimization = SurfaceOptimizationSettings(
        loss_name=surface_loss,
        sampler=surface_sampler,
        trials=surface_trials,
        seed=seed,
        sampler_options=dict(surface_sampler_options or {}),
        edge_uncertainty_ratio=edge_uncertainty_ratio,
        radial_power=radial_uncertainty_power,
    )

    primary, ranking, selected_fit, test_prediction, train, test = _split_evaluation(
        train_cases,
        test_case,
        response_structure=response_structure,
        response_model=response_model,
        model_families=model_families,
        candidate_id=candidate_id,
        surface_optimization=surface_optimization,
        reaction_input_mode=reaction_input_mode,
        record_optimization_history=True,
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
        surface_optimization=surface_optimization,
    )
    coefficient_rows = _coefficient_rows(selected_fit, bootstrap)

    test_prediction, test_diagnostics = _predict_response(selected_fit, test)
    train_chemical_prediction, _ = _predict_response(selected_fit, train)
    spatial_fit = fit_spatial_response(
        spatial_response, train, train_chemical_prediction
    )
    spatial_prediction, spatial_factor = apply_spatial_response(
        spatial_fit, test, test_prediction
    )
    chemical_test_metrics = _transfer_metrics(
        test.rate, test_prediction, _condition_mean_rate(train)
    )
    spatial_test_metrics = _transfer_metrics(
        test.rate, spatial_prediction, _condition_mean_rate(train)
    )
    primary.update(
        {
            "chemical_test_rmse_nm_s": chemical_test_metrics["rmse_nm_s"],
            "chemical_test_centered_spatial_r2": chemical_test_metrics[
                "centered_spatial_r2"
            ],
            "spatial_response_model": spatial_fit.mode,
            "spatial_response_test_rmse_nm_s": spatial_test_metrics["rmse_nm_s"],
            "spatial_response_test_centered_spatial_r2": spatial_test_metrics[
                "centered_spatial_r2"
            ],
            "spatial_response_test_spatial_correlation": spatial_test_metrics[
                "spatial_correlation"
            ],
            "spatial_response_preserves_chemical_mean": True,
        }
    )
    selected_input_metadata = (
        test.reaction_input_metadata(selected_fit.candidate.transport_mode)
        if isinstance(selected_fit, SurfaceKineticFit)
        else {
            "mode": "bulk_as_surface",
            "quantity": "concentration",
            "location": "reference_plane",
            "unit": "kmol/m^3",
            "interpretation": "empirical bulk-concentration response",
        }
    )
    model_input_concentrations = (
        test.reaction_inputs_for(selected_fit.candidate.transport_mode)
        if isinstance(selected_fit, SurfaceKineticFit)
        else test.bulk_concentrations
    )
    reference_input_shares = (
        selected_fit.reference_input_shares
        if isinstance(selected_fit, SurfaceKineticFit)
        else selected_fit.reference_species_fractions
    )
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
            "transport_mode": getattr(
                selected_fit.candidate, "transport_mode", "empirical"
            ),
            "reaction_input_quantity": selected_input_metadata["quantity"],
            "reaction_input_location": selected_input_metadata["location"],
            "reaction_input_unit": selected_input_metadata["unit"],
            "x": float(test.xyz[index, 0]),
            "y": float(test.xyz[index, 1]),
            "z": float(test.xyz[index, 2]),
            "measured_rate_nm_s": float(test.rate[index]),
            "predicted_rate_nm_s": float(test_prediction[index]),
            "chemical_prediction_nm_s": float(test_prediction[index]),
            "spatial_response_prediction_nm_s": float(spatial_prediction[index]),
            "spatial_response_factor": float(spatial_factor[index]),
            "spatial_response_residual_nm_s": float(
                spatial_prediction[index] - test.rate[index]
            ),
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
            "reduction_id": getattr(selected_fit.candidate, "reduction_id", "empirical"),
        }
        for name in test.species:
            row[f"bulk_concentration_{name}_kmol_m3"] = float(
                test.bulk_concentrations[name][index]
            )
            if selected_input_metadata["quantity"] == "transport_capacity_flux":
                row[f"model_input_flux_{name}_kmol_m2_s"] = float(
                    model_input_concentrations[name][index]
                )
            else:
                row[f"model_input_concentration_{name}_kmol_m3"] = float(
                    model_input_concentrations[name][index]
                )
            if name in test.surface_concentrations:
                row[f"surface_concentration_{name}_kmol_m3"] = float(
                    test.surface_concentrations[name][index]
                )
            row[f"bulk_fraction_{name}"] = float(test.species_fractions[name][index])
            row[f"fraction_{name}"] = float(test.species_fractions[name][index])
            row[f"reference_reaction_input_share_{name}"] = float(
                reference_input_shares[name]
            )
            if selected_input_metadata["mode"] == "bulk_as_surface":
                row[f"reference_fraction_{name}"] = float(reference_input_shares[name])
            if isinstance(selected_fit, SurfaceKineticFit):
                reference_label = (
                    f"reference_flux_{name}_kmol_m2_s"
                    if selected_input_metadata["quantity"] == "transport_capacity_flux"
                    else f"reference_concentration_{name}_kmol_m3"
                )
                row[reference_label] = float(selected_fit.reference_concentrations[name])
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
        (
            split_result,
            split_ranking,
            split_fit,
            split_chemical_prediction,
            split_train_fields,
            split_test_fields,
        ) = _split_evaluation(split_train, case_lookup[held_out],
                                                                  response_structure=response_structure,
                                                                  response_model=response_model,
                                                                  model_families=model_families,
                                                                  candidate_id=candidate_id,
                                                                  surface_optimization=surface_optimization,
                                                                  reaction_input_mode=reaction_input_mode)
        split_train_prediction, _ = _predict_response(split_fit, split_train_fields)
        split_spatial_fit = fit_spatial_response(
            spatial_response, split_train_fields, split_train_prediction
        )
        split_spatial_prediction, _ = apply_spatial_response(
            split_spatial_fit, split_test_fields, split_chemical_prediction
        )
        split_spatial_metrics = _transfer_metrics(
            split_test_fields.rate,
            split_spatial_prediction,
            _condition_mean_rate(split_train_fields),
        )
        split_result.update(
            {
                "chemical_test_centered_spatial_r2": split_result[
                    "test_centered_spatial_r2"
                ],
                "spatial_response_model": split_spatial_fit.mode,
                "spatial_response_test_rmse_nm_s": split_spatial_metrics[
                    "rmse_nm_s"
                ],
                "spatial_response_test_relative_rmse": split_spatial_metrics[
                    "relative_rmse_vs_test_mean"
                ],
                "spatial_response_test_centered_spatial_r2": split_spatial_metrics[
                    "centered_spatial_r2"
                ],
                "spatial_response_test_spatial_correlation": split_spatial_metrics[
                    "spatial_correlation"
                ],
            }
        )
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
    role_stability_warning = bool(stability["warning"])
    selected_families = [str(row["selected_equation_family"]) for row in split_rows]
    family_counts = {
        name: selected_families.count(name) for name in sorted(set(selected_families))
    }
    family_warning = len(family_counts) > 1
    selected_structures = [
        json.dumps(_selection_structure(row), ensure_ascii=True, sort_keys=True)
        for row in split_rows
    ]
    structure_counts = {
        name: selected_structures.count(name)
        for name in sorted(set(selected_structures))
    }
    structure_warning = len(structure_counts) > 1
    stability["equation_family_counts"] = family_counts
    stability["equation_family_warning"] = family_warning
    stability["model_structure_counts"] = structure_counts
    stability["model_structure_warning"] = structure_warning
    stability["warning"] = bool(stability["warning"] or structure_warning)
    equation_family_assessments = (
        _equation_family_assessments(
            ranking,
            split_rows,
            _surface_families(response_model, model_families),
            train.available_inputs(),
        )
        if _uses_surface_response(response_model)
        else []
    )
    reaction_mechanism_assessments = (
        _reaction_mechanism_assessments(ranking, equation_family_assessments)
        if _uses_surface_response(response_model)
        else []
    )
    optimization_history_rows = _optimization_history_rows(
        ranking, equation_family_assessments
    )
    family_role_rows = _best_family_role_rows(
        ranking, equation_family_assessments, selected_input_metadata
    )
    input_correlation_rows = _condition_mean_input_correlation_rows(
        all_fields, reaction_input_mode
    )
    if isinstance(selected_fit, SurfaceKineticFit):
        role_sensitivity_rows = role_input_sensitivity_rows(selected_fit, all_fields)
        role_response_rows = role_response_curve_rows(selected_fit, all_fields)
        parameter_sensitivity_diagnostic_rows = parameter_sensitivity_rows(selected_fit)
        parameter_loss_rows = parameter_loss_slice_rows(
            selected_fit,
            train,
            optimization=surface_optimization,
        )
        family_prediction_rows, family_state_rows = _best_family_diagnostic_rows(
            ranking,
            equation_family_assessments,
            train,
            test,
            selected_fit,
            test_prediction,
            surface_optimization,
        )
    else:
        role_sensitivity_rows = []
        role_response_rows = []
        parameter_sensitivity_diagnostic_rows = []
        parameter_loss_rows = []
        family_prediction_rows = []
        family_state_rows = []
    role_importance_rows = _role_importance_and_stability_rows(
        role_sensitivity_rows,
        split_rows,
        held_out_rmse_nm_s=float(primary["test_rmse_nm_s"]),
    )
    reaction_state_rows = _reaction_state_summary_rows(
        test_prediction_rows,
        selected_fit if isinstance(selected_fit, SurfaceKineticFit) else None,
    )
    workflow_layers = _workflow_layers(
        equation_family_assessments,
        selected_reaction_input=selected_input_metadata,
        spatial_response_mode=spatial_fit.mode,
    )
    role_stability_rows.extend(
        {
            "slot": "equation_family",
            "species": name,
            "count": count,
            "frequency": count / len(selected_families),
            "refit_count": len(selected_families),
            "basis": "outer_condition_cv_selection",
        }
        for name, count in family_counts.items()
    )
    role_stability_rows.extend(
        {
            "slot": "model_structure",
            "species": name,
            "count": count,
            "frequency": count / len(selected_structures),
            "refit_count": len(selected_structures),
            "basis": "outer_condition_cv_selection",
        }
        for name, count in structure_counts.items()
    )
    split_role_rows = [
        {
            "held_out_condition": row["test_case"],
            "identification_conditions": row["train_cases"],
            "selected_model": row["selected_model"],
            "selected_role_model_id": row["selected_role_model_id"],
            "selected_equation_family": row["selected_equation_family"],
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

    extrapolation_rows = _extrapolation_summary(train, test)
    model_uncertainty_rows, model_uncertainty = _model_structure_uncertainty(
        train,
        test,
        (primary, *split_rows),
        response_model=response_model,
        selected_prediction=test_prediction,
        model_families=model_families,
        candidate_id=candidate_id,
        surface_optimization=surface_optimization,
    )
    quality_rows = _condition_quality_rows(cases)
    scaling_rows = _scaling_rows(cases)
    correlation_rows = _correlation_rows(cases)

    all_concentrations = np.vstack(
        [np.column_stack([case.bulk_concentrations[name] for name in case.species]) for case in cases]
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
        ranking, score_epsilon=0.0, role_stability_warning=role_stability_warning,
        model_structure_stability_warning=structure_warning,
        parameter_identifiability_warning=not selected_row["design_identifiable"],
        application=application,
    )
    assessment = role_summary_rows[0]
    role_ambiguous = assessment["role_support"] == "unresolved"
    cv_supported = selected_row.get("validation_skill", 0.0) > 0.0
    chemical_spatial_supported = primary["test_centered_spatial_r2"] > 0.0
    decision = assessment["decision"]
    validity = {
        "overall_assessment": "needs_revision" if decision == "reject_prediction" else "share_with_caveats",
        "condition_mean_transfer_assessment": ("improves_constant_baseline" if
            float(np.mean(test_prediction - test.rate))**2 < (float(np.mean(test.rate)) - _condition_mean_rate(train))**2
            else "not_supported"),
        "chemical_spatial_prediction": (
            "improves_centered_constant_baseline"
            if chemical_spatial_supported
            else "not_supported"
        ),
        "spatial_response_assessment": (
            "not_enabled"
            if spatial_fit.mode == "none"
            else "improves_chemical_spatial_prediction"
            if spatial_test_metrics["centered_spatial_r2"]
            > chemical_test_metrics["centered_spatial_r2"]
            else "no_holdout_improvement"
        ),
        "species_role_assessment": assessment["role_support"],
        "species_role_adoption": decision,
        "composition_role_cross_condition_validation": "unresolved" if role_ambiguous else "see_condition_refits",
        "equation_family_cross_condition_validation": (
            "unstable" if family_warning else "stable"
        ),
        "model_structure_cross_condition_validation": (
            "unstable" if structure_warning else "stable"
        ),
        "condition_holdout_cv_assessment": "improves_constant_baseline" if cv_supported else "not_supported",
        "elementary_kinetics_validated": False,
        "test_was_refit": False,
        "test_relative_rmse": primary["test_relative_rmse_vs_test_mean"],
        "test_spatial_r2": primary["test_centered_spatial_r2"],
        "test_range_capture_fraction": primary["test_range_capture_fraction"],
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
        "numerical_prediction_winner": primary["selected_model"],
        "adopted_model": (
            primary["selected_model"] if decision == "adopt_candidate" else None
        ),
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

    chemical_spatial_transfer_supported = bool(
        split_rows
        and all(float(row["test_centered_spatial_r2"]) > 0.0 for row in split_rows)
    )
    spatial_response_transfer_supported = bool(
        spatial_fit.mode != "none"
        and split_rows
        and all(
            float(row["spatial_response_test_centered_spatial_r2"]) > 0.0
            for row in split_rows
        )
    )
    role_supported = bool(
        assessment["role_support"] != "unresolved"
        and selected_row.get("contrast_status") == "sufficient"
        and not role_stability_warning
    )
    capability_assessments, data_requirement_rows = build_capability_requirements(
        spatial_supported=(
            spatial_response_transfer_supported
            if spatial_fit.mode != "none"
            else chemical_spatial_transfer_supported
        ),
        role_supported=role_supported,
        parameter_identifiability_status=str(
            selected_row.get("parameter_identifiability_status", "not_assessed")
        ),
        concentration_location=str(
            getattr(selected_fit.candidate, "transport_mode", "empirical")
        ),
        has_measurement_uncertainty=train.rate_sigma is not None,
        family_stable=not family_warning,
    )

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
                       "loss": (surface_optimization.loss_name if _uses_surface_response(response_model)
                                 else "condition-balanced squared log-rate error"),
                       "sampler": (surface_optimization.sampler if _uses_surface_response(response_model)
                                   else "analytic_regularized_linear_fit"),
                       "sampler_trials": (
                           surface_optimization.trials
                           if _uses_surface_response(response_model)
                           and surface_optimization.sampler != "pattern"
                           else None
                       ),
                       "edge_uncertainty_ratio": (surface_optimization.edge_uncertainty_ratio
                                                  if _uses_surface_response(response_model) else None),
                       "radial_uncertainty_power": (surface_optimization.radial_power
                                                    if _uses_surface_response(response_model) else None),
                       "penalty": ("none; exact site-balance reductions and condition CV control complexity"
                                   if _uses_surface_response(response_model) else
                                   "shared: lambda*||beta||^2; within_between: lambda*((||between||^2+||within||^2)/2+||between-within||^2); intercept unpenalized"),
                       "regularization_grid": [] if _uses_surface_response(response_model) else list(REGULARIZATION_GRID),
                       "response_structure_policy": "site_balance_qss_family_comparison" if response_model == "surface_compare" else response_structure,
                       "response_structures": (list(_surface_families(response_model, model_families)) if _uses_surface_response(response_model) else
                                               list(RESPONSE_STRUCTURES) if response_structure == "select" else [response_structure]),
                       "candidate_filter": candidate_id,
                       "map_centering": ("not used; absolute concentrations are normalized by identification-data references"
                                         if _uses_surface_response(response_model) else
                                         "full supplied Fluent input map, independent of measured rates and prediction batch"),
                       "selection": ("equation-family/role/reduction fitted by the named loss; ranked by no-refit condition CV RMSE in rate units"
                                     if _uses_surface_response(response_model) else
                                     "joint role/response-structure/regularization inner condition CV MSE in rate units"),
                       "reaction_input": selected_input_metadata,
                       "spatial_response": {
                           "mode": spatial_fit.mode,
                           "fit_stage": "after chemical-model selection",
                           "participates_in_chemical_selection": False,
                       }},
        "analysis_type": (
            f"{len(train_case_ids)}-condition identification plus one-condition no-refit test"
        ),
        "surface_state_assumption": (
            "quasi-steady site balance with observable lumped parameters, uniform wafer temperature, and no fitted condition-specific offsets"
            if _uses_surface_response(response_model) else
            "effective response coefficients transfer across conditions; no fitted condition-specific offsets"
        ),
        "primary_split": primary,
        # Compatibility name for the model used to make the fixed prediction.
        # Adoption remains an independent evidence decision in ``validity``.
        "selected_model": {
            "model_id": primary["selected_model"],
            "equation_family": primary["selected_equation_family"],
            "formula": _fit_formula(selected_fit),
            "reference_rate_nm_s": selected_fit.reference_rate_nm_s,
            "reference_total_concentration_kmol_m3": (
                selected_fit.reference_total_concentration
                if selected_input_metadata["quantity"] == "concentration"
                else None
            ),
            "reference_species_fractions": (
                selected_fit.reference_species_fractions
                if selected_input_metadata["mode"] == "bulk_as_surface"
                else {}
            ),
            "common_total_order": selected_fit.common_order,
            "within_total_order": selected_fit.within_order,
            "response_structure": selected_fit.response_structure,
            "effect_scopes": selected_fit.effect_scopes,
            "species_role_terms": [name for name, (_, term) in zip(selected_fit.effect_names, selected_fit.coefficient_terms)
                                   if term != "common_total_order"],
            "selection_reason": primary["selection_reason"],
            "regularization": selected_fit.regularization,
            "effective_roles": selected_fit.effective_roles,
            "reduction_id": getattr(selected_fit.candidate, "reduction_id", "empirical"),
            "observable_parameters": (
                selected_fit.shape_parameters if isinstance(selected_fit, SurfaceKineticFit) else {}
            ),
            "applicability_status": selected_row.get(
                "applicability_status", "production"
            ),
            "contrast_status": selected_row.get(
                "contrast_status", "not_assessed"
            ),
            "distinguishable": bool(selected_row.get("distinguishable", False)),
            "supported_claims": selected_row.get("supported_claims", []),
            "missing_evidence": selected_row.get("missing_evidence", []),
            "reference_reaction_inputs": (
                selected_fit.reference_concentrations if isinstance(selected_fit, SurfaceKineticFit) else {}
            ),
            "reference_reaction_input_total": (
                selected_fit.reference_input_total
                if isinstance(selected_fit, SurfaceKineticFit)
                else selected_fit.reference_total_concentration
            ),
            "reference_reaction_input_shares": (
                selected_fit.reference_input_shares
                if isinstance(selected_fit, SurfaceKineticFit)
                else selected_fit.reference_species_fractions
            ),
            "reaction_input": selected_input_metadata,
        },
        "model_inputs": {
            "available": list(train.available_inputs()),
            "selected_reaction_input": selected_input_metadata,
            "bulk_as_surface_approximation": getattr(
                selected_fit.candidate, "transport_mode", "empirical"
            ) == "bulk_as_surface",
            "wafer_flux_supplied": (
                selected_input_metadata["quantity"] == "transport_capacity_flux"
            ),
            "realized_reactive_flux_used_as_model_input": False,
            "wafer_temperature": {
                "spatial_mode": "uniform",
                "value_K": wafer_temperature_k,
                "used_as_spatial_correction": False,
            },
        },
        "spatial_response": {
            "model": spatial_fit.mode,
            "terms": list(spatial_fit.terms),
            "coefficients": spatial_fit.coefficients,
            "center_xy_source_units": spatial_fit.center_xy,
            "radius_scale_source_units": spatial_fit.radius_scale,
            "weighting": spatial_fit.weighting,
            "mean_rate_policy": "conditionwise chemical mean preserved",
            "participates_in_role_or_equation_selection": False,
            "fixed_holdout": {
                "chemical_centered_r2": chemical_test_metrics[
                    "centered_spatial_r2"
                ],
                "corrected_centered_r2": spatial_test_metrics[
                    "centered_spatial_r2"
                ],
                "chemical_rmse_nm_s": chemical_test_metrics["rmse_nm_s"],
                "corrected_rmse_nm_s": spatial_test_metrics["rmse_nm_s"],
            },
            "outer_condition_transfer_supported": spatial_response_transfer_supported,
        },
        "unrestricted_numerical_winner": unrestricted_row,
        "equation_family_assessments": equation_family_assessments,
        "reaction_mechanism_assessments": reaction_mechanism_assessments,
        "reaction_model_prediction_comparison": family_prediction_rows,
        "reaction_model_state_comparison": family_state_rows,
        "role_importance_and_stability": role_importance_rows,
        "parameter_sensitivity_correlations": parameter_sensitivity_diagnostic_rows,
        "workflow_layers": workflow_layers,
        "capability_assessments": capability_assessments,
        "data_requirements": data_requirement_rows,
        "validity": validity,
        "model_structure_uncertainty": model_uncertainty,
        "grid_alignment": grid_alignment,
        "data_quality": [case.quality for case in cases],
        "missing_information": list(dict.fromkeys([
            *selected_row.get("missing_evidence", []),
            *[
                measurement
                for capability in (
                    "wafer_spatial_correction",
                    "anonymous_species_role_assignment",
                    "elementary_kinetic_parameter_estimation",
                )
                for measurement in required_measurements_for(
                    data_requirement_rows, capability
                )
            ],
        ])),
        "interpretation_limits": [
            ("Roles and exact kinetic reductions are chosen by inner condition CV; only outer predictions evaluate the selected procedure."
             if _uses_surface_response(response_model) else
             "Regularization and roles are jointly chosen by inner condition CV; only outer predictions evaluate the selected procedure."),
            "A numerical score tie is not statistical or practical equivalence. A no-inhibitor steady AB response cannot identify A/B direction.",
            ("Observable dimensionless groups are not separate elementary rate constants or a surface relaxation time."
             if _uses_surface_response(response_model) else
             "The common order describes the supplied condition scaling and is not an elementary reaction order."),
            "A species excluded by the adoption gate is not proven inert.",
            "The optional spatial residual response is evaluated after chemical selection and cannot support a reaction-role or mechanism claim.",
            "Uniform wafer temperature is assumed; no radial temperature field is fitted.",
            "Negative test R2 can coexist with low relative RMSE because within-map variation is much smaller than the absolute condition-level rate.",
            "See test_extrapolation.csv for the supplied test condition; a reduced physical form does not establish the true mechanism out of domain.",
            "Bootstrap intervals condition on the same identification conditions. model_structure_uncertainty.csv separately shows the prediction envelope of structures selected across outer condition folds, but is not a confidence interval.",
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

    spatial_response_rows = [
        {
            "held_out_condition": row["test_case"],
            "spatial_model": row["spatial_response_model"],
            "selected_chemical_model": row["selected_model"],
            "chemical_rmse_nm_s": row["test_rmse_nm_s"],
            "corrected_rmse_nm_s": row["spatial_response_test_rmse_nm_s"],
            "chemical_centered_spatial_r2": row["test_centered_spatial_r2"],
            "corrected_centered_spatial_r2": row[
                "spatial_response_test_centered_spatial_r2"
            ],
            "corrected_spatial_correlation": row[
                "spatial_response_test_spatial_correlation"
            ],
            "chemical_selection_uses_spatial_response": False,
        }
        for row in split_rows
    ]

    _write_rows(output_dir / "condition_quality.csv", quality_rows)
    _write_rows(output_dir / "concentration_scaling.csv", scaling_rows)
    _write_rows(output_dir / "pooled_concentration_correlations.csv", correlation_rows)
    _write_rows(
        output_dir / "condition_mean_input_correlations.csv", input_correlation_rows
    )
    _write_rows(output_dir / "role_ranking.csv", ranking)
    _write_rows(output_dir / "role_summary.csv", role_summary_rows)
    _write_rows(output_dir / "best_model_role_assignments.csv", family_role_rows)
    _write_rows(output_dir / "optimization_history.csv", optimization_history_rows)
    _write_rows(output_dir / "role_input_sensitivity.csv", role_sensitivity_rows)
    _write_rows(
        output_dir / "role_importance_and_stability.csv", role_importance_rows
    )
    _write_rows(output_dir / "role_response_curves.csv", role_response_rows)
    _write_rows(output_dir / "reaction_state_summary.csv", reaction_state_rows)
    _write_rows(output_dir / "reaction_model_predictions.csv", family_prediction_rows)
    _write_rows(output_dir / "reaction_model_states.csv", family_state_rows)
    _write_rows(
        output_dir / "parameter_sensitivity_correlations.csv",
        parameter_sensitivity_diagnostic_rows,
    )
    _write_rows(output_dir / "parameter_loss_slices.csv", parameter_loss_rows)
    _write_rows(output_dir / "coefficients.csv", coefficient_rows)
    _write_rows(
        output_dir / "spatial_response_coefficients.csv",
        spatial_coefficient_rows(spatial_fit),
    )
    _write_rows(
        output_dir / "spatial_response_summary.csv", spatial_response_rows
    )
    _write_rows(output_dir / "test_predictions.csv", test_prediction_rows)
    _write_rows(output_dir / "split_sensitivity.csv", split_rows)
    _write_rows(output_dir / "role_stability.csv", role_stability_rows)
    _write_rows(output_dir / "condition_scores.csv", build_condition_scores(ranking))
    _write_rows(output_dir / "test_extrapolation.csv", extrapolation_rows)
    _write_rows(output_dir / "model_structure_uncertainty.csv", model_uncertainty_rows)
    _write_rows(output_dir / "condition_means.csv", condition_mean_rows)
    _write_rows(output_dir / "data_requirements.csv", data_requirement_rows)
    _write_json(output_dir / "analysis_summary.json", summary)
    condition_predictions = {
        case.case_id: _predict_response(selected_fit, _combine_cases([case]))[0]
        for case in cases
    }
    plot_paths = plot_multicond_results(
        output_dir,
        cases,
        condition_predictions,
        train,
        test,
        test_prediction,
        ranking,
        equation_family_assessments,
        split_rows,
        model_uncertainty_rows,
        test_prediction_rows,
        spatial_prediction=(
            spatial_prediction if spatial_fit.mode != "none" else None
        ),
        reaction_input_mode=reaction_input_mode,
        optimization_history_rows=optimization_history_rows,
        family_role_rows=family_role_rows,
        input_correlation_rows=input_correlation_rows,
        role_sensitivity_rows=role_sensitivity_rows,
        role_importance_rows=role_importance_rows,
        role_response_rows=role_response_rows,
        reaction_state_rows=reaction_state_rows,
        family_prediction_rows=family_prediction_rows,
        family_state_rows=family_state_rows,
        parameter_sensitivity_rows=parameter_sensitivity_diagnostic_rows,
        parameter_loss_rows=parameter_loss_rows,
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
        "label": (
            "CVD Fluent reaction-input maps and deposition-rate maps for "
            f"conditions {list(all_case_ids)}"
        ),
        "files": source_files,
        "filters": [
            f"Identification conditions: {list(train_case_ids)}",
            f"Held-out no-refit test condition: {test_case_id}",
            f"{sum(case.rate.size for case in cases)} matched observations across {len(cases)} conditions",
        ],
    }
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
                            "label": ("Observable surface-response parameters" if _uses_surface_response(response_model)
                                      else "Common total-concentration order"),
                            "definition": (
                                "Dimensionless groups of the quasi-steady site balance; they are not separate elementary constants."
                                if _uses_surface_response(response_model) else
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
            "equation_family_assessments": {
                "rows": equation_family_assessments,
                "source": source_metadata,
            },
            "reaction_mechanism_assessments": {
                "rows": reaction_mechanism_assessments,
                "source": source_metadata,
            },
            "capability_assessments": {
                "rows": capability_assessments,
                "source": source_metadata,
            },
            "data_requirements": {
                "rows": data_requirement_rows,
                "source": source_metadata,
            },
            "model_ranking": {
                "rows": ranking,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Candidate selection",
                            "definition": (
                                ("Training-condition CV compares independently refitted role assignments and exact kinetic reductions. "
                                 if _uses_surface_response(response_model) else
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
            "spatial_response": {
                "rows": spatial_response_rows,
                "source": source_metadata,
            },
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
            "role_importance_and_stability": {
                "rows": role_importance_rows,
                "source": source_metadata,
            },
            "reaction_model_predictions": {
                "rows": family_prediction_rows,
                "source": source_metadata,
            },
            "parameter_sensitivity_correlations": {
                "rows": parameter_sensitivity_diagnostic_rows,
                "source": source_metadata,
            },
            "test_extrapolation": {"rows": extrapolation_rows, "source": source_metadata},
            "model_structure_uncertainty": {
                "rows": model_uncertainty_rows,
                "source": source_metadata,
            },
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
        output_dir / "condition_mean_input_correlations.csv",
        output_dir / "role_ranking.csv",
        output_dir / "role_summary.csv",
        output_dir / "best_model_role_assignments.csv",
        output_dir / "optimization_history.csv",
        output_dir / "role_input_sensitivity.csv",
        output_dir / "role_importance_and_stability.csv",
        output_dir / "role_response_curves.csv",
        output_dir / "reaction_state_summary.csv",
        output_dir / "reaction_model_predictions.csv",
        output_dir / "reaction_model_states.csv",
        output_dir / "parameter_sensitivity_correlations.csv",
        output_dir / "parameter_loss_slices.csv",
        output_dir / "coefficients.csv",
        output_dir / "spatial_response_coefficients.csv",
        output_dir / "spatial_response_summary.csv",
        output_dir / "test_predictions.csv",
        output_dir / "split_sensitivity.csv",
        output_dir / "role_stability.csv",
        output_dir / "condition_scores.csv",
        output_dir / "test_extrapolation.csv",
        output_dir / "model_structure_uncertainty.csv",
        output_dir / "condition_means.csv",
        output_dir / "data_requirements.csv",
        output_dir / "analysis_summary.json",
        output_dir / "report.md",
        output_dir / "report_snapshot.json",
        notebook_path,
        *plot_paths,
    ]
    manifest = {
        "generated_at": summary["generated_at"],
        "analysis_type": summary["analysis_type"],
        "estimation": summary["estimation"],
        "sources": summary["sources"],
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
