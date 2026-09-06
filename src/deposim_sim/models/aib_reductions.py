"""Observable steady reductions of role-based surface balances.

The registry in this module owns equation-family behavior. Parameter estimation
belongs to :mod:`deposim_opt.surface_fit`; raw species names remain anonymous.
Adding a family requires one descriptor and its equation functions, without
adding family-name branches to candidate enumeration or reporting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Callable, Iterable, Mapping
from urllib.parse import quote

import numpy as np


AIB_QSS = "aib_qss"
PARALLEL_QSS = "parallel_a_ab_qss"
LH_QSS = "langmuir_hinshelwood_qss"
BULK_AS_SURFACE = "bulk_as_surface"
DIRECT_SURFACE = "direct_surface"
DIRECT_FLUX = "direct_flux"
TRANSPORT_NOT_APPLICABLE = "not_applicable"
TOTAL_POWER_BASELINE = "total_power"
TRANSPORT_MODES = (BULK_AS_SURFACE, DIRECT_SURFACE, DIRECT_FLUX)


@dataclass(frozen=True)
class SurfaceCandidateDefinition:
    parameter_names: tuple[str, ...]
    effect_groups: dict[str, list[str]]
    reductions: tuple["SurfaceKineticCandidate", ...]
    formula: str
    role_symmetry: str = ""


@dataclass(frozen=True)
class SurfaceModelFamily:
    name: str
    result_family: str
    description: str
    supported_processes: tuple[str, ...]
    observables: tuple[str, ...]
    enabled_by_default: bool
    required_classes: tuple[str, ...]
    required_inputs: tuple[str, ...]
    reduction_ids: tuple[str, ...]
    symmetric_classes: tuple[str, ...]
    physical_question: str
    evidence_requirements: tuple[str, ...]
    mechanism: str
    pathways: tuple[str, ...]
    state_variables: tuple[str, ...]
    define: Callable[["SurfaceKineticCandidate"], SurfaceCandidateDefinition]
    evaluate: Callable[
        [
            "SurfaceKineticCandidate",
            Mapping[str, np.ndarray],
            dict[str, float],
            dict[str, float],
        ],
        tuple[np.ndarray, dict[str, np.ndarray]],
    ]


_FAMILIES: dict[str, SurfaceModelFamily] = {}


def _declared_inputs(available_inputs: Iterable[str]) -> set[str]:
    supplied = {str(name) for name in available_inputs}
    if supplied.intersection(
        {"concentration", "bulk_concentration", "surface_concentration", "transport_capacity_flux"}
    ):
        supplied.add("reaction_driver")
    return supplied


def available_surface_model_families(
    available_inputs: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return registered families, optionally restricted by available inputs."""

    if available_inputs is None:
        return tuple(_FAMILIES)
    supplied = _declared_inputs(available_inputs)
    return tuple(
        name
        for name, family in _FAMILIES.items()
        if set(family.required_inputs).issubset(supplied)
    )


def default_surface_model_families(
    available_inputs: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return families supported by the routine production comparison."""

    supplied = None if available_inputs is None else _declared_inputs(available_inputs)
    return tuple(
        name
        for name, family in _FAMILIES.items()
        if family.enabled_by_default
        and (supplied is None or set(family.required_inputs).issubset(supplied))
    )


def get_surface_model_family(name: str) -> SurfaceModelFamily:
    try:
        return _FAMILIES[str(name)]
    except KeyError as exc:
        supported = ", ".join(available_surface_model_families())
        raise ValueError(f"Unknown surface model family {name!r}; supported: {supported}") from exc


def candidate_physical_question(candidate: "SurfaceKineticCandidate") -> str:
    """Return the physical comparison owned by a candidate equation."""

    if candidate.class_id == "baseline":
        return "Can a condition-independent deposition rate explain the observations?"
    if candidate.class_id == TOTAL_POWER_BASELINE:
        return "Can total concentration alone explain the condition response?"
    if candidate.class_id == "A":
        return "Does one raw species provide a saturating growth-related supply response?"
    if candidate.class_id == "AI":
        return "Does a second raw species add competitive suppression to an A response?"
    question = get_surface_model_family(candidate.family).physical_question
    if candidate.reduction_id == "no_desorption":
        return question + " Is the finite nonproductive-loss group unnecessary?"
    return question


def candidate_evidence_requirements(
    candidate: "SurfaceKineticCandidate",
) -> tuple[str, ...]:
    """Describe the smallest observations needed to interpret a fitted candidate."""

    if candidate.class_id in {"baseline", TOTAL_POWER_BASELINE}:
        return ()
    if candidate.class_id == "A":
        return (
            "independent variation of the A candidate across low-response and saturation regimes",
        )
    if candidate.class_id == "AI":
        return (
            "independent variation of the A and I candidates",
            "an I perturbation spanning weak and strong suppression",
        )
    requirements = list(get_surface_model_family(candidate.family).evidence_requirements)
    if candidate.I is not None:
        requirements.append("independent variation of the inhibitor candidate")
    return tuple(dict.fromkeys(requirements))


def candidate_parameter_units(
    candidate: "SurfaceKineticCandidate",
) -> dict[str, str]:
    """Return observable parameter units for the normalized steady equation."""

    return {
        "rate_scale_nm_s": "nm/s",
        **{name: "1" for name in candidate.parameter_names},
    }


def reduction_removed_effects(
    candidate: "SurfaceKineticCandidate",
    reduction: "SurfaceKineticCandidate",
) -> tuple[str, ...]:
    """Name the physical effects removed by one declared exact reduction."""

    if candidate.class_id == "A" and reduction.class_id == "baseline":
        return ("A",)
    if (
        candidate.class_id == TOTAL_POWER_BASELINE
        and reduction.class_id == "baseline"
    ):
        return ("total_concentration",)
    if candidate.class_id == "AI" and reduction.class_id == "A":
        return ("I",)
    if (
        candidate.reduction_id == "full"
        and reduction.reduction_id == "no_desorption"
        and candidate.family == reduction.family
        and candidate.class_id == reduction.class_id
    ):
        return ("finite_loss",)
    if candidate.family == LH_QSS and candidate.class_id == "AIB":
        return ("I",)
    if candidate.family == AIB_QSS:
        if reduction.class_id == "AB" and candidate.class_id == "AIB":
            return ("I",)
        if reduction.class_id == "baseline" and candidate.class_id == "AB":
            return ("AB",)
    if candidate.family == PARALLEL_QSS:
        if reduction.family == AIB_QSS and reduction.class_id == candidate.class_id:
            return ("A",)
        if reduction.class_id in {"A", "AI"}:
            return ("AB",)
        if reduction.family == PARALLEL_QSS and reduction.class_id == "AB":
            return ("I",)
    return ()


@dataclass(frozen=True)
class SurfaceKineticCandidate:
    """One raw-species role assignment and one physical reduction."""

    class_id: str
    A: str | None = None
    I: str | None = None
    B: str | None = None
    reduction_id: str = "full"
    family: str = AIB_QSS
    transport_mode: str = BULK_AS_SURFACE

    def __post_init__(self) -> None:
        family = get_surface_model_family(self.family)
        if self.class_id in {"baseline", TOTAL_POWER_BASELINE}:
            object.__setattr__(self, "transport_mode", TRANSPORT_NOT_APPLICABLE)
        common = {"baseline", TOTAL_POWER_BASELINE, "A", "AI"}
        if self.class_id not in common and self.class_id not in family.required_classes:
            raise ValueError(
                f"{self.family} does not support role class {self.class_id!r}"
            )
        if self.class_id in common and self.family != AIB_QSS:
            raise ValueError("baseline, A, and AI use the shared AIB reduction family")
        if self.reduction_id not in family.reduction_ids:
            raise ValueError(
                f"{self.family} reduction_id must be one of {family.reduction_ids}"
            )
        if (
            self.class_id not in {"baseline", TOTAL_POWER_BASELINE}
            and self.transport_mode not in TRANSPORT_MODES
        ):
            raise ValueError(
                f"transport_mode must be one of {TRANSPORT_MODES}, got {self.transport_mode!r}"
            )

    @property
    def model_family(self) -> str:
        if self.class_id == TOTAL_POWER_BASELINE:
            return "observation_baseline"
        return get_surface_model_family(self.family).result_family

    @property
    def model_id(self) -> str:
        roles = ",".join(
            f"{slot}={quote(str(value), safe='._-')}"
            for slot, value in (("A", self.A), ("I", self.I), ("B", self.B))
            if value is not None
        ) or "none"
        family = (
            "observation_baseline"
            if self.class_id == TOTAL_POWER_BASELINE
            else self.family
        )
        return (
            f"cvd:{family}:{self.class_id}:{self.reduction_id}:"
            f"{self.transport_mode}:{roles}"
        )

    @property
    def effect_groups(self) -> dict[str, list[str]]:
        if self.class_id in {"baseline", TOTAL_POWER_BASELINE}:
            return {}
        if self.class_id == "A":
            return {"A": [str(self.A)]}
        if self.class_id == "AI":
            return {"A": [str(self.A)], "I": [str(self.I)]}
        return get_surface_model_family(self.family).define(self).effect_groups

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self.class_id == "baseline":
            return ()
        if self.class_id == TOTAL_POWER_BASELINE:
            return ("common_total_order",)
        if self.class_id == "A":
            return ("half_saturation_ratio",)
        if self.class_id == "AI":
            return ("half_saturation_ratio", "inhibition_ratio")
        return get_surface_model_family(self.family).define(self).parameter_names

    @property
    def parameter_log10_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {name: (-10.0, 10.0) for name in self.parameter_names}
        if self.class_id == TOTAL_POWER_BASELINE:
            # The exponent is positive and intentionally limited to a broad,
            # finite response range for a nuisance concentration baseline.
            bounds["common_total_order"] = (-2.0, 1.0)
        return bounds

    @property
    def role_symmetry(self) -> str:
        if self.class_id in {"baseline", TOTAL_POWER_BASELINE, "A", "AI"}:
            return ""
        return get_surface_model_family(self.family).define(self).role_symmetry

    def reductions(self) -> tuple["SurfaceKineticCandidate", ...]:
        if self.class_id == "baseline":
            return ()
        if self.class_id == TOTAL_POWER_BASELINE:
            return (SurfaceKineticCandidate("baseline"),)
        if self.class_id == "A":
            return (
                SurfaceKineticCandidate(
                    "baseline", transport_mode=self.transport_mode
                ),
            )
        if self.class_id == "AI":
            return (
                SurfaceKineticCandidate(
                    "A", A=self.A, transport_mode=self.transport_mode
                ),
            )
        return get_surface_model_family(self.family).define(self).reductions


def _normalized(
    concentrations: Mapping[str, np.ndarray], refs: dict[str, float], role: str | None
) -> np.ndarray:
    if role is None:
        first = next(iter(concentrations.values()))
        return np.ones(np.asarray(first).shape, dtype=float)
    return np.asarray(concentrations[role], dtype=float) / refs[role]


def _empty_state(concentrations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    first = next(iter(concentrations.values()))
    nan = np.full(np.asarray(first).shape, np.nan, dtype=float)
    return {
        "theta_free": nan.copy(),
        "theta_A": nan.copy(),
        "theta_B": nan.copy(),
        "theta_I": nan.copy(),
        "path_A_fraction": nan.copy(),
        "path_AB_fraction": nan.copy(),
        "inhibition_availability": nan.copy(),
    }


def _shared_response_state(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if candidate.class_id == "baseline":
        first = next(iter(concentrations.values()))
        return np.ones(np.asarray(first).shape, dtype=float), _empty_state(concentrations)
    if candidate.class_id == TOTAL_POWER_BASELINE:
        total = sum(np.asarray(value, dtype=float) for value in concentrations.values())
        reference = max(float(sum(refs.values())), np.finfo(float).tiny)
        order = parameters["common_total_order"]
        return np.power(np.maximum(total / reference, np.finfo(float).tiny), order), _empty_state(concentrations)
    ua = _normalized(concentrations, refs, candidate.A)
    ui = _normalized(concentrations, refs, candidate.I)
    inhibition_ratio = parameters.get("inhibition_ratio", 0.0) * ui
    inhibition = 1.0 + inhibition_ratio
    loss = np.full(ua.shape, parameters["half_saturation_ratio"], dtype=float)
    denominator = ua + loss * inhibition
    state = {
        "theta_free": loss / denominator,
        "theta_A": ua / denominator,
        "theta_B": np.zeros(ua.shape, dtype=float),
        "theta_I": inhibition_ratio * loss / denominator,
        "path_A_fraction": np.ones(ua.shape, dtype=float),
        "path_AB_fraction": np.zeros(ua.shape, dtype=float),
        "inhibition_availability": 1.0 / inhibition,
    }
    return ua / denominator, state


def _pathway_response_state(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
    *,
    include_single_path: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ua = _normalized(concentrations, refs, candidate.A)
    ub = _normalized(concentrations, refs, candidate.B)
    ui = _normalized(concentrations, refs, candidate.I)
    inhibition_ratio = parameters.get("inhibition_ratio", 0.0) * ui
    inhibition = 1.0 + inhibition_ratio
    conversion_a = np.zeros(ua.shape, dtype=float)
    if include_single_path:
        conversion_a.fill(parameters["single_conversion_ratio"])
    conversion_ab = parameters["conversion_ratio"] * ub
    conversion = conversion_a + conversion_ab
    loss = parameters.get("desorption_ratio", 0.0) + conversion
    denominator = ua + loss * inhibition
    tiny = np.finfo(float).tiny
    state = {
        "theta_free": loss / denominator,
        "theta_A": ua / denominator,
        "theta_B": np.zeros(ua.shape, dtype=float),
        "theta_I": inhibition_ratio * loss / denominator,
        "path_A_fraction": conversion_a / np.maximum(conversion, tiny),
        "path_AB_fraction": conversion_ab / np.maximum(conversion, tiny),
        "inhibition_availability": 1.0 / inhibition,
    }
    return ua * conversion / denominator, state


def _sequential_definition(
    candidate: SurfaceKineticCandidate,
) -> SurfaceCandidateDefinition:
    pair = tuple(sorted((str(candidate.A), str(candidate.B))))
    names: list[str] = []
    if candidate.reduction_id != "no_desorption":
        names.append("desorption_ratio")
    names.append("conversion_ratio")
    if candidate.class_id == "AIB":
        names.append("inhibition_ratio")

    ab = SurfaceKineticCandidate(
        "AB", A=pair[0], B=pair[1], transport_mode=candidate.transport_mode
    )
    ab_zero = SurfaceKineticCandidate(
        "AB",
        A=pair[0],
        B=pair[1],
        reduction_id="no_desorption",
        transport_mode=candidate.transport_mode,
    )
    if candidate.class_id == "AB":
        reductions = (
            (
                ab_zero,
                SurfaceKineticCandidate(
                    "baseline", transport_mode=candidate.transport_mode
                ),
            )
            if candidate.reduction_id == "full"
            else (
                SurfaceKineticCandidate(
                    "baseline", transport_mode=candidate.transport_mode
                ),
            )
        )
    elif candidate.reduction_id == "full":
        reductions = (
            SurfaceKineticCandidate(
                "AIB",
                A=candidate.A,
                I=candidate.I,
                B=candidate.B,
                reduction_id="no_desorption",
                transport_mode=candidate.transport_mode,
            ),
            ab,
        )
    else:
        reductions = (ab_zero,)

    delta = "0" if candidate.reduction_id == "no_desorption" else "delta"
    inhibitor = "*(1 + kappa*uI)" if candidate.class_id == "AIB" else ""
    return SurfaceCandidateDefinition(
        parameter_names=tuple(names),
        effect_groups=(
            {"AB": list(pair)}
            if candidate.class_id == "AB"
            else {
                "A": [str(candidate.A)],
                "B": [str(candidate.B)],
                "I": [str(candidate.I)],
            }
        ),
        reductions=reductions,
        formula=f"v = R*uA*b*uB / (uA + ({delta} + b*uB){inhibitor})",
        role_symmetry=(
            "A/B exchange in the no-inhibitor sequential steady response"
            if candidate.class_id == "AB"
            else ""
        ),
    )


def _parallel_definition(
    candidate: SurfaceKineticCandidate,
) -> SurfaceCandidateDefinition:
    names = ["single_conversion_ratio"]
    if candidate.reduction_id != "no_desorption":
        names.append("desorption_ratio")
    names.append("conversion_ratio")
    if candidate.class_id == "AIB":
        names.append("inhibition_ratio")

    reductions: list[SurfaceKineticCandidate] = []
    if candidate.reduction_id == "full":
        reductions.append(
            SurfaceKineticCandidate(
                candidate.class_id,
                A=candidate.A,
                I=candidate.I,
                B=candidate.B,
                reduction_id="no_desorption",
                family=PARALLEL_QSS,
                transport_mode=candidate.transport_mode,
            )
        )
    reductions.extend(
        [
            SurfaceKineticCandidate(
                candidate.class_id,
                A=candidate.A,
                I=candidate.I,
                B=candidate.B,
                reduction_id=candidate.reduction_id,
                family=AIB_QSS,
                transport_mode=candidate.transport_mode,
            ),
            SurfaceKineticCandidate(
                "AI" if candidate.I is not None else "A",
                A=candidate.A,
                I=candidate.I,
                transport_mode=candidate.transport_mode,
            ),
        ]
    )
    if candidate.I is not None:
        reductions.append(
            SurfaceKineticCandidate(
                "AB",
                A=candidate.A,
                B=candidate.B,
                reduction_id=candidate.reduction_id,
                family=PARALLEL_QSS,
                transport_mode=candidate.transport_mode,
            )
        )
    unique = {row.model_id: row for row in reductions}
    delta = "0" if candidate.reduction_id == "no_desorption" else "delta"
    inhibitor = "*(1 + kappa*uI)" if candidate.class_id == "AIB" else ""
    return SurfaceCandidateDefinition(
        parameter_names=tuple(names),
        effect_groups={
            "A": [str(candidate.A)],
            "AB": [str(candidate.A), str(candidate.B)],
            **({"I": [str(candidate.I)]} if candidate.class_id == "AIB" else {}),
        },
        reductions=tuple(unique.values()),
        formula=(
            "v = R*uA*(c + b*uB) / "
            f"(uA + ({delta} + c + b*uB){inhibitor})"
        ),
    )


def _lh_definition(candidate: SurfaceKineticCandidate) -> SurfaceCandidateDefinition:
    pair = tuple(sorted((str(candidate.A), str(candidate.B))))
    reductions: tuple[SurfaceKineticCandidate, ...] = ()
    if candidate.class_id == "AIB":
        reductions = (
            SurfaceKineticCandidate(
                "AB",
                A=pair[0],
                B=pair[1],
                family=LH_QSS,
                transport_mode=candidate.transport_mode,
            ),
        )
    inhibitor = "+ kappa*uI" if candidate.class_id == "AIB" else ""
    return SurfaceCandidateDefinition(
        parameter_names=(
            "adsorption_ratio_A",
            "adsorption_ratio_B",
            *(("inhibition_ratio",) if candidate.class_id == "AIB" else ()),
        ),
        effect_groups={
            "AB": list(pair),
            **({"I": [str(candidate.I)]} if candidate.class_id == "AIB" else {}),
        },
        reductions=reductions,
        formula=(
            "v = R*(a*uA)*(b*uB) / "
            f"(1 + a*uA + b*uB {inhibitor})^2"
        ),
        role_symmetry="A/B exchange with adsorption-parameter exchange",
    )


def _sequential_evaluate(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return _pathway_response_state(
        candidate, concentrations, refs, parameters, include_single_path=False
    )


def _parallel_evaluate(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return _pathway_response_state(
        candidate, concentrations, refs, parameters, include_single_path=True
    )


def _lh_evaluate(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ua = _normalized(concentrations, refs, candidate.A)
    ub = _normalized(concentrations, refs, candidate.B)
    ui = _normalized(concentrations, refs, candidate.I)
    adsorption_a = parameters["adsorption_ratio_A"] * ua
    adsorption_b = parameters["adsorption_ratio_B"] * ub
    inhibition = parameters.get("inhibition_ratio", 0.0) * ui
    denominator = 1.0 + adsorption_a + adsorption_b + inhibition
    theta_free = 1.0 / denominator
    theta_a = adsorption_a * theta_free
    theta_b = adsorption_b * theta_free
    theta_i = inhibition * theta_free
    zeros = np.zeros(theta_free.shape, dtype=float)
    ones = np.ones(theta_free.shape, dtype=float)
    return theta_a * theta_b, {
        "theta_free": theta_free,
        "theta_A": theta_a,
        "theta_B": theta_b,
        "theta_I": theta_i,
        "path_A_fraction": zeros,
        "path_AB_fraction": ones,
        "inhibition_availability": 1.0 / np.maximum(1.0 + inhibition, 1.0),
    }


_FAMILIES.update(
    {
        AIB_QSS: SurfaceModelFamily(
            name=AIB_QSS,
            result_family="surface_qss",
            description="A adsorbs, B converts adsorbed A, and I blocks free sites.",
            supported_processes=("cvd",),
            observables=("film_rate",),
            enabled_by_default=True,
            required_classes=("AB", "AIB"),
            required_inputs=("reaction_driver",),
            reduction_ids=("full", "no_desorption"),
            symmetric_classes=("AB",),
            physical_question=(
                "Does adsorbed A require a sequential B-assisted conversion path, "
                "with optional blocking by I?"
            ),
            evidence_requirements=(
                "independent A/B perturbations including a low-B regime",
            ),
            mechanism="Langmuir-Rideal-type adsorbed-A/gas-B conversion",
            pathways=("AB",),
            state_variables=("theta_A", "theta_I"),
            define=_sequential_definition,
            evaluate=_sequential_evaluate,
        ),
        PARALLEL_QSS: SurfaceModelFamily(
            name=PARALLEL_QSS,
            result_family="surface_parallel_qss",
            description="A converts alone while B provides an additional conversion path.",
            supported_processes=("cvd",),
            observables=("film_rate",),
            enabled_by_default=True,
            required_classes=("AB", "AIB"),
            required_inputs=("reaction_driver",),
            reduction_ids=("full", "no_desorption"),
            symmetric_classes=(),
            physical_question=(
                "Does growth contain an A-only path plus a B-assisted parallel path?"
            ),
            evidence_requirements=(
                "independent B perturbations from near-zero B into the B-responsive regime",
            ),
            mechanism="Adsorbed-A conversion with parallel A-only and gas-B paths",
            pathways=("A", "AB"),
            state_variables=("theta_A", "theta_I"),
            define=_parallel_definition,
            evaluate=_parallel_evaluate,
        ),
        LH_QSS: SurfaceModelFamily(
            name=LH_QSS,
            result_family="surface_lh_qss",
            description="A and B compete for one site pool and react as adsorbates.",
            supported_processes=("cvd",),
            observables=("film_rate",),
            enabled_by_default=False,
            required_classes=("AB", "AIB"),
            required_inputs=("reaction_driver",),
            reduction_ids=("full",),
            symmetric_classes=("AB", "AIB"),
            physical_question=(
                "Do A and B compete for one site pool and react as two adsorbates?"
            ),
            evidence_requirements=(
                "independent A/B perturbations spanning low coverage and saturation",
                "B adsorption, retention, or time-response evidence",
            ),
            mechanism="Langmuir-Hinshelwood coadsorption on one site pool",
            pathways=("AB",),
            state_variables=("theta_A", "theta_B", "theta_I"),
            define=_lh_definition,
            evaluate=_lh_evaluate,
        ),
    }
)


def _family_assignments(
    names: tuple[str, ...], family: SurfaceModelFamily, class_id: str
) -> Iterable[tuple[str, str | None, str]]:
    if class_id == "AB":
        pairs = (
            combinations(names, 2)
            if class_id in family.symmetric_classes
            else permutations(names, 2)
        )
        return ((a, None, b) for a, b in pairs)
    if class_id not in family.symmetric_classes:
        return ((a, inhibitor, b) for a, inhibitor, b in permutations(names, 3))
    return (
        (a, inhibitor, b)
        for inhibitor in names
        for a, b in combinations(tuple(name for name in names if name != inhibitor), 2)
    )


def enumerate_surface_kinetic_candidates(
    species: Iterable[str],
    *,
    include_boundaries: bool = True,
    families: Iterable[str] = (AIB_QSS,),
    available_inputs: Iterable[str] = ("concentration",),
    transport_modes: Iterable[str] = (BULK_AS_SURFACE,),
) -> list[SurfaceKineticCandidate]:
    """Enumerate applicable families while collapsing declared role symmetries."""

    supplied = _declared_inputs(available_inputs)
    requested = tuple(dict.fromkeys(str(name) for name in families))
    selected: list[SurfaceModelFamily] = []
    for name in requested:
        family = get_surface_model_family(name)
        if set(family.required_inputs).issubset(supplied):
            selected.append(family)

    names = tuple(sorted(str(name) for name in species))
    modes = tuple(dict.fromkeys(str(mode) for mode in transport_modes))
    invalid_modes = sorted(set(modes) - set(TRANSPORT_MODES))
    if invalid_modes:
        raise ValueError(f"Unknown transport modes: {invalid_modes}")
    candidates: list[SurfaceKineticCandidate] = [SurfaceKineticCandidate("baseline")]
    if "concentration" in supplied and BULK_AS_SURFACE in modes:
        candidates.append(SurfaceKineticCandidate(TOTAL_POWER_BASELINE))
    for transport_mode in modes:
        if any(
            family.name in {AIB_QSS, PARALLEL_QSS} for family in selected
        ):
            candidates.extend(
                SurfaceKineticCandidate(
                    "A", A=name, transport_mode=transport_mode
                )
                for name in names
            )
            candidates.extend(
                SurfaceKineticCandidate(
                    "AI", A=a, I=i, transport_mode=transport_mode
                )
                for a, i in permutations(names, 2)
            )
        for family in selected:
            for class_id in family.required_classes:
                for a, inhibitor, b in _family_assignments(names, family, class_id):
                    limits = (
                        family.reduction_ids
                        if include_boundaries
                        else family.reduction_ids[:1]
                    )
                    for reduction_id in limits:
                        candidates.append(
                            SurfaceKineticCandidate(
                                class_id,
                                A=a,
                                I=inhibitor,
                                B=b,
                                reduction_id=reduction_id,
                                family=family.name,
                                transport_mode=transport_mode,
                            )
                        )
    return list({candidate.model_id: candidate for candidate in candidates}.values())


def response_shape(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> np.ndarray:
    """Return the dimensionless steady response for one physical candidate."""

    if candidate.class_id in {"baseline", TOTAL_POWER_BASELINE, "A", "AI"}:
        response, _state = _shared_response_state(
            candidate, concentrations, refs, parameters
        )
    else:
        response, _state = get_surface_model_family(candidate.family).evaluate(
            candidate, concentrations, refs, parameters
        )
    return response


def surface_state(
    candidate: SurfaceKineticCandidate,
    concentrations: Mapping[str, np.ndarray],
    refs: dict[str, float],
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Reconstruct coverages and pathway fractions identified by the reduction."""

    if candidate.class_id in {"baseline", TOTAL_POWER_BASELINE, "A", "AI"}:
        _response, state = _shared_response_state(
            candidate, concentrations, refs, parameters
        )
    else:
        _response, state = get_surface_model_family(candidate.family).evaluate(
            candidate, concentrations, refs, parameters
        )
    return state


def surface_formula(candidate: SurfaceKineticCandidate) -> str:
    """Human-readable equation in observable dimensionless groups."""

    if candidate.class_id == "baseline":
        return "v = R"
    if candidate.class_id == TOTAL_POWER_BASELINE:
        return "v = R*(C_total/C_total,0)^n"
    if candidate.class_id == "A":
        return "v = Vmax*uA / (uA + h)"
    if candidate.class_id == "AI":
        return "v = Vmax*uA / (uA + h*(1 + kappa*uI))"
    return get_surface_model_family(candidate.family).define(candidate).formula


__all__ = [
    "AIB_QSS",
    "BULK_AS_SURFACE",
    "DIRECT_SURFACE",
    "DIRECT_FLUX",
    "LH_QSS",
    "PARALLEL_QSS",
    "TRANSPORT_MODES",
    "TRANSPORT_NOT_APPLICABLE",
    "TOTAL_POWER_BASELINE",
    "SurfaceCandidateDefinition",
    "SurfaceKineticCandidate",
    "SurfaceModelFamily",
    "available_surface_model_families",
    "candidate_evidence_requirements",
    "candidate_parameter_units",
    "candidate_physical_question",
    "default_surface_model_families",
    "enumerate_surface_kinetic_candidates",
    "get_surface_model_family",
    "reduction_removed_effects",
    "response_shape",
    "surface_formula",
    "surface_state",
]
