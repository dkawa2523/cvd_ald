"""Translate evaluation limits into measurements that can resolve them."""

from __future__ import annotations

from typing import Any, Iterable


def build_capability_requirements(
    *,
    spatial_supported: bool,
    role_supported: bool,
    parameter_identifiability_status: str,
    concentration_location: str,
    has_measurement_uncertainty: bool,
    family_stable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return capability summaries and one row per useful next measurement.

    Requirements describe the experimental contrast and the claim it unlocks.
    They are independent of raw species names and therefore apply to new data
    directories with a different number or identity of Fluent species.
    """

    summaries = [
        {
            "capability": "wafer_spatial_correction",
            "current_status": (
                "demonstrated_on_supplied_holdouts"
                if spatial_supported
                else "additional_data_required"
            ),
            "ready_when": (
                "centered spatial prediction is positive on every independent holdout, "
                "residual structure is acceptably small, and the declared spatial tolerance is met"
            ),
        },
        {
            "capability": "anonymous_species_role_assignment",
            "current_status": (
                "demonstrated_on_supplied_holdouts"
                if role_supported and family_stable
                else "additional_data_required"
            ),
            "ready_when": (
                "condition contrasts have full role rank and the same assignment and effect "
                "necessity transfer across independent condition refits"
            ),
        },
        {
            "capability": "elementary_kinetic_parameter_estimation",
            "current_status": "additional_data_required",
            "ready_when": (
                "absolute surface balances are observed, transient and temperature responses "
                "resolve every fitted direction, and uncertainty intervals remain finite on external data"
            ),
        },
    ]

    requirements: list[dict[str, Any]] = []

    def add(
        capability: str,
        measurement: str,
        experimental_design: str,
        resolves: str,
        code_use: str,
        needed: bool = True,
    ) -> None:
        requirements.append(
            {
                "capability": capability,
                "needed_for_current_evidence": bool(needed),
                "required_measurement": measurement,
                "experimental_design": experimental_design,
                "resolves": resolves,
                "workflow_use": code_use,
            }
        )

    add(
        "wafer_spatial_correction",
        "replicated, coordinate-registered film maps with pointwise uncertainty",
        "repeat the same condition without changing wafer orientation or metrology coordinates",
        "measurement noise versus reproducible in-plane residual structure",
        "use uncertainty-weighted centered spatial validation",
        needed=not has_measurement_uncertainty,
    )
    add(
        "wafer_spatial_correction",
        "wall or near-wall species/transport fields; spatial temperature only if the uniform-temperature assumption is invalid",
        "save the fields on the same wafer coordinates for conditions that change flow or pressure, while recording the uniform wafer temperature",
        "surface-response variation versus local delivery under the stated uniform-temperature assumption",
        "fit spatial drivers on identification conditions and evaluate centered residuals on held-out wafers",
        needed=(
            not spatial_supported
            or concentration_location not in {"direct_surface", "direct_flux"}
        ),
    )
    add(
        "wafer_spatial_correction",
        "independent wafer maps spanning the intended operating window",
        "reserve complete conditions before choosing equations or spatial corrections",
        "transfer of the correction rather than interpolation of one map",
        "apply the frozen correction and compare centered R2, range capture, and residual maps",
        needed=not spatial_supported,
    )

    add(
        "anonymous_species_role_assignment",
        "independently varied candidate-species concentrations",
        "use factorial or near-orthogonal A/B/I perturbations, including low or off levels for one candidate at a time",
        "confounded role directions and A/B exchange alternatives",
        "require full condition-contrast rank and stable assignment across condition refits",
        needed=not role_supported,
    )
    add(
        "anonymous_species_role_assignment",
        "low-coverage and saturation conditions",
        "span dilute supply, transition, and plateau regimes without changing all species by the same factor",
        "linear supply response versus adsorption saturation and competitive site occupation",
        "compare each parent equation with its exact reductions",
        needed=not role_supported or not family_stable,
    )
    add(
        "anonymous_species_role_assignment",
        "surface-state or outlet-species observation tied to the anonymous inputs",
        "measure coverage, oxidation state, or consumption/byproduct response during an independent perturbation",
        "an empirical role association versus a chemically assigned pathway",
        "combine film prediction with the measured state or pathway observation",
        needed=not role_supported,
    )

    add(
        "elementary_kinetic_parameter_estimation",
        "time-resolved uptake, thickness, or surface-state response",
        "apply concentration steps or separated pulses with sampling faster than the observed relaxation",
        "storage, release, conversion, and redox time constants",
        "fit the dynamic AIB, ALD, or MvK state history rather than a steady lumped group",
    )
    add(
        "elementary_kinetic_parameter_estimation",
        "multiple calibrated substrate temperatures",
        "repeat independent concentration transients at several temperatures with matched transport characterization",
        "Arrhenius prefactors and activation energies from concentration effects",
        "share activation parameters across temperature conditions and validate temperature transfer",
    )
    add(
        "elementary_kinetic_parameter_estimation",
        "site density and absolute wall concentration or reacting-wall molar flux",
        "measure or calibrate active-site capacity, stoichiometry, diffusivity, and wall-normal transport",
        "coverage turnover frequency and elementary constants in physical units",
        "close molar surface and transport fluxes with Gamma_s and propagate their uncertainty",
        needed=(concentration_location != "direct_surface" or parameter_identifiability_status != "sufficient"),
    )
    add(
        "elementary_kinetic_parameter_estimation",
        "replicated dynamic observations with uncertainty",
        "repeat the transients used for estimation and retain a new recipe as an external test",
        "practical parameter intervals and predictive uncertainty",
        "check sensitivity rank, profile intervals, parameter correlation, and no-refit prediction",
        needed=not has_measurement_uncertainty,
    )
    return summaries, requirements


def required_measurements_for(
    rows: Iterable[dict[str, Any]], capability: str
) -> list[str]:
    return [
        str(row["required_measurement"])
        for row in rows
        if row["capability"] == capability and row["needed_for_current_evidence"]
    ]


__all__ = ["build_capability_requirements", "required_measurements_for"]
