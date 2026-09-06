"""Internal aligned fields used by reaction-role fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from deposim_sim.models.aib_reductions import (
    BULK_AS_SURFACE,
    DIRECT_FLUX,
    DIRECT_SURFACE,
    TRANSPORT_NOT_APPLICABLE,
)


REACTION_INPUT_ALIASES = {
    "bulk_concentration": BULK_AS_SURFACE,
    "surface_concentration": DIRECT_SURFACE,
    "transport_capacity_flux": DIRECT_FLUX,
    BULK_AS_SURFACE: BULK_AS_SURFACE,
    DIRECT_SURFACE: DIRECT_SURFACE,
    DIRECT_FLUX: DIRECT_FLUX,
}


@dataclass(frozen=True)
class RoleFieldSet:
    """Aligned condition fields after file-format and coordinate handling.

    The object deliberately contains arrays only.  CSV column discovery and
    simulator configuration stay in their adapters; kinetic equations receive
    only the concentration mapping selected here.
    """

    case_ids: tuple[int, ...]
    xyz: np.ndarray
    condition_id: np.ndarray
    species: tuple[str, ...]
    bulk_concentrations: dict[str, np.ndarray]
    species_fractions: dict[str, np.ndarray]
    total_concentration: np.ndarray
    rate: np.ndarray
    surface_concentrations: dict[str, np.ndarray] = field(default_factory=dict)
    transport_capacity_flux: dict[str, np.ndarray] = field(default_factory=dict)
    realized_reactive_flux: dict[str, np.ndarray] = field(default_factory=dict)
    rate_sigma: np.ndarray | None = None

    def available_reaction_input_modes(self) -> tuple[str, ...]:
        """Return complete input representations, without ranking them.

        Input selection is an analysis decision.  It is deliberately not part
        of reaction-family selection because choosing a sampling location or a
        CFD flux product is not evidence for one chemical mechanism.
        """

        modes = [BULK_AS_SURFACE]
        if set(self.surface_concentrations) == set(self.species):
            modes.append(DIRECT_SURFACE)
        if set(self.transport_capacity_flux) == set(self.species):
            modes.append(DIRECT_FLUX)
        return tuple(modes)

    def reaction_inputs_for(self, input_mode: str) -> dict[str, np.ndarray]:
        """Select the one field family supplied to a steady role equation."""

        if input_mode in {BULK_AS_SURFACE, TRANSPORT_NOT_APPLICABLE}:
            return self.bulk_concentrations
        if input_mode == DIRECT_SURFACE:
            missing = sorted(set(self.species) - set(self.surface_concentrations))
            if missing:
                raise ValueError(
                    "direct_surface requires surface concentrations for every species; "
                    f"missing: {missing}"
                )
            return self.surface_concentrations
        if input_mode == DIRECT_FLUX:
            missing = sorted(set(self.species) - set(self.transport_capacity_flux))
            if missing:
                raise ValueError(
                    "direct_flux requires transport-capacity flux for every species; "
                    f"missing: {missing}"
                )
            return self.transport_capacity_flux
        raise ValueError(f"Unknown reaction input mode: {input_mode!r}")

    def resolve_reaction_input_mode(self, requested: str) -> str:
        """Resolve one explicit user-facing input choice and verify availability."""

        key = str(requested).strip().lower()
        try:
            mode = REACTION_INPUT_ALIASES[key]
        except KeyError as exc:
            choices = sorted(
                {"bulk_concentration", "surface_concentration", "transport_capacity_flux"}
            )
            raise ValueError(
                f"Unknown reaction input {requested!r}; choose one of {choices}"
            ) from exc
        if mode not in self.available_reaction_input_modes():
            metadata = {
                BULK_AS_SURFACE: "concentration_<species>",
                DIRECT_SURFACE: "surface_concentration_<species>",
                DIRECT_FLUX: "transport_capacity_flux_<species>",
            }
            raise ValueError(
                f"Reaction input {requested!r} is unavailable; every condition must "
                f"supply {metadata[mode]} columns for every species"
            )
        return mode

    def reaction_input_metadata(self, input_mode: str) -> dict[str, str]:
        """Describe the selected input without assigning a chemical role."""

        if input_mode == BULK_AS_SURFACE:
            return {
                "mode": input_mode,
                "quantity": "concentration",
                "location": "reference_plane",
                "unit": "kmol/m^3",
                "interpretation": "reference concentration used as a surface proxy",
            }
        if input_mode == DIRECT_SURFACE:
            return {
                "mode": input_mode,
                "quantity": "concentration",
                "location": "wafer_surface",
                "unit": "kmol/m^3",
                "interpretation": "supplied wall-adjacent concentration",
            }
        if input_mode == DIRECT_FLUX:
            return {
                "mode": input_mode,
                "quantity": "transport_capacity_flux",
                "location": "wafer_surface",
                "unit": "kmol/(m^2 s)",
                "interpretation": (
                    "nonnegative flux toward the wafer, calculated independently "
                    "of the fitted surface reaction"
                ),
            }
        if input_mode == TRANSPORT_NOT_APPLICABLE:
            return {
                "mode": input_mode,
                "quantity": "none",
                "location": "not_applicable",
                "unit": "1",
                "interpretation": "observation baseline",
            }
        raise ValueError(f"Unknown reaction input mode: {input_mode!r}")

    # Kept as narrow API aliases for callers outside the main analysis path.
    def available_transport_modes(self) -> tuple[str, ...]:
        return self.available_reaction_input_modes()

    def concentrations_for(self, transport_mode: str) -> dict[str, np.ndarray]:
        return self.reaction_inputs_for(transport_mode)

    def available_inputs(self) -> tuple[str, ...]:
        inputs = {
            "concentration",
            "bulk_concentration",
            "reaction_driver",
            "film_rate",
        }
        if DIRECT_SURFACE in self.available_reaction_input_modes():
            inputs.add("surface_concentration")
        if self.transport_capacity_flux:
            inputs.add("transport_capacity_flux")
        if self.realized_reactive_flux:
            inputs.add("realized_reactive_flux")
        if self.rate_sigma is not None:
            inputs.add("uncertainty")
        return tuple(sorted(inputs))


def condition_contrast_summary(
    data: RoleFieldSet,
    species: Iterable[str],
    *,
    transport_mode: str = BULK_AS_SURFACE,
) -> dict[str, object]:
    """Measure between-condition excitation for one anonymous role assignment.

    This is an input-only screen.  It neither reads the measured rate nor drops
    candidates.  Centered log condition means test whether the assigned species
    can vary independently enough to support a cross-condition role claim.
    """

    names = tuple(dict.fromkeys(str(name) for name in species))
    conditions = np.unique(np.asarray(data.condition_id))
    if not names:
        return {
            "status": "not_required",
            "condition_count": int(conditions.size),
            "species_count": 0,
            "rank": 0,
            "condition_number": 1.0,
            "max_abs_correlation": 0.0,
            "log10_span": {},
            "confounded_species": [],
        }

    concentrations = data.reaction_inputs_for(transport_mode)
    missing = sorted(set(names) - set(concentrations))
    if missing:
        raise ValueError(f"Contrast assessment is missing species: {missing}")
    means = np.asarray(
        [
            [
                float(np.mean(np.asarray(concentrations[name])[data.condition_id == condition]))
                for name in names
            ]
            for condition in conditions
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(means) & (means > 0.0)):
        return {
            "status": "limited",
            "condition_count": int(conditions.size),
            "species_count": len(names),
            "rank": 0,
            "condition_number": float("inf"),
            "max_abs_correlation": float("nan"),
            "log10_span": {},
            "confounded_species": list(names),
        }

    log_means = np.log(means)
    centered = log_means - np.mean(log_means, axis=0)
    scale = np.sqrt(np.mean(np.square(centered), axis=0))
    varying = scale > 64.0 * np.finfo(float).eps
    standardized = np.zeros_like(centered)
    standardized[:, varying] = centered[:, varying] / scale[varying]
    singular = np.linalg.svd(standardized, compute_uv=False)
    if singular.size and singular[0] > 0.0:
        # Contrasts smaller than 0.1% of the dominant condition contrast are
        # treated as practically unresolved by this data-only screen.
        rank = int(np.count_nonzero(singular / singular[0] >= 1.0e-3))
        condition_number = (
            float(singular[0] / singular[-1])
            if singular[-1] > np.finfo(float).eps * singular[0]
            else float("inf")
        )
    else:
        rank = 0
        condition_number = float("inf")

    max_correlation = 0.0
    confounded: set[str] = {name for name, changes in zip(names, varying) if not changes}
    if len(names) > 1:
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                if not (varying[left] and varying[right]):
                    continue
                numerator = float(np.dot(centered[:, left], centered[:, right]))
                denominator = float(
                    np.linalg.norm(centered[:, left])
                    * np.linalg.norm(centered[:, right])
                )
                value = numerator / denominator
                max_correlation = max(max_correlation, abs(value))
                if abs(value) >= 0.98:
                    confounded.update((names[left], names[right]))
    status = "sufficient" if rank == len(names) and not confounded else "limited"
    return {
        "status": status,
        "condition_count": int(conditions.size),
        "species_count": len(names),
        "rank": rank,
        "condition_number": condition_number,
        "max_abs_correlation": max_correlation,
        "log10_span": {
            name: float(np.ptp(log_means[:, index]) / np.log(10.0))
            for index, name in enumerate(names)
        },
        "confounded_species": sorted(confounded),
    }


__all__ = [
    "BULK_AS_SURFACE",
    "DIRECT_FLUX",
    "DIRECT_SURFACE",
    "REACTION_INPUT_ALIASES",
    "RoleFieldSet",
    "condition_contrast_summary",
]
