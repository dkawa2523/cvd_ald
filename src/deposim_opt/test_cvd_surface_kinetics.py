from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

import numpy as np

from .cvd_surface_kinetics import (
    SurfaceKineticCandidate,
    enumerate_surface_kinetic_candidates,
    fit_surface_kinetic,
    predict_surface_kinetic,
    response_shape,
    surface_state,
)


@dataclass(frozen=True)
class _Data:
    species: tuple[str, ...]
    concentrations: dict[str, np.ndarray]
    condition_id: np.ndarray
    rate: np.ndarray


def _data() -> _Data:
    x = np.linspace(0.25, 2.5, 30)
    concentrations = {
        "s0": x,
        "s1": 0.3 + 1.2 * x[::-1],
        "s2": 0.4 + 0.35 * np.sin(np.linspace(0.0, 3.0, x.size)) ** 2,
    }
    return _Data(
        species=("s0", "s1", "s2"),
        concentrations=concentrations,
        condition_id=np.repeat((1, 2, 3), 10),
        rate=np.ones(x.size),
    )


class SurfaceKineticTests(unittest.TestCase):
    def test_qss_coverages_close_the_site_balance(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        refs = {name: float(np.median(values)) for name, values in data.concentrations.items()}
        parameters = {
            "desorption_ratio": 0.4,
            "conversion_ratio": 1.7,
            "inhibition_ratio": 2.2,
        }
        state = surface_state(candidate, data, refs, parameters)
        np.testing.assert_allclose(
            state["theta_free"] + state["theta_A"] + state["theta_I"],
            1.0,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_fit_recovers_aib_observable_response(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        train = np.arange(data.rate.size)
        refs = {name: float(np.median(values)) for name, values in data.concentrations.items()}
        parameters = {
            "desorption_ratio": 1.0,
            "conversion_ratio": 0.1,
            "inhibition_ratio": 0.01,
        }
        truth = 0.23 * response_shape(candidate, data, refs, parameters)
        fitted = fit_surface_kinetic(candidate, replace(data, rate=truth), train)
        prediction, _ = predict_surface_kinetic(fitted, data)
        self.assertLess(float(np.max(np.abs(prediction - truth))), 1.0e-7)

    def test_references_make_concentration_units_invariant(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AI", A="s0", I="s2")
        refs = {name: float(np.median(values)) for name, values in data.concentrations.items()}
        parameters = {"half_saturation_ratio": 0.8, "inhibition_ratio": 1.3}
        first = response_shape(candidate, data, refs, parameters)
        factors = {"s0": 1.0e3, "s1": 1.0e-2, "s2": 1.0e6}
        scaled = replace(
            data,
            concentrations={name: values * factors[name] for name, values in data.concentrations.items()},
        )
        scaled_refs = {name: refs[name] * factors[name] for name in refs}
        second = response_shape(candidate, scaled, scaled_refs, parameters)
        np.testing.assert_allclose(first, second, rtol=1.0e-14, atol=0.0)

    def test_no_inhibitor_ab_response_has_a_b_exchange_symmetry(self):
        data = _data()
        refs = {name: float(np.median(values)) for name, values in data.concentrations.items()}
        forward = SurfaceKineticCandidate("AB", A="s0", B="s1")
        reverse = SurfaceKineticCandidate("AB", A="s1", B="s0")
        delta, conversion, scale = 0.7, 2.5, 0.31
        first = scale * response_shape(
            forward, data, refs,
            {"desorption_ratio": delta, "conversion_ratio": conversion},
        )
        second = (scale * conversion) * response_shape(
            reverse, data, refs,
            {
                "desorption_ratio": delta / conversion,
                "conversion_ratio": 1.0 / conversion,
            },
        )
        np.testing.assert_allclose(first, second, rtol=1.0e-14, atol=1.0e-14)

    def test_candidate_reductions_are_explicit_physical_boundaries(self):
        candidates = enumerate_surface_kinetic_candidates(("s0", "s1", "s2"))
        ids = {candidate.model_id for candidate in candidates}
        self.assertIn("surface_AB:s0|s1:no_desorption", ids)
        full = next(
            candidate for candidate in candidates
            if candidate.model_id == "surface_AIB:s0|s2|s1"
        )
        reductions = {candidate.model_id for candidate in full.reductions()}
        self.assertEqual(
            reductions,
            {
                "surface_AIB:s0|s2|s1:no_desorption",
                "surface_AB:s0|s1",
            },
        )

    def test_prediction_reuses_identification_references(self):
        data = _data()
        candidate = SurfaceKineticCandidate("A", A="s0")
        train = np.arange(20)
        target = 0.2 * data.concentrations["s0"] / (data.concentrations["s0"] + 0.5)
        fitted = fit_surface_kinetic(candidate, replace(data, rate=target), train)
        changed = replace(
            data,
            concentrations={**data.concentrations, "s0": data.concentrations["s0"] * 10.0},
        )
        predict_surface_kinetic(fitted, changed)
        self.assertEqual(
            fitted.reference_concentrations["s0"],
            float(np.median(data.concentrations["s0"][train])),
        )


if __name__ == "__main__":
    unittest.main()
