from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import ModelSpec

from .models import rate_laws


@unittest.skipIf(np is None, "NumPy is required for rate-law tests")
class TestRateLaws(unittest.TestCase):
    def test_registry_resolves_known_models_and_alias(self) -> None:
        names = rate_laws.available_rate_law_models()
        self.assertIn("powerlaw", names)
        self.assertIn("power_law", names)
        self.assertIn("saturation_inhibition", names)
        self.assertIs(rate_laws.resolve_rate_law_model("powerlaw"), rate_laws.powerlaw)
        self.assertIs(rate_laws.resolve_rate_law_model("power_law"), rate_laws.powerlaw)

    def test_unknown_model_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown rate-law model"):
            rate_laws.compute_rate("missing_model", Cs={"A": 1.0})

    def test_compute_rate_from_model_config_with_alias(self) -> None:
        model = ModelSpec(
            kinetics_name="power_law",
            kinetics_params={"k0": 2.0, "orders": {"A": 1.5}},
        )
        cs = {"A": np.array([0.25, 1.0, 4.0], dtype=float)}
        rate = rate_laws.compute_rate_from_model_config(model, Cs=cs)
        expected = 2.0 * (cs["A"] ** 1.5)
        np.testing.assert_allclose(rate, expected)

    def test_powerlaw_supports_negative_and_fractional_orders(self) -> None:
        cs = {
            "A": np.array([0.2, 0.5, 1.0], dtype=float),
            "B": 4.0,
        }
        params = {"k0": 3.0, "orders": {"A": -0.5, "B": 0.25}}
        rate = rate_laws.compute_rate("powerlaw", Cs=cs, params=params)
        expected = 3.0 * (cs["A"] ** -0.5) * (4.0 ** 0.25)
        self.assertTrue(np.all(np.isfinite(rate)))
        np.testing.assert_allclose(rate, expected)

    def test_powerlaw_apparent_orders_matches_finite_difference(self) -> None:
        cs = {"A": np.array([0.3, 0.7, 1.3], dtype=float)}
        params = {"k0": 1.7, "orders": {"A": -0.25}}
        eps = 1.0e-6
        rate = rate_laws.compute_rate("powerlaw", Cs=cs, params=params)
        rate_perturbed = rate_laws.compute_rate("powerlaw", Cs={"A": cs["A"] * (1.0 + eps)}, params=params)
        fd = (np.log(rate_perturbed) - np.log(rate)) / np.log(1.0 + eps)
        analytical = rate_laws.apparent_orders("powerlaw", Cs=cs, params=params)["A"]
        np.testing.assert_allclose(fd, analytical, rtol=2.0e-5, atol=2.0e-6)

    def test_saturation_inhibition_apparent_orders_matches_finite_difference(self) -> None:
        cs = {
            "A": np.array([0.3, 0.7, 1.4], dtype=float),
            "B": np.array([0.2, 0.4, 0.8], dtype=float),
        }
        params = {
            "k0": 1.2,
            "numerator_orders": {"A": 1.0, "B": -0.3},
            "denominator_coeffs": {"A": 0.8, "B": 0.5},
            "denominator_orders": {"A": 1.0, "B": 1.0},
            "denominator_power": 1.0,
        }
        eps = 1.0e-6
        rate = rate_laws.compute_rate("saturation_inhibition", Cs=cs, params=params)
        rate_perturbed = rate_laws.compute_rate(
            "saturation_inhibition",
            Cs={"A": cs["A"] * (1.0 + eps), "B": cs["B"]},
            params=params,
        )
        fd = (np.log(rate_perturbed) - np.log(rate)) / np.log(1.0 + eps)
        analytical = rate_laws.apparent_orders("saturation_inhibition", Cs=cs, params=params)["A"]
        np.testing.assert_allclose(fd, analytical, rtol=2.0e-5, atol=2.0e-6)

    def test_saturation_inhibition_rejects_nonpositive_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator must stay > 0"):
            rate_laws.compute_rate(
                "saturation_inhibition",
                Cs={"A": np.array([1.0, 2.0], dtype=float)},
                params={
                    "k0": 1.0,
                    "numerator_orders": {"A": 1.0},
                    "denominator_coeffs": {"A": -10.0},
                    "denominator_orders": {"A": 1.0},
                    "denominator_base": 0.1,
                },
            )

    def test_cs_state_t_broadcast_shape_alignment(self) -> None:
        cs = {"A": np.array([[0.2], [0.4], [0.8]], dtype=float)}
        state = {"site": np.ones((3, 4), dtype=float)}
        temperature = np.array([[650.0, 700.0, 750.0, 800.0]], dtype=float)
        rate = rate_laws.compute_rate(
            "powerlaw",
            Cs=cs,
            state=state,
            T=temperature,
            params={"k0": 2.0, "orders": {"A": 0.5}, "ea_j_mol": 0.0},
        )
        self.assertEqual(rate.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(rate)))


if __name__ == "__main__":
    unittest.main()
