from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .transport_provider import CfdFluxSinkKmProvider, FitScalarKmProvider


@unittest.skipIf(np is None, "NumPy is required")
class TestTransportProvider(unittest.TestCase):
    def test_scalar_provider_broadcast(self) -> None:
        provider = FitScalarKmProvider.from_transport_params(
            transport={
                "km_A": {"mode": "constant", "value": 0.12},
                "km_B": {"mode": "constant", "value": 0.03},
            },
            reference_shape=(4,),
            time_dependent=False,
        )
        km_a = provider.get_km("A")
        km_b = provider.get_km("B")
        self.assertTrue(np.allclose(km_a, 0.12))
        self.assertTrue(np.allclose(km_b, 0.03))
        self.assertEqual(km_a.shape, (4,))

    def test_scalar_provider_supports_2d_spatial_shape(self) -> None:
        provider = FitScalarKmProvider.from_transport_params(
            transport={"km_A": 0.08, "km_B": 0.02},
            reference_shape=(3, 2),
            time_dependent=False,
        )
        km_a = provider.get_km("A", t_index=0)
        self.assertEqual(km_a.shape, (3, 2))
        self.assertTrue(np.allclose(km_a, 0.08))

    def test_flux_sink_to_km_finite(self) -> None:
        cref = np.array(
            [
                [1.0, 2.0, 1.5],
                [0.5, 1.0, 2.0],
            ],
            dtype=float,
        )
        flux = np.array(
            [
                [0.1, 0.2, 0.15],
                [0.05, 0.1, 0.2],
            ],
            dtype=float,
        )
        provider = CfdFluxSinkKmProvider.from_arrays(
            cref_a=cref,
            cref_b=cref,
            flux_a=flux,
            flux_b=flux,
            transport={
                "gamma_km_A": 2.0,
                "gamma_km_B": 1.0,
                "from_cfd_flux_sink": {
                    "eps_cref": 1.0e-12,
                    "km_clip": [1.0e-8, 10.0],
                    "flux_negative_policy": "error",
                },
            },
            time_dependent=True,
        )
        km_a_t0 = provider.get_km("A", t_index=0)
        self.assertTrue(np.all(np.isfinite(km_a_t0)))
        self.assertTrue(np.allclose(km_a_t0, np.array([0.2, 0.2, 0.2], dtype=float)))

    def test_flux_negative_policy_error(self) -> None:
        cref = np.array([[1.0, 1.0]], dtype=float)
        flux = np.array([[0.1, -0.2]], dtype=float)
        provider = CfdFluxSinkKmProvider.from_arrays(
            cref_a=cref,
            cref_b=cref,
            flux_a=flux,
            flux_b=flux,
            transport={
                "from_cfd_flux_sink": {
                    "eps_cref": 1.0e-12,
                    "km_clip": [1.0e-8, 10.0],
                    "flux_negative_policy": "error",
                },
            },
        )
        with self.assertRaises(ValueError):
            provider.get_km("A")


if __name__ == "__main__":
    unittest.main()
