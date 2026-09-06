from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .transport_provider import (
    CfdFluxSinkKmProvider,
    DirectSurfaceConcentrationProvider,
    FitScalarKmProvider,
)


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

    def test_transport_capacity_uses_declared_driving_concentration(self) -> None:
        cref = np.array([1.0, 2.0], dtype=float)
        flux = np.array([0.08, 0.16], dtype=float)
        provider = CfdFluxSinkKmProvider.from_arrays(
            cref_a=cref,
            cref_b=cref,
            flux_a=flux,
            flux_b=flux,
            transport={
                "from_cfd_flux_sink": {
                    "flux_semantics": "transport_capacity",
                    "boundary_concentration_A": 0.2,
                    "boundary_concentration_B": 0.2,
                    "km_clip": [1.0e-8, 10.0],
                }
            },
        )
        np.testing.assert_allclose(provider.get_km("A"), [0.1, 0.16 / 1.8])
        diagnostics = provider.get_diagnostics("A")
        np.testing.assert_allclose(diagnostics["boundary_concentration"], 0.2)
        np.testing.assert_allclose(diagnostics["driving_concentration"], [0.8, 1.8])

    def test_realized_reactive_flux_is_not_reused_as_transport_capacity(self) -> None:
        cref = np.ones(2, dtype=float)
        with self.assertRaisesRegex(ValueError, "realized reactive flux"):
            CfdFluxSinkKmProvider.from_arrays(
                cref_a=cref,
                cref_b=cref,
                flux_a=0.1 * cref,
                flux_b=0.1 * cref,
                transport={
                    "from_cfd_flux_sink": {
                        "flux_semantics": "realized_reactive_flux"
                    }
                },
            )

    def test_direct_surface_provider_marks_wall_concentration(self) -> None:
        provider = DirectSurfaceConcentrationProvider.from_reference_shape(
            reference_shape=(3,)
        )
        self.assertTrue(np.all(np.isinf(provider.get_km("A"))))
        self.assertEqual(
            provider.get_diagnostics("A")["concentration_location"], "wall"
        )


if __name__ == "__main__":
    unittest.main()
