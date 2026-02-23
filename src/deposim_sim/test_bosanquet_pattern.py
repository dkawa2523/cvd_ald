from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .input_builder import build_domain_from_fluent_xy
from .models.mass_transfer import compute_km, compute_km_from_model_config


@unittest.skipIf(np is None, "NumPy is required for Bosanquet tests")
class TestBosanquetPattern(unittest.TestCase):
    def _grid(self):
        xy = np.array([[-20.0, -15.0], [-10.0, 5.0], [0.0, 0.0], [12.0, 9.0], [22.0, -8.0]], dtype=float)
        return build_domain_from_fluent_xy(xy=xy, xy_unit="mm", wafer_radius_mm=40.0)

    def test_bosanquet_diffusivity_harmonic_mean(self) -> None:
        grid = self._grid()
        km = compute_km(
            "stagnant_film",
            grid=grid,
            params={
                "diffusivity_model": "bosanquet",
                "d_m_m2_s": 2.0e-5,
                "d_k_m2_s": 1.0e-5,
                "delta_eff_m": 5.0e-4,
            },
        )
        d_eff = 1.0 / (1.0 / 2.0e-5 + 1.0 / 1.0e-5)
        expected = d_eff / 5.0e-4
        np.testing.assert_allclose(km, np.full(grid.shape, expected))

    def test_rotating_disk_fallback_matches_stagnant_on_zero_omega(self) -> None:
        grid = self._grid()
        model = SimpleNamespace(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.62,
                "diffusivity_model": "bosanquet",
                "d_m_m2_s": 2.0e-5,
                "d_k_m2_s": 1.0e-5,
                "delta_eff_m": 8.0e-4,
                "nu_m2_s": 1.5e-5,
                "omega_zero_guard": "fallback_stagnant_film",
            },
        )
        km = compute_km_from_model_config(model, grid=grid, omega_rad_s=np.zeros(grid.shape, dtype=float))
        d_eff = 1.0 / (1.0 / 2.0e-5 + 1.0 / 1.0e-5)
        expected = np.full(grid.shape, d_eff / 8.0e-4)
        np.testing.assert_allclose(km, expected)

    def test_rotating_disk_nonzero_omega_differs_from_fallback(self) -> None:
        grid = self._grid()
        omega = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=float)
        km_rot = compute_km(
            "rotating_disk",
            grid=grid,
            params={
                "ck": 0.62,
                "diffusivity_m2_s": 2.0e-5,
                "nu_m2_s": 1.5e-5,
                "omega_zero_guard": "fallback_stagnant_film",
                "delta_eff_m": 8.0e-4,
            },
            omega_rad_s=omega,
        )
        km_fallback = compute_km(
            "stagnant_film",
            grid=grid,
            params={"diffusivity_m2_s": 2.0e-5, "delta_eff_m": 8.0e-4},
        )
        self.assertGreater(float(np.max(np.abs(km_rot - km_fallback))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
