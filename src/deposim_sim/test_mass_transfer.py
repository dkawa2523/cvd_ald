from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .input_builder import build_domain_from_fluent_xy
from .models import mass_transfer


@unittest.skipIf(np is None, "NumPy is required for mass-transfer tests")
class TestMassTransferModels(unittest.TestCase):
    def _build_grid(self):
        xy = np.array(
            [[-20.0, -20.0], [-10.0, 0.0], [0.0, 0.0], [10.0, 15.0], [20.0, -10.0], [25.0, 20.0]],
            dtype=float,
        )
        return build_domain_from_fluent_xy(xy=xy, xy_unit="mm", wafer_radius_mm=50.0)

    def test_registry_resolves_from_model_config(self) -> None:
        grid = self._build_grid()
        model = SimpleNamespace(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
        )
        km = mass_transfer.compute_km_from_model_config(model, grid=grid)
        self.assertEqual(km.shape, grid.shape)
        np.testing.assert_allclose(km, np.full(grid.shape, 0.05))

    def test_rotating_disk_omega_zero_guard_error(self) -> None:
        grid = self._build_grid()
        model = SimpleNamespace(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.62,
                "diffusivity_m2_s": 1.0e-4,
                "nu_m2_s": 1.5e-5,
                "omega_zero_guard": "error",
            },
        )
        with self.assertRaisesRegex(ValueError, "omega_rad_s=0"):
            mass_transfer.compute_km_from_model_config(model, grid=grid, omega_rad_s=0.0)

    def test_rotating_disk_omega_zero_fallback_to_stagnant_film(self) -> None:
        grid = self._build_grid()
        model = SimpleNamespace(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.5,
                "diffusivity_m2_s": 2.0e-5,
                "nu_m2_s": 1.0e-5,
                "delta_eff_m": 1.0e-3,
                "omega_zero_guard": "fallback_stagnant_film",
            },
        )
        omega = np.array([0.0, 9.0, 16.0, 0.0, 4.0, 1.0], dtype=float)
        km = mass_transfer.compute_km_from_model_config(model, grid=grid, omega_rad_s=omega)
        expected_rot = 0.5 * (2.0e-5 ** (2.0 / 3.0)) * np.sqrt(omega) * (1.0e-5 ** (-1.0 / 6.0))
        expected_fallback = np.full(grid.shape, 2.0e-5 / 1.0e-3)
        expected = np.where(omega == 0.0, expected_fallback, expected_rot)
        np.testing.assert_allclose(km, expected)

    def test_scalar_or_spatial_inputs_are_aligned_to_grid(self) -> None:
        grid = self._build_grid()
        diffusivity_by_point = np.array([1.0e-5, 2.0e-5, 3.0e-5, 1.5e-5, 2.5e-5, 3.5e-5], dtype=float)
        km = mass_transfer.compute_km(
            "stagnant_film",
            grid=grid,
            diffusivity_m2_s=diffusivity_by_point,
            delta_eff_m=1.0e-3,
        )
        self.assertEqual(km.shape, grid.shape)
        np.testing.assert_allclose(km, diffusivity_by_point / 1.0e-3)

    def test_run_config_forwards_input_omega(self) -> None:
        grid = self._build_grid()
        omega = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0], dtype=float)
        model = SimpleNamespace(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.5,
                "diffusivity_m2_s": 2.0e-5,
                "nu_m2_s": 1.0e-5,
                "omega_zero_guard": "error",
            },
        )
        run_config = SimpleNamespace(model=model, inputs=SimpleNamespace(omega_rad_s=omega))
        km_from_run = mass_transfer.compute_km_from_run_config(run_config, grid=grid)
        km_direct = mass_transfer.compute_km_from_model_config(model, grid=grid, omega_rad_s=omega)
        np.testing.assert_allclose(km_from_run, km_direct)


if __name__ == "__main__":
    unittest.main()
