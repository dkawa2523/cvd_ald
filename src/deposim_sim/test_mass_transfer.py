from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, ModelSpec

from .domain import build_domain_grid
from .models import mass_transfer


@unittest.skipIf(np is None, "NumPy is required for mass-transfer tests")
class TestMassTransferModels(unittest.TestCase):
    def _build_grid(self):
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=100.0,
            nr=3,
            ntheta=4,
            edge_exclusion_mm=0.0,
        )
        return build_domain_grid(spec)

    def test_registry_resolves_from_model_config(self) -> None:
        grid = self._build_grid()
        model = ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
        )

        km = mass_transfer.compute_km_from_model_config(model, grid=grid)

        self.assertEqual(km.shape, grid.shape)
        np.testing.assert_allclose(km, np.full(grid.shape, 0.05))

    def test_rotating_disk_omega_zero_guard_error(self) -> None:
        grid = self._build_grid()
        model = ModelSpec(
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
        model = ModelSpec(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.5,
                "diffusivity_m2_s": 2.0e-5,
                "nu_m2_s": 1.0e-5,
                "delta_eff_m": 1.0e-3,
                "omega_zero_guard": "fallback_stagnant_film",
            },
        )
        omega_field = np.array(
            [
                [0.0, 9.0, 16.0, 0.0],
                [4.0, 0.0, 1.0, 25.0],
                [0.0, 0.0, 36.0, 49.0],
            ],
            dtype=float,
        )

        km = mass_transfer.compute_km_from_model_config(model, grid=grid, omega_rad_s=omega_field)

        self.assertEqual(km.shape, grid.shape)
        expected_rot = 0.5 * (2.0e-5 ** (2.0 / 3.0)) * np.sqrt(omega_field) * (1.0e-5 ** (-1.0 / 6.0))
        expected_fallback = np.full(grid.shape, 2.0e-5 / 1.0e-3)
        expected = np.where(omega_field == 0.0, expected_fallback, expected_rot)
        np.testing.assert_allclose(km, expected)

    def test_scalar_or_spatial_inputs_are_aligned_to_grid(self) -> None:
        grid = self._build_grid()
        diffusivity_by_radius = np.array([[1.0e-5], [2.0e-5], [3.0e-5]])

        km = mass_transfer.compute_km(
            "stagnant_film",
            grid=grid,
            diffusivity_m2_s=diffusivity_by_radius,
            delta_eff_m=1.0e-3,
        )

        self.assertEqual(km.shape, grid.shape)
        expected = diffusivity_by_radius / 1.0e-3
        np.testing.assert_allclose(km, np.broadcast_to(expected, grid.shape))

    def test_run_config_forwards_input_omega(self) -> None:
        grid = self._build_grid()
        omega_field = np.array(
            [
                [1.0, 4.0, 9.0, 16.0],
                [25.0, 36.0, 49.0, 64.0],
                [81.0, 100.0, 121.0, 144.0],
            ],
            dtype=float,
        )
        model = ModelSpec(
            mass_transfer_name="rotating_disk",
            mass_transfer_params={
                "ck": 0.5,
                "diffusivity_m2_s": 2.0e-5,
                "nu_m2_s": 1.0e-5,
                "omega_zero_guard": "error",
            },
        )
        run_config = SimpleNamespace(model=model, inputs=SimpleNamespace(omega_rad_s=omega_field))

        km_from_run = mass_transfer.compute_km_from_run_config(run_config, grid=grid)
        km_direct = mass_transfer.compute_km_from_model_config(model, grid=grid, omega_rad_s=omega_field)
        np.testing.assert_allclose(km_from_run, km_direct)


if __name__ == "__main__":
    unittest.main()
