from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, ModelSpec, SolverSpec

from .domain import build_domain_grid
from .physics.cvd_steady import FieldBundle, run_cvd_steady


@unittest.skipIf(np is None, "NumPy is required for cvd_steady tests")
class TestCVDSteady(unittest.TestCase):
    def _build_grid(self):
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=100.0,
            nr=3,
            ntheta=4,
            edge_exclusion_mm=0.0,
        )
        return build_domain_grid(spec)

    def _build_single_species_model(self) -> ModelSpec:
        return ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
            kinetics_name="power_law",
            kinetics_params={"k0": 0.2, "orders": {"A": 1.0}, "nu": {"A": 1.0}},
        )

    def test_run_returns_aligned_outputs_and_required_diagnostics(self) -> None:
        grid = self._build_grid()
        fields = FieldBundle(
            C_ref={"A": np.full(grid.shape, 1.0, dtype=float)},
            T=np.full(grid.shape, 700.0, dtype=float),
            scalars={"omega_rad_s": 0.0},
        )
        solver = SolverSpec(max_iter=60, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=True)
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._build_single_species_model(),
            process_time_s=10.0,
            solver_config=solver,
        )

        self.assertEqual(result.thickness.shape, grid.shape)
        self.assertEqual(result.deposition_rate.shape, grid.shape)
        self.assertEqual(result.R.shape, grid.shape)
        self.assertIn("A", result.Cs)
        self.assertEqual(result.Cs["A"].shape, grid.shape)

        required_diag_keys = {
            "Cs_over_Cref",
            "Da_proxy",
            "apparent_orders",
            "root_iteration_count",
            "root_status_map",
            "root_failure_mask",
            "root_failure_fraction",
            "sign_convention",
        }
        self.assertTrue(required_diag_keys.issubset(result.diagnostics.keys()))
        self.assertEqual(result.diagnostics["Da_proxy"].shape, grid.shape)
        self.assertEqual(result.diagnostics["root_iteration_count"].shape, grid.shape)
        self.assertEqual(result.diagnostics["root_status_map"].shape, grid.shape)
        self.assertEqual(result.diagnostics["root_failure_mask"].shape, grid.shape)
        self.assertIsInstance(result.diagnostics["root_failure_fraction"], float)

    def test_thickness_relation_and_sign_convention(self) -> None:
        grid = self._build_grid()
        process_time_s = 12.5
        fields = FieldBundle(C_ref={"A": np.full(grid.shape, 1.0, dtype=float)})
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._build_single_species_model(),
            process_time_s=process_time_s,
            solver_config=SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False),
        )

        np.testing.assert_allclose(result.thickness, result.deposition_rate * process_time_s)
        self.assertEqual(result.diagnostics["sign_convention"], "deposit_positive_etch_negative")

    def test_invalid_process_time_raises(self) -> None:
        grid = self._build_grid()
        fields = FieldBundle(C_ref={"A": np.full(grid.shape, 1.0, dtype=float)})
        with self.assertRaisesRegex(ValueError, "process_time_s must be > 0"):
            run_cvd_steady(
                grid=grid,
                fields=fields,
                model_config=self._build_single_species_model(),
                process_time_s=0.0,
            )

    def test_multi_species_uses_nu_from_kinetics_params(self) -> None:
        grid = self._build_grid()
        fields = FieldBundle(
            C_ref={
                "A": np.full(grid.shape, 1.2, dtype=float),
                "B": np.full(grid.shape, 0.8, dtype=float),
            },
            T=np.full(grid.shape, 700.0, dtype=float),
        )
        model = ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
            kinetics_name="power_law",
            kinetics_params={
                "k0": 0.05,
                "orders": {"A": 1.0, "B": 0.5},
                "nu": {"A": 1.0, "B": 0.5},
            },
        )
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=model,
            process_time_s=5.0,
            solver_config=SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False),
        )

        self.assertIn("A", result.Cs)
        self.assertIn("B", result.Cs)
        self.assertIn("A", result.diagnostics["Cs_over_Cref"])
        self.assertIn("B", result.diagnostics["Cs_over_Cref"])
        self.assertEqual(result.Cs["A"].shape, grid.shape)
        self.assertEqual(result.Cs["B"].shape, grid.shape)
        self.assertTrue(np.all(result.Cs["A"] >= 0.0))
        self.assertTrue(np.all(result.Cs["B"] >= 0.0))

    def test_solver_health_diagnostic_types(self) -> None:
        grid = self._build_grid()
        fields = FieldBundle(C_ref={"A": np.full(grid.shape, 1.0, dtype=float)})
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._build_single_species_model(),
            process_time_s=2.0,
            solver_config=SolverSpec(max_iter=2, rtol=1.0e-14, atol=1.0e-20, monotonicity_check=False),
        )

        failure_mask = result.diagnostics["root_failure_mask"]
        status_map = result.diagnostics["root_status_map"]
        iteration_count = result.diagnostics["root_iteration_count"]
        self.assertEqual(failure_mask.shape, grid.shape)
        self.assertEqual(status_map.shape, grid.shape)
        self.assertEqual(iteration_count.shape, grid.shape)
        self.assertTrue(np.issubdtype(failure_mask.dtype, np.bool_))
        self.assertTrue(np.issubdtype(status_map.dtype, np.integer))
        self.assertTrue(np.issubdtype(iteration_count.dtype, np.integer))
        self.assertGreaterEqual(result.diagnostics["root_failure_fraction"], 0.0)
        self.assertLessEqual(result.diagnostics["root_failure_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
