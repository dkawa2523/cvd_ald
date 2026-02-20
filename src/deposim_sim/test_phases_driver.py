from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, DriversSpec, InputsSpec, ModelSpec, ReferencePlaneSpec, RunSpec, SolverSpec, TimeSpec

from .phases_driver import build_phase_input_preview, run_phased_synthetic


@unittest.skipIf(np is None, "NumPy is required for phases/driver tests")
class TestPhasesDriver(unittest.TestCase):
    def _run_spec(self) -> RunSpec:
        return RunSpec(
            run_name="phase_test",
            domain=DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=60.0, nr=3, ntheta=6, edge_exclusion_mm=0.0),
            reference_plane=ReferencePlaneSpec(z_ref_mm=5.0, species=["A"]),
            time=TimeSpec(
                mode="ald_cycle",
                process_time_s=6.0,
                phases=[
                    {"name": "expose", "duration_s": 2.0, "scalar_overrides": {"c_ref_mol_m3": 1.5}},
                    {"name": "purge", "duration_s": 4.0, "scalar_overrides": {"c_ref_mol_m3": 0.6}},
                ],
            ),
            inputs=InputsSpec(synthetic_case="uniform", c_ref_mol_m3=1.0, temperature_k=700.0, pressure_pa=1200.0, omega_rad_s=0.0),
            drivers=DriversSpec(
                enable_time_driver=True,
                scalar_schedule={
                    "purge": {"omega_rad_s": 20.0},
                },
            ),
            model=ModelSpec(
                mass_transfer_name="stagnant_film",
                mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
                kinetics_name="power_law",
                kinetics_params={"k0": 0.2, "orders": {"A": 1.0}, "nu": {"A": 1.0}},
                net_name="deposition_only",
            ),
            solver=SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False),
        )

    def test_preview_reflects_phase_and_driver_scalars(self) -> None:
        spec = self._run_spec()
        preview = build_phase_input_preview(spec)
        self.assertEqual(len(preview), 2)
        self.assertEqual(preview[0]["phase_name"], "expose")
        self.assertEqual(preview[1]["phase_name"], "purge")
        self.assertAlmostEqual(preview[0]["effective_scalars"]["c_ref_mol_m3"], 1.5)
        self.assertAlmostEqual(preview[1]["effective_scalars"]["c_ref_mol_m3"], 0.6)
        self.assertAlmostEqual(preview[1]["effective_scalars"]["omega_rad_s"], 20.0)

    def test_run_phased_synthetic_executes_and_accumulates(self) -> None:
        spec = self._run_spec()
        result = run_phased_synthetic(spec)
        self.assertEqual(len(result.phase_thickness), 2)
        self.assertEqual(len(result.input_preview), 2)
        self.assertEqual(result.total_thickness.shape, result.phase_thickness[0].shape)
        expected = result.phase_thickness[0] + result.phase_thickness[1]
        np.testing.assert_allclose(result.total_thickness, expected)
        self.assertTrue(np.all(np.isfinite(result.total_thickness)))


if __name__ == "__main__":
    unittest.main()
