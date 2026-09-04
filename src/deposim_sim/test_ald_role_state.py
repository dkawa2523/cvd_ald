from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .pipeline import run_sim_from_spec


@unittest.skipIf(np is None, "NumPy is required")
class TestALDRoleState(unittest.TestCase):
    def test_mean_rate_observations_use_growth_and_original_coordinates(self):
        with TemporaryDirectory() as tmp:
            fluent = Path(tmp) / "ald.npz"
            measurement = Path(tmp) / "rate.npz"
            self._write_ald_input(fluent)
            spec = compose_sim_config("ald_state_min", overrides=[
                f"sim.inputs.fluent.file={fluent}", "sim.initial_conditions.h_nm.value=5.0",
            ])
            predicted = run_sim_from_spec(spec)
            expected_rate = predicted.thickness - 5.0  # one-second process
            np.testing.assert_allclose(predicted.deposition_rate, expected_rate)
            indices = [2, 0]
            xy_m = np.asarray(predicted.diagnostics["xy_mm"])[indices] / 1000.
            np.savez(measurement, rate=expected_rate[indices], xy=xy_m, noise=np.full(2, .01))
            spec.measurement.enabled = True
            spec.measurement.file = str(measurement)
            spec.measurement.keys = {"h": "rate", "xy": "xy", "sigma": "noise"}
            spec.measurement.quantity = "mean_rate"
            spec.measurement.xy_unit = "m"
            compared = run_sim_from_spec(spec)
            observed = compared.diagnostics["observation"]
            self.assertEqual(observed["count"], 2)
            np.testing.assert_allclose(observed["residual_nm"], 0., atol=1e-12)
            np.testing.assert_allclose(observed["sigma_nm"], .01)

    def _write_ald_input(self, path: Path, *, dose_scale: float = 1.0) -> None:
        xy = np.asarray([[0.0, 0.0], [40.0, 0.0], [-40.0, 0.0]], dtype=float)
        time = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
        carrier = np.full(3, 0.02, dtype=float)
        a = dose_scale * np.asarray([1.0, 0.9, 0.85], dtype=float)
        b = np.asarray([1.0, 0.95, 0.9], dtype=float)
        i = np.asarray([0.05, 0.06, 0.04], dtype=float)
        residual = 0.01
        frames = [
            np.stack([a, residual * b, i, carrier], axis=1),
            np.stack([residual * a, residual * b, residual * i, carrier], axis=1),
            np.stack([residual * a, b, residual * i, carrier], axis=1),
            np.stack([residual * a, residual * b, residual * i, carrier], axis=1),
            np.stack([residual * a, residual * b, residual * i, carrier], axis=1),
        ]
        cref = np.stack(frames, axis=0)
        np.savez(path, xy=xy, time=time, cref=cref)

    def test_role_ald_state_runs_and_keeps_latent_states_bounded(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = Path(tmp) / "ald.npz"
            self._write_ald_input(fluent)
            spec = compose_sim_config("ald_state_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            out = run_sim_from_spec(spec)
            self.assertEqual(out.thickness.shape, (3,))
            self.assertIn("theta_I", out.fields)
            self.assertIn("theta_free", out.fields)
            self.assertEqual(out.diagnostics["process_model_implementation"], "ald_role_state")
            self.assertEqual(out.diagnostics["solver_kind"], "explicit_substep_bounded")
            self.assertFalse(bool(out.diagnostics["root_metrics_applicable"]))
            self.assertEqual(out.diagnostics["bounded_violation_count"], 0)
            self.assertGreater(float(np.mean(out.thickness)), 0.0)
            self.assertTrue(np.all(np.asarray(out.fields["theta_A"]) >= 0.0))
            self.assertTrue(np.all(np.asarray(out.fields["theta_A"]) <= 1.0))
            self.assertTrue(np.all(np.asarray(out.fields["theta_I"]) >= 0.0))
            self.assertTrue(np.all(np.asarray(out.fields["theta_free"]) >= 0.0))

    def test_role_ald_state_has_saturating_dose_trend(self) -> None:
        with TemporaryDirectory() as tmp:
            low = Path(tmp) / "low.npz"
            nominal = Path(tmp) / "nominal.npz"
            high = Path(tmp) / "high.npz"
            self._write_ald_input(low, dose_scale=0.3)
            self._write_ald_input(nominal, dose_scale=1.0)
            self._write_ald_input(high, dose_scale=4.0)

            means = []
            for path in (low, nominal, high):
                spec = compose_sim_config("ald_state_min", overrides=[f"sim.inputs.fluent.file={path}"])
                out = run_sim_from_spec(spec)
                means.append(float(np.mean(out.thickness)))

            self.assertLess(means[0], means[1])
            self.assertLess(means[1], means[2])
            self.assertLess(means[2] - means[1], means[1] - means[0])

    def test_a_only_channel_can_grow_without_role_b(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = Path(tmp) / "ald_a_only.npz"
            self._write_ald_input(fluent)
            spec = compose_sim_config(
                "ald_state_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent}",
                    "sim.roles.B=null",
                    "sim.model.params.kinetics.k_convert_A=0.3",
                ],
            )
            out = run_sim_from_spec(spec)
            self.assertGreater(float(np.mean(out.thickness)), 0.0)
            self.assertEqual(out.diagnostics["ald_role_state"]["event_channel"], "A")

    def test_time_step_refinement_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = Path(tmp) / "ald_refine.npz"
            self._write_ald_input(fluent)
            means = []
            for dt_s in (0.01, 0.005, 0.0025):
                spec = compose_sim_config(
                    "ald_state_min",
                    overrides=[f"sim.inputs.fluent.file={fluent}", f"sim.time.dt_s={dt_s}"],
                )
                out = run_sim_from_spec(spec)
                means.append(float(np.mean(out.thickness)))
            coarse_delta = abs(means[1] - means[0])
            fine_delta = abs(means[2] - means[1])
            self.assertLessEqual(fine_delta, coarse_delta + 1.0e-12)


if __name__ == "__main__":
    unittest.main()
