from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_schema import compose_sim_config

from .models.mvk_state import run_mvk_state
from .pipeline import run_sim_from_spec
from .transport_provider import (
    DirectSurfaceConcentrationProvider,
    FitScalarKmProvider,
)


class TestMVKState(unittest.TestCase):
    def test_direct_surface_converges_to_analytic_redox_steady_state(self) -> None:
        shape = (3,)
        provider = DirectSurfaceConcentrationProvider(spatial_shape=shape)
        c_a = np.full((2, *shape), 2.0)
        c_b = np.full((2, *shape), 1.0)
        result = run_mvk_state(
            c_a=c_a,
            c_b=c_b,
            km_provider=provider,
            time_s=np.array([0.0, 20.0]),
            dt_max_s=0.2,
            oxidized_fraction0=np.ones(shape),
            h0_nm=np.zeros(shape),
            k_reduce=np.ones(shape),
            k_regenerate=np.ones(shape),
            gamma_s=np.ones(shape),
            nu_b=np.ones(shape),
            alpha_h=np.ones(shape),
            max_iter=60,
            state_tol=1.0e-12,
        )
        np.testing.assert_allclose(result.oxidized_fraction, 1.0 / 3.0, atol=1.0e-10)
        np.testing.assert_allclose(
            result.reduction_rate, result.regeneration_rate, atol=1.0e-10
        )
        self.assertTrue(np.all(result.h_nm > 0.0))
        np.testing.assert_allclose(result.time_s, [0.0, 20.0])
        np.testing.assert_allclose(result.oxidized_fraction_history[0], 1.0)
        np.testing.assert_allclose(
            result.oxidized_fraction_history[-1], result.oxidized_fraction
        )
        np.testing.assert_allclose(result.h_nm_history[-1], result.h_nm)

    def test_scalar_transport_closes_both_surface_fluxes(self) -> None:
        shape = (2,)
        km_a = np.full(shape, 0.7)
        km_b = np.full(shape, 0.4)
        provider = FitScalarKmProvider(km_a=km_a, km_b=km_b, spatial_shape=shape)
        c_a = np.full((2, *shape), 1.2)
        c_b = np.full((2, *shape), 0.8)
        result = run_mvk_state(
            c_a=c_a,
            c_b=c_b,
            km_provider=provider,
            time_s=np.array([0.0, 5.0]),
            dt_max_s=0.1,
            oxidized_fraction0=np.full(shape, 0.5),
            h0_nm=np.zeros(shape),
            k_reduce=np.full(shape, 0.6),
            k_regenerate=np.full(shape, 0.9),
            gamma_s=np.full(shape, 0.2),
            nu_b=np.full(shape, 2.0),
            alpha_h=np.ones(shape),
            max_iter=60,
            state_tol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.j_a_surface, km_a * (c_a[0] - result.cs_a), rtol=1.0e-12
        )
        np.testing.assert_allclose(
            result.j_b_surface, km_b * (c_b[0] - result.cs_b), rtol=1.0e-12
        )

    def test_redox_state_retains_feed_history(self) -> None:
        shape = (1,)
        provider = DirectSurfaceConcentrationProvider(spatial_shape=shape)

        def execute(first_a: float, first_b: float):
            return run_mvk_state(
                c_a=np.array([[first_a], [0.2], [0.2]]),
                c_b=np.array([[first_b], [0.2], [0.2]]),
                km_provider=provider,
                time_s=np.array([0.0, 1.0, 1.1]),
                dt_max_s=0.01,
                oxidized_fraction0=np.array([0.5]),
                h0_nm=np.zeros(shape),
                k_reduce=np.ones(shape),
                k_regenerate=np.ones(shape),
                gamma_s=np.ones(shape),
                nu_b=np.ones(shape),
                alpha_h=np.ones(shape),
                max_iter=60,
                state_tol=1.0e-12,
            )

        reduced_first = execute(2.0, 0.0)
        regenerated_first = execute(0.0, 2.0)
        self.assertLess(
            float(reduced_first.oxidized_fraction[0]),
            float(regenerated_first.oxidized_fraction[0]),
        )
        self.assertNotAlmostEqual(
            float(reduced_first.reduction_rate[0]),
            float(regenerated_first.reduction_rate[0]),
        )

    def test_pipeline_dispatches_mvk_and_exposes_units(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mvk_transient.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]])
            time = np.array([0.0, 0.5, 1.0])
            cref = np.array(
                [
                    [[1.0, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
                    [[0.2, 1.0, 0.0, 0.0], [0.3, 0.9, 0.0, 0.0]],
                    [[0.2, 1.0, 0.0, 0.0], [0.3, 0.9, 0.0, 0.0]],
                ]
            )
            np.savez(path, xy=xy, time=time, cref=cref)
            spec = compose_sim_config(
                "cvd_mvk_transient_min",
                overrides=[
                    f"sim.inputs.fluent.file={path}",
                    "sim.time.dt_s=0.05",
                ],
            )
            result = run_sim_from_spec(spec)
            self.assertEqual(result.diagnostics["process_model_implementation"], "mvk_state")
            self.assertEqual(result.diagnostics["units"]["oxidized_fraction"], "1")
            self.assertEqual(result.diagnostics["units"]["surface_flux"], "kmol/(m^2 s)")
            self.assertIn("oxidized_fraction", result.fields)
            self.assertIn("regeneration_rate_s-1", result.fields)
            self.assertIn("oxidized_fraction_history", result.fields)
            self.assertIn("J_A_surface_history", result.fields)
            self.assertEqual(result.fields["oxidized_fraction_history"].shape, (3, 2))
            np.testing.assert_allclose(result.fields["time_s"], time)
            np.testing.assert_allclose(
                result.fields["oxidized_fraction_history"][-1],
                result.fields["oxidized_fraction"],
            )
            np.testing.assert_allclose(
                result.fields["oxidized_fraction"]
                + result.fields["reduced_fraction"],
                1.0,
            )

    def test_pipeline_runs_steady_fluent_input_as_finite_process_interval(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mvk_steady.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]])
            cref = np.array(
                [[1.0, 0.4, 0.0, 0.0], [0.8, 0.5, 0.0, 0.0]]
            )
            np.savez(path, xy=xy, cref=cref)
            spec = compose_sim_config(
                "cvd_mvk_transient_min",
                overrides=[
                    "sim.time_mode=steady",
                    "sim.inputs.fluent.mode=steady",
                    f"sim.inputs.fluent.file={path}",
                    "sim.time.t_proc_s=1.0",
                    "sim.time.dt_s=0.05",
                ],
            )
            result = run_sim_from_spec(spec)
            self.assertEqual(result.diagnostics["dispatch_mode"], "steady")
            self.assertTrue(np.all(np.isfinite(result.thickness)))
            self.assertTrue(np.all(result.thickness > 0.0))

    def test_pipeline_adapts_film_and_redox_history_observations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "mvk_transient.npz"
            measurement_path = root / "mvk_measurement.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]])
            time = np.array([0.0, 0.5, 1.0])
            cref = np.array(
                [
                    [[1.0, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
                    [[0.2, 1.0, 0.0, 0.0], [0.3, 0.9, 0.0, 0.0]],
                    [[0.2, 1.0, 0.0, 0.0], [0.3, 0.9, 0.0, 0.0]],
                ]
            )
            np.savez(fluent_path, xy=xy, time=time, cref=cref)
            spec = compose_sim_config(
                "cvd_mvk_transient_min",
                overrides=[f"sim.inputs.fluent.file={fluent_path}"],
            )
            truth = run_sim_from_spec(spec)
            order = np.array([1, 0])
            np.savez(
                measurement_path,
                xy=xy[order],
                h_nm=truth.thickness[order],
                h_sigma=np.full(xy.shape[0], 0.01),
                time=time,
                chi=truth.fields["oxidized_fraction_history"][:, order],
                chi_sigma=np.full(
                    truth.fields["oxidized_fraction_history"].shape, 0.02
                ),
            )
            spec.measurement.enabled = True
            spec.measurement.file = str(measurement_path)
            spec.measurement.keys = {
                "xy": "xy",
                "h": "h_nm",
                "sigma": "h_sigma",
                "time": "time",
                "oxidized_fraction_history": "chi",
                "oxidized_fraction_history_sigma": "chi_sigma",
            }
            compared = run_sim_from_spec(spec)
            observations = compared.diagnostics["observations"]
            self.assertEqual(set(observations), {"film", "oxidized_fraction_history"})
            np.testing.assert_allclose(
                observations["oxidized_fraction_history"]["target"],
                observations["oxidized_fraction_history"]["prediction"],
            )

    def test_thickness_history_does_not_duplicate_final_film(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "mvk_transient.npz"
            measurement_path = root / "mvk_measurement.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]])
            time = np.array([0.0, 0.5, 1.0])
            cref = np.ones((3, 2, 4), dtype=float)
            np.savez(fluent_path, xy=xy, time=time, cref=cref)
            spec = compose_sim_config(
                "cvd_mvk_transient_min",
                overrides=[f"sim.inputs.fluent.file={fluent_path}"],
            )
            truth = run_sim_from_spec(spec)
            history = truth.fields["h_nm_history"]
            np.savez(
                measurement_path,
                xy=xy,
                h_nm=truth.thickness,
                h_sigma=np.full(2, 0.01),
                time=time,
                h_history=history,
                h_history_sigma=np.full(history.shape, 0.02),
            )
            spec.measurement.enabled = True
            spec.measurement.file = str(measurement_path)
            spec.measurement.keys = {
                "xy": "xy",
                "h": "h_nm",
                "sigma": "h_sigma",
                "time": "time",
                "h_nm_history": "h_history",
                "h_nm_history_sigma": "h_history_sigma",
            }
            observations = run_sim_from_spec(spec).diagnostics["observations"]
            self.assertEqual(observations["film"]["target"].size, 2)
            self.assertEqual(observations["h_nm_history"]["target"].size, 4)

    def test_mvk_requires_b_and_keeps_inhibition_separate(self) -> None:
        for override, message in (
            ("sim.roles.B=null", "requires sim.roles.B"),
            ("sim.roles.I=s2", "does not use role I"),
        ):
            spec = compose_sim_config(
                "cvd_mvk_transient_min", overrides=[override]
            )
            with self.assertRaisesRegex(ValueError, message):
                run_sim_from_spec(spec)


if __name__ == "__main__":
    unittest.main()
