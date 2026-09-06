from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .pipeline import run_aib_from_spec


@unittest.skipIf(np is None, "NumPy is required")
class TestPipelineAIB(unittest.TestCase):
    def _write_fluent(self, path: Path) -> np.ndarray:
        xy = np.array(
            [
                [-20.0, -10.0],
                [0.0, -5.0],
                [10.0, 12.0],
                [25.0, -15.0],
            ],
            dtype=float,
        )
        cref = np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.9, 0.5, 0.1, 0.0],
                [0.7, 0.4, 0.1, 0.0],
                [0.6, 0.3, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref)
        return xy

    def _write_fluent_with_flux(self, path: Path) -> np.ndarray:
        xy = np.array(
            [
                [-20.0, -10.0],
                [0.0, -5.0],
                [10.0, 12.0],
                [25.0, -15.0],
            ],
            dtype=float,
        )
        cref = np.array(
            [
                [1.0, 0.4, 0.2, 0.1],
                [0.9, 0.5, 0.2, 0.1],
                [0.7, 0.4, 0.2, 0.1],
                [0.6, 0.3, 0.2, 0.1],
            ],
            dtype=float,
        )
        flux_sink = np.array(
            [
                [0.10, 0.02, 0.03, 0.01],
                [0.09, 0.02, 0.03, 0.01],
                [0.06, 0.02, 0.03, 0.01],
                [0.05, 0.02, 0.03, 0.01],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref, flux_sink=flux_sink)
        return xy

    def _write_fluent_transient(self, path: Path) -> np.ndarray:
        xy = np.array(
            [
                [-20.0, -10.0],
                [0.0, -5.0],
                [10.0, 12.0],
                [25.0, -15.0],
            ],
            dtype=float,
        )
        time = np.array([0.0, 0.5, 1.0], dtype=float)
        cref = np.array(
            [
                [[1.00, 0.40, 0.10, 0.00], [0.90, 0.50, 0.10, 0.00], [0.70, 0.40, 0.10, 0.00], [0.60, 0.30, 0.10, 0.00]],
                [[1.05, 0.40, 0.10, 0.00], [0.95, 0.50, 0.10, 0.00], [0.75, 0.40, 0.10, 0.00], [0.65, 0.30, 0.10, 0.00]],
                [[1.10, 0.40, 0.10, 0.00], [1.00, 0.50, 0.10, 0.00], [0.80, 0.40, 0.10, 0.00], [0.70, 0.30, 0.10, 0.00]],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, time=time, cref=cref)
        return xy

    def test_run_aib_from_spec_steady(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            self._write_fluent(fluent_path)

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[f"sim.inputs.fluent.file={fluent_path}"],
            )
            out = run_aib_from_spec(spec)
            self.assertEqual(out.thickness.shape, (4,))
            self.assertIn("h_nm", out.fields)
            self.assertIn("phi_B", out.fields)
            self.assertIn("f_I", out.fields)
            self.assertIn("residual_nm", out.fields)
            self.assertIn("xy_mm", out.diagnostics)
            self.assertIn("root_iteration_count", out.diagnostics)
            self.assertIn("root_status_map", out.diagnostics)
            self.assertIn("root_non_bracket_count_map", out.diagnostics)
            self.assertEqual(np.asarray(out.diagnostics["root_iteration_count"]).shape, (4,))
            self.assertEqual(np.asarray(out.diagnostics["root_status_map"]).shape, (4,))
            self.assertEqual(out.grid.kind, "from_fluent_xy")

    def test_measurement_align_changes_residual(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            xy = self._write_fluent(fluent_path)

            meas_xy = xy[::-1].copy()
            meas_h = 0.2 * meas_xy[:, 0] - 0.1 * meas_xy[:, 1]
            meas_path = Path(tmp) / "meas.npz"
            np.savez(meas_path, h_nm=meas_h, xy=meas_xy)

            spec_no_align = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.measurement.enabled=true",
                    f"sim.measurement.file={meas_path}",
                    "sim.measurement.align.enable=false",
                ],
            )
            out_no_align = run_aib_from_spec(spec_no_align)

            spec_align = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.measurement.enabled=true",
                    f"sim.measurement.file={meas_path}",
                    "sim.measurement.align.enable=true",
                    "sim.measurement.align.shift_mm=[0.0,0.0]",
                    "sim.measurement.align.rotate_deg=0.0",
                    "sim.measurement.align.mask_radius_mm=150.0",
                ],
            )
            out_align = run_aib_from_spec(spec_align)

            residual_no_align = np.asarray(out_no_align.fields["residual_nm"], dtype=float)
            residual_align = np.asarray(out_align.fields["residual_nm"], dtype=float)
            self.assertGreater(float(np.nanmax(np.abs(residual_no_align - residual_align))), 0.0)
            self.assertIn("measurement_valid_mask", out_align.diagnostics)

    def test_pipeline_uses_configured_io_loaders(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent_payload.data"
            meas_path = root / "meas_payload.data"
            fluent_path.write_text(
                "x,y,s0,s1,s2,s3\n"
                "-20.0,-10.0,1.0,0.4,0.1,0.0\n"
                "0.0,-5.0,0.9,0.5,0.1,0.0\n"
                "10.0,12.0,0.7,0.4,0.1,0.0\n"
                "25.0,-15.0,0.6,0.3,0.1,0.0\n",
                encoding="utf-8",
            )
            meas_path.write_text(
                "x,y,h_nm\n"
                "-20.0,-10.0,1.0\n"
                "0.0,-5.0,1.1\n"
                "10.0,12.0,1.2\n"
                "25.0,-15.0,1.3\n",
                encoding="utf-8",
            )

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.inputs.fluent.io_loader_name=csv",
                    "sim.measurement.enabled=true",
                    f"sim.measurement.file={meas_path}",
                    "sim.measurement.io_loader_name=csv",
                    "sim.measurement.align.enable=false",
                ],
            )
            out = run_aib_from_spec(spec)
            self.assertEqual(out.thickness.shape, (4,))
            self.assertTrue(np.all(np.isfinite(np.asarray(out.fields["residual_nm"], dtype=float))))

    def test_fit_scalar_explicit_matches_default(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            self._write_fluent(fluent_path)
            base = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_path}"])
            explicit = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.model.params.transport.km_source=fit_scalar",
                ],
            )
            out_base = run_aib_from_spec(base)
            out_explicit = run_aib_from_spec(explicit)
            self.assertTrue(np.allclose(np.asarray(out_base.thickness), np.asarray(out_explicit.thickness)))

    def test_from_cfd_flux_sink_uses_flux_field(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent_flux.npz"
            self._write_fluent_with_flux(fluent_path)
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.model.params.transport.km_source=from_cfd_flux_sink",
                    "sim.model.params.transport.gamma_km_A=1.0",
                    "sim.model.params.transport.from_cfd_flux_sink.flux_negative_policy=error",
                ],
            )
            out = run_aib_from_spec(spec)
            km_map = np.asarray(out.diagnostics["km_A_map"], dtype=float)
            self.assertTrue(np.all(np.isfinite(km_map)))
            self.assertGreater(float(np.nanmax(km_map) - np.nanmin(km_map)), 0.0)
            self.assertEqual(out.diagnostics["km_source"], "from_cfd_flux_sink")

    def test_from_cfd_flux_sink_requires_flux_input(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent_no_flux.npz"
            self._write_fluent(fluent_path)
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.model.params.transport.km_source=from_cfd_flux_sink",
                ],
            )
            with self.assertRaises(ValueError):
                run_aib_from_spec(spec)

    def test_direct_surface_bypasses_transport_closure(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "wall_concentration.npz"
            self._write_fluent(fluent_path)
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.roles.B=s1",
                    "sim.model.params.transport.km_source=direct_surface",
                    "sim.time.t_proc_s=0.1",
                ],
            )
            out = run_aib_from_spec(spec)
            self.assertEqual(out.diagnostics["concentration_location"], "wall")
            np.testing.assert_allclose(out.fields["CsA_over_CrefA"], 1.0)
            np.testing.assert_allclose(out.fields["CsB_over_CrefB"], 1.0)
            self.assertTrue(np.all(np.isnan(out.fields["J_A_transport"])))

    def test_surface_and_transport_fluxes_share_stoichiometric_closure(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "reference_concentration.npz"
            self._write_fluent(fluent_path)
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.roles.B=s1",
                    "sim.model.params.transport.nu_B=2.0",
                    "sim.time.t_proc_s=0.1",
                ],
            )
            out = run_aib_from_spec(spec)
            np.testing.assert_allclose(
                out.fields["J_A_surface"], out.fields["J_A_transport"], rtol=1.0e-12
            )
            np.testing.assert_allclose(
                out.fields["J_B_surface"], out.fields["J_B_transport"], rtol=1.0e-12
            )
            np.testing.assert_allclose(out.fields["tau_A_s"], 0.05)
            np.testing.assert_allclose(out.fields["tau_B_s"], 0.05)

    def test_run_aib_from_spec_wafer2d_xy_steady(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            self._write_fluent(fluent_path)

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.domain.kind=wafer_2d_xy",
                    "sim.domain.nr=4",
                    "sim.domain.nx=6",
                    "sim.domain.ny=5",
                ],
            )
            out = run_aib_from_spec(spec)
            self.assertEqual(out.grid.kind, "wafer_2d_xy")
            self.assertEqual(out.thickness.shape, (5, 6))
            self.assertEqual(np.asarray(out.diagnostics["xy_mm"]).shape, (30, 2))
            self.assertEqual(np.asarray(out.fields["phi_B"]).shape, (5, 6))

    def test_run_aib_from_spec_wafer2d_xy_transient(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent_t.npz"
            self._write_fluent_transient(fluent_path)

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.inputs.fluent.mode=transient",
                    "sim.time_mode=transient",
                    "sim.domain.kind=wafer_2d_xy",
                    "sim.domain.nr=4",
                    "sim.domain.nx=6",
                    "sim.domain.ny=5",
                ],
            )
            out = run_aib_from_spec(spec)
            self.assertEqual(out.grid.kind, "wafer_2d_xy")
            self.assertEqual(out.thickness.shape, (5, 6))
            self.assertEqual(out.diagnostics["dispatch_mode"], "transient")


if __name__ == "__main__":
    unittest.main()
