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


if __name__ == "__main__":
    unittest.main()
