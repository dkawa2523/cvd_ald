from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .multiz import run_multi_z_synthetic
from .pipeline import run_aib_from_spec


@unittest.skipIf(np is None, "NumPy is required for multi-z tests")
class TestMultiZ(unittest.TestCase):
    def _write_fluent(self, root: Path) -> Path:
        path = root / "fluent.npz"
        xy = np.array([[-15.0, -8.0], [0.0, 0.0], [20.0, 12.0], [30.0, -10.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.3, 0.1, 0.0],
                [0.9, 0.3, 0.1, 0.0],
                [0.8, 0.2, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref)
        return path

    def test_single_plane_mode_matches_direct_aib_run(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_fluent(Path(tmp))
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[f"sim.inputs.fluent.file={fluent}", "sim.reference_plane.z_ref_mm=5.0"],
            )
            baseline = run_aib_from_spec(spec)
            out = run_multi_z_synthetic(spec)
            np.testing.assert_allclose(out.thickness, baseline.thickness)
            self.assertEqual(out.diagnostics["plane_count"], 1)
            self.assertTrue(out.diagnostics["single_plane_compat_mode"])

    def test_multi_plane_mode_returns_plane_stack_and_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_fluent(Path(tmp))
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[f"sim.inputs.fluent.file={fluent}", "sim.reference_plane.z_ref_mm=5.0"],
            )
            # Compatibility path: z_ref_mm_list is optional dynamic attribute.
            spec.reference_plane.z_ref_mm_list = [3.0, 5.0, 7.0]  # type: ignore[attr-defined]
            out = run_multi_z_synthetic(spec)
            self.assertEqual(out.diagnostics["plane_count"], 3)
            self.assertEqual(out.plane_thickness.shape[0], 3)
            self.assertEqual(out.thickness.shape, out.plane_thickness.shape[1:])
            self.assertEqual(len(out.diagnostics["plane_thickness_mean"]), 3)
            self.assertEqual(len(out.diagnostics["plane_metadata"]), 3)
            self.assertIn("A", out.Cs)


if __name__ == "__main__":
    unittest.main()
