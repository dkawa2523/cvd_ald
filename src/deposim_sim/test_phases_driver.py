from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .phases_driver import build_phase_input_preview, run_phased_synthetic


@unittest.skipIf(np is None, "NumPy is required for phases/driver tests")
class TestPhasesDriver(unittest.TestCase):
    def _write_fluent(self, root: Path) -> Path:
        path = root / "fluent.npz"
        xy = np.array([[-15.0, -10.0], [0.0, 0.0], [18.0, 9.0], [25.0, -12.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.5, 0.1, 0.0],
                [0.9, 0.4, 0.1, 0.0],
                [0.8, 0.3, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref)
        return path

    def test_preview_reflects_phase_schedule(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_fluent(Path(tmp))
            spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            spec.phase_schedule = [  # type: ignore[attr-defined]
                {"name": "boost", "duration_s": 4.0, "fluent_scale": 1.4},
                {"name": "settle", "duration_s": 2.0, "fluent_scale": 0.8},
            ]
            preview = build_phase_input_preview(spec)
            self.assertEqual(len(preview), 2)
            self.assertEqual(preview[0]["phase_name"], "boost")
            self.assertAlmostEqual(float(preview[0]["fluent_scale"]), 1.4)
            self.assertEqual(preview[1]["phase_name"], "settle")

    def test_run_phased_synthetic_executes_and_accumulates(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_fluent(Path(tmp))
            spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            spec.phase_schedule = [  # type: ignore[attr-defined]
                {"name": "phase_1", "duration_s": 3.0, "fluent_scale": 1.1},
                {"name": "phase_2", "duration_s": 3.0, "fluent_scale": 0.9},
            ]
            result = run_phased_synthetic(spec)
            self.assertEqual(len(result.phase_thickness), 2)
            self.assertEqual(len(result.input_preview), 2)
            self.assertEqual(len(result.phase_diagnostics), 2)
            expected = result.phase_thickness[0] + result.phase_thickness[1]
            np.testing.assert_allclose(result.total_thickness, expected)
            self.assertTrue(np.all(np.isfinite(result.total_thickness)))

    def test_legacy_scalar_override_phase_is_converted(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_fluent(Path(tmp))
            spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            spec.inputs.c_ref_mol_m3 = 1.0  # type: ignore[attr-defined]
            spec.time.phases = [  # type: ignore[attr-defined]
                {"name": "legacy", "duration_s": 2.0, "scalar_overrides": {"c_ref_mol_m3": 1.8}}
            ]
            preview = build_phase_input_preview(spec)
            self.assertEqual(len(preview), 1)
            self.assertAlmostEqual(float(preview[0]["fluent_scale"]), 1.8)


if __name__ == "__main__":
    unittest.main()
