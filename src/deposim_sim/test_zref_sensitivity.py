from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .doe import run_zref_sensitivity


@unittest.skipIf(np is None, "NumPy is required for z_ref sensitivity tests")
class TestZrefSensitivity(unittest.TestCase):
    def test_zref_sensitivity_outputs_metrics_and_plot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent.npz"
            xy = np.array([[-10.0, -10.0], [0.0, 0.0], [20.0, 10.0], [30.0, -20.0]], dtype=float)
            cref = np.array(
                [
                    [1.0, 0.4, 0.1, 0.0],
                    [0.9, 0.3, 0.1, 0.0],
                    [0.8, 0.3, 0.1, 0.0],
                    [0.7, 0.2, 0.1, 0.0],
                ],
                dtype=float,
            )
            np.savez(fluent_path, xy=xy, cref=cref)

            result = run_zref_sensitivity(
                config_name="cvd_steady_min",
                z_ref_values_mm=[2.0, 5.0, 8.0],
                base_overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=zref_test",
                    f"sim.inputs.fluent.file={fluent_path}",
                ],
            )
            run_dir = result.run_dir
            npz_path = run_dir / "outputs" / "zref_sensitivity.npz"
            plot_path = run_dir / "plots" / "zref_sensitivity.png"
            self.assertTrue(npz_path.exists())
            self.assertTrue(plot_path.exists())
            payload = np.load(npz_path)
            self.assertEqual(payload["z_ref_mm"].shape[0], 3)
            self.assertEqual(payload["nu_percent"].shape[0], 3)
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("z_ref Sensitivity", report)
            self.assertIn("outputs/zref_sensitivity.npz", report)
            self.assertIn("plots/zref_sensitivity.png", report)


if __name__ == "__main__":
    unittest.main()
