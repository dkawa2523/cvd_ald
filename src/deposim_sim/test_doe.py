from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .doe import run_doe


@unittest.skipIf(np is None, "NumPy is required for DOE tests")
class TestDoeRunner(unittest.TestCase):
    def test_grid_doe_writes_case_dimension_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent.npz"
            xy = np.array([[-20.0, -10.0], [0.0, 0.0], [18.0, 9.0], [30.0, -12.0]], dtype=float)
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

            result = run_doe(
                config_name="cvd_steady_min",
                base_overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=doe_test",
                    "sim.output.run_name=doe_run",
                    f"sim.inputs.fluent.file={fluent_path}",
                ],
                sweep={
                    "sim.model.params.kinetics.k_rxn": [0.006, 0.010],
                    "sim.model.params.transport.km_A": [0.01, 0.02],
                },
                sampling="grid",
            )
            run_dir = result.run_dir
            self.assertTrue((run_dir / "outputs" / "doe_cases.npz").exists())
            self.assertTrue((run_dir / "outputs" / "doe_cases.json").exists())
            self.assertTrue((run_dir / "outputs" / "manifest.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report.html").exists())
            self.assertEqual(result.case_count, 4)

            arr = np.load(run_dir / "outputs" / "doe_cases.npz")
            self.assertEqual(arr["thickness"].shape[0], 4)
            self.assertEqual(arr["deposition_rate"].shape[0], 4)
            self.assertEqual(arr["nu_percent"].shape[0], 4)

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["case_count"], 4)
            self.assertIn("ranking_top_nu", summary)
            self.assertGreaterEqual(len(summary["ranking_top_nu"]), 1)
            self.assertEqual(summary.get("manifest_path"), "outputs/manifest.json")

            runs = [p for p in (Path(tmp) / "doe_test" / "runs").iterdir() if p.is_dir()]
            self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()
