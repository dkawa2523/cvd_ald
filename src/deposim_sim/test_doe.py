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
            result = run_doe(
                config_name="smoke",
                base_overrides=[
                    f"output.project_dir={tmp}",
                    "domain.nr=6",
                    "domain.ntheta=10",
                    "time.process_time_s=2.0",
                ],
                sweep={
                    "inputs.c_ref_mol_m3": [1.2, 1.8],
                    "model.kinetics_params.k0": [0.8, 1.1],
                },
                sampling="grid",
            )
            run_dir = result.run_dir
            self.assertTrue((run_dir / "outputs" / "doe_cases.npz").exists())
            self.assertTrue((run_dir / "outputs" / "doe_cases.json").exists())
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

            runs = [p for p in (Path(tmp) / "runs").iterdir() if p.is_dir()]
            self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()
