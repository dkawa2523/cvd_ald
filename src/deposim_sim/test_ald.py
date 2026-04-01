from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_schema import compose_sim_config

from .physics.ald import run_ald_synthetic
from .smoke import main as smoke_main


@unittest.skip("legacy ALD synthetic path is isolated pending schema-aligned rewrite")
class TestAldSynthetic(unittest.TestCase):
    def test_ald_cycle_runs_and_keeps_coverage_bounded(self) -> None:
        run_spec = compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=6",
                "domain.ntheta=10",
                "time.mode=ald_cycle",
            ],
        )
        run_spec.time.phases = [
            {"name": "dose", "duration_s": 0.2, "scalar_overrides": {"precursor_scale": 1.0, "react_scale": 1.0}},
            {"name": "purge", "duration_s": 0.2, "scalar_overrides": {"precursor_scale": 0.0, "react_scale": 0.0}},
            {"name": "dose2", "duration_s": 0.2, "scalar_overrides": {"precursor_scale": 1.0, "react_scale": 1.0}},
        ]
        run_spec.model.state_params = {"sticking_coeff": 0.4, "desorption_rate_s": 0.08, "growth_rate_nm_s": 1.1}

        result = run_ald_synthetic(run_spec)
        self.assertEqual(result.thickness.shape, result.coverage.shape)
        self.assertGreaterEqual(float(np.min(result.coverage)), 0.0)
        self.assertLessEqual(float(np.max(result.coverage)), 1.0)
        self.assertEqual(result.diagnostics["phase_count"], 3)

    def test_invalid_mode_is_rejected(self) -> None:
        run_spec = compose_sim_config("smoke", overrides=["time.mode=cvd_steady"])
        with self.assertRaises(ValueError):
            run_ald_synthetic(run_spec)

    def test_smoke_dispatch_runs_ald_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "out"
            rc = smoke_main(
                [
                    "--config-name",
                    "ald_synthetic",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    f"output.project_dir={project_dir}",
                    "output.run_dir_name=ald_smoke",
                ]
            )
            self.assertEqual(rc, 0)
            runs = sorted([p for p in (project_dir / "runs").iterdir() if p.is_dir()])
            latest = runs[-1]
            diag = np.load(latest / "outputs" / "diagnostics.npz")
            self.assertIn("phase_count", diag.files)
            self.assertGreaterEqual(int(np.asarray(diag["phase_count"]).item()), 1)


if __name__ == "__main__":
    unittest.main()
