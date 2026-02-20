from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_schema import compose_sim_config

from .domain import build_domain_grid
from .multiz import run_multi_z_synthetic
from .physics.cvd_steady import run_cvd_steady
from .smoke import main as smoke_main
from .synthetic_inputs import build_synthetic_field_bundle


class TestMultiZ(unittest.TestCase):
    def _base_spec(self):
        return compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=6",
                "domain.ntheta=10",
                "time.process_time_s=2.0",
                "reference_plane.z_ref_mm=5.0",
            ],
        )

    def test_single_plane_mode_matches_baseline(self) -> None:
        run_spec = self._base_spec()
        grid = build_domain_grid(run_spec.domain)
        fields = build_synthetic_field_bundle(run_spec, grid)
        baseline = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=run_spec.model,
            process_time_s=run_spec.time.process_time_s,
            solver_config=run_spec.solver,
        )
        out = run_multi_z_synthetic(run_spec)
        np.testing.assert_allclose(out.thickness, baseline.thickness)
        self.assertEqual(out.diagnostics["plane_count"], 1)

    def test_multi_plane_mode_returns_diagnostics(self) -> None:
        run_spec = self._base_spec()
        run_spec.reference_plane.z_ref_mm_list = [3.0, 5.0, 7.0]
        out = run_multi_z_synthetic(run_spec)
        self.assertEqual(out.diagnostics["plane_count"], 3)
        self.assertEqual(out.plane_thickness.shape[0], 3)
        self.assertEqual(out.thickness.shape, out.plane_thickness.shape[1:])
        self.assertIn("plane_thickness_mean", out.diagnostics)

    def test_smoke_path_emits_multiz_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "out"
            rc = smoke_main(
                [
                    "--config-name",
                    "smoke",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                    "reference_plane.z_ref_mm=5.0",
                    "reference_plane.z_ref_mm_list=[3.0,5.0]",
                    f"output.project_dir={project_dir}",
                    "output.run_dir_name=multiz_smoke",
                ]
            )
            self.assertEqual(rc, 0)
            runs = sorted([p for p in (project_dir / "runs").iterdir() if p.is_dir()])
            latest = runs[-1]
            diag = np.load(latest / "outputs" / "diagnostics.npz")
            self.assertIn("plane_count", diag.files)
            self.assertEqual(int(np.asarray(diag["plane_count"]).item()), 2)


if __name__ == "__main__":
    unittest.main()
