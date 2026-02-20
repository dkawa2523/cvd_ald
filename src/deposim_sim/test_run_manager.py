from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .domain import build_domain_grid
from .run_manager import save_run_outputs


@unittest.skipIf(np is None, "NumPy is required for run-manager tests")
class TestRunManagerOutputs(unittest.TestCase):
    def _build_run_spec(self, project_dir: str):
        return compose_sim_config(
            "example_cvd",
            overrides=[
                f"output.project_dir={project_dir}",
                "output.run_dir_name=test_run",
                "domain.nr=8",
                "domain.ntheta=12",
            ],
        )

    def _fake_result(self, shape: tuple[int, ...]):
        thickness = np.full(shape, 2.5, dtype=float)
        deposition_rate = np.full(shape, 0.25, dtype=float)
        root_status = np.zeros(shape, dtype=int)
        root_iters = np.full(shape, 7, dtype=int)
        return SimpleNamespace(
            thickness=thickness,
            deposition_rate=deposition_rate,
            R=deposition_rate.copy(),
            Cs={"precursor": np.full(shape, 1.1, dtype=float)},
            diagnostics={
                "Da_proxy": np.full(shape, 0.3, dtype=float),
                "Cs_over_Cref": {"precursor": np.full(shape, 0.8, dtype=float)},
                "apparent_orders": {"precursor": np.full(shape, 1.0, dtype=float)},
                "root_iteration_count": root_iters,
                "root_status_map": root_status,
                "root_failure_mask": root_status.astype(bool),
                "root_failure_fraction": 0.0,
            },
        )

    def test_save_run_outputs_creates_artifacts_and_updates_latest(self) -> None:
        with TemporaryDirectory(prefix="deposim_p0008_") as tmpdir:
            run_spec = self._build_run_spec(tmpdir)
            grid = build_domain_grid(run_spec.domain)
            result = self._fake_result(grid.shape)

            run_dir_1 = save_run_outputs(
                run_spec=run_spec,
                config_name="example_cvd",
                config_overrides=[f"output.project_dir={tmpdir}", "output.run_dir_name=test_run"],
                grid=grid,
                result=result,
            )
            self._assert_run_artifacts(run_dir_1, tmpdir)

            run_dir_2 = save_run_outputs(
                run_spec=run_spec,
                config_name="example_cvd",
                config_overrides=[f"output.project_dir={tmpdir}", "output.run_dir_name=test_run"],
                grid=grid,
                result=result,
            )
            self._assert_run_artifacts(run_dir_2, tmpdir)

            self.assertNotEqual(run_dir_1.name, run_dir_2.name)

            project_summary = json.loads((Path(tmpdir) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(project_summary["run_count"], 2)
            self.assertEqual(project_summary["latest_run_id"], run_dir_2.name)

            index_html = (Path(tmpdir) / "index.html").read_text(encoding="utf-8")
            self.assertIn(f'runs/{run_dir_2.name}/report.html', index_html)

            run_dirs = sorted([p for p in (Path(tmpdir) / "runs").iterdir() if p.is_dir()])
            self.assertEqual(len(run_dirs), 2)
            self.assertEqual({run_dir_1, run_dir_2}, set(run_dirs))

    def _assert_run_artifacts(self, run_dir: Path, project_dir: str) -> None:
        project = Path(project_dir)
        self.assertTrue((project / "index.html").is_file())
        self.assertTrue((project / "summary.json").is_file())
        self.assertTrue((run_dir / "config_resolved.yaml").is_file())
        self.assertTrue((run_dir / "report.html").is_file())
        self.assertTrue((run_dir / "summary.json").is_file())

        outputs_dir = run_dir / "outputs"
        self.assertTrue((outputs_dir / "thickness.npz").is_file())
        self.assertTrue((outputs_dir / "cs_fields.npz").is_file())
        self.assertTrue((outputs_dir / "diagnostics.npz").is_file())
        self.assertTrue((outputs_dir / "radial_profile.npz").is_file())

        plots_dir = run_dir / "plots"
        expected_plots = {
            "thickness_map.png",
            "radial_profile.png",
            "cs_over_cref_precursor.png",
            "da_proxy_map.png",
            "n_app_precursor.png",
            "solver_health_map.png",
        }
        actual_plots = {p.name for p in plots_dir.glob("*.png")}
        self.assertTrue(expected_plots.issubset(actual_plots))


if __name__ == "__main__":
    unittest.main()
