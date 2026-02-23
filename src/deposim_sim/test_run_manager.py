from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .output_manifest import SCHEMA_VERSION
from .pipeline import run_aib_from_spec
from .results_index import update_root_files
from .run_manager import save_run_outputs


@unittest.skipIf(np is None, "NumPy is required for run-manager tests")
class TestRunManagerOutputs(unittest.TestCase):
    def _write_fluent(self, root: Path) -> Path:
        path = root / "fluent.npz"
        xy = np.array([[-15.0, -10.0], [0.0, 0.0], [20.0, 8.0], [30.0, -12.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.9, 0.3, 0.1, 0.0],
                [0.8, 0.3, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref)
        return path

    def _build_run_spec(self, tmp: str, fluent_path: Path):
        return compose_sim_config(
            "cvd_steady_min",
            overrides=[
                f"sim.output.root_dir={tmp}",
                "sim.output.project=run_manager_unit",
                "sim.output.run_name=test_run",
                f"sim.inputs.fluent.file={fluent_path}",
                "sim.output.save_fields=[h_nm,theta_A,residual_nm]",
            ],
        )

    def test_save_run_outputs_creates_current_aib_artifacts(self) -> None:
        with TemporaryDirectory(prefix="deposim_runmgr_") as tmp:
            fluent = self._write_fluent(Path(tmp))
            run_spec = self._build_run_spec(tmp, fluent)
            result = run_aib_from_spec(run_spec)

            run_dir_1 = save_run_outputs(
                run_spec=run_spec,
                config_name="cvd_steady_min",
                config_overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=run_manager_unit",
                    "sim.output.run_name=test_run",
                    f"sim.inputs.fluent.file={fluent}",
                    "sim.output.save_fields=[h_nm,theta_A,residual_nm]",
                ],
                result=result,
            )
            self._assert_run_artifacts(run_dir_1)

            run_dir_2 = save_run_outputs(
                run_spec=run_spec,
                config_name="cvd_steady_min",
                config_overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=run_manager_unit",
                    "sim.output.run_name=test_run",
                    f"sim.inputs.fluent.file={fluent}",
                    "sim.output.save_fields=[h_nm,theta_A,residual_nm]",
                ],
                result=result,
            )
            self._assert_run_artifacts(run_dir_2)
            self.assertNotEqual(run_dir_1.name, run_dir_2.name)

            project_dir = Path(tmp) / "run_manager_unit"
            project_summary = json.loads((project_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(int(project_summary["run_count"]), 2)
            self.assertEqual(str(project_summary["latest_run_id"]), run_dir_2.name)
            index_html = (project_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn(f"runs/{run_dir_2.name}/report.html", index_html)
            root_index = (Path(tmp) / "index.html").read_text(encoding="utf-8")
            self.assertIn("run_manager_unit/index.html", root_index)

    def test_root_index_update_tolerates_broken_project_summary(self) -> None:
        with TemporaryDirectory(prefix="deposim_rootidx_") as tmp:
            root = Path(tmp)
            broken = root / "broken_project"
            broken.mkdir(parents=True, exist_ok=True)
            (broken / "index.html").write_text("<html></html>", encoding="utf-8")
            (broken / "summary.json").write_text("{not-json", encoding="utf-8")

            fluent = self._write_fluent(root)
            run_spec = self._build_run_spec(tmp, fluent)
            result = run_aib_from_spec(run_spec)
            run_dir = save_run_outputs(
                run_spec=run_spec,
                config_name="cvd_steady_min",
                config_overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=run_manager_unit",
                    "sim.output.run_name=test_run",
                    f"sim.inputs.fluent.file={fluent}",
                    "sim.output.save_fields=[h_nm,theta_A,residual_nm]",
                ],
                result=result,
            )
            self.assertTrue((run_dir / "report.html").exists())
            update_root_files(root)
            root_index = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("run_manager_unit/index.html", root_index)

    def _assert_run_artifacts(self, run_dir: Path) -> None:
        self.assertTrue((run_dir / "config_resolved.yaml").is_file())
        self.assertTrue((run_dir / "report.html").is_file())
        self.assertTrue((run_dir / "summary.json").is_file())

        outputs_dir = run_dir / "outputs"
        fields = outputs_dir / "fields.npz"
        metrics = outputs_dir / "metrics.json"
        manifest = outputs_dir / "manifest.json"
        self.assertTrue(fields.is_file())
        self.assertTrue(metrics.is_file())
        self.assertTrue(manifest.is_file())
        with np.load(fields) as data:
            self.assertEqual(set(data.files), {"h_nm", "theta_A", "residual_nm"})
        metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
        self.assertIn("kpi", metrics_payload)
        self.assertIn("dispatch_mode", metrics_payload)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest_payload["schema_version"], SCHEMA_VERSION)
        artifact_ids = {row["id"] for row in manifest_payload["artifacts"]}
        self.assertTrue({"fields", "metrics", "summary", "report", "config", "manifest"}.issubset(artifact_ids))

        plots_dir = run_dir / "plots"
        self.assertTrue((plots_dir / "thickness_map.png").is_file())
        self.assertTrue((plots_dir / "radial_profile.png").is_file())
        self.assertTrue((plots_dir / "solver_health_map.png").is_file())


if __name__ == "__main__":
    unittest.main()
