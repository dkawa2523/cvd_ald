from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .benchmark_wafer2d import build_wafer2d_cases, run_wafer2d_benchmark, write_case_input_npz
from .output_manifest import SCHEMA_VERSION


@unittest.skipIf(np is None, "NumPy is required for wafer2d benchmark tests")
class TestWafer2DBenchmark(unittest.TestCase):
    def test_case_matrix_is_deterministic(self) -> None:
        cases_a = build_wafer2d_cases()
        cases_b = build_wafer2d_cases()
        self.assertEqual([c.case_id for c in cases_a], [c.case_id for c in cases_b])
        self.assertEqual([c.class_id for c in cases_a], [c.class_id for c in cases_b])
        self.assertEqual({c.class_id for c in cases_a}, {"A", "AI", "AB", "AIB"})

    def test_file_payload_keys_and_shapes(self) -> None:
        case = build_wafer2d_cases()[0]
        with TemporaryDirectory() as tmp:
            payload_path, xy_mm, cref = write_case_input_npz(case=case, output_dir=Path(tmp))
            self.assertTrue(payload_path.exists())
            with np.load(payload_path) as payload:
                self.assertIn("xy", payload.files)
                self.assertIn("cref", payload.files)
                self.assertEqual(payload["xy"].shape, xy_mm.shape)
                self.assertEqual(payload["cref"].shape, cref.shape)
                self.assertEqual(payload["cref"].shape[1], 4)

    def test_runner_emits_aib_metric_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="cvd_steady_min",
                overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=benchtest",
                ],
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report.html").exists())
            self.assertTrue((run_dir / "outputs" / "benchmark_case_metrics.json").exists())
            self.assertTrue((run_dir / "outputs" / "benchmark_cases.npz").exists())
            self.assertTrue((run_dir / "outputs" / "class_compare.csv").exists())
            self.assertTrue((run_dir / "outputs" / "ranking.csv").exists())
            self.assertTrue((run_dir / "outputs" / "manifest.json").exists())

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sim_model"], "aib_ode")
            self.assertEqual(summary["case_count"], len(build_wafer2d_cases()))
            self.assertIn("overall_passed", summary["trend_assertions"])
            self.assertEqual(summary.get("manifest_path"), "outputs/manifest.json")
            self.assertIn("ranking", summary.get("artifact_paths", {}))

            manifest = json.loads((run_dir / "outputs" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            artifact_ids = {row["id"] for row in manifest["artifacts"]}
            self.assertTrue({"ranking", "class_compare", "benchmark_case_metrics", "benchmark_cases"}.issubset(artifact_ids))

            payload = np.load(run_dir / "outputs" / "benchmark_cases.npz")
            self.assertIn("phi_B", payload.files)
            self.assertIn("f_I", payload.files)
            self.assertIn("residual_nm", payload.files)
            self.assertEqual(payload["h_nm"].shape[0], len(build_wafer2d_cases()))

            rows = json.loads((run_dir / "outputs" / "benchmark_case_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), len(build_wafer2d_cases()))
            required = {
                "case_id",
                "class_id",
                "mean_phi_B",
                "mean_f_I",
                "mean_CsA_over_CrefA",
                "mean_CsB_over_CrefB",
                "mean_abs_residual_nm",
            }
            self.assertTrue(required.issubset(set(rows[0].keys())))

            with (run_dir / "outputs" / "class_compare.csv").open("r", encoding="utf-8") as fh:
                class_rows = list(csv.DictReader(fh))
            classes = {row["class_id"] for row in class_rows}
            self.assertEqual(classes, {"A", "AI", "AB", "AIB"})

    def test_runner_supports_legacy_cli_override_aliases(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="cvd_steady_min",
                overrides=[
                    f"output.project_dir={tmp}",
                    "output.run_dir_name=bench_alias",
                    "domain.nr=8",
                    "domain.ntheta=12",
                ],
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue(run_dir.exists())
            self.assertIn("bench_alias", run_dir.name)

    def test_report_and_index_links_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="cvd_steady_min",
                overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=benchtest",
                ],
            )
            run_dir = Path(out["run_dir"])
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Trend Assertions", report)
            self.assertIn("benchmark_case_metrics.json", report)
            manifest = json.loads((run_dir / "outputs" / "manifest.json").read_text(encoding="utf-8"))
            for row in manifest["artifacts"]:
                self.assertIn(str(row["path"]), report)
            index = (Path(tmp) / "benchtest" / "index.html").read_text(encoding="utf-8")
            self.assertIn(f"runs/{run_dir.name}/report.html", index)


if __name__ == "__main__":
    unittest.main()
