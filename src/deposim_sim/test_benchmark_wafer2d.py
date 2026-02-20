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

from .benchmark_wafer2d import (
    build_wafer2d_cases,
    run_wafer2d_benchmark,
    write_case_input_npz,
)
from .domain import build_domain_grid


@unittest.skipIf(np is None, "NumPy is required for wafer2d benchmark tests")
class TestWafer2DBenchmark(unittest.TestCase):
    def test_case_matrix_is_deterministic(self) -> None:
        cases_a = build_wafer2d_cases()
        cases_b = build_wafer2d_cases()
        self.assertEqual([c.case_id for c in cases_a], [c.case_id for c in cases_b])
        self.assertEqual([c.overrides for c in cases_a], [c.overrides for c in cases_b])

    def test_file_payload_keys_and_shapes(self) -> None:
        run_spec = compose_sim_config("smoke", overrides=["domain.kind=wafer_2d_polar", "domain.nr=8", "domain.ntheta=16"])
        grid = build_domain_grid(run_spec.domain)
        file_cases = [case for case in build_wafer2d_cases() if case.file_pattern is not None]
        self.assertGreaterEqual(len(file_cases), 2)

        with TemporaryDirectory() as tmp:
            payload_path, cref = write_case_input_npz(
                case=file_cases[0],
                grid=grid,
                output_dir=Path(tmp),
                c_ref_mol_m3=1.8,
                temperature_k=710.0,
                species="precursor",
            )
            self.assertTrue(payload_path.exists())
            with np.load(payload_path) as payload:
                self.assertIn("C_ref__precursor", payload.files)
                self.assertIn("T", payload.files)
                self.assertEqual(payload["C_ref__precursor"].shape, grid.shape)
                self.assertEqual(payload["T"].shape, grid.shape)
                np.testing.assert_allclose(payload["C_ref__precursor"], cref)

    def test_runner_emits_case_dimension_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="smoke",
                overrides=[
                    f"output.project_dir={tmp}",
                    "domain.nr=8",
                    "domain.ntheta=12",
                    "time.process_time_s=2.0",
                ],
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report.html").exists())
            self.assertTrue((run_dir / "outputs" / "benchmark_case_metrics.json").exists())

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            artifact_path = run_dir / summary["artifact_paths"]["benchmark_cases"]
            self.assertTrue(artifact_path.exists())
            case_count = len(build_wafer2d_cases())
            self.assertEqual(summary["case_count"], case_count)

            if artifact_path.suffix == ".npz":
                payload = np.load(artifact_path)
                self.assertEqual(payload["thickness"].shape[0], case_count)
                self.assertEqual(payload["da_proxy"].shape[0], case_count)
                self.assertEqual(payload["cs_over_cref"].shape[0], case_count)

    def test_trend_assertions_regime_split(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="smoke",
                overrides=[
                    f"output.project_dir={tmp}",
                    "domain.nr=8",
                    "domain.ntheta=12",
                    "time.process_time_s=2.0",
                ],
            )
            trend = out["summary"]["trend_assertions"]
            self.assertTrue(trend["assert_regime_cs_ratio"])
            self.assertTrue(trend["assert_regime_da_proxy"])
            self.assertTrue(trend["assert_radial_trend"])
            self.assertTrue(trend["assert_file_theta_transfer"])
            self.assertTrue(trend["assert_solver_health"])
            self.assertTrue(trend["overall_passed"])

    def test_report_and_index_links_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="smoke",
                overrides=[
                    f"output.project_dir={tmp}",
                    "domain.nr=8",
                    "domain.ntheta=12",
                    "time.process_time_s=2.0",
                ],
            )
            run_dir = Path(out["run_dir"])
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Trend Assertions", report)
            self.assertIn("benchmark_case_metrics.json", report)
            index = (Path(tmp) / "index.html").read_text(encoding="utf-8")
            self.assertIn(f"runs/{run_dir.name}/report.html", index)


if __name__ == "__main__":
    unittest.main()
