from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_report import write_run_report
from deposim_schema import compose_sim_config

from .benchmark import run_benchmark
from .compute_engine import is_jax_available
from .domain import build_domain_grid


class TestBenchmark(unittest.TestCase):
    def test_benchmark_returns_timing_and_throughput(self) -> None:
        run_spec = compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=6",
                "domain.ntheta=12",
                "time.process_time_s=2.0",
                "compute.engine=numpy",
            ],
        )
        out = run_benchmark(run_spec, repeats=2)
        self.assertEqual(out["engine_requested"], "numpy")
        self.assertEqual(out["engine_selected"], "numpy")
        self.assertEqual(out["engine_execution_backend"], "numpy")
        self.assertEqual(out["requested_engine"], "numpy")
        self.assertEqual(out["engine_used"], "numpy")
        self.assertGreater(out["best_timing_sec"], 0.0)
        self.assertGreater(out["mean_timing_sec"], 0.0)
        self.assertGreater(out["throughput_cells_per_s"], 0.0)

    def test_benchmark_respects_user_engine_choice(self) -> None:
        run_spec = compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=4",
                "domain.ntheta=8",
                "time.process_time_s=1.0",
                "compute.engine=jax",
            ],
        )
        if is_jax_available():
            out = run_benchmark(run_spec, repeats=1)
            self.assertEqual(out["engine_selected"], "jax")
            self.assertEqual(out["engine_execution_backend"], "numpy")
            self.assertEqual(out["engine_used"], "jax")
        else:
            with self.assertRaisesRegex(RuntimeError, "deposim\\[jax\\]"):
                run_benchmark(run_spec, repeats=1)

    def test_report_can_include_benchmark_summary(self) -> None:
        run_spec = compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=4",
                "domain.ntheta=8",
                "compute.engine=numpy",
            ],
        )
        benchmark = run_benchmark(run_spec, repeats=1)
        grid = build_domain_grid(run_spec.domain)
        thickness = np.ones(grid.shape, dtype=float)
        diagnostics = {
            "Da_proxy": np.ones(grid.shape, dtype=float),
            "Cs_over_Cref": {},
            "apparent_orders": {},
            "root_iteration_count": np.zeros(grid.shape, dtype=int),
            "root_status_map": np.zeros(grid.shape, dtype=int),
        }
        summary = {"run_id": "bench_test", "benchmark": benchmark}

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_run_report(
                run_dir=run_dir,
                run_id="bench_test",
                grid=grid,
                thickness=thickness,
                diagnostics=diagnostics,
                summary=summary,
                output_links=[],
            )
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Benchmark", report)
            self.assertIn("best_timing_sec", report)
            self.assertIn("throughput_cells_per_s", report)


if __name__ == "__main__":
    unittest.main()
