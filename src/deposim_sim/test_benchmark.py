from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_report import write_run_report
from deposim_schema import compose_sim_config

from .benchmark import run_benchmark
from .pipeline import run_aib_from_spec


@unittest.skipIf(np is None, "NumPy is required for benchmark tests")
class TestBenchmark(unittest.TestCase):
    def _write_inputs(self, root: Path) -> Path:
        path = root / "fluent.npz"
        xy = np.array([[-20.0, -15.0], [0.0, 0.0], [25.0, 10.0], [40.0, -10.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.9, 0.4, 0.1, 0.0],
                [0.8, 0.3, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(path, xy=xy, cref=cref)
        return path

    def test_benchmark_returns_timing_and_aib_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = self._write_inputs(Path(tmp))
            spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            out = run_benchmark(spec, repeats=2)
            self.assertGreater(out["best_timing_sec"], 0.0)
            self.assertGreater(out["mean_timing_sec"], 0.0)
            self.assertGreater(out["throughput_cells_per_s"], 0.0)
            self.assertIn("phi_B_mean", out)
            self.assertIn("f_I_mean", out)
            self.assertIn("CsA_over_CrefA_mean", out)
            self.assertIn("residual_nm_mae", out)

    def test_report_can_include_benchmark_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent = self._write_inputs(root)
            spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent}"])
            result = run_aib_from_spec(spec)
            benchmark = run_benchmark(spec, repeats=1)

            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = {"run_id": "bench_test", "benchmark": benchmark}
            write_run_report(
                run_dir=run_dir,
                run_id="bench_test",
                grid=result.grid,
                thickness=result.thickness,
                diagnostics=result.diagnostics,
                summary=summary,
                output_links=[],
            )
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Benchmark", report)
            self.assertIn("best_timing_sec", report)
            self.assertIn("throughput_cells_per_s", report)


if __name__ == "__main__":
    unittest.main()
