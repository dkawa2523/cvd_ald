from __future__ import annotations

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .benchmark_wafer2d import run_wafer2d_benchmark


@unittest.skipIf(np is None, "NumPy is required for physviz tests")
class TestPhysViz(unittest.TestCase):
    def test_benchmark_runner_writes_physviz_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="cvd_steady_min",
                overrides=[
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=benchviz",
                ],
                with_physviz=True,
                physviz_fast=True,
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "outputs" / "physviz_maps.npz").exists())
            self.assertTrue((run_dir / "outputs" / "manifest.json").exists())
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Physviz Maps", report)
            self.assertIn("plots/physviz_", report)
            with np.load(run_dir / "outputs" / "physviz_maps.npz") as data:
                self.assertIn("phi_B", data.files)
                self.assertIn("f_I", data.files)
                self.assertIn("h_nm", data.files)
                self.assertIn("km_A", data.files)
                self.assertIn("tau_A", data.files)
                self.assertIn("input_cref_A", data.files)
                self.assertIn("input_flux_A", data.files)

    def test_benchmark_wafer2d_physviz_command(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        runs_dir = repo_root / "results" / "demo" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        before = sorted([p.name for p in runs_dir.glob("benchmark_wafer2d*") if p.is_dir()])
        if os.name == "nt":
            cmd = ["bash", "-lc", "./scripts/commands.sh benchmark_wafer2d_physviz"]
        else:
            cmd = ["./scripts/commands.sh", "benchmark_wafer2d_physviz"]
        completed = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        )
        after = sorted([p.name for p in runs_dir.glob("benchmark_wafer2d*") if p.is_dir()])
        self.assertGreater(len(after), len(before))
        latest = runs_dir / after[-1]
        self.assertTrue((latest / "summary.json").exists())
        self.assertTrue((latest / "report.html").exists())
        self.assertTrue((latest / "outputs" / "physviz_maps.npz").exists())


if __name__ == "__main__":
    unittest.main()
