from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config
from deposim_report.physviz_report import _draw_map

from .benchmark_wafer2d import run_wafer2d_benchmark
from .domain import build_domain_grid
from .input_builder import build_field_bundle
from .models import mass_transfer
from .physics.cvd_steady import run_cvd_steady
from .physviz import (
    build_ald_phase_snapshots,
    build_cvd_pseudo_time_snapshots,
    compute_reaction_term_importance,
    compute_transport_term_maps,
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


@unittest.skipIf(np is None, "NumPy is required for physviz tests")
class TestPhysViz(unittest.TestCase):
    def _cvd_spec(self):
        return compose_sim_config(
            "smoke",
            overrides=[
                "domain.kind=wafer_2d_polar",
                "domain.nr=6",
                "domain.ntheta=12",
                "time.process_time_s=2.0",
                "inputs.synthetic_case=radial_gradient",
            ],
        )

    def test_physviz_cvd_time_snapshots_shape(self) -> None:
        spec = self._cvd_spec()
        out = build_cvd_pseudo_time_snapshots(spec, [0.1, 0.5, 1.0])
        self.assertEqual(out["thickness_snapshots"].shape[0], 3)
        self.assertEqual(out["delta_thickness_snapshots"].shape[0], 2)
        self.assertEqual(out["linearity_residual_max"].shape, out["thickness_snapshots"][0].shape)

    def test_physviz_ald_phase_snapshots_shape(self) -> None:
        spec = compose_sim_config(
            "ald_synthetic",
            overrides=[
                "domain.kind=wafer_2d_polar",
                "domain.nr=4",
                "domain.ntheta=8",
            ],
        )
        out = build_ald_phase_snapshots(spec)
        self.assertEqual(out["phase_thickness_snapshots"].shape[0], len(out["phase_names"]))
        self.assertEqual(out["phase_coverage_snapshots"].shape[0], len(out["phase_names"]))
        self.assertEqual(out["cumulative_thickness_snapshots"].shape[0], len(out["phase_names"]))

    def test_physviz_transport_terms_finite(self) -> None:
        spec = self._cvd_spec()
        grid = build_domain_grid(spec.domain)
        fields = build_field_bundle(spec, grid)
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=spec.model,
            process_time_s=spec.time.process_time_s,
            solver_config=spec.solver,
        )
        km = mass_transfer.compute_km_from_model_config(spec.model, grid=grid, omega_rad_s=spec.inputs.omega_rad_s)
        maps = compute_transport_term_maps(result, fields, km, {"precursor": 1.0})
        for key in (
            "transport_capacity__precursor",
            "reaction_demand__precursor",
            "depletion_ratio__precursor",
            "utilization__precursor",
        ):
            arr = np.asarray(maps[key], dtype=float)
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertTrue(np.all(arr >= 0.0))

    def test_physviz_reaction_importance_outputs(self) -> None:
        spec = self._cvd_spec()
        out = compute_reaction_term_importance(spec, mode="sensitivity+ablation")
        self.assertIn("scores", out)
        self.assertIsInstance(out["scores"], list)
        self.assertGreaterEqual(len(out["scores"]), 1)
        self.assertIn("sensitivity_maps", out)
        self.assertIn("ablation_maps", out)

    @unittest.skipIf(plt is None, "Matplotlib is required for plot tests")
    def test_physviz_polar_map_uses_wafer_coordinates(self) -> None:
        spec = self._cvd_spec()
        grid = build_domain_grid(spec.domain)
        values = np.asarray(grid.r_grid_mm, dtype=float)
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            _draw_map(ax, grid, values)
            self.assertEqual(ax.get_aspect(), 1.0)
            radius = float(grid.wafer_radius_mm)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self.assertAlmostEqual(xlim[0], -radius, places=6)
            self.assertAlmostEqual(xlim[1], radius, places=6)
            self.assertAlmostEqual(ylim[0], -radius, places=6)
            self.assertAlmostEqual(ylim[1], radius, places=6)
        finally:
            plt.close(fig)

    def test_physviz_report_links_and_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="smoke",
                overrides=[
                    f"output.project_dir={tmp}",
                    "domain.nr=8",
                    "domain.ntheta=12",
                    "time.process_time_s=2.0",
                ],
                with_physviz=True,
                physviz_fast=True,
            )
            run_dir = Path(out["run_dir"])
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Input Field Maps", report)
            self.assertIn("Time-Space Maps", report)
            self.assertIn("Transport Term Importance", report)
            self.assertIn("Reaction Term Importance (Sensitivity + Ablation)", report)
            self.assertIn("Net Term Importance", report)
            self.assertIn("plots/physviz_", report)
            self.assertIn("physviz_input_cref_", report)

    def test_benchmark_wafer2d_physviz_command(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        runs_dir = repo_root / "results" / "runs"
        before = sorted([p.name for p in runs_dir.glob("benchmark_wafer2d_*") if p.is_dir()])
        cmd = ["./scripts/commands.sh", "benchmark_wafer2d_physviz", "domain.nr=6", "domain.ntheta=10"]
        completed = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        )
        after = sorted([p.name for p in runs_dir.glob("benchmark_wafer2d_*") if p.is_dir()])
        self.assertGreater(len(after), len(before))
        latest = runs_dir / after[-1]
        self.assertTrue((latest / "summary.json").exists())
        self.assertTrue((latest / "report.html").exists())
        self.assertTrue((latest / "outputs" / "physviz_maps.npz").exists() or (latest / "outputs" / "physviz_maps.zarr").exists())


if __name__ == "__main__":
    unittest.main()
