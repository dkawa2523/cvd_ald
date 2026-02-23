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

from .input_builder import build_domain_from_fluent_xy
from .output_manifest import build_manifest
from .pipeline import run_aib_from_spec
from .run_manager import save_run_outputs


@unittest.skipIf(np is None, "NumPy is required for report comparison tests")
class TestReportComparison(unittest.TestCase):
    def test_report_includes_comparison_artifacts_when_measurement_present(self) -> None:
        xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=float)
        grid = build_domain_from_fluent_xy(xy=xy, xy_unit="mm", wafer_radius_mm=50.0)
        thickness = np.full(grid.shape, 10.0, dtype=float)
        measurement = thickness + 0.5
        diagnostics = {
            "Da_proxy": np.ones(grid.shape, dtype=float),
            "measurement_thickness": measurement,
            "xy_mm": xy,
            "phi_B": np.array([0.1, 0.2, np.nan, 0.4], dtype=float),
            "f_I": np.array([0.0, 0.3, 0.2, np.nan], dtype=float),
            "Cs_over_Cref": {},
            "apparent_orders": {},
            "root_iteration_count": np.zeros(grid.shape, dtype=int),
            "root_status_map": np.zeros(grid.shape, dtype=int),
        }
        summary = {"run_id": "dummy", "thickness_mean": 10.0}

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_run_report(
                run_dir=run_dir,
                run_id="dummy",
                grid=grid,
                thickness=thickness,
                diagnostics=diagnostics,
                summary=summary,
                output_links=[],
            )
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("comparison_error_map.png", report)
            self.assertIn("comparison_mae", report)
            self.assertTrue((run_dir / "plots" / "comparison_error_map.png").exists())
            self.assertTrue((run_dir / "plots" / "measurement_map.png").exists())
            self.assertTrue((run_dir / "plots" / "thickness_map.png").exists())
            self.assertTrue((run_dir / "plots" / "phi_B_map.png").exists())
            self.assertTrue((run_dir / "plots" / "f_I_map.png").exists())
            self.assertTrue((run_dir / "plots" / "solver_health_map.png").exists())

    def test_run_manager_respects_save_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            xy = np.array([[-10.0, -10.0], [0.0, 0.0], [20.0, 5.0], [30.0, -15.0]], dtype=float)
            cref = np.array(
                [
                    [1.0, 0.4, 0.1, 0.0],
                    [0.9, 0.3, 0.1, 0.0],
                    [0.8, 0.3, 0.1, 0.0],
                    [0.7, 0.2, 0.1, 0.0],
                ],
                dtype=float,
            )
            np.savez(fluent_path, xy=xy, cref=cref)

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=save_fields_test",
                    "sim.output.run_name=save_fields_test",
                    "sim.output.save_fields=[h_nm,phi_B]",
                ],
            )
            result = run_aib_from_spec(spec)
            run_dir = save_run_outputs(
                run_spec=spec,
                config_name="cvd_steady_min",
                config_overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=save_fields_test",
                    "sim.output.run_name=save_fields_test",
                    "sim.output.save_fields=[h_nm,phi_B]",
                ],
                result=result,
            )
            self.assertTrue((run_dir / "outputs" / "manifest.json").exists())
            with np.load(run_dir / "outputs" / "fields.npz") as data:
                self.assertEqual(set(data.files), {"h_nm", "phi_B"})

    def test_write_run_report_uses_manifest_links_when_provided(self) -> None:
        xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=float)
        grid = build_domain_from_fluent_xy(xy=xy, xy_unit="mm", wafer_radius_mm=50.0)
        thickness = np.full(grid.shape, 10.0, dtype=float)
        diagnostics = {
            "Da_proxy": np.ones(grid.shape, dtype=float),
            "xy_mm": xy,
            "phi_B": np.array([0.1, 0.2, np.nan, 0.4], dtype=float),
            "f_I": np.array([0.0, 0.3, 0.2, np.nan], dtype=float),
            "Cs_over_Cref": {},
            "apparent_orders": {},
            "root_iteration_count": np.zeros(grid.shape, dtype=int),
            "root_status_map": np.zeros(grid.shape, dtype=int),
        }
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = build_manifest(
                run_id="dummy",
                mode="simulation",
                created_at_utc="2026-02-23T00:00:00Z",
                artifacts=[
                    {"id": "summary", "path": "summary.json", "kind": "json", "required": True},
                    {"id": "report", "path": "report.html", "kind": "html", "required": True},
                ],
                plots=[],
            )
            write_run_report(
                run_dir=run_dir,
                run_id="dummy",
                grid=grid,
                thickness=thickness,
                diagnostics=diagnostics,
                summary={"run_id": "dummy"},
                output_links=[],
                manifest=manifest,
            )
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("summary.json", report)
            self.assertIn("report.html", report)


if __name__ == "__main__":
    unittest.main()
