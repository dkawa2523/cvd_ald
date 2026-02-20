from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_report import write_run_report
from deposim_schema import DomainSpec

from .domain import build_domain_grid


@unittest.skipIf(np is None, "NumPy is required for report comparison tests")
class TestReportComparison(unittest.TestCase):
    def test_report_includes_comparison_artifacts_when_measurement_present(self) -> None:
        grid = build_domain_grid(DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=50.0, nr=4, ntheta=8))
        thickness = np.full(grid.shape, 10.0, dtype=float)
        measurement = thickness + 0.5
        diagnostics = {
            "Da_proxy": np.ones(grid.shape, dtype=float),
            "measurement_thickness": measurement,
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


if __name__ == "__main__":
    unittest.main()
