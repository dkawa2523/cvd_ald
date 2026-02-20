from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec

from .domain import build_domain_grid
from .metrics import compute_kpi_metrics


@unittest.skipIf(np is None, "NumPy is required for KPI tests")
class TestKpiMetrics(unittest.TestCase):
    def test_compute_kpis_has_expected_keys(self) -> None:
        spec = DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=100.0, nr=8, ntheta=12, edge_exclusion_mm=0.0)
        grid = build_domain_grid(spec)
        thickness = 10.0 + 0.5 * (grid.r_grid_mm / grid.wafer_radius_mm)

        kpi = compute_kpi_metrics(thickness, grid, spec_min=9.8, spec_max=10.6, ring_count=4)
        self.assertIn("nu_percent", kpi)
        self.assertIn("center_edge_delta", kpi)
        self.assertIn("ring_means", kpi)
        self.assertIn("out_of_spec_area_fraction", kpi)
        self.assertEqual(kpi["ring_count"], 4)
        self.assertEqual(len(kpi["ring_means"]), 4)
        self.assertGreaterEqual(kpi["out_of_spec_area_fraction"], 0.0)
        self.assertLessEqual(kpi["out_of_spec_area_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
