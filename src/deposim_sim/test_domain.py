from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec

from .domain import build_domain_grid, edge_exclusion_mask, radial_profile


@unittest.skipIf(np is None, "NumPy is required for domain grid tests")
class TestDomainGrid(unittest.TestCase):
    def test_polar_grid_shape_and_area(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=100.0,
            nr=4,
            ntheta=8,
            edge_exclusion_mm=0.0,
        )
        grid = build_domain_grid(spec)

        self.assertEqual(grid.shape, (4, 8))
        self.assertEqual(grid.r_mm.shape, (4,))
        self.assertEqual(grid.theta_rad.shape, (8,))
        self.assertEqual(grid.area_weights_mm2.shape, (4, 8))
        self.assertTrue(np.all(grid.edge_mask))
        self.assertAlmostEqual(float(np.sum(grid.area_weights_mm2)), np.pi * 100.0**2, places=10)

    def test_polar_edge_exclusion(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=100.0,
            nr=4,
            ntheta=6,
            edge_exclusion_mm=30.0,
        )
        grid = build_domain_grid(spec)

        self.assertTrue(np.all(grid.edge_mask[0:3, :]))
        self.assertFalse(np.any(grid.edge_mask[3, :]))

    def test_radial_grid_shape_area_and_mask(self) -> None:
        spec = DomainSpec(
            kind="wafer_1d_radial",
            wafer_radius_mm=50.0,
            nr=5,
            ntheta=1,
            edge_exclusion_mm=10.0,
        )
        grid = build_domain_grid(spec)

        self.assertEqual(grid.shape, (5,))
        self.assertEqual(grid.area_weights_mm2.shape, (5,))
        self.assertAlmostEqual(float(np.sum(grid.area_weights_mm2)), np.pi * 50.0**2, places=10)
        np.testing.assert_array_equal(grid.edge_mask, np.array([True, True, True, True, False]))

    def test_xy_grid_shape_and_mask(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_xy",
            wafer_radius_mm=100.0,
            nr=5,
            nx=8,
            ny=10,
            edge_exclusion_mm=10.0,
        )
        grid = build_domain_grid(spec)

        self.assertEqual(grid.shape, (10, 8))
        self.assertEqual(grid.r_mm.shape, (5,))
        self.assertEqual(grid.r_edges_mm.shape, (6,))
        self.assertEqual(grid.x_mm.shape, (8,))
        self.assertEqual(grid.y_mm.shape, (10,))
        self.assertEqual(grid.x_grid_mm.shape, (10, 8))
        self.assertEqual(grid.y_grid_mm.shape, (10, 8))
        self.assertEqual(grid.area_weights_mm2.shape, (10, 8))
        self.assertTrue(np.any(grid.edge_mask))
        self.assertFalse(np.all(grid.edge_mask))

    def test_radial_profile_for_xy_grid(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_xy",
            wafer_radius_mm=120.0,
            nr=6,
            nx=32,
            ny=32,
            edge_exclusion_mm=0.0,
        )
        grid = build_domain_grid(spec)
        values = grid.r_grid_mm.copy()

        r_mm, profile = radial_profile(values, grid)
        self.assertEqual(r_mm.shape, (6,))
        self.assertEqual(profile.shape, (6,))
        self.assertTrue(np.isfinite(profile).any())
        finite = np.isfinite(profile)
        self.assertTrue(np.all(np.diff(profile[finite]) >= -1.0e-9))

    def test_radial_profile_for_polar_grid(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=120.0,
            nr=3,
            ntheta=4,
            edge_exclusion_mm=0.0,
        )
        grid = build_domain_grid(spec)
        values = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ]
        )

        r_mm, profile = radial_profile(values, grid)
        np.testing.assert_allclose(r_mm, grid.r_mm)
        np.testing.assert_allclose(profile, np.array([1.0, 2.0, 3.0]))

    def test_radial_profile_for_radial_grid_uses_mask(self) -> None:
        spec = DomainSpec(
            kind="wafer_1d_radial",
            wafer_radius_mm=60.0,
            nr=4,
            ntheta=1,
            edge_exclusion_mm=15.0,
        )
        grid = build_domain_grid(spec)
        values = np.array([10.0, 20.0, 30.0, 40.0])

        _, profile = radial_profile(values, grid)
        self.assertTrue(np.isnan(profile[-1]))
        np.testing.assert_allclose(profile[:3], np.array([10.0, 20.0, 30.0]))

    def test_edge_exclusion_mask_override(self) -> None:
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=80.0,
            nr=4,
            ntheta=4,
            edge_exclusion_mm=0.0,
        )
        grid = build_domain_grid(spec)

        override_mask = edge_exclusion_mask(grid, edge_exclusion_mm=20.0)
        self.assertTrue(np.all(override_mask[0:3, :]))
        self.assertFalse(np.any(override_mask[3, :]))

if __name__ == "__main__":
    unittest.main()
