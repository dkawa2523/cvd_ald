from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec

from .domain import build_domain_grid
from .measurement_adapter import MeasurementMap, align_measurement_to_grid


@unittest.skipIf(np is None, "NumPy is required for measurement adapter tests")
class TestMeasurementAdapter(unittest.TestCase):
    def _measurement(self) -> MeasurementMap:
        x = np.linspace(-100.0, 100.0, 21)
        y = np.linspace(-100.0, 100.0, 21)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        values = 0.01 * xx + 0.02 * yy
        valid = np.ones(values.shape, dtype=bool)
        return MeasurementMap(x_mm=x, y_mm=y, values=values, valid_mask=valid)

    def _grid(self):
        spec = DomainSpec(kind="wafer_2d_xy", wafer_radius_mm=100.0, nr=8, nx=31, ny=31, edge_exclusion_mm=0.0)
        return build_domain_grid(spec)

    def test_alignment_is_deterministic(self) -> None:
        meas = self._measurement()
        grid = self._grid()
        a1, m1 = align_measurement_to_grid(meas, grid, interpolation="nearest")
        a2, m2 = align_measurement_to_grid(meas, grid, interpolation="nearest")
        np.testing.assert_allclose(a1, a2, equal_nan=True)
        np.testing.assert_array_equal(m1, m2)

    def test_shift_changes_aligned_map(self) -> None:
        meas = self._measurement()
        grid = self._grid()
        base, _ = align_measurement_to_grid(meas, grid, interpolation="nearest")
        shifted, _ = align_measurement_to_grid(meas, grid, dx_mm=10.0, dy_mm=-5.0, interpolation="nearest")
        self.assertGreater(float(np.nanmax(np.abs(base - shifted))), 0.0)

    def test_edge_exclusion_reduces_valid_region(self) -> None:
        meas = self._measurement()
        grid = self._grid()
        _, mask0 = align_measurement_to_grid(meas, grid, edge_exclusion_mm=0.0)
        _, mask1 = align_measurement_to_grid(meas, grid, edge_exclusion_mm=20.0)
        self.assertLess(int(np.sum(mask1)), int(np.sum(mask0)))

    def test_bilinear_and_nearest_return_same_shape(self) -> None:
        meas = self._measurement()
        grid = self._grid()
        nearest, nmask = align_measurement_to_grid(meas, grid, interpolation="nearest")
        bilinear, bmask = align_measurement_to_grid(meas, grid, interpolation="bilinear")
        self.assertEqual(nearest.shape, grid.shape)
        self.assertEqual(bilinear.shape, grid.shape)
        self.assertEqual(nmask.shape, grid.shape)
        self.assertEqual(bmask.shape, grid.shape)


if __name__ == "__main__":
    unittest.main()
