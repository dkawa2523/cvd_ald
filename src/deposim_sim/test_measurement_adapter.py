from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .measurement_adapter import align_point_measurement_to_points, compare_point_observations


@unittest.skipIf(np is None, "NumPy is required for measurement adapter tests")
class TestMeasurementAdapter(unittest.TestCase):
    def test_refining_mesh_does_not_duplicate_observations(self):
        measured_xy = np.array([[0., 0.], [2., 0.]])
        values = np.array([1., 3.])
        outputs = []
        for grid in (measured_xy, np.array([[0., 0.], [1., 0.], [2., 0.]])):
            outputs.append(compare_point_observations(
                prediction_nm=grid[:, 0] + 2, model_xy_mm=grid, measured=values,
                measurement_xy_mm=measured_xy, align={"enable": True},
            ))
        self.assertEqual(outputs[1]["count"], 2)
        np.testing.assert_allclose(outputs[0]["residual_nm"], outputs[1]["residual_nm"])

    def test_mean_rate_and_uncertainty_convert_with_duration_and_initial_film(self):
        out = compare_point_observations(
            prediction_nm=np.array([7., 12.]), model_xy_mm=np.array([[0., 0.], [1., 0.]]),
            measured=np.array([1., 2.]), measurement_xy_mm=np.array([[0., 0.], [1., 0.]]),
            align={}, quantity="mean_rate", duration_s=5, initial_nm=2., sigma=.1,
        )
        np.testing.assert_allclose(out["residual_nm"], 0)
        np.testing.assert_allclose(out["sigma_nm"], .5)

    def _point_measurement(self) -> tuple[np.ndarray, np.ndarray]:
        xy = np.array(
            [
                [-20.0, -20.0],
                [-20.0, 20.0],
                [20.0, -20.0],
                [20.0, 20.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        values = 0.05 * xy[:, 0] + 0.02 * xy[:, 1]
        return xy, values

    def test_alignment_is_deterministic(self) -> None:
        src_xy, values = self._point_measurement()
        tgt_xy = np.array([[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0]], dtype=float)
        a1, m1 = align_point_measurement_to_points(values=values, source_xy_mm=src_xy, target_xy_mm=tgt_xy)
        a2, m2 = align_point_measurement_to_points(values=values, source_xy_mm=src_xy, target_xy_mm=tgt_xy)
        np.testing.assert_allclose(a1, a2, equal_nan=True)
        np.testing.assert_array_equal(m1, m2)

    def test_shift_changes_aligned_values(self) -> None:
        src_xy, values = self._point_measurement()
        tgt_xy = np.array([[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0]], dtype=float)
        base, _ = align_point_measurement_to_points(values=values, source_xy_mm=src_xy, target_xy_mm=tgt_xy)
        shifted, _ = align_point_measurement_to_points(
            values=values,
            source_xy_mm=src_xy,
            target_xy_mm=tgt_xy,
            shift_mm=(8.0, -4.0),
        )
        self.assertGreater(float(np.nanmax(np.abs(base - shifted))), 0.0)

    def test_mask_radius_applies_nan_mask(self) -> None:
        src_xy, values = self._point_measurement()
        tgt_xy = np.array([[0.0, 0.0], [60.0, 0.0]], dtype=float)
        aligned, valid = align_point_measurement_to_points(
            values=values,
            source_xy_mm=src_xy,
            target_xy_mm=tgt_xy,
            mask_radius_mm=30.0,
        )
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))
        self.assertTrue(np.isnan(aligned[1]))

    def test_max_nearest_distance_rejects_remote_match(self) -> None:
        src_xy, values = self._point_measurement()
        tgt_xy = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=float)
        aligned, valid = align_point_measurement_to_points(
            values=values,
            source_xy_mm=src_xy,
            target_xy_mm=tgt_xy,
            max_nearest_distance_mm=10.0,
        )
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))
        self.assertTrue(np.isnan(aligned[1]))


if __name__ == "__main__":
    unittest.main()
