from __future__ import annotations

import unittest
import warnings

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from .solvers.root_solve import (
    STATUS_BRACKET_NOT_FOUND,
    STATUS_CS_CLIPPED,
    STATUS_FALLBACK_INTERVAL_SPLIT,
    STATUS_MAX_ITER_REACHED,
    STATUS_NON_MONOTONIC,
    STATUS_NON_MONOTONIC_FAILURE,
    solve_progress_R,
)


@unittest.skipIf(np is None, "NumPy is required for root solver tests")
class TestRootSolve(unittest.TestCase):
    def test_monotonic_solution_shapes_and_bounds(self) -> None:
        c_ref = np.array([1.0, 2.0, 4.0], dtype=float)
        alpha = 0.3

        def rate_fn(Cs, state=None, T=None, params=None):
            return float(params["alpha"]) * Cs["A"]

        R, Cs, iteration_count, status_map = solve_progress_R(
            c_ref={"A": c_ref},
            k_m={"A": 1.0},
            nu={"A": 1.0},
            rate_fn=rate_fn,
            rate_params={"alpha": alpha},
        )

        expected_R = alpha / (1.0 + alpha) * c_ref
        np.testing.assert_allclose(R, expected_R, rtol=1.0e-6, atol=1.0e-8)
        np.testing.assert_allclose(Cs["A"], c_ref - expected_R, rtol=1.0e-6, atol=1.0e-8)
        self.assertEqual(R.shape, c_ref.shape)
        self.assertEqual(iteration_count.shape, c_ref.shape)
        self.assertEqual(status_map.shape, c_ref.shape)
        self.assertEqual(iteration_count.dtype, np.int32)
        self.assertEqual(status_map.dtype, np.int32)
        self.assertTrue(np.all(R >= 0.0))
        self.assertTrue(np.all(R <= c_ref))
        self.assertTrue(np.all(Cs["A"] >= 0.0))
        self.assertTrue(np.all(status_map == 0))

    def test_bracket_not_found_sets_status(self) -> None:
        def constant_high_rate(Cs, state=None, T=None, params=None):
            return np.full_like(Cs["A"], 10.0)

        R, Cs, iteration_count, status_map = solve_progress_R(
            c_ref={"A": np.array([1.0, 2.0], dtype=float)},
            k_m={"A": 1.0},
            nu={"A": 1.0},
            rate_fn=constant_high_rate,
            monotonicity_check=False,
        )

        self.assertTrue(np.all((status_map & STATUS_BRACKET_NOT_FOUND) != 0))
        self.assertTrue(np.all(iteration_count == 0))
        np.testing.assert_allclose(R, np.zeros_like(R))
        np.testing.assert_allclose(Cs["A"], np.array([1.0, 2.0], dtype=float))

    def test_max_iter_reached_sets_status(self) -> None:
        def rate_fn(Cs, state=None, T=None, params=None):
            return 0.3 * Cs["A"]

        R, Cs, iteration_count, status_map = solve_progress_R(
            c_ref={"A": np.array([1.0], dtype=float)},
            k_m={"A": 1.0},
            nu={"A": 1.0},
            rate_fn=rate_fn,
            max_iter=1,
            rtol=1.0e-14,
            atol=1.0e-20,
            monotonicity_check=False,
        )

        self.assertTrue(bool(status_map[0] & STATUS_MAX_ITER_REACHED))
        self.assertEqual(int(iteration_count[0]), 1)
        self.assertGreaterEqual(float(R[0]), 0.0)
        self.assertGreaterEqual(float(Cs["A"][0]), 0.0)

    def test_non_monotonic_warn_and_fail_sets_flags(self) -> None:
        def non_monotonic_rate(Cs, state=None, T=None, params=None):
            return 0.2 + 0.6 * np.sin(8.0 * Cs["A"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, _, status_map = solve_progress_R(
                c_ref={"A": np.array([1.0], dtype=float)},
                k_m={"A": 1.0},
                nu={"A": 1.0},
                rate_fn=non_monotonic_rate,
                non_monotonic_mode="warn_and_fail",
            )

        self.assertTrue(bool(status_map[0] & STATUS_NON_MONOTONIC))
        self.assertTrue(bool(status_map[0] & STATUS_NON_MONOTONIC_FAILURE))

    def test_non_monotonic_warn_and_split_sets_split_flag(self) -> None:
        def non_monotonic_rate(Cs, state=None, T=None, params=None):
            return 0.2 + 0.6 * np.sin(8.0 * Cs["A"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, _, status_map = solve_progress_R(
                c_ref={"A": np.array([1.0], dtype=float)},
                k_m={"A": 1.0},
                nu={"A": 1.0},
                rate_fn=non_monotonic_rate,
                non_monotonic_mode="warn_and_split",
            )

        self.assertTrue(bool(status_map[0] & STATUS_NON_MONOTONIC))
        self.assertTrue(bool(status_map[0] & STATUS_FALLBACK_INTERVAL_SPLIT))
        self.assertFalse(bool(status_map[0] & STATUS_NON_MONOTONIC_FAILURE))

    def test_cs_clipped_status_is_set_when_roundoff_drives_negative(self) -> None:
        c_ref = 3.818252938586784
        k_m = 3536.009194432506
        nu = 0.0001492588392223162
        r_max = k_m * c_ref / nu

        def target_rate(Cs, state=None, T=None, params=None):
            return np.full_like(Cs["A"], float(params["target"]))

        R, Cs, _, status_map = solve_progress_R(
            c_ref={"A": np.array([c_ref], dtype=float)},
            k_m={"A": np.array([k_m], dtype=float)},
            nu={"A": np.array([nu], dtype=float)},
            rate_fn=target_rate,
            rate_params={"target": r_max},
            monotonicity_check=False,
        )

        self.assertTrue(bool(status_map[0] & STATUS_CS_CLIPPED))
        self.assertGreaterEqual(float(R[0]), 0.0)
        self.assertGreaterEqual(float(Cs["A"][0]), 0.0)

    def test_inputs_broadcast_to_common_shape(self) -> None:
        temperature = np.array([[700.0, 750.0, 800.0]], dtype=float)
        state = {"s": np.ones((2, 3), dtype=float)}

        def rate_fn(Cs, state=None, T=None, params=None):
            alpha = float((params or {}).get("alpha", 0.2))
            return alpha * Cs["A"]

        R, Cs, iteration_count, status_map = solve_progress_R(
            c_ref={"A": 1.0},
            k_m={"A": np.array([[1.0], [1.2]], dtype=float)},
            nu={"A": 1.0},
            rate_fn=rate_fn,
            T=temperature,
            state=state,
            rate_params={"alpha": 0.2},
        )

        self.assertEqual(R.shape, (2, 3))
        self.assertEqual(Cs["A"].shape, (2, 3))
        self.assertEqual(iteration_count.shape, (2, 3))
        self.assertEqual(status_map.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(R)))
        self.assertTrue(np.all(Cs["A"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
