from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec
from deposim_sim.domain import build_domain_grid
from deposim_sim.models import mass_transfer
from deposim_sim.solvers.root_solve import solve_progress_R


@unittest.skipIf(np is None, "NumPy is required for P0-010 sanity tests")
class TestP0010UnitSanity(unittest.TestCase):
    def _grid(self):
        spec = DomainSpec(
            kind="wafer_2d_polar",
            wafer_radius_mm=100.0,
            nr=2,
            ntheta=3,
            edge_exclusion_mm=0.0,
        )
        return build_domain_grid(spec)

    def test_root_bracket_and_nonnegativity(self) -> None:
        c_ref = np.array([0.0, 1.0, 2.0], dtype=float)
        k_m = 0.8
        nu = 1.0

        def first_order_rate(Cs, state=None, T=None, params=None):
            return float(params["k"]) * Cs["A"]

        R, Cs, _, _ = solve_progress_R(
            c_ref={"A": c_ref},
            k_m={"A": k_m},
            nu={"A": nu},
            rate_fn=first_order_rate,
            rate_params={"k": 0.3},
        )

        r_max = (k_m * c_ref) / nu
        self.assertTrue(np.all(R >= 0.0))
        self.assertTrue(np.all(R <= r_max + 1.0e-12))
        self.assertTrue(np.all(Cs["A"] >= 0.0))
        np.testing.assert_allclose(R[0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(Cs["A"][0], 0.0, atol=1.0e-12)

    def test_regime_limits_reaction_vs_transport(self) -> None:
        c_ref = np.full((5,), 1.5, dtype=float)
        k_m = 0.4

        def first_order_rate(Cs, state=None, T=None, params=None):
            return float(params["k"]) * Cs["A"]

        R_reaction, _, _, _ = solve_progress_R(
            c_ref={"A": c_ref},
            k_m={"A": k_m},
            nu={"A": 1.0},
            rate_fn=first_order_rate,
            rate_params={"k": 1.0e-5},
            max_iter=120,
            rtol=1.0e-10,
            atol=1.0e-16,
            monotonicity_check=False,
        )
        R_transport, _, _, _ = solve_progress_R(
            c_ref={"A": c_ref},
            k_m={"A": k_m},
            nu={"A": 1.0},
            rate_fn=first_order_rate,
            rate_params={"k": 1.0e5},
            max_iter=120,
            rtol=1.0e-10,
            atol=1.0e-16,
            monotonicity_check=False,
        )

        r_max = k_m * c_ref
        np.testing.assert_allclose(R_reaction, 1.0e-5 * c_ref, rtol=1.0e-3, atol=1.0e-12)
        np.testing.assert_allclose(R_transport, r_max, rtol=1.0e-3, atol=1.0e-9)
        self.assertTrue(np.all(R_reaction < 1.0e-2 * r_max))

    def test_rotating_disk_omega_zero_guard_error(self) -> None:
        grid = self._grid()
        with self.assertRaisesRegex(ValueError, "omega_rad_s=0"):
            mass_transfer.compute_km(
                "rotating_disk",
                grid=grid,
                diffusivity_m2_s=2.0e-5,
                nu_m2_s=1.0e-5,
                ck=0.62,
                omega_rad_s=0.0,
                omega_zero_guard="error",
            )

    def test_rotating_disk_omega_zero_guard_fallback(self) -> None:
        grid = self._grid()
        omega = np.array([[0.0, 4.0, 9.0], [16.0, 0.0, 1.0]], dtype=float)
        km = mass_transfer.compute_km(
            "rotating_disk",
            grid=grid,
            diffusivity_m2_s=2.0e-5,
            nu_m2_s=1.0e-5,
            ck=0.62,
            omega_rad_s=omega,
            omega_zero_guard="fallback_stagnant_film",
            delta_eff_m=1.0e-3,
        )

        expected_rot = 0.62 * (2.0e-5 ** (2.0 / 3.0)) * np.sqrt(omega) * (1.0e-5 ** (-1.0 / 6.0))
        expected_fallback = np.full(grid.shape, 2.0e-5 / 1.0e-3)
        expected = np.where(omega == 0.0, expected_fallback, expected_rot)
        np.testing.assert_allclose(km, expected)


if __name__ == "__main__":
    unittest.main()
