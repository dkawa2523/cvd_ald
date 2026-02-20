from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, ModelSpec, SolverSpec

from .domain import build_domain_grid
from .models.mass_transfer import compute_km
from .physics.cvd_steady import FieldBundle, run_cvd_steady


@unittest.skipIf(np is None, "NumPy is required for Bosanquet/pattern tests")
class TestBosanquetPattern(unittest.TestCase):
    def _grid(self):
        return build_domain_grid(DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=80.0, nr=3, ntheta=6, edge_exclusion_mm=0.0))

    def _model(self, *, pattern: float | np.ndarray | None = None) -> ModelSpec:
        params = {
            "k0": 0.2,
            "orders": {"A": 1.0},
            "nu": {"A": 1.0},
        }
        if pattern is not None:
            params["pattern_loading"] = pattern
        return ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
            kinetics_name="power_law",
            kinetics_params=params,
            net_name="deposition_only",
        )

    def test_bosanquet_diffusivity_harmonic_mean(self) -> None:
        grid = self._grid()
        km = compute_km(
            "stagnant_film",
            grid=grid,
            params={
                "diffusivity_model": "bosanquet",
                "d_m_m2_s": 2.0e-5,
                "d_k_m2_s": 1.0e-5,
                "delta_eff_m": 5.0e-4,
            },
        )
        d_eff = 1.0 / (1.0 / 2.0e-5 + 1.0 / 1.0e-5)
        expected = d_eff / 5.0e-4
        np.testing.assert_allclose(km, np.full(grid.shape, expected))

    def test_pattern_loading_changes_tc_solution(self) -> None:
        grid = self._grid()
        fields = FieldBundle(C_ref={"A": np.full(grid.shape, 1.0, dtype=float)}, T=np.full(grid.shape, 700.0, dtype=float))
        solver = SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False)

        baseline = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._model(pattern=None),
            process_time_s=5.0,
            solver_config=solver,
        )
        reduced = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._model(pattern=0.5),
            process_time_s=5.0,
            solver_config=solver,
        )
        self.assertGreater(
            float(np.max(np.abs(reduced.deposition_rate - baseline.deposition_rate))),
            1.0e-12,
        )
        self.assertGreater(
            float(np.max(np.abs(reduced.Cs["A"] - baseline.Cs["A"]))),
            1.0e-12,
        )
        self.assertTrue(reduced.diagnostics["pattern_loading_enabled"])

    def test_pattern_loading_one_matches_baseline(self) -> None:
        grid = self._grid()
        fields = FieldBundle(C_ref={"A": np.full(grid.shape, 1.0, dtype=float)}, T=np.full(grid.shape, 700.0, dtype=float))
        solver = SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False)

        baseline = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._model(pattern=None),
            process_time_s=5.0,
            solver_config=solver,
        )
        one = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=self._model(pattern=1.0),
            process_time_s=5.0,
            solver_config=solver,
        )
        np.testing.assert_allclose(one.deposition_rate, baseline.deposition_rate)
        np.testing.assert_allclose(one.thickness, baseline.thickness)


if __name__ == "__main__":
    unittest.main()
