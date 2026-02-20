from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, ModelSpec, SolverSpec

from .domain import build_domain_grid
from .physics.cvd_steady import FieldBundle, run_cvd_steady
from .state_closure import dynamic_ode_closure, steady_state_closure


@unittest.skipIf(np is None, "NumPy is required for state closure tests")
class TestStateClosure(unittest.TestCase):
    def test_dynamic_ode_closure_bounds(self) -> None:
        cs = {"A": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=float)}
        out = dynamic_ode_closure(
            Cs=cs,
            params={"species": "A", "A": 0.8, "B": 0.1, "m": 1.0, "theta0": 0.2},
            dt_s=2.0,
            initial_state={"theta": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=float)},
        )
        theta = out["theta"]
        self.assertEqual(theta.shape, (2, 2))
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))

    def test_steady_state_closure_bounds(self) -> None:
        cs = {"A": np.array([0.2, 0.6, 1.2], dtype=float)}
        out = steady_state_closure(Cs=cs, params={"species": "A", "A": 1.5, "B": 0.2})
        theta = out["theta"]
        self.assertEqual(theta.shape, (3,))
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))

    def test_sticking_flux_runs_via_model_selection(self) -> None:
        grid = build_domain_grid(
            DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=80.0, nr=3, ntheta=6, edge_exclusion_mm=0.0)
        )
        fields = FieldBundle(
            C_ref={"A": np.full(grid.shape, 1.0, dtype=float)},
            T=np.full(grid.shape, 700.0, dtype=float),
            scalars={"omega_rad_s": 0.0},
        )
        model = ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
            kinetics_name="sticking_flux",
            kinetics_params={"species": "A", "alpha_stick": 0.3, "molar_mass_kg_mol": 0.1, "nu": {"A": 1.0}},
            state_name="dynamic_ode",
            state_params={"species": "A", "A": 0.5, "B": 0.1, "m": 1.0, "theta0": 0.2},
            net_name="deposition_only",
        )
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=model,
            process_time_s=1.0,
            solver_config=SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False),
        )
        self.assertTrue(np.all(np.isfinite(result.deposition_rate)))
        self.assertIn("state_snapshot", result.diagnostics)
        self.assertIn("theta", result.diagnostics["state_snapshot"])
        theta = np.asarray(result.diagnostics["state_snapshot"]["theta"], dtype=float)
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))


if __name__ == "__main__":
    unittest.main()
