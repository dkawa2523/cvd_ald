from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment guard
    np = None  # type: ignore[assignment]

from deposim_schema import DomainSpec, ModelSpec, SolverSpec

from .domain import build_domain_grid
from .models import net_models, rate_laws
from .physics.cvd_steady import FieldBundle, run_cvd_steady


@unittest.skipIf(np is None, "NumPy is required for net model tests")
class TestNetModels(unittest.TestCase):
    def test_registry_resolves_models_and_lhhw_kinetics(self) -> None:
        net_names = net_models.available_net_models()
        self.assertIn("deposition_only", net_names)
        self.assertIn("dep_etch_loss", net_names)
        self.assertIs(net_models.resolve_net_model("deposition_only"), net_models.deposition_only)
        self.assertIs(net_models.resolve_net_model("dep_etch_loss"), net_models.dep_etch_loss)

        rate_names = rate_laws.available_rate_law_models()
        self.assertIn("lhhw_competition", rate_names)
        self.assertIn("competition_lhhw", rate_names)

    def test_dep_etch_loss_sign_convention(self) -> None:
        dep = np.array([1.0, 2.0, 3.0], dtype=float)
        net, comp = net_models.compute_net_rate(
            "dep_etch_loss",
            deposition_rate=dep,
            params={"etch_fraction": 0.2, "loss_fraction": 0.1},
        )
        np.testing.assert_allclose(comp["dep_rate"], dep)
        np.testing.assert_allclose(comp["etch_rate"], dep * 0.2)
        np.testing.assert_allclose(comp["loss_rate"], dep * 0.1)
        np.testing.assert_allclose(net, dep * 0.7)

    def test_cvd_steady_with_lhhw_and_dep_etch_loss(self) -> None:
        grid = build_domain_grid(
            DomainSpec(kind="wafer_2d_polar", wafer_radius_mm=80.0, nr=3, ntheta=6, edge_exclusion_mm=0.0)
        )
        fields = FieldBundle(
            C_ref={"A": np.full(grid.shape, 1.2, dtype=float)},
            T=np.full(grid.shape, 700.0, dtype=float),
            scalars={"omega_rad_s": 0.0},
        )
        model = ModelSpec(
            mass_transfer_name="stagnant_film",
            mass_transfer_params={"diffusivity_m2_s": 1.0e-5, "delta_eff_m": 2.0e-4},
            kinetics_name="lhhw_competition",
            kinetics_params={
                "k0": 0.4,
                "numerator_orders": {"A": 1.0},
                "denominator_coeffs": {"A": 0.2},
                "denominator_orders": {"A": 1.0},
                "nu": {"A": 1.0},
            },
            net_name="dep_etch_loss",
            net_params={"etch_fraction": 0.15, "loss_fraction": 0.05},
        )
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=model,
            process_time_s=5.0,
            solver_config=SolverSpec(max_iter=80, rtol=1.0e-7, atol=1.0e-12, monotonicity_check=False),
        )
        gross = result.diagnostics["gross_deposition_rate"]
        components = result.diagnostics["net_rate_components"]
        np.testing.assert_allclose(components["dep_rate"], gross)
        np.testing.assert_allclose(result.deposition_rate, gross - components["etch_rate"] - components["loss_rate"])
        np.testing.assert_allclose(result.thickness, result.deposition_rate * 5.0)
        self.assertEqual(result.diagnostics["net_model_name"], "dep_etch_loss")
        self.assertIn("Da_proxy", result.diagnostics)
        self.assertIn("apparent_orders", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
