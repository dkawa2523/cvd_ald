from __future__ import annotations

import unittest

from deposim_schema import compose_opt_config, compose_sim_config


class TestSimConfigV2(unittest.TestCase):
    def test_compose_sim_v2(self) -> None:
        spec = compose_sim_config("cvd_steady_min")
        self.assertEqual(spec.model.name, "aib_ode")
        self.assertEqual(spec.time_mode, "steady")
        self.assertTrue(spec.roles.A)

    def test_compose_opt_v2(self) -> None:
        spec = compose_opt_config("fit_cvd_steady_min")
        self.assertEqual(spec.sim.model.name, "aib_ode")
        self.assertEqual(spec.opt.task, "fit_roles_and_params")
        self.assertTrue(spec.opt.role_enumeration.enabled)
        self.assertEqual(spec.opt.parameter_fit.pruner, "none")
        self.assertIn("levels", spec.opt.parameter_fit.fidelity)
        self.assertIn("penalties", spec.opt.parameter_fit.objective)
        self.assertIn("analysis", spec.opt.parameter_fit.__dict__)
        self.assertIn("identifiability", spec.opt.parameter_fit.analysis)

    def test_deprecated_aliases_resolve_to_aib_min_configs(self) -> None:
        sim_spec = compose_sim_config("smoke")
        self.assertEqual(sim_spec.model.name, "aib_ode")
        self.assertEqual(sim_spec.time_mode, "steady")

        opt_spec = compose_opt_config("stub")
        self.assertEqual(opt_spec.sim.model.name, "aib_ode")
        self.assertEqual(opt_spec.opt.task, "fit_roles_and_params")


if __name__ == "__main__":
    unittest.main()
