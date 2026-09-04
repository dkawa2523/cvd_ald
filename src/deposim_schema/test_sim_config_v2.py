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

    def test_compose_ald_state_opt_v2(self) -> None:
        spec = compose_opt_config("fit_ald_state_min")
        self.assertEqual(spec.sim.model.name, "role_ald_state")
        self.assertEqual(spec.sim.process, "ald")
        self.assertEqual(spec.opt.parameter_fit.objective["profile"], "generic")

        multicond = compose_opt_config("fit_ald_state_multicond_min")
        self.assertEqual(multicond.sim.model.name, "role_ald_state")
        self.assertEqual(len(multicond.opt.measurement["conditions"]), 5)

    def test_deprecated_aliases_resolve_to_aib_min_configs(self) -> None:
        sim_spec = compose_sim_config("smoke")
        self.assertEqual(sim_spec.model.name, "aib_ode")
        self.assertEqual(sim_spec.time_mode, "steady")

        opt_spec = compose_opt_config("stub")
        self.assertEqual(opt_spec.sim.model.name, "aib_ode")
        self.assertEqual(opt_spec.opt.task, "fit_roles_and_params")

    def test_process_model_alias_configs_compose(self) -> None:
        ald_spec = compose_sim_config("ald_transient_min")
        self.assertEqual(ald_spec.model.name, "role_ald_compat")
        self.assertEqual(ald_spec.process, "ald")
        self.assertEqual(ald_spec.time_mode, "transient")

        ald_state = compose_sim_config("ald_state_min")
        self.assertEqual(ald_state.model.name, "role_ald_state")
        self.assertEqual(ald_state.process, "ald")
        self.assertEqual(ald_state.time_mode, "transient")
        self.assertEqual(ald_state.time.solver.name, "explicit_substep_bounded")

    def test_ald_state_rejects_misleading_implicit_solver_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit_substep_bounded"):
            compose_sim_config("ald_state_min", overrides=["sim.time.solver.name=implicit_euler_bisect"])

    def test_domain_kind_overrides_are_supported(self) -> None:
        spec_xy = compose_sim_config(
            "cvd_steady_min",
            overrides=[
                "sim.domain.kind=wafer_2d_xy",
                "sim.domain.nr=4",
                "sim.domain.nx=8",
                "sim.domain.ny=6",
            ],
        )
        self.assertEqual(spec_xy.domain.kind, "wafer_2d_xy")

        spec_polar = compose_sim_config(
            "cvd_steady_min",
            overrides=[
                "sim.domain.kind=wafer_2d_polar",
                "sim.domain.nr=4",
                "sim.domain.ntheta=12",
            ],
        )
        self.assertEqual(spec_polar.domain.kind, "wafer_2d_polar")


if __name__ == "__main__":
    unittest.main()
