from __future__ import annotations

import unittest

from deposim_schema import compose_opt_config, compose_sim_config


class TestSimConfigV2(unittest.TestCase):
    def test_compose_sim_v2(self) -> None:
        spec = compose_sim_config("cvd_steady_min")
        self.assertEqual(spec.model.name, "role_cvd_aib")
        self.assertEqual(spec.time_mode, "steady")
        self.assertTrue(spec.roles.A)

    def test_compose_opt_v2(self) -> None:
        spec = compose_opt_config("fit_cvd_steady_min")
        self.assertEqual(spec.sim.model.name, "role_cvd_aib")
        self.assertEqual(spec.opt.task, "fit_roles_and_params")
        self.assertTrue(spec.opt.role_enumeration.enabled)
        self.assertEqual(spec.opt.parameter_fit.search["pruner"], "none")
        self.assertIn("levels", spec.opt.parameter_fit.fidelity)
        self.assertIn("penalties", spec.opt.parameter_fit.objective)
        self.assertIn("analysis", spec.opt.parameter_fit.__dict__)
        self.assertIn("identifiability", spec.opt.parameter_fit.analysis)

    def test_compose_ald_state_opt_v2(self) -> None:
        spec = compose_opt_config("fit_ald_state_min")
        self.assertEqual(spec.sim.model.name, "role_ald_state")
        self.assertEqual(spec.sim.process, "ald")
        self.assertEqual(spec.opt.parameter_fit.objective["loss"]["name"], "mse")

        multicond = compose_opt_config("fit_ald_state_multicond_min")
        self.assertEqual(multicond.sim.model.name, "role_ald_state")
        self.assertEqual(len(multicond.opt.measurement["conditions"]), 5)

    def test_process_model_configs_compose(self) -> None:
        ald_state = compose_sim_config("ald_state_min")
        self.assertEqual(ald_state.model.name, "role_ald_state")
        self.assertEqual(ald_state.process, "ald")
        self.assertEqual(ald_state.time_mode, "transient")
        self.assertEqual(ald_state.time.solver.name, "explicit_substep_bounded")

        mvk = compose_sim_config("cvd_mvk_transient_min")
        self.assertEqual(mvk.model.name, "role_cvd_mvk")
        self.assertEqual(mvk.roles.B, "s1")
        self.assertEqual(mvk.initial_conditions.redox_fraction.value, 1.0)

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
