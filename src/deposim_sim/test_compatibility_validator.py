from __future__ import annotations

import unittest

from deposim_schema import compose_sim_config

from .validation.compatibility import validate_run_spec


class TestCompatibilityValidator(unittest.TestCase):
    def test_smoke_config_is_valid(self) -> None:
        spec = compose_sim_config("smoke")
        validate_run_spec(spec)

    def test_rotating_disk_zero_omega_error_guard_fails(self) -> None:
        spec = compose_sim_config(
            "smoke",
            overrides=[
                "model.mass_transfer_name=rotating_disk",
                "+model.mass_transfer_params.omega_zero_guard=error",
                "+model.mass_transfer_params.diffusivity_m2_s=1e-4",
                "+model.mass_transfer_params.nu_m2_s=1e-6",
                "inputs.omega_rad_s=0.0",
            ],
        )
        with self.assertRaises(ValueError):
            validate_run_spec(spec)

    def test_dynamic_state_rejected_for_steady_mode(self) -> None:
        spec = compose_sim_config(
            "smoke",
            overrides=[
                "model.state_name=dynamic_ode",
                "time.mode=cvd_steady",
            ],
        )
        with self.assertRaises(ValueError):
            validate_run_spec(spec)


if __name__ == "__main__":
    unittest.main()
