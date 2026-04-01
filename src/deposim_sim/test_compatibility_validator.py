from __future__ import annotations

import unittest

from deposim_schema import compose_sim_config

from .validation.compatibility import validate_run_spec


class TestCompatibilityValidator(unittest.TestCase):
    def test_smoke_config_is_valid(self) -> None:
        spec = compose_sim_config("smoke")
        validate_run_spec(spec)

    def test_domain_kind_validation(self) -> None:
        spec = compose_sim_config("smoke", overrides=["sim.domain.kind=wafer_2d_xy"])
        validate_run_spec(spec)

    def test_invalid_xy_grid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spec = compose_sim_config(
                "smoke",
                overrides=[
                    "sim.domain.kind=wafer_2d_xy",
                    "sim.domain.nx=1",
                ],
            )
            validate_run_spec(spec)

    def test_flux_policy_validation(self) -> None:
        spec = compose_sim_config(
            "smoke",
            overrides=[
                "sim.model.params.transport.km_source=from_cfd_flux_sink",
                "sim.model.params.transport.from_cfd_flux_sink.flux_negative_policy=invalid",
            ],
        )
        with self.assertRaises(ValueError):
            validate_run_spec(spec)


if __name__ == "__main__":
    unittest.main()
