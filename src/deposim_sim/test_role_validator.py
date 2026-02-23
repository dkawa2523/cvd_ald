from __future__ import annotations

import unittest

from deposim_schema import compose_sim_config

from .validation.compatibility import validate_roles_v2, validate_sim_spec_v2


class TestRoleValidator(unittest.TestCase):
    def test_valid_spec(self) -> None:
        spec = compose_sim_config("cvd_steady_min")
        validate_roles_v2(spec)
        validate_sim_spec_v2(spec)

    def test_disjoint_violation(self) -> None:
        spec = compose_sim_config("cvd_steady_min")
        spec.roles.I = spec.roles.A
        with self.assertRaises(ValueError):
            validate_roles_v2(spec)


if __name__ == "__main__":
    unittest.main()
