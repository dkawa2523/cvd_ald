from __future__ import annotations

import unittest

from .models.process_models import (
    available_process_models,
    canonical_process_implementation,
    get_process_model_info,
    primary_process_models,
    validate_process_model_choice,
)


class TestProcessModels(unittest.TestCase):
    def test_registry_contains_public_role_models(self) -> None:
        names = set(available_process_models())
        self.assertEqual(names, {"role_cvd_aib", "role_cvd_mvk", "role_ald_state"})
        self.assertEqual(canonical_process_implementation("role_cvd_aib"), "aib_ode")
        self.assertEqual(canonical_process_implementation("role_ald_state"), "ald_role_state")
        self.assertEqual(canonical_process_implementation("role_cvd_mvk"), "mvk_state")
        mvk = get_process_model_info("role_cvd_mvk")
        self.assertEqual(mvk.required_roles, ("A", "B"))
        self.assertEqual(mvk.steady_observable_equivalence, "aib_qss:AB:no_desorption")
        self.assertEqual(dict(mvk.quantity_units)["k_reduce"], "m^3/(kmol s)")
        self.assertEqual({info.name for info in primary_process_models()}, names)

    def test_registry_rejects_process_mismatch(self) -> None:
        validate_process_model_choice(name="role_cvd_aib", process="cvd", time_mode="steady")
        validate_process_model_choice(name="role_ald_state", process="ald", time_mode="transient")
        validate_process_model_choice(name="role_cvd_mvk", process="cvd", time_mode="transient")
        with self.assertRaises(ValueError):
            validate_process_model_choice(name="role_ald_state", process="cvd", time_mode="steady")


if __name__ == "__main__":
    unittest.main()
