from __future__ import annotations

import unittest

from .models.process_models import available_process_models, canonical_process_implementation, validate_process_model_choice


class TestProcessModels(unittest.TestCase):
    def test_registry_contains_simple_cvd_and_ald_aliases(self) -> None:
        names = set(available_process_models())
        self.assertIn("aib_ode", names)
        self.assertIn("role_cvd_aib", names)
        self.assertIn("role_ald_compat", names)
        self.assertIn("role_ald_state", names)
        self.assertEqual(canonical_process_implementation("role_cvd_aib"), "aib_ode")
        self.assertEqual(canonical_process_implementation("role_ald_compat"), "aib_ode")
        self.assertEqual(canonical_process_implementation("role_ald_state"), "ald_role_state")

    def test_registry_rejects_process_mismatch(self) -> None:
        validate_process_model_choice(name="role_cvd_aib", process="cvd", time_mode="steady")
        validate_process_model_choice(name="role_ald_compat", process="ald", time_mode="transient")
        validate_process_model_choice(name="role_ald_state", process="ald", time_mode="transient")
        with self.assertRaises(ValueError):
            validate_process_model_choice(name="role_ald_compat", process="cvd", time_mode="steady")
        with self.assertRaises(ValueError):
            validate_process_model_choice(name="role_ald_state", process="cvd", time_mode="steady")


if __name__ == "__main__":
    unittest.main()
