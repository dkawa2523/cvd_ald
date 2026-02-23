from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .objective import evaluate_candidate_score


@unittest.skipIf(np is None, "NumPy is required")
class TestObjective(unittest.TestCase):
    def _objective(self, *, lambda_phys: float, lambda_solver: float, lambda_prior: float) -> dict[str, object]:
        return {
            "loss": "huber",
            "huber_delta_nm": 1.0,
            "phi_B_min": 0.2,
            "penalties": {
                "lambda_solver": lambda_solver,
                "lambda_phys": lambda_phys,
                "lambda_prior": lambda_prior,
            },
        }

    def test_inhibitor_penalty_increases_with_large_f_i(self) -> None:
        residual = np.array([0.0, 0.0, 0.0], dtype=float)
        fields_low = {"h_nm": np.zeros(3), "f_I": np.array([0.1, 0.2, 0.1]), "phi_B": np.array([0.5, 0.5, 0.5])}
        fields_high = {"h_nm": np.zeros(3), "f_I": np.array([0.9, 1.0, 0.8]), "phi_B": np.array([0.5, 0.5, 0.5])}

        low = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields_low,
            diagnostics={},
            role_has_i=True,
            role_has_b=False,
            objective=self._objective(lambda_phys=1.0, lambda_solver=0.0, lambda_prior=0.0),
            lambda_complex=0.0,
            prior_terms=None,
        )
        high = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields_high,
            diagnostics={},
            role_has_i=True,
            role_has_b=False,
            objective=self._objective(lambda_phys=1.0, lambda_solver=0.0, lambda_prior=0.0),
            lambda_complex=0.0,
            prior_terms=None,
        )
        self.assertGreater(high["penalty_phys"], low["penalty_phys"])

    def test_phi_b_penalty_increases_when_b_effect_is_low(self) -> None:
        residual = np.array([0.0, 0.0, 0.0], dtype=float)
        fields_low = {"h_nm": np.zeros(3), "f_I": np.array([0.0, 0.0, 0.0]), "phi_B": np.array([0.01, 0.02, 0.03])}
        fields_high = {"h_nm": np.zeros(3), "f_I": np.array([0.0, 0.0, 0.0]), "phi_B": np.array([0.4, 0.5, 0.6])}

        low = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields_low,
            diagnostics={},
            role_has_i=False,
            role_has_b=True,
            objective=self._objective(lambda_phys=1.0, lambda_solver=0.0, lambda_prior=0.0),
            lambda_complex=0.0,
            prior_terms=None,
        )
        high = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields_high,
            diagnostics={},
            role_has_i=False,
            role_has_b=True,
            objective=self._objective(lambda_phys=1.0, lambda_solver=0.0, lambda_prior=0.0),
            lambda_complex=0.0,
            prior_terms=None,
        )
        self.assertGreater(low["penalty_phys"], high["penalty_phys"])

    def test_prior_penalty_reflects_sigma_weighting(self) -> None:
        residual = np.array([0.0, 0.0], dtype=float)
        fields = {"h_nm": np.zeros(2), "f_I": np.array([0.0, 0.0]), "phi_B": np.array([0.0, 0.0])}

        weak = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields,
            diagnostics={},
            role_has_i=False,
            role_has_b=False,
            objective=self._objective(lambda_phys=0.0, lambda_solver=0.0, lambda_prior=0.1),
            lambda_complex=0.0,
            prior_terms=[0.1],
        )
        strong = evaluate_candidate_score(
            residual_nm=residual,
            fields=fields,
            diagnostics={},
            role_has_i=False,
            role_has_b=False,
            objective=self._objective(lambda_phys=0.0, lambda_solver=0.0, lambda_prior=2.0),
            lambda_complex=0.0,
            prior_terms=[0.1],
        )
        self.assertGreater(strong["penalty_prior"], weak["penalty_prior"])


if __name__ == "__main__":
    unittest.main()
