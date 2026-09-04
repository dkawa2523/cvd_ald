from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .objective import data_loss, evaluate_candidate_score


@unittest.skipIf(np is None, "NumPy is required")
class TestObjective(unittest.TestCase):
    def test_measurement_uncertainty_makes_score_invariant_to_consistent_unit_scaling(self):
        scores = []
        for scale in (1., 1.e-3):
            observation = {"target_nm": scale * np.array([1., 2.]),
                           "prediction_nm": scale * np.array([1.1, 2.2]),
                           "residual_nm": scale * np.array([.1, .2]),
                           "sigma_nm": scale * np.array([.1, .1]),
                           "xy_mm": np.array([[0., 0.], [1., 0.]])}
            out = evaluate_candidate_score(
                residual_nm=np.zeros(100), fields={}, diagnostics={"observation": observation},
                role_has_i=False, role_has_b=False, objective={}, lambda_complex=0.,
            )
            self.assertEqual(out["observation_count"], 2)
            scores.append(out["score_total"])
        self.assertAlmostEqual(*scores)

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

    def test_cvd_profile_can_penalize_center_edge_residual_bias(self) -> None:
        fields = {"h_nm": np.zeros(4), "residual_nm": np.array([0.0, 0.0, 1.0, 1.0])}
        diagnostics = {"xy_mm": np.array([[0.0, 0.0], [10.0, 0.0], [90.0, 0.0], [100.0, 0.0]])}
        out = evaluate_candidate_score(
            residual_nm=fields["residual_nm"],
            fields=fields,
            diagnostics=diagnostics,
            role_has_i=False,
            role_has_b=False,
            objective={
                "profile": "cvd_map",
                "loss": "huber",
                "huber_delta_nm": 1.0,
                "weights": {"measurement": 1.0, "spatial_bias": 0.5},
                "penalties": {},
            },
            lambda_complex=0.0,
            prior_terms=None,
        )
        self.assertGreater(out["penalty_profile"], 0.0)

    def test_ald_profile_uses_available_cycle_metrics(self) -> None:
        out = evaluate_candidate_score(
            residual_nm=np.zeros(3),
            fields={"h_nm": np.zeros(3)},
            diagnostics={"ald_metrics": {"plateau_gain_ratio": 2.0, "cycle_gpc_cv": 0.2, "purge_growth_fraction": 0.03}},
            role_has_i=False,
            role_has_b=False,
            objective={
                "profile": "ald_cycle",
                "loss": "huber",
                "huber_delta_nm": 1.0,
                "weights": {"measurement": 1.0, "plateau": 0.2, "cycle": 0.1, "purge": 0.1},
                "penalties": {},
            },
            lambda_complex=0.0,
            prior_terms=None,
        )
        self.assertGreater(out["penalty_profile"], 0.0)

    def test_data_loss_rejects_missing_measurement_residual_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "no finite residual_nm"):
            data_loss(
                residual_nm=np.array([np.nan, np.nan], dtype=float),
                loss_kind="huber",
                huber_delta_nm=1.0,
            )

    def test_prediction_fallback_requires_explicit_objective_flag(self) -> None:
        residual = np.array([np.nan, np.nan], dtype=float)
        with self.assertRaisesRegex(ValueError, "no finite residual_nm"):
            evaluate_candidate_score(
                residual_nm=residual,
                fields={"h_nm": np.array([2.0, 2.0], dtype=float)},
                diagnostics={},
                role_has_i=False,
                role_has_b=False,
                objective={"loss": "huber", "huber_delta_nm": 1.0, "weights": {"measurement": 1.0}},
                lambda_complex=0.0,
                prior_terms=None,
            )

        out = evaluate_candidate_score(
            residual_nm=residual,
            fields={"h_nm": np.array([2.0, 2.0], dtype=float)},
            diagnostics={},
            role_has_i=False,
            role_has_b=False,
            objective={
                "loss": "huber",
                "huber_delta_nm": 1.0,
                "weights": {"measurement": 1.0},
                "allow_prediction_fallback_when_no_measurement": True,
            },
            lambda_complex=0.0,
            prior_terms=None,
        )
        self.assertGreater(out["loss_data"], 0.0)


if __name__ == "__main__":
    unittest.main()
