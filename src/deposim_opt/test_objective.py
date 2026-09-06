from __future__ import annotations

import unittest

import numpy as np

from .losses import data_loss, multi_observation_loss
from .objective import evaluate_candidate_score


class TestObjective(unittest.TestCase):
    def test_multi_observation_loss_is_unit_invariant(self) -> None:
        losses = []
        for scale in (1.0, 1.0e6):
            loss, components = multi_observation_loss(
                {
                    "film_rate": {
                        "target": scale * np.array([1.0, 2.0]),
                        "prediction": scale * np.array([1.1, 1.8]),
                        "sigma": scale * np.array([0.1, 0.2]),
                    },
                    "reactive_flux": {
                        "target": np.array([4.0, 5.0]),
                        "prediction": np.array([4.2, 4.7]),
                        "sigma": np.array([0.2, 0.3]),
                        "weight": 0.5,
                    },
                }
            )
            self.assertEqual(set(components), {"film_rate", "reactive_flux"})
            losses.append(loss)
        self.assertAlmostEqual(*losses)

    def test_multi_observation_loss_requires_uncertainty(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            multi_observation_loss({"film_rate": {"target": [1.0], "prediction": [1.1]}})

    def test_measurement_uncertainty_makes_score_unit_invariant(self) -> None:
        scores = []
        for scale in (1.0, 1.0e-3):
            observation = {
                "target_nm": scale * np.array([1.0, 2.0]),
                "prediction_nm": scale * np.array([1.1, 2.2]),
                "residual_nm": scale * np.array([0.1, 0.2]),
                "sigma_nm": scale * np.array([0.1, 0.1]),
                "xy_mm": np.array([[0.0, 0.0], [1.0, 0.0]]),
            }
            out = evaluate_candidate_score(
                residual_nm=np.zeros(100), fields={}, diagnostics={"observation": observation},
                objective={"loss": {"name": "mse", "standardized": "auto"}},
            )
            self.assertEqual(out["observation_count"], 2)
            scores.append(out["score_total"])
        self.assertAlmostEqual(*scores)

    def test_prior_penalty_is_kept_separate_from_data_loss(self) -> None:
        base = evaluate_candidate_score(
            residual_nm=np.array([1.0, -1.0]), fields={}, diagnostics={},
            objective={"loss": {"name": "mse", "standardized": False},
                       "penalties": {"lambda_prior": 0.0}}, prior_terms=[0.5],
        )
        regularized = evaluate_candidate_score(
            residual_nm=np.array([1.0, -1.0]), fields={}, diagnostics={},
            objective={"loss": {"name": "mse", "standardized": False},
                       "penalties": {"lambda_prior": 2.0}}, prior_terms=[0.5],
        )
        self.assertEqual(base["loss_data"], regularized["loss_data"])
        self.assertGreater(regularized["penalty_prior"], base["penalty_prior"])
        self.assertAlmostEqual(
            regularized["score_total"],
            regularized["loss_data"] + regularized["penalty_solver"] + regularized["penalty_prior"],
        )

    def test_data_loss_rejects_missing_residual_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "no finite residual"):
            data_loss(residual=np.array([np.nan, np.nan]), loss_name="huber", huber_delta=1.0)

    def test_prediction_fallback_requires_explicit_objective_flag(self) -> None:
        residual = np.array([np.nan, np.nan])
        with self.assertRaisesRegex(ValueError, "no finite residual"):
            evaluate_candidate_score(
                residual_nm=residual, fields={"h_nm": np.array([2.0, 2.0])}, diagnostics={},
                objective={"loss": {"name": "huber", "standardized": False, "delta_nm": 1.0}},
            )

        out = evaluate_candidate_score(
            residual_nm=residual, fields={"h_nm": np.array([2.0, 2.0])}, diagnostics={},
            objective={
                "loss": {"name": "huber", "standardized": False, "delta_nm": 1.0},
                "allow_prediction_fallback_when_no_measurement": True,
            },
        )
        self.assertGreater(out["loss_data"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
