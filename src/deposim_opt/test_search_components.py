from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from deposim_schema import compose_sim_config

from .losses import data_loss, multi_observation_loss, wafer_loss
from .parameter_space import compile_parameter_space, draw_parameter_sample, parameter_dimension
from .samplers import parse_search_settings, repetition_seeds, run_search, trial_budget


def _settings(**overrides):
    values = {
        "method": "random", "seed": 7, "min_trials": 3, "max_trials": 20,
        "trials_per_dimension": 4, "patience": 2,
        "relative_improvement": 1.0e-4, "repetitions": 2,
        "pruner": "none", "sampler_options": {}, "storage": {},
    }
    values.update(overrides)
    return parse_search_settings(SimpleNamespace(search=values))


class TestLossLibrary(unittest.TestCase):
    def test_unknown_loss_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss name"):
            data_loss(residual=[1.0], loss_name="typo")

    def test_observation_weights_are_normalized(self) -> None:
        observations = {
            "a": {"target": [0.0], "prediction": [1.0], "sigma": [1.0], "weight": 1.0},
            "b": {"target": [0.0], "prediction": [3.0], "sigma": [1.0], "weight": 3.0},
        }
        total, components = multi_observation_loss(observations, loss_name="mse")
        self.assertEqual(components, {"a": 1.0, "b": 9.0})
        self.assertAlmostEqual(total, 7.0)

    def test_wafer_normalized_losses_remove_absolute_rate_scale(self) -> None:
        target = np.array([1.0, 1.0, 10.0, 10.0])
        prediction = 2.0 * target
        conditions = np.array([1, 1, 2, 2])
        weights = np.full(4, 0.25)
        self.assertAlmostEqual(
            wafer_loss(
                target=target,
                prediction=prediction,
                condition_id=conditions,
                weights=weights,
                loss_name="mse",
            ),
            50.5,
        )
        self.assertAlmostEqual(
            wafer_loss(
                target=target,
                prediction=prediction,
                condition_id=conditions,
                weights=weights,
                loss_name="wafer_normalized_mse",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            wafer_loss(
                target=target,
                prediction=prediction,
                condition_id=conditions,
                weights=weights,
                loss_name="wafer_normalized_mae",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            wafer_loss(
                target=target,
                prediction=prediction,
                condition_id=conditions,
                weights=weights,
                loss_name="symmetric_normalized_mse",
            ),
            0.4,
        )


class TestParameterSpace(unittest.TestCase):
    def test_role_conditions_and_fixed_values_define_actual_dimension(self) -> None:
        sim = compose_sim_config("cvd_steady_min")
        space = compile_parameter_space(
            [
                {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1e-3, "high": 1.0},
                {"name": "model.params.transport.km_B", "type": "uniform", "low": 2.0, "high": 2.0, "condition": "role_has_B"},
                {"name": "model.params.inhibitor.K_I", "type": "uniform", "low": 0.0, "high": 1.0, "condition": "role_has_I"},
            ],
            sim_spec=sim, task="fit_roles_and_params", role_has_i=False, role_has_b=True,
        )
        self.assertEqual(parameter_dimension(space, ["one", "two"]), 1)
        sample = draw_parameter_sample(space=space, condition_names=["one", "two"], lambda_prior=0.0, rng=np.random.default_rng(1))
        self.assertEqual(sample["flat_params"]["model.params.transport.km_B"], 2.0)
        self.assertNotIn("model.params.inhibitor.K_I", sample["flat_params"])


class TestSamplerLibrary(unittest.TestCase):
    def test_budget_scales_with_active_dimension_and_is_capped(self) -> None:
        settings = _settings(min_trials=5, max_trials=30, trials_per_dimension=6)
        self.assertEqual(trial_budget(settings, 3), 18)
        self.assertEqual(trial_budget(settings, 10), 30)

    def test_repetition_seeds_are_deterministic_and_distinct(self) -> None:
        self.assertEqual(repetition_seeds(_settings(repetitions=3)), [7, 104736, 209465])

    def test_sampler_options_from_cli_are_typed(self) -> None:
        settings = _settings(sampler_options={"n_startup_trials": "2", "multivariate": "true"})
        self.assertEqual(settings.sampler_options["n_startup_trials"], 2)
        self.assertIs(settings.sampler_options["multivariate"], True)

    def test_priority_sampler_names_are_explicit(self) -> None:
        for method in ("de", "pso", "levy", "cma_mae"):
            self.assertEqual(_settings(method=method).method, method)

    def test_random_search_records_plateau_termination(self) -> None:
        settings = _settings(min_trials=3, max_trials=20, patience=2, repetitions=1)

        def objective(_rng):
            return 1.0, {"constant": True}

        run = run_search(
            settings, seed=7, dimension=1, random_objective=objective,
            optuna_objective=lambda _trial: (1.0, {"constant": True}),
        )
        self.assertTrue(run.converged)
        self.assertEqual(run.termination_reason, "score_plateau")
        self.assertEqual(run.trial_count, 3)
        self.assertEqual(len(run.trace), 3)

    def test_cmaes_rejects_one_dimensional_fallback(self) -> None:
        settings = _settings(method="cmaes", repetitions=1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            run_search(
                settings, seed=7, dimension=1,
                random_objective=lambda _rng: (1.0, {}),
                optuna_objective=lambda _trial: (1.0, {}),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
