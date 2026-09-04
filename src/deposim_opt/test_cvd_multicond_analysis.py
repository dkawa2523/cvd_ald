from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dataclasses import replace
from .cvd_multicond_analysis import (
    analyze_cvd_multicond_case, _load_case, _split_evaluation,
    _combine_cases, _fit_transfer,
    _condition_holdout_predictions, _ranking_for_training, CombinedCases,
    _predict_transfer, _transfer_design, _coefficient_rows,
)
from .cvd_spatial_analysis import _fit_nonnegative_effects, RoleResponseCandidate, enumerate_role_response_candidates


class TestCvdMulticondAnalysis(unittest.TestCase):
    def _grouped_response(self, ids=(1, 2, 3, 4, 5)):
        means = np.repeat(np.linspace(-.8, .8, len(ids)), 9)
        within = np.tile(np.linspace(-.15, .15, 9), len(ids))
        total = np.exp(means + within)
        return CombinedCases(case_ids=ids, xyz=np.zeros((total.size, 3)),
                             condition_id=np.repeat(ids, 9), species=("raw:species", "other"),
                             concentrations={}, species_fractions={"raw:species": np.full(total.size, .2),
                                                                     "other": np.full(total.size, .8)},
                             total_concentration=total, rate=np.exp(-2. + 1.6 * means + .15 * within))

    def test_separate_response_recovers_between_and_within_slopes(self):
        data = self._grouped_response()
        candidate = RoleResponseCandidate("baseline", "baseline")
        train = np.flatnonzero(data.condition_id != 5)
        separate = _fit_transfer(candidate, data, train, np.arange(data.rate.size), response_structure="within_between")
        shared = _fit_transfer(candidate, data, train, np.arange(data.rate.size))
        np.testing.assert_allclose([separate.common_order, separate.within_order], [1.6, .15], atol=1.e-12)
        np.testing.assert_allclose(separate.prediction, data.rate, rtol=1.e-12)
        self.assertGreater(np.mean((shared.prediction - data.rate)**2), 1.e-5)
        coupled = _fit_transfer(candidate, data, train, response_structure="within_between", regularization=.01)
        self.assertLess(abs(coupled.common_order - coupled.within_order), 1.6 - .15)

    def test_map_centering_is_input_only_and_independent_of_prediction_batch(self):
        data = self._grouped_response()
        candidate = RoleResponseCandidate("baseline", "baseline")
        train = np.flatnonzero(data.condition_id != 5)
        fit = _fit_transfer(candidate, data, train, response_structure="within_between")
        changed = replace(data, rate=np.where(data.condition_id == 5, data.rate * 1000., data.rate))
        refit = _fit_transfer(candidate, changed, train, response_structure="within_between")
        np.testing.assert_array_equal(fit.coefficients, refit.coefficients)
        full, design = _predict_transfer(fit, changed)
        batch = np.array([36, 39, 40])
        partial = _transfer_design(candidate, data, batch, response_structure="within_between",
                                   reference_total_concentration=fit.reference_total_concentration,
                                   reference_species_fractions=fit.reference_species_fractions)
        np.testing.assert_array_equal(partial, design[batch])
        np.testing.assert_allclose(full, data.rate, rtol=1.e-12)
        replicated = np.concatenate([train, np.flatnonzero(data.condition_id == 1)])
        duplicate_fit = _fit_transfer(candidate, data, replicated, np.arange(data.rate.size),
                                      response_structure="within_between", regularization=.01)
        original = _fit_transfer(candidate, data, train, np.arange(data.rate.size),
                                 response_structure="within_between", regularization=.01)
        np.testing.assert_allclose(original.prediction, duplicate_fit.prediction, rtol=1.e-11)

    def test_coefficient_outputs_preserve_scope_and_model_defined_roles(self):
        data = self._grouped_response()
        candidate = RoleResponseCandidate("display only", "A", A="raw:species")
        fraction = .2 * np.exp(np.tile(np.linspace(-.1, .1, 9), 5))
        data = replace(data, species_fractions={"raw:species": fraction, "other": 1. - fraction})
        fit = _fit_transfer(candidate, data, np.arange(data.rate.size), response_structure="within_between", regularization=.01)
        rows = _coefficient_rows(fit, np.tile(fit.coefficients, (3, 1)))
        self.assertEqual(len({row["term"] for row in rows}), len(rows))
        role_rows = [row for row in rows if row["role"] == "A"]
        self.assertEqual({row["response_scope"] for row in role_rows}, {"between", "within"})

    def test_training_cv_can_select_separate_response_for_known_grouped_generator(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = []
            for i, scale in enumerate((.6, .8, 1., 1.4, 1.8), start=1):
                self._write_case(root, i, scale, 1.)
                case = _load_case(i, root / f"condition_{i}.csv", root / f"validation_{i}.csv")
                x = np.log(case.total_concentration / 2.03e-4)
                cases.append(replace(case, rate=np.exp(-2. + 1.6 * x.mean() + .15 * (x - x.mean()))))
            result, _, fit, prediction, _, test = _split_evaluation(cases[:4], cases[4], response_structure="select")
            self.assertEqual(fit.response_structure, "within_between")
            self.assertEqual(result["selected_role_model_id"], "baseline")
            np.testing.assert_allclose(prediction, test.rate, rtol=1.e-10)

    def test_one_numerically_failed_setting_does_not_abort_candidate_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, scale in enumerate((.8, 1., 1.2), start=1):
                self._write_case(root, i, scale, scale**1.5)
            cases = [_load_case(i, root / f"condition_{i}.csv", root / f"validation_{i}.csv") for i in (1, 2, 3)]
            def evaluate(candidate, data, **kwargs):
                if kwargs["response_structure"] == "within_between" and kwargs["regularization"] == 0:
                    raise FloatingPointError("unresolved held-out prediction")
                return _condition_holdout_predictions(candidate, data, **kwargs)
            with patch("deposim_opt.cvd_multicond_analysis._condition_holdout_predictions", side_effect=evaluate):
                rows, _, _, _, _ = _ranking_for_training(cases, response_structure="select")
            selected = next(row for row in rows if row["selected"])
            self.assertTrue(np.isfinite(selected["selection_score"]))
            self.assertFalse(selected["response_structure"] == "within_between" and selected["regularization"] == 0)
            self.assertTrue(any(score["numerical_failure"] for score in selected["regularization_scores"]))

    def test_full_fit_zero_does_not_remove_an_effect_from_training_folds(self):
        x = np.repeat([-1., 0., 1.], 2)
        fraction = .1 * np.exp(x)
        data = CombinedCases(case_ids=(1, 2, 3), xyz=np.zeros((6, 3)), condition_id=np.repeat([1, 2, 3], 2),
                             species=("unusual:A|name", "other"),
                             concentrations={}, species_fractions={"unusual:A|name": fraction, "other": 1 - fraction},
                             total_concentration=np.ones(6), rate=np.exp(np.repeat([2., 0., 1.], 2)))
        candidate = RoleResponseCandidate("arbitrary display name", "A", A="unusual:A|name")
        full = _fit_transfer(candidate, data, np.arange(6))
        self.assertEqual(full.effect_groups, {})
        _, folds = _condition_holdout_predictions(candidate, data)
        self.assertEqual(folds[0]["effect_groups"], {"A": ["unusual:A|name"]})
        self.assertEqual(folds[0]["effective_roles"]["A"], "unusual:A|name")
        self.assertNotIn("mse_nm2", folds[0])

    def test_empirical_product_symmetry_and_reduction_closure(self):
        ab = RoleResponseCandidate("forward", "AB", A="left", B="right")
        ba = RoleResponseCandidate("reverse", "AB", A="right", B="left")
        self.assertEqual(ab.effect_groups, ba.effect_groups)
        candidates = enumerate_role_response_candidates(["left", "right", "inhibitor"], include_reductions=True)
        keys = {str(c.effect_groups) for c in candidates}
        self.assertEqual(len(keys), len(candidates))
        for candidate in candidates:
            self.assertTrue(all(str(reduced.effect_groups) in keys for reduced in candidate.reductions()))
        self.assertTrue(any(c.A is None and c.I == "inhibitor" for c in candidates))

    def test_inhibitory_response_is_refitted_without_inventing_a_driver(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id, scale in enumerate((.8, 1., 1.2), start=1):
                self._write_case(root, case_id, scale, 1.)
            cases = [_load_case(i, root / f"condition_{i}.csv", root / f"validation_{i}.csv") for i in range(1, 4)]
            cases = [replace(case, rate=.1 * (case.total_concentration / 2.03e-4)**1.5 *
                             (case.concentrations["s1"] / case.total_concentration / .005)**-2.) for case in cases]
            rows, fits, _, _, _ = _ranking_for_training(cases)
            selected = next(row for row in rows if row["selected"])
            self.assertEqual(selected["role_model_id"], "I_response:s1")
            self.assertIsNone(selected["effective_roles"]["A"])
            self.assertEqual(selected["effect_groups"], {"I": ["s1"]})
            self.assertTrue(selected["reduced_model_comparisons"])
            self.assertIn("I_response:s1", fits)

    def test_regularization_stabilizes_a_weak_direction_without_changing_intercept_units(self):
        x = np.column_stack([np.ones(9), np.linspace(-1.e-7, 1.e-7, 9)])
        y = 2. + np.linspace(-.01, .01, 9)
        raw, _ = _fit_nonnegative_effects(x, y)
        stable, _ = _fit_nonnegative_effects(x, y, regularization=1.e-4)
        shifted, _ = _fit_nonnegative_effects(x, y + 5., regularization=1.e-4)
        self.assertGreater(raw[1], 1.e4)
        self.assertLess(stable[1], .01)
        np.testing.assert_allclose(shifted, stable + [5., 0.], atol=1.e-12)

    def test_joint_selection_excludes_outer_observations_and_balances_conditions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id, scale in enumerate((.8, 1., 1.2, 1.4), start=1):
                self._write_case(root, case_id, scale, scale**1.5, role_rate=True)
            cases = [_load_case(i, root / f"condition_{i}.csv", root / f"validation_{i}.csv") for i in range(1, 5)]
            first = _split_evaluation(cases[:3], cases[3], response_structure="select")
            second = _split_evaluation(cases[:3], replace(cases[3], rate=cases[3].rate * 100.), response_structure="select")
            self.assertEqual(first[0]["selected_role_model_id"], second[0]["selected_role_model_id"])
            self.assertEqual(first[2].regularization, second[2].regularization)
            self.assertEqual(first[2].response_structure, second[2].response_structure)
            np.testing.assert_array_equal(first[2].coefficients, second[2].coefficients)
            train = _combine_cases(cases[:3])
            original = np.arange(train.rate.size)
            duplicated = np.concatenate([original, np.flatnonzero(train.condition_id == 1)])
            fit_a = _fit_transfer(first[2].candidate, train, original, regularization=1.e-4)
            fit_b = _fit_transfer(first[2].candidate, train, duplicated, original, regularization=1.e-4)
            np.testing.assert_allclose(fit_a.prediction, fit_b.prediction, rtol=1.e-10)

    def _write_case(
        self,
        root: Path,
        case_id: int,
        scale: float,
        rate_scale: float,
        *,
        s0_multiplier: float = 1.0,
        role_rate: bool = False,
    ) -> None:
        condition_path = root / f"condition_{case_id}.csv"
        validation_path = root / f"validation_{case_id}.csv"
        condition_rows: list[list[float]] = []
        validation_rows: list[list[float]] = []
        for radius in (0.03, 0.07, 0.11):
            for sector in range(8):
                angle = 2.0 * np.pi * sector / 8.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                c0 = scale * s0_multiplier * 2.0e-6 * (1.0 + 0.06 * np.cos(angle))
                c1 = scale * 1.0e-6 * (1.0 + 0.04 * np.sin(angle))
                c2 = scale * 2.0e-4 * (1.0 + 0.002 * np.cos(2.0 * angle))
                total = c0 + c1 + c2
                if role_rate:
                    reference_total = 2.03e-4
                    reference_fraction_s0 = 2.0e-6 / reference_total
                    rate = (
                        0.1
                        * (total / reference_total) ** 1.5
                        * ((c0 / total) / reference_fraction_s0) ** 0.9
                    )
                else:
                    rate = rate_scale * 0.1 * (1.0 + 0.004 * np.cos(angle))
                condition_rows.append(
                    [x, y, 0.0, c0, c1, c2, c0 / total, c1 / total, c2 / total, 30.0 * total]
                )
                validation_rows.append([x, y, 0.0, rate])
        with condition_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "x", "y", "z", "concentration_s0", "concentration_s1", "concentration_s2",
                    "molef_s0", "molef_s1", "molef_s2", "density",
                ]
            )
            writer.writerows(condition_rows)
        with validation_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "dr_nm_per_sec"])
            writer.writerows(validation_rows)

    def test_two_train_one_test_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_case(root, 1, 1.0, 1.0)
            self._write_case(root, 2, 0.8, 0.8**1.5)
            self._write_case(root, 3, 1.2, 1.2**1.5)
            output = root / "results"
            summary = analyze_cvd_multicond_case(
                data_dir=root,
                train_case_ids=(1, 2),
                test_case_id=3,
                output_dir=output,
                bootstrap_samples=100,
                seed=7,
                response_model="empirical_power",
            )
            self.assertFalse(summary["validity"]["test_was_refit"])
            self.assertEqual(summary["primary_split"]["test_case"], 3)
            self.assertLess(summary["primary_split"]["test_relative_rmse_vs_test_mean"], 0.02)
            for name in (
                "analysis_summary.json",
                "condition_quality.csv",
                "model_ranking.csv",
                "role_ranking.csv",
                "role_summary.csv",
                "role_stability.csv",
                "coefficients.csv",
                "test_predictions.csv",
                "candidate_test_diagnostics.csv",
                "split_sensitivity.csv",
                "report.md",
                "report_snapshot.json",
                "cvd_multicond_transfer_analysis.ipynb",
                "manifest.json",
            ):
                self.assertTrue((output / name).exists(), name)
            with (output / "test_predictions.csv").open(encoding="utf-8") as handle:
                prediction = next(csv.DictReader(handle))
            for species in ("s0", "s1", "s2"):
                self.assertEqual(
                    float(prediction[f"assumed_wall_concentration_{species}_kmol_m3"]),
                    0.0,
                )
                self.assertAlmostEqual(
                    float(prediction[f"wall_zero_driving_concentration_{species}_kmol_m3"]),
                    float(prediction[f"concentration_{species}_kmol_m3"]),
                )
            notebook = json.loads(
                (output / "cvd_multicond_transfer_analysis.ipynb").read_text(encoding="utf-8")
            )
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(
                any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
            )

    def test_added_composition_condition_selects_interpretable_raw_species_role(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_case(root, 1, 1.0, 1.0, role_rate=True)
            self._write_case(root, 2, 0.8, 1.0, role_rate=True)
            self._write_case(root, 3, 1.2, 1.0, role_rate=True)
            self._write_case(
                root,
                4,
                1.0,
                1.0,
                s0_multiplier=0.65,
                role_rate=True,
            )
            output = root / "results"
            summary = analyze_cvd_multicond_case(
                data_dir=root,
                train_case_ids=(1, 2, 4),
                test_case_id=3,
                output_dir=output,
                bootstrap_samples=100,
                seed=7,
                response_model="empirical_power",
            )
            self.assertEqual(summary["primary_split"]["selected_role_model_id"], "A:s0")
            self.assertEqual(summary["primary_split"]["response_structure"], "shared")
            self.assertLess(summary["primary_split"]["test_relative_rmse_vs_test_mean"], 0.01)
            with (output / "model_ranking.csv").open(encoding="utf-8") as handle:
                ranking = list(csv.DictReader(handle))
            selected = next(row for row in ranking if row["role_model_id"] == "A:s0")
            self.assertEqual(selected["eligible_for_adoption"], "True")
            rejected_interaction = next(
                row for row in ranking if row["role_model_id"] == "AB:s0|s1"
            )
            # A partner's arbitrary fraction-range cutoff is no longer a gate.
            # The known A-only generator must favor A through held-out predictions.
            self.assertGreater(float(rejected_interaction["condition_cv_rmse_nm_s"]),
                               float(selected["condition_cv_rmse_nm_s"]))
            self.assertIn("condition_cv_rmse_nm_s", selected)


if __name__ == "__main__":
    unittest.main()
