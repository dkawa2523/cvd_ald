from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from copy import deepcopy

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config, compose_opt_config

from .class_compare import build_role_summary, rank_role_candidates, build_role_stability, build_class_compare, effect_signature, build_condition_scores
from .metrics import prediction_metrics
from .enumerate_roles import RoleCandidate
from .parameter_fit import fit_candidate_parameters, _persistent_study_name


def _search(trials: int, seed: int) -> dict:
    return {
        "method": "random",
        "seed": seed,
        "min_trials": trials,
        "max_trials": trials,
        "trials_per_dimension": 1,
        "patience": max(trials, 1),
        "relative_improvement": 0.0,
        "repetitions": 1,
        "pruner": "none",
        "sampler_options": {},
        "storage": {"url": "", "study_name": "", "load_if_exists": False},
    }
from .run_fit import run_fit
from .fit_roles import fit_role_candidates


class TestSelectionEvidence(unittest.TestCase):
    def record(self, effects=None):
        metrics = prediction_metrics([1., 2., 3.], [1.05, 2.05, 3.05], baseline=0.)
        return {"class_id": "A", "roles": {"A": "s0", "I": None, "B": None},
                "effect_groups": effects or {"A": ["s0"]}, "best_score": 1.,
                "validation_conditions": [{"condition": "inner", "weight": 1., **metrics}],
                "fit_diagnostics": {"identifiability": {"assessed": True}},
                "holdout_metrics": {"external": metrics}}

    def summarize(self, record, **kwargs):
        return build_role_summary(rank_role_candidates([record]), score_epsilon=0.,
                                  role_stability_warning=False, **kwargs)[0]

    def test_ranking_does_not_mutate_fit_evidence(self):
        record = self.record()
        before = deepcopy(record)
        ranked = rank_role_candidates([record])
        ranked[0]["validation_conditions"][0]["mse"] = 999.
        self.assertEqual(record, before)

    def test_stability_uses_each_refit_and_counts_duplicate_effects_once(self):
        a = self.record({"A": ["s0"], "I": ["s2"]})
        a["selection_refits"] = [{"condition": "left", "selected": True, "effect_groups": {"I": ["s2"]}}]
        alias = deepcopy(a)
        alias["roles"]["A"] = "another_name"
        b = self.record()
        b["selection_refits"] = [{"condition": "left", "selected": True, "effect_groups": {"A": ["s1"]}}]
        first, _ = build_role_stability([a, b], score_epsilon=0.)
        second, _ = build_role_stability([alias, b, a], score_epsilon=0.)
        self.assertEqual(first, second)
        self.assertEqual({r["species"]: r["frequency"] for r in first if r["slot"] == "A"},
                         {"__NONE__": .5, "s1": .5})

    def test_rate_and_thickness_scaling_have_the_same_decision(self):
        results = []
        for scale, unit in ((1., "nm/s"), (1.e-9, "m/s")):
            metrics = prediction_metrics(np.array([1., 2., 3.]) * scale,
                                         np.array([1.05, 2.05, 3.05]) * scale, baseline=0., sigma=.1 * scale)
            record = self.record()
            record["validation_conditions"] = [{"condition": str(i), "weight": 1., "unit": unit, **metrics} for i in range(2)]
            record["holdout_metrics"] = {"external": metrics}
            record["declared_effect_groups"] = {"A": ["s0"]}
            record["reduced_effect_groups"] = [{}]
            reduced, alternative = deepcopy(record), deepcopy(record)
            for other, groups, error in ((reduced, {}, .2), (alternative, {"A": ["s1"]}, .1)):
                other["effect_groups"] = other["declared_effect_groups"] = groups
                other["validation_conditions"] = [{"condition": str(i), "weight": 1., "unit": unit,
                    **prediction_metrics(np.array([1., 2., 3.]) * scale,
                                         np.array([1. + error, 2. + error, 3. + error]) * scale, baseline=0.)}
                    for i in range(2)]
            row = build_role_summary(rank_role_candidates([record, reduced, alternative]), score_epsilon=0.,
                                     role_stability_warning=False,
                                     application={"conditions": ["external"], "max_relative_rmse": .1})[0]
            results.append((row["decision"], row["prediction_status"], row["spatial_status"]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], "adopt_candidate")

    def comparison_records(self, full_predictions, reduced_predictions, alternative_predictions=None):
        rows = []
        for predictions, groups in ((full_predictions, {"A": ["s0"]}), (reduced_predictions, {}),
                                    (alternative_predictions, {"A": ["s1"]})):
            if predictions is None:
                continue
            row = self.record()
            row["effect_groups"] = row["declared_effect_groups"] = groups
            row["reduced_effect_groups"] = [{}] if groups else []
            row["validation_conditions"] = [{"condition": str(i), "weight": 1.,
                                              **prediction_metrics([1., 2., 3.], prediction, baseline=0.)}
                                             for i, prediction in enumerate(predictions)]
            rows.append(row)
        return rank_role_candidates(rows)

    def test_crossing_reduction_losses_do_not_establish_role_necessity(self):
        rows = self.comparison_records([[1., 2., 3.], [2., 3., 4.]],
                                       [[3., 4., 5.], [1., 2., 3.]])
        self.assertEqual(rows[0]["effect_groups"], {"A": ["s0"]})
        self.assertEqual(rows[0]["role_evidence"][0]["necessity"], "mixed")
        summary = build_role_summary(rows, score_epsilon=0., role_stability_warning=False)[0]
        self.assertEqual(summary["role_support"], "unresolved")
        self.assertIn("term removal", summary["reason"])

    def test_mean_and_spatial_contributions_are_not_confounded_in_role_evidence(self):
        rows = self.comparison_records([[.8, 2., 3.2]] * 2, [[2., 3., 4.]] * 2)
        comparison = rows[0]["reduced_model_comparisons"][0]
        self.assertEqual(comparison["mean_status"], "consistent_benefit")
        self.assertEqual(comparison["spatial_status"], "no_benefit")
        for fold in comparison["conditions"]:
            self.assertAlmostEqual(fold["mse_increase"], fold["mean_mse_increase"] + fold["centered_mse_increase"])
        scores = build_condition_scores(rows)
        self.assertTrue(any(row["reduction_comparisons"] for row in scores if row["evaluation_scope"] == "inner_selection"))

    def test_necessary_response_does_not_identify_an_interchangeable_species(self):
        rows = self.comparison_records([[1., 2., 3.], [1.2, 2.2, 3.2]], [[3., 4., 5.]] * 2,
                                       [[1.3, 2.3, 3.3], [1., 2., 3.]])
        evidence = rows[0]["role_evidence"][0]
        self.assertEqual(evidence["necessity"], "consistent_benefit")
        self.assertEqual(evidence["assignment"], "unresolved")
        summary = build_role_summary(rows, score_epsilon=0., role_stability_warning=False)[0]
        self.assertEqual(summary["role_support"], "unresolved")

    def test_missing_comparisons_cannot_support_automatic_role_adoption(self):
        summary = self.summarize(self.record(), application={"conditions": ["external"], "max_relative_rmse": .1})
        self.assertEqual(summary["prediction_status"], "improves_baseline")
        self.assertEqual(summary["role_support"], "unresolved")
        self.assertEqual(summary["decision"], "review")

    def test_application_tolerance_is_required_even_for_a_good_fixed_model(self):
        row = self.summarize(self.record())
        self.assertEqual(row["decision"], "review")
        self.assertEqual(row["application_status"], "not_specified")

    def test_outer_procedure_success_does_not_validate_the_fixed_model(self):
        record = self.record()
        record["evaluation_conditions"] = [{"condition": "outer", **record["holdout_metrics"]["external"]}]
        row = self.summarize(record, application={"conditions": ["outer"], "max_relative_rmse": .1})
        self.assertEqual(row["procedure_assessment"]["application_status"], "meets_tolerance")
        self.assertEqual(row["fixed_model_assessment"]["application_status"], "scope_not_tested")
        self.assertEqual(row["decision"], "review")

    def test_primary_success_does_not_hide_a_failed_outer_condition(self):
        record = self.record()
        record["evaluation_conditions"] = [{"condition": "outer_bad", **prediction_metrics([1., 2., 3.], [10., 10., 10.], baseline=2.)}]
        row = self.summarize(record)
        self.assertEqual(row["fixed_model_assessment"]["prediction_status"], "improves_baseline")
        self.assertEqual(row["decision"], "reject_prediction")
        self.assertIn("outer_bad", row["reason"])
        scores = build_condition_scores([record])
        self.assertEqual({r["evaluation_scope"] for r in scores},
                         {"inner_selection", "fixed_model_holdout", "outer_selection_procedure"})

    def test_native_state_roles_do_not_exchange_a_and_b(self):
        ab = RoleCandidate("left", None, "right", "AB")
        ba = RoleCandidate("right", None, "left", "AB")
        self.assertNotEqual(effect_signature({"effect_groups": ab.effect_groups}),
                            effect_signature({"effect_groups": ba.effect_groups}))


@unittest.skipIf(np is None, "NumPy is required")
class TestFitDiagnostics(unittest.TestCase):
    def test_stored_trials_are_separated_by_roles_and_observation_contents(self):
        with TemporaryDirectory() as tmp:
            fluent, measured = self._write_inputs(tmp)
            spec = compose_sim_config("cvd_steady_min", overrides=[
                f"sim.inputs.fluent.file={fluent}", "sim.measurement.enabled=true",
                f"sim.measurement.file={measured}",
            ])
            first = _persistent_study_name("resume", [spec], {})
            self.assertEqual(first, _persistent_study_name("resume", [spec], {}))
            spec.roles.A = "s1"
            self.assertNotEqual(first, _persistent_study_name("resume", [spec], {}))
            spec.roles.A = "s0"
            with np.load(measured) as data:
                xy = data["xy"]
            np.savez(measured, h_nm=np.ones(xy.shape[0]), xy=xy)
            self.assertNotEqual(first, _persistent_study_name("resume", [spec], {}))

    def test_good_mean_does_not_hide_failed_spatial_prediction(self):
        record = self._prediction_record("s0", .1)
        for fold in record["validation_conditions"]:
            fold["centered_r2"] = -1.
        rows = build_role_summary(rank_role_candidates([record]), score_epsilon=1e-6, role_stability_warning=False)
        self.assertEqual(rows[0]["decision"], "review")
        self.assertIn("spatial", rows[0]["reason"])

    def test_unassessed_parameters_cannot_be_adopted(self):
        records = rank_role_candidates([self._prediction_record("s0", .1)])
        rows = build_role_summary(records, score_epsilon=1e-6, role_stability_warning=False)
        self.assertIn("not been assessed", rows[0]["reason"])
        records[0]["fit_diagnostics"] = {"identifiability": {"assessed": True}}
        rows = build_role_summary(records, score_epsilon=1e-6, role_stability_warning=False)
        self.assertEqual(rows[0]["decision"], "review")
        self.assertIn("independent", rows[0]["reason"])

    def test_class_comparison_uses_selection_score_for_ties(self):
        records = rank_role_candidates([self._prediction_record("s0", .1, 10.), self._prediction_record("s1", .1, 20.)])
        rows = build_class_compare(records)
        self.assertEqual(rows[0]["tie_group_size"], 2)

    def test_real_condition_refits_exclude_external_measurements(self):
        with TemporaryDirectory() as tmp:
            fluent_a, fluent_b, meas_a, meas_b = self._write_multi_inputs(tmp)
            external = Path(tmp) / "external.npz"
            with np.load(meas_a) as data:
                xy = data["xy"]
            np.savez(external, h_nm=np.ones(4), xy=xy)
            spec = compose_opt_config("fit_cvd_steady_min", overrides=[
                f"sim.inputs.fluent.file={fluent_a}", "sim.time.t_proc_s=0.02", "sim.time.dt_s=0.01",
            ])
            spec.opt.measurement = {"align": {"enable": False}, "conditions": [
                {"name": "first", "fluent_file": str(fluent_a), "file": str(meas_a)},
                {"name": "second", "fluent_file": str(fluent_b), "file": str(meas_b)},
                {"name": "third", "fluent_file": str(fluent_a), "file": str(meas_a)},
                {"name": "external", "split": "holdout", "fluent_file": str(fluent_a), "file": str(external)},
            ]}
            spec.opt.role_enumeration.roles = {"A": {"candidates": ["s0", "s1"]}, "I": {"candidates": []}, "B": {"candidates": []}}
            spec.opt.class_compare.classes = ["A"]
            spec.opt.parameter_fit.search.update({
                "method": "random", "min_trials": 1, "max_trials": 1,
                "trials_per_dimension": 1, "patience": 1, "repetitions": 1,
            })
            spec.opt.parameter_fit.search_space = [{"name": "model.params.kinetics.k_rxn", "type": "uniform", "low": .01, "high": .01}]
            before = fit_role_candidates(spec.sim, spec.opt)
            self.assertTrue(all(len(r["selection_refits"]) == 3 for r in before))
            spec.opt.parameter_fit.analysis["role_stability"]["enabled"] = False
            np.savez(external, h_nm=np.full(4, 1000.), xy=xy)
            after = fit_role_candidates(spec.sim, spec.opt)
            self.assertEqual([r["roles"] for r in before], [r["roles"] for r in after])
            np.testing.assert_allclose([r["selection_score"] for r in before], [r["selection_score"] for r in after])
            for record in after:
                self.assertEqual({f["condition"] for f in record["validation_conditions"]}, {"first", "second", "third"})
                self.assertEqual(record["selection_basis"], "condition_cv")

    def _prediction_record(self, species, mse, train_score=1.0):
        return {"roles": {"A": species, "I": None, "B": None}, "class_id": "A",
                "best_score": train_score, "loss_data": train_score,
                "validation_conditions": [
                    {"condition": str(i), "weight": 1., "mse_nm2": mse,
                     "baseline_mse_nm2": 1., "refit_score": train_score}
                    for i in range(3)]}

    def test_condition_cv_ranks_predictions_and_external_holdout_cannot_select(self):
        a = self._prediction_record("s0", .1, 10.)
        b = self._prediction_record("s1", .4, .001)
        a["holdout_metrics"] = {"external": {"mse_nm2": 100., "baseline_mse_nm2": 1.}}
        ranked = rank_role_candidates([b, a])
        self.assertEqual(ranked[0]["roles"]["A"], "s0")
        rows = build_role_summary(ranked, score_epsilon=1e-6, role_stability_warning=False)
        self.assertEqual(rows[0]["decision"], "reject_prediction")

    def test_equivalent_roles_remain_unresolved_after_renaming(self):
        for species in (("s0", "s1"), ("unfamiliar_b", "unfamiliar_a")):
            records = rank_role_candidates([self._prediction_record(name, .1) for name in species])
            stability, diagnosis = build_role_stability(records, topk_window=1, score_epsilon=1e-6)
            self.assertTrue(diagnosis["warning"])
            self.assertTrue(all(r["frequency"] == .5 for r in stability if r["slot"] == "A"))
            rows = build_role_summary(records, score_epsilon=1e-6, role_stability_warning=False)
            self.assertEqual(rows[0]["decision"], "review")

    def test_single_failed_condition_is_not_equivalent_to_good_predictions(self):
        best = self._prediction_record("s0", 1.)
        best["roles"]["I"] = "s2"
        worse = self._prediction_record("s1", 1.)
        for row in (best, worse):
            row["validation_conditions"].append(dict(row["validation_conditions"][-1], condition="3"))
        worse["validation_conditions"][0]["mse_nm2"] += 1.0e6
        ranked = rank_role_candidates([worse, best])
        self.assertEqual(ranked[0]["roles"], best["roles"])
        self.assertFalse(ranked[1]["equivalent_to_best"])
        self.assertNotIn("equivalent_to_best", worse)

    def test_stability_counts_repeated_selection_not_training_fit(self):
        a, b = self._prediction_record("s0", .1, 100.), self._prediction_record("s1", .2, .001)
        a["selection_refits"] = [{"condition": "left", "selected": True}]
        b["selection_refits"] = [{"condition": "left", "selected": False}]
        rows, diagnosis = build_role_stability([a, b], score_epsilon=1e-6)
        self.assertEqual(next(r["species"] for r in rows if r["slot"] == "A"), "s0")
        self.assertEqual(diagnosis["basis"], "repeated_condition_cv_selection")
        self.assertFalse(diagnosis["warning"])

    def test_all_inadequate_candidates_can_be_rejected(self):
        records = rank_role_candidates([self._prediction_record("s0", 2.), self._prediction_record("s1", 4.)])
        rows = build_role_summary(records, score_epsilon=1e-6, role_stability_warning=False)
        self.assertEqual(rows[0]["decision"], "reject_prediction")

    def test_parameter_degeneracy_requires_review(self) -> None:
        rows = build_role_summary(
            [
                {
                    "class_id": "A",
                    "roles": {"A": "s0", "I": None, "B": None},
                    "best_score": 1.0,
                    "loss_data": 1.0,
                }
            ],
            score_epsilon=1.0e-8,
            role_stability_warning=False,
            parameter_identifiability_warning=True,
        )
        self.assertEqual(rows[0]["decision"], "review")
        self.assertIn("identifiable", rows[0]["reason"])

    def _write_inputs(self, tmp: str) -> tuple[Path, Path]:
        fluent_path = Path(tmp) / "fluent.npz"
        meas_path = Path(tmp) / "meas.npz"
        xy = np.array([[-10.0, -10.0], [0.0, 0.0], [20.0, 10.0], [30.0, -15.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.5, 0.1, 0.0],
                [0.8, 0.4, 0.1, 0.0],
                [0.6, 0.3, 0.1, 0.0],
                [0.5, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(fluent_path, xy=xy, cref=cref)
        np.savez(meas_path, h_nm=0.2 * xy[:, 0] - 0.1 * xy[:, 1], xy=xy)
        return fluent_path, meas_path

    def _write_multi_inputs(self, tmp: str) -> tuple[Path, Path, Path, Path]:
        fluent_a = Path(tmp) / "fluent_a.npz"
        fluent_b = Path(tmp) / "fluent_b.npz"
        meas_a = Path(tmp) / "meas_a.npz"
        meas_b = Path(tmp) / "meas_b.npz"

        xy = np.array([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0], [35.0, 5.0]], dtype=float)
        cref_a = np.array(
            [
                [1.0, 0.5, 0.2, 0.0],
                [0.9, 0.4, 0.2, 0.0],
                [0.8, 0.3, 0.2, 0.0],
                [0.7, 0.2, 0.2, 0.0],
            ],
            dtype=float,
        )
        cref_b = np.array(
            [
                [0.8, 0.2, 0.5, 0.0],
                [0.7, 0.2, 0.4, 0.0],
                [0.6, 0.2, 0.3, 0.0],
                [0.5, 0.1, 0.2, 0.0],
            ],
            dtype=float,
        )
        np.savez(fluent_a, xy=xy, cref=cref_a)
        np.savez(fluent_b, xy=xy, cref=cref_b)
        np.savez(meas_a, h_nm=0.4 + 0.1 * xy[:, 0], xy=xy)
        np.savez(meas_b, h_nm=0.1 + 0.05 * xy[:, 0], xy=xy)
        return fluent_a, fluent_b, meas_a, meas_b

    def test_run_fit_emits_role_stability_and_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
            out = run_fit(
                config_name="fit_cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.output.project=fit_diag_unit",
                    f"sim.output.root_dir={tmp}",
                    "opt.output.project=fit_diag_unit",
                    f"opt.output.root_dir={tmp}",
                    "opt.output.run_name=fit_diag_unit",
                    f"opt.measurement.file={meas_path}",
                    "opt.parameter_fit.search.method=random",
                    "opt.parameter_fit.search.min_trials=1",
                    "opt.parameter_fit.search.max_trials=1",
                    "opt.parameter_fit.search.trials_per_dimension=1",
                    "opt.parameter_fit.search.patience=1",
                    "opt.parameter_fit.search.repetitions=1",
                    "opt.role_enumeration.roles.A.candidates=[s0]",
                    "opt.role_enumeration.roles.I.candidates=[s1,s2]",
                    "opt.role_enumeration.roles.B.candidates=[s3]",
                    "opt.parameter_fit.analysis.role_stability.score_epsilon=1000000000.0",
                ],
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "tables" / "role_stability.csv").exists())
            diag = json.loads((run_dir / "outputs" / "fit_diagnostics.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(diag.get("role_stability_warning")))
            self.assertIn("cache_stats", diag)

    def test_fidelity_cache_hits_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_a, fluent_b, meas_a, meas_b = self._write_multi_inputs(tmp)
            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_a}"])
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                measurement={
                    "keys": {"h": "h_nm", "xy": "xy"},
                    "conditions": [
                        {"name": "c1", "weight": 1.0, "fluent_file": str(fluent_a), "measurement_file": str(meas_a)},
                        {"name": "c2", "weight": 1.0, "fluent_file": str(fluent_b), "measurement_file": str(meas_b)},
                    ],
                },
                parameter_fit=SimpleNamespace(
                    search=_search(1, 5),
                    fidelity={"levels": [1, 2]},
                    analysis={
                        "cache": {"enabled": True, "max_entries": 64},
                        "preflight": {"enabled": True, "min_finite_ratio": 0.5},
                        "identifiability": {"enabled": False},
                    },
                    objective={"loss": {"name": "huber", "standardized": False, "delta_nm": 10.0}, "penalties": {}},
                    search_space=[
                        {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1.0e-6, "high": 1.0e-2}
                    ],
                ),
                class_compare=SimpleNamespace(),
            )

            out = fit_candidate_parameters(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            stats = dict(out.get("cache_stats", {}))
            self.assertGreaterEqual(int(stats.get("trial_hits", 0)), 1)

    def test_preflight_fails_on_low_finite_ratio(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, _ = self._write_inputs(tmp)
            meas_path = Path(tmp) / "meas_bad.npz"
            xy = np.array([[-10.0, -10.0], [0.0, 0.0], [20.0, 10.0], [30.0, -15.0]], dtype=float)
            h = np.array([np.nan, np.nan, 1.0, np.nan], dtype=float)
            np.savez(meas_path, h_nm=h, xy=xy)

            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_path}"])
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                measurement={"file": str(meas_path), "keys": {"h": "h_nm", "xy": "xy"}},
                parameter_fit=SimpleNamespace(
                    search=_search(1, 3),
                    fidelity={"levels": [1]},
                    analysis={"preflight": {"enabled": True, "min_finite_ratio": 0.9}},
                    objective={"loss": {"name": "huber", "standardized": False, "delta_nm": 10.0}, "penalties": {}},
                    search_space=[
                        {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1.0e-6, "high": 1.0e-2}
                    ],
                ),
                class_compare=SimpleNamespace(),
            )

            with self.assertRaises(ValueError):
                fit_candidate_parameters(
                    sim_spec=sim_spec,
                    role_candidate=role,
                    order_candidate=order,
                    opt_spec=opt_spec,
                )

    def test_holdout_condition_is_scored_without_training_leakage(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_a, fluent_b, meas_a, meas_b = self._write_multi_inputs(tmp)
            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_a}"])
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                measurement={
                    "keys": {"h": "h_nm", "xy": "xy"},
                    "conditions": [
                        {"name": "train", "split": "train", "fluent_file": str(fluent_a), "measurement_file": str(meas_a)},
                        {"name": "holdout", "split": "holdout", "fluent_file": str(fluent_b), "measurement_file": str(meas_b)},
                    ],
                },
                parameter_fit=SimpleNamespace(
                    search=_search(1, 11),
                    fidelity={"levels": [1]},
                    analysis={"identifiability": {"enabled": False}},
                    objective={"loss": {"name": "huber", "standardized": False, "delta_nm": 10.0}, "penalties": {}},
                    search_space=[
                        {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1.0e-6, "high": 1.0e-2}
                    ],
                ),
                class_compare=SimpleNamespace(),
            )
            out = fit_candidate_parameters(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            self.assertEqual(out["train_condition_count"], 1)
            self.assertEqual(out["holdout_condition_count"], 1)
            self.assertIn("train", out["condition_scores"])
            self.assertNotIn("holdout", out["condition_scores"])
            self.assertIn("holdout", out["holdout_scores"])
            self.assertIn("rmse_nm", out["holdout_metrics"]["holdout"])


if __name__ == "__main__":
    unittest.main()
