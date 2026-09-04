from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from deposim_sim.output_manifest import SCHEMA_VERSION
from .enumerate_roles import RoleCandidate
from .fit_optuna import fit_candidate_with_optuna
from .run_fit import _json_default, run_fit


@unittest.skipIf(np is None, "NumPy is required")
class TestFitOptuna(unittest.TestCase):
    def test_fit_diagnostics_numpy_values_are_json_serializable(self) -> None:
        payload = {
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "rank": np.int64(2),
        }
        encoded = json.dumps(payload, default=_json_default)
        self.assertEqual(json.loads(encoded), {"matrix": [[1.0, 2.0], [3.0, 4.0]], "rank": 2})

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

    def test_fit_candidate_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
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
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=2,
                    seed=123,
                    objective={
                        "loss": "huber",
                        "huber_delta_nm": 10.0,
                        "penalties": {
                            "lambda_solver": 0.1,
                            "lambda_phys": 0.0,
                            "lambda_prior": 0.0,
                            "lambda_complex": 0.0,
                            "lambda_role": 0.0,
                        },
                    },
                    search_space=[
                        {
                            "name": "model.params.kinetics.k_rxn",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                        }
                    ],
                ),
                class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.1}),
            )

            out = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            self.assertIn("best_score", out)
            self.assertIn("best_params", out)
            self.assertIn("best_components", out)
            self.assertEqual(out["class_id"], "A")
            for key in ["loss_data", "penalty_solver", "penalty_phys", "penalty_prior", "score_total"]:
                self.assertIn(key, out["best_components"])

    def test_fit_candidate_uses_configured_csv_loaders(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent_payload.data"
            meas_path = root / "meas_payload.data"
            fluent_path.write_text(
                "x,y,s0,s1,s2,s3\n"
                "-10.0,-10.0,1.0,0.5,0.1,0.0\n"
                "0.0,0.0,0.8,0.4,0.1,0.0\n"
                "20.0,10.0,0.6,0.3,0.1,0.0\n"
                "30.0,-15.0,0.5,0.2,0.1,0.0\n",
                encoding="utf-8",
            )
            meas_path.write_text(
                "x,y,h_nm\n"
                "-10.0,-10.0,-1.0\n"
                "0.0,0.0,0.0\n"
                "20.0,10.0,3.0\n"
                "30.0,-15.0,7.5\n",
                encoding="utf-8",
            )
            sim_spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.inputs.fluent.io_loader_name=csv",
                    "sim.measurement.io_loader_name=csv",
                ],
            )
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                measurement={
                    "file": str(meas_path),
                    "keys": {"x": "x", "y": "y", "h": "h_nm"},
                    "align": {"enable": False},
                },
                parameter_fit=SimpleNamespace(
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=1,
                    seed=123,
                    objective={"loss": "huber", "huber_delta_nm": 10.0, "penalties": {}},
                    search_space=[
                        {
                            "name": "model.params.kinetics.k_rxn",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                        }
                    ],
                ),
                class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.0}),
            )

            out = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            self.assertIn("best_score", out)
            self.assertGreaterEqual(float(out["best_components"]["loss_data"]), 0.0)

    def test_fit_candidate_multi_condition_hierarchical(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_a, fluent_b, meas_a, meas_b = self._write_multi_inputs(tmp)
            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_a}"])
            role = RoleCandidate(A="s0", I="s1", B="s2", class_id="AIB")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 1,
            }
            opt_spec = SimpleNamespace(
                measurement={
                    "keys": {"h": "h_nm", "xy": "xy"},
                    "conditions": [
                        {
                            "name": "c1",
                            "weight": 1.0,
                            "fluent_file": str(fluent_a),
                            "measurement_file": str(meas_a),
                            "overrides": [],
                        },
                        {
                            "name": "c2",
                            "weight": 2.0,
                            "fluent_file": str(fluent_b),
                            "measurement_file": str(meas_b),
                            "overrides": [],
                        },
                    ],
                },
                parameter_fit=SimpleNamespace(
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1, 2]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=2,
                    seed=7,
                    objective={
                        "loss": "huber",
                        "huber_delta_nm": 10.0,
                        "penalties": {
                            "lambda_solver": 0.0,
                            "lambda_phys": 0.1,
                            "lambda_prior": 0.5,
                            "lambda_complex": 0.0,
                            "lambda_role": 0.0,
                        },
                    },
                    search_space=[
                        {
                            "name": "model.params.kinetics.k_rxn",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                        },
                        {
                            "name": "model.params.transport.km_A",
                            "type": "loguniform",
                            "low": 1.0e-4,
                            "high": 1.0e-1,
                            "per_condition": True,
                            "hierarchical": {
                                "mode": "log_offset",
                                "sigma": 0.2,
                                "delta_low": -0.2,
                                "delta_high": 0.2,
                            },
                        },
                    ],
                ),
                class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.1}),
            )

            out = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            self.assertEqual(out["condition_count"], 2)
            self.assertEqual(out["fidelity_levels"], [1, 2])
            self.assertIn("model.params.transport.km_A__base", out["best_params"])
            self.assertIn("condition_scores", out)
            self.assertGreaterEqual(out["best_components"]["penalty_prior"], 0.0)

    def test_flux_km_requires_gamma_not_km_a(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
            sim_spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.model.params.transport.km_source=from_cfd_flux_sink",
                ],
            )
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                measurement={"file": str(meas_path), "keys": {"h": "h_nm", "xy": "xy"}},
                parameter_fit=SimpleNamespace(
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=1,
                    seed=123,
                    objective={"loss": "huber", "huber_delta_nm": 10.0, "penalties": {}},
                    search_space=[
                        {
                            "name": "model.params.transport.km_A",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                        }
                    ],
                ),
                class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.0}),
            )

            with self.assertRaises(ValueError):
                fit_candidate_with_optuna(
                    sim_spec=sim_spec,
                    role_candidate=role,
                    order_candidate=order,
                    opt_spec=opt_spec,
                )

    def test_search_space_metadata_filters_disabled_and_non_calibration_items(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_path}"])
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            opt_spec = SimpleNamespace(
                task="fit_roles_and_params",
                measurement={"file": str(meas_path), "keys": {"h": "h_nm", "xy": "xy"}},
                parameter_fit=SimpleNamespace(
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=1,
                    seed=11,
                    objective={"loss": "huber", "huber_delta_nm": 10.0, "penalties": {}},
                    search_space=[
                        {
                            "name": "model.params.kinetics.k_rxn",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                            "group": "surface_kinetics",
                            "stage": "calibration",
                            "symbol": "k_rxn",
                            "unit": "s^-1",
                            "enabled": True,
                        },
                        {
                            "name": "model.params.transport.km_A",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                            "group": "effective_transport",
                            "stage": "screening",
                            "symbol": "k_mA",
                            "unit": "m s^-1",
                            "enabled": True,
                        },
                        {
                            "name": "model.params.inhibitor.K_I",
                            "type": "loguniform",
                            "low": 1.0e-6,
                            "high": 1.0e-2,
                            "group": "surface_kinetics",
                            "stage": "calibration",
                            "symbol": "K_I",
                            "unit": "a.u.",
                            "enabled": False,
                        },
                    ],
                ),
                class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.0}),
            )

            out = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=opt_spec,
            )
            self.assertIn("model.params.kinetics.k_rxn", out["best_params"])
            self.assertNotIn("model.params.transport.km_A", out["best_params"])
            self.assertNotIn("model.params.inhibitor.K_I", out["best_params"])

    @unittest.skipIf(optuna is None, "optuna is required for resume test")
    def test_fit_candidate_optuna_resume_storage(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
            sim_spec = compose_sim_config("cvd_steady_min", overrides=[f"sim.inputs.fluent.file={fluent_path}"])
            role = RoleCandidate(A="s0", I=None, B=None, class_id="A")
            order = {
                "adsorption_site_order": 1,
                "reaction_site_order_A": 1,
                "reaction_site_order_star": 0,
            }
            db_path = Path(tmp) / "study.sqlite3"
            study_name = "resume_unit"

            def build_spec(n_trials: int) -> SimpleNamespace:
                return SimpleNamespace(
                    measurement={"file": str(meas_path), "keys": {"h": "h_nm", "xy": "xy"}},
                    parameter_fit=SimpleNamespace(
                        engine="optuna",
                        sampler="tpe",
                        pruner="median",
                        fidelity={"levels": [1]},
                        storage={"url": f"sqlite:///{db_path}", "study_name": study_name, "load_if_exists": True},
                        n_trials_per_candidate=n_trials,
                        seed=11,
                        objective={
                            "loss": "huber",
                            "huber_delta_nm": 10.0,
                            "penalties": {
                                "lambda_solver": 0.0,
                                "lambda_phys": 0.0,
                                "lambda_prior": 0.0,
                                "lambda_complex": 0.0,
                                "lambda_role": 0.0,
                            },
                        },
                        search_space=[
                            {
                                "name": "model.params.kinetics.k_rxn",
                                "type": "loguniform",
                                "low": 1.0e-6,
                                "high": 1.0e-2,
                            }
                        ],
                    ),
                    class_compare=SimpleNamespace(complexity_penalty={"lambda_role": 0.0}),
                )

            out1 = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=build_spec(1),
            )
            out2 = fit_candidate_with_optuna(
                sim_spec=sim_spec,
                role_candidate=role,
                order_candidate=order,
                opt_spec=build_spec(2),
            )
            self.assertGreaterEqual(int(out2["study_trial_count"]), int(out1["study_trial_count"]))
            self.assertGreaterEqual(int(out2["study_trial_count"]), 3)

    def test_run_fit_emits_topk_assignments(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path, meas_path = self._write_inputs(tmp)
            out = run_fit(
                config_name="fit_cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.output.project=fit_unit",
                    f"sim.output.root_dir={tmp}",
                    "opt.output.project=fit_unit",
                    f"opt.output.root_dir={tmp}",
                    "opt.output.run_name=fit_unit",
                    f"opt.measurement.file={meas_path}",
                    "opt.parameter_fit.engine=random",
                    "opt.parameter_fit.n_trials_per_candidate=1",
                    "opt.role_enumeration.roles.A.candidates=[s0]",
                    "opt.role_enumeration.roles.I.candidates=[s1]",
                    "opt.role_enumeration.roles.B.candidates=[s2]",
                    "opt.selection.topk_overall=3",
                    "opt.selection.topk_per_class=1",
                ],
            )
            run_dir = Path(out["run_dir"])
            ranking_path = run_dir / "tables" / "ranking.csv"
            role_summary_path = run_dir / "tables" / "role_summary.csv"
            role_ranking_path = run_dir / "tables" / "role_ranking.csv"
            condition_scores_path = run_dir / "tables" / "condition_scores.csv"
            topk_path = run_dir / "tables" / "topk_assignments.csv"
            class_path = run_dir / "tables" / "class_compare.csv"
            role_stability_path = run_dir / "tables" / "role_stability.csv"
            complexity_sensitivity_path = run_dir / "tables" / "complexity_sensitivity.csv"
            manifest_path = run_dir / "outputs" / "manifest.json"
            diagnostics_path = run_dir / "outputs" / "fit_diagnostics.json"
            self.assertTrue(ranking_path.exists())
            self.assertTrue(role_summary_path.exists())
            self.assertTrue(role_ranking_path.exists())
            self.assertTrue(condition_scores_path.exists())
            self.assertTrue(topk_path.exists())
            self.assertTrue(class_path.exists())
            self.assertTrue(role_stability_path.exists())
            self.assertTrue(complexity_sensitivity_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(diagnostics_path.exists())

            with ranking_path.open("r", encoding="utf-8") as fh:
                ranking_rows = list(csv.DictReader(fh))
            with role_summary_path.open("r", encoding="utf-8") as fh:
                role_summary_rows = list(csv.DictReader(fh))
            with role_ranking_path.open("r", encoding="utf-8") as fh:
                role_ranking_rows = list(csv.DictReader(fh))
            summary = out["summary"]
            self.assertEqual(len(ranking_rows), int(summary["candidate_count"]))
            self.assertEqual(len(role_summary_rows), int(summary["candidate_count"]))
            self.assertEqual(len(role_ranking_rows), int(summary["candidate_count"]))
            self.assertEqual(int(summary["ranking_count"]), int(summary["candidate_count"]))
            self.assertEqual(int(summary["role_summary_count"]), int(summary["candidate_count"]))
            self.assertEqual(int(summary["role_ranking_count"]), int(summary["candidate_count"]))
            self.assertTrue(bool(summary["consistency"]["ranking_equals_candidates"]))
            self.assertIn("loss_data", ranking_rows[0])
            self.assertIn("penalty_solver", ranking_rows[0])
            self.assertIn("penalty_phys", ranking_rows[0])
            self.assertIn("penalty_prior", ranking_rows[0])
            self.assertIn("penalty_profile", ranking_rows[0])
            self.assertIn("score_total", ranking_rows[0])
            self.assertIn("rmse_nm", ranking_rows[0])
            self.assertIn("mae_nm", ranking_rows[0])
            self.assertIn("max_abs_nm", ranking_rows[0])
            self.assertIn("decision", role_summary_rows[0])
            self.assertIn("reason", role_summary_rows[0])
            self.assertIn("score_gap_from_best", role_summary_rows[0])
            self.assertIn(role_summary_rows[0]["decision"], {"adopt_candidate", "review"})
            self.assertIn("role_A", role_ranking_rows[0])
            self.assertIn("role_I", role_ranking_rows[0])
            self.assertIn("role_B", role_ranking_rows[0])

            with class_path.open("r", encoding="utf-8") as fh:
                class_rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(class_rows), 1)
            self.assertIn("tie_flag", class_rows[0])
            self.assertIn("tie_group_size", class_rows[0])
            self.assertIn("delta_from_best", class_rows[0])

            with topk_path.open("r", encoding="utf-8") as fh:
                topk_rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(topk_rows), 1)
            self.assertLessEqual(len(topk_rows), len(ranking_rows))
            self.assertIn("selected_by_class", topk_rows[0])
            self.assertIn("selected_by_overall", topk_rows[0])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(manifest["metadata"]["workflow_name"], "fit")
            self.assertIn("config_fingerprint", manifest["metadata"])
            artifact_ids = {row["id"] for row in manifest["artifacts"]}
            self.assertTrue(
                {
                    "ranking",
                    "role_summary",
                    "role_ranking",
                    "condition_scores",
                    "class_compare",
                    "topk_assignments",
                    "role_stability",
                    "complexity_sensitivity",
                    "fit_diagnostics",
                    "summary",
                    "report",
                }.issubset(artifact_ids)
            )
            summary = out["summary"]
            self.assertEqual(summary.get("diagnostics_path"), "outputs/fit_diagnostics.json")
            self.assertIn("cache_stats", summary)
            report_text = (run_dir / "report.html").read_text(encoding="utf-8")
            for row in manifest["artifacts"]:
                self.assertIn(str(row["path"]), report_text)


if __name__ == "__main__":
    unittest.main()
