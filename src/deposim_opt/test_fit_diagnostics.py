from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .enumerate_roles import RoleCandidate
from .fit_optuna import fit_candidate_with_optuna
from .run_fit import run_fit


@unittest.skipIf(np is None, "NumPy is required")
class TestFitDiagnostics(unittest.TestCase):
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
                    "opt.parameter_fit.engine=random",
                    "opt.parameter_fit.n_trials_per_candidate=1",
                    "opt.role_enumeration.roles.A.candidates=[s0]",
                    "opt.role_enumeration.roles.I.candidates=[s1,s2]",
                    "opt.role_enumeration.roles.B.candidates=[s3]",
                    "opt.parameter_fit.analysis.role_stability.topk_window=6",
                    "opt.parameter_fit.analysis.role_stability.score_epsilon=1000000000.0",
                ],
            )
            run_dir = Path(out["run_dir"])
            self.assertTrue((run_dir / "tables" / "role_stability.csv").exists())
            diag = json.loads((run_dir / "outputs" / "fit_diagnostics.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(diag.get("role_identifiability_warning")))
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
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1, 2]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=1,
                    seed=5,
                    analysis={
                        "cache": {"enabled": True, "max_entries": 64},
                        "preflight": {"enabled": True, "min_finite_ratio": 0.5},
                        "identifiability": {"enabled": False},
                    },
                    objective={"loss": "huber", "huber_delta_nm": 10.0, "penalties": {}},
                    search_space=[
                        {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1.0e-6, "high": 1.0e-2}
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
                    engine="random",
                    sampler="tpe",
                    pruner="none",
                    fidelity={"levels": [1]},
                    storage={"url": "", "study_name": "", "load_if_exists": False},
                    n_trials_per_candidate=1,
                    seed=3,
                    analysis={"preflight": {"enabled": True, "min_finite_ratio": 0.9}},
                    objective={"loss": "huber", "huber_delta_nm": 10.0, "penalties": {}},
                    search_space=[
                        {"name": "model.params.kinetics.k_rxn", "type": "loguniform", "low": 1.0e-6, "high": 1.0e-2}
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


if __name__ == "__main__":
    unittest.main()
