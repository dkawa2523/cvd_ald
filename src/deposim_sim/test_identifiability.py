from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from deposim_report import write_run_report
from deposim_schema import compose_sim_config

from .identifiability import analyze_sensitivities, compute_identifiability_diagnostics
from .pipeline import run_aib_from_spec


class TestIdentifiability(unittest.TestCase):
    def test_joint_dependence_is_found_without_high_pairwise_correlation(self):
        out = analyze_sensitivities(np.array([[1., 0.], [0., 1.], [1., 1.]]), ["a", "b", "c"])
        self.assertFalse(out["high_correlation_pairs"])
        self.assertEqual(out["effective_rank"], 2)
        self.assertTrue(out["degeneracy_warning"])
        self.assertEqual(len(out["weak_parameter_combinations"]), 1)

    def test_complementary_conditions_resolve_parameters_in_observation_space(self):
        def simulate(spec):
            observed = np.array([spec.a + spec.sign * spec.b]) * spec.unit
            # The unobserved mesh value must not improve identifiability.
            return SimpleNamespace(thickness=np.r_[observed, spec.b], fields={}, diagnostics={
                "observation": {"residual_nm": observed - 2 * spec.unit,
                                "target_nm": np.array([2 * spec.unit]), "sigma_nm": None}})
        first = SimpleNamespace(a=2., b=1., sign=1., unit=1.)
        second = SimpleNamespace(a=2., b=1., sign=-1., unit=1.)
        with patch("deposim_sim.identifiability.run_sim_from_spec", side_effect=simulate):
            single = compute_identifiability_diagnostics(first, parameter_paths=["a", "b"])
            joint = compute_identifiability_diagnostics(first, run_specs=[first, second], parameter_paths=["a", "b"])
            first.unit = second.unit = 1.0e-6
            scaled = compute_identifiability_diagnostics(first, run_specs=[first, second], parameter_paths=["a", "b"])
        self.assertEqual(single["effective_rank"], 1)
        self.assertEqual(joint["effective_rank"], 2)
        self.assertEqual(joint["observation_count"], 2)
        np.testing.assert_allclose(joint["singular_values"], scaled["singular_values"])

    def _run_spec(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fluent = Path(tmp.name) / "fluent.npz"
        xy = np.array([[-10.0, -10.0], [0.0, 0.0], [20.0, 10.0], [30.0, -10.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.9, 0.3, 0.1, 0.0],
                [0.8, 0.3, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(fluent, xy=xy, cref=cref)
        return compose_sim_config(
            "cvd_steady_min",
            overrides=[
                f"sim.inputs.fluent.file={fluent}",
                "sim.model.params.kinetics.k_rxn=0.012",
                "sim.model.params.transport.km_A=0.02",
            ],
        )

    def test_sensitivity_and_correlation_are_generated(self) -> None:
        run_spec = self._run_spec()
        out = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.params.kinetics.k_rxn",
                "model.params.transport.km_A",
            ],
        )
        self.assertEqual(len(out["parameter_paths"]), 2)
        self.assertEqual(np.asarray(out["correlation_matrix"]).shape, (2, 2))
        self.assertIn("model.params.kinetics.k_rxn", out["sensitivity_norms"])
        self.assertIn("model.params.transport.km_A", out["sensitivity_norms"])
        self.assertTrue(np.isfinite(out["sensitivity_norms"]["model.params.kinetics.k_rxn"]))

    def test_degeneracy_warning_is_reported(self) -> None:
        run_spec = self._run_spec()
        out = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.params.kinetics.k_rxn",
                "model.params.transport.km_A",
            ],
            low_sensitivity_threshold=1.0e8,
        )
        self.assertTrue(out["degeneracy_warning"])
        self.assertGreater(len(out["warnings"]), 0)

    def test_report_includes_identifiability_section(self) -> None:
        run_spec = self._run_spec()
        result = run_aib_from_spec(run_spec)
        ident = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.params.kinetics.k_rxn",
                "model.params.transport.km_A",
            ],
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["identifiability"] = ident
        summary = {"run_id": "ident_test", "thickness_mean": float(np.mean(result.thickness))}

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_run_report(
                run_dir=run_dir,
                run_id="ident_test",
                grid=result.grid,
                thickness=result.thickness,
                diagnostics=diagnostics,
                summary=summary,
                output_links=[],
            )
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Identifiability", report)
            self.assertIn("identifiability_correlation.png", report)
            self.assertTrue((run_dir / "plots" / "identifiability_correlation.png").exists())


if __name__ == "__main__":
    unittest.main()
