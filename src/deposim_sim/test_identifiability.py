from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_report import write_run_report
from deposim_schema import compose_sim_config

from .domain import build_domain_grid
from .identifiability import compute_identifiability_diagnostics
from .physics.cvd_steady import run_cvd_steady
from .synthetic_inputs import build_synthetic_field_bundle


class TestIdentifiability(unittest.TestCase):
    def _run_spec(self):
        return compose_sim_config(
            "smoke",
            overrides=[
                "domain.nr=6",
                "domain.ntheta=12",
                "time.process_time_s=3.0",
                "model.kinetics_params.k0=1.2",
                "model.mass_transfer_params.k_m_m_s=0.02",
            ],
        )

    def test_sensitivity_and_correlation_are_generated(self) -> None:
        run_spec = self._run_spec()
        out = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.kinetics_params.k0",
                "model.mass_transfer_params.k_m_m_s",
            ],
        )
        self.assertEqual(len(out["parameter_paths"]), 2)
        self.assertEqual(np.asarray(out["correlation_matrix"]).shape, (2, 2))
        self.assertIn("model.kinetics_params.k0", out["sensitivity_norms"])
        self.assertIn("model.mass_transfer_params.k_m_m_s", out["sensitivity_norms"])
        self.assertTrue(np.isfinite(out["sensitivity_norms"]["model.kinetics_params.k0"]))

    def test_degeneracy_warning_is_reported(self) -> None:
        run_spec = self._run_spec()
        out = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.kinetics_params.k0",
                "model.mass_transfer_params.k_m_m_s",
            ],
            low_sensitivity_threshold=1.0e8,
        )
        self.assertTrue(out["degeneracy_warning"])
        self.assertGreater(len(out["warnings"]), 0)

    def test_report_includes_identifiability_section(self) -> None:
        run_spec = self._run_spec()
        grid = build_domain_grid(run_spec.domain)
        fields = build_synthetic_field_bundle(run_spec, grid)
        result = run_cvd_steady(
            grid=grid,
            fields=fields,
            model_config=run_spec.model,
            process_time_s=run_spec.time.process_time_s,
            solver_config=run_spec.solver,
        )
        ident = compute_identifiability_diagnostics(
            run_spec,
            parameter_paths=[
                "model.kinetics_params.k0",
                "model.mass_transfer_params.k_m_m_s",
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
                grid=grid,
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
