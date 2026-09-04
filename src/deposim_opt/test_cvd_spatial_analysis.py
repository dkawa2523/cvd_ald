from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .cvd_spatial_analysis import (
    RoleResponseCandidate,
    analyze_cvd_spatial_case,
    enumerate_role_response_candidates,
    fit_candidate,
)


class TestCvdSpatialAnalysis(unittest.TestCase):
    def test_candidate_enumeration_for_three_species(self) -> None:
        candidates = enumerate_role_response_candidates(["s0", "s1", "s2"])
        self.assertEqual(len(candidates), 16)
        self.assertEqual(len({candidate.model_id for candidate in candidates}), 16)

    def test_centered_a_coefficient_is_recovered(self) -> None:
        concentration = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
        reference = float(np.median(concentration))
        target = 0.2 + 0.04 * (concentration / reference - 1.0)
        fit = fit_candidate(
            RoleResponseCandidate(model_id="A:s0", class_id="A", A="s0"),
            {"s0": concentration},
            target,
            np.arange(target.size),
        )
        self.assertAlmostEqual(float(fit.coefficients[0]), 0.2, places=12)
        self.assertAlmostEqual(float(fit.coefficients[1]), 0.04, places=12)
        np.testing.assert_allclose(fit.prediction, target, rtol=0.0, atol=1.0e-12)

    def test_end_to_end_outputs_and_unit_consistency(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition_path = root / "condition.csv"
            validation_path = root / "validation.csv"
            output_dir = root / "results"

            condition_rows: list[list[float]] = []
            validation_rows: list[list[float]] = []
            for radius in (0.03, 0.07, 0.11):
                for sector in range(8):
                    angle = 2.0 * np.pi * sector / 8.0
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    c0 = 2.0e-6 * (1.0 + 0.08 * np.cos(angle))
                    c1 = 1.0e-6 * (1.0 + 0.05 * np.sin(angle))
                    c2 = 2.0e-4 * (1.0 + 0.001 * np.cos(2.0 * angle))
                    total = c0 + c1 + c2
                    rate = 0.1 + 0.01 * (c0 / 2.0e-6 - 1.0)
                    condition_rows.append([x, y, 0.0, c0, c1, c2, c0 / total, c1 / total, c2 / total, 30.0 * total])
                    validation_rows.append([x, y, 0.0, rate])

            with condition_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "x",
                        "y",
                        "z",
                        "concentration_s0",
                        "concentration_s1",
                        "concentration_s2",
                        "molef_s0",
                        "molef_s1",
                        "molef_s2",
                        "density",
                    ]
                )
                writer.writerows(condition_rows)
            with validation_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "y", "z", "dr_nm_per_sec"])
                writer.writerows(validation_rows)

            summary = analyze_cvd_spatial_case(
                condition_path=condition_path,
                validation_path=validation_path,
                output_dir=output_dir,
                bootstrap_samples=100,
                seed=7,
            )
            self.assertEqual(summary["data_quality"]["row_count"], 24)
            self.assertLess(summary["data_quality"]["mole_fraction_sum_max_abs_error_from_one"], 1.0e-12)
            for name in (
                "analysis_summary.json",
                "model_ranking.csv",
                "coefficients.csv",
                "contribution_summary.csv",
                "spatial_contributions.csv",
                "cvd_condition_1_analysis.ipynb",
                "report.md",
                "manifest.json",
            ):
                self.assertTrue((output_dir / name).exists(), name)
            notebook = json.loads((output_dir / "cvd_condition_1_analysis.ipynb").read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
