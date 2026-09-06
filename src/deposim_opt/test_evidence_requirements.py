from __future__ import annotations

import unittest

from .evidence_requirements import build_capability_requirements


class EvidenceRequirementTests(unittest.TestCase):
    def test_requirements_name_data_and_the_claim_it_resolves(self):
        summaries, rows = build_capability_requirements(
            spatial_supported=False,
            role_supported=False,
            parameter_identifiability_status="weak",
            concentration_location="bulk_as_surface",
            has_measurement_uncertainty=False,
            family_stable=False,
        )
        self.assertEqual(len(summaries), 3)
        self.assertTrue(all(row["required_measurement"] for row in rows))
        self.assertTrue(all(row["experimental_design"] for row in rows))
        self.assertTrue(all(row["resolves"] for row in rows))
        self.assertTrue(all(row["workflow_use"] for row in rows))
        capabilities = {row["capability"] for row in rows}
        self.assertEqual(
            capabilities,
            {
                "wafer_spatial_correction",
                "anonymous_species_role_assignment",
                "elementary_kinetic_parameter_estimation",
            },
        )


if __name__ == "__main__":
    unittest.main()
