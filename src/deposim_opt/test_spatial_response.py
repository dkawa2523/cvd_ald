from __future__ import annotations

import unittest

import numpy as np

from .role_fields import RoleFieldSet
from .spatial_response import (
    RADIAL_QUARTIC,
    apply_spatial_response,
    fit_spatial_response,
)


class SpatialResponseTests(unittest.TestCase):
    def test_shared_residual_preserves_condition_means_and_transfers(self) -> None:
        x = np.linspace(-1.0, 1.0, 15)
        radius = np.abs(x)
        xy = np.column_stack([x, np.zeros_like(x)])
        xyz = np.column_stack([np.vstack([xy, xy]), np.zeros(x.size * 2)])
        condition_id = np.repeat((1, 2), x.size)
        chemistry = np.concatenate(
            [1.0 + 0.2 * radius, 2.2 - 0.3 * radius]
        )
        shape = 0.12 * radius**2 - 0.11 * radius**4
        rate = chemistry * np.exp(np.tile(shape, 2))
        for condition in (1, 2):
            mask = condition_id == condition
            rate[mask] *= np.mean(chemistry[mask]) / np.mean(rate[mask])
        concentrations = {"s0": np.ones_like(rate)}
        data = RoleFieldSet(
            case_ids=(1, 2),
            xyz=xyz,
            condition_id=condition_id,
            species=("s0",),
            bulk_concentrations=concentrations,
            species_fractions={"s0": np.ones_like(rate)},
            total_concentration=np.ones_like(rate),
            rate=rate,
        )

        fit = fit_spatial_response(RADIAL_QUARTIC, data, chemistry)
        corrected, factor = apply_spatial_response(fit, data, chemistry)

        self.assertLess(float(np.max(np.abs(corrected - rate))), 1.0e-12)
        self.assertTrue(np.all(factor > 0.0))
        for condition in (1, 2):
            mask = condition_id == condition
            self.assertAlmostEqual(
                float(np.mean(corrected[mask])), float(np.mean(chemistry[mask]))
            )


if __name__ == "__main__":
    unittest.main()
