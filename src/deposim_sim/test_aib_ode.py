from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .models.aib_ode import step_theta_implicit


@unittest.skipIf(np is None, "NumPy is required")
class TestAIBODE(unittest.TestCase):
    def test_step_bounds_and_growth(self) -> None:
        shape = (4,)
        theta = np.zeros(shape, dtype=float)
        h = np.zeros(shape, dtype=float)
        ones = np.ones(shape, dtype=float)

        out = step_theta_implicit(
            theta_n=theta,
            h_n=h,
            dt_s=0.1,
            cref_a=ones,
            cref_i=np.zeros(shape, dtype=float),
            cref_b=np.zeros(shape, dtype=float),
            km_a=0.02 * ones,
            km_b=0.02 * ones,
            k_ads=1.0 * ones,
            k_des=0.1 * ones,
            k_rxn=0.01 * ones,
            K_I=np.zeros(shape, dtype=float),
            gamma_s=1.0 * ones,
            nu_a=1.0 * ones,
            alpha_h=1.0 * ones,
            c_b_scale=1.0 * ones,
            m_ads=1,
            p_a=1,
            p_star=0,
            has_b=False,
            max_iter=60,
            theta_tol=1.0e-10,
        )
        self.assertTrue(np.all(out.theta_next >= 0.0))
        self.assertTrue(np.all(out.theta_next <= 1.0))
        self.assertTrue(np.all(out.h_next >= h))
        self.assertIn("iteration_count", out.diagnostics)
        self.assertIn("fallback_mask", out.diagnostics)
        self.assertEqual(np.asarray(out.diagnostics["iteration_count"]).shape, shape)
        self.assertEqual(np.asarray(out.diagnostics["fallback_mask"]).shape, shape)


if __name__ == "__main__":
    unittest.main()
