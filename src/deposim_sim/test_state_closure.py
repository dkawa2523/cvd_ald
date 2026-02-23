from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .state_closure import dynamic_ode_closure, resolve_state_from_model_config, steady_state_closure


@unittest.skipIf(np is None, "NumPy is required for state closure tests")
class TestStateClosure(unittest.TestCase):
    def test_dynamic_ode_closure_bounds(self) -> None:
        cs = {"A": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=float)}
        out = dynamic_ode_closure(
            Cs=cs,
            params={"species": "A", "A": 0.8, "B": 0.1, "m": 1.0, "theta0": 0.2},
            dt_s=2.0,
            initial_state={"theta": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=float)},
        )
        theta = out["theta"]
        self.assertEqual(theta.shape, (2, 2))
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))

    def test_steady_state_closure_bounds(self) -> None:
        cs = {"A": np.array([0.2, 0.6, 1.2], dtype=float)}
        out = steady_state_closure(Cs=cs, params={"species": "A", "A": 1.5, "B": 0.2})
        theta = out["theta"]
        self.assertEqual(theta.shape, (3,))
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))

    def test_resolve_state_dynamic_and_steady_modes(self) -> None:
        cs = {"A": np.array([0.5, 1.0, 1.5], dtype=float)}
        model_dyn = SimpleNamespace(
            state_name="dynamic_ode",
            state_params={"species": "A", "A": 0.4, "B": 0.05, "m": 1.0, "theta0": 0.1},
        )
        dyn = resolve_state_from_model_config(
            model_dyn,
            Cs=cs,
            dt_s=1.5,
            initial_state={"theta": np.array([0.1, 0.2, 0.3], dtype=float)},
        )
        self.assertIsNotNone(dyn)
        self.assertIn("theta", dyn)
        self.assertTrue(np.all(np.asarray(dyn["theta"], dtype=float) <= 1.0))

        model_steady = SimpleNamespace(state_name="steady_state", state_params={"species": "A", "A": 1.2, "B": 0.3})
        st = resolve_state_from_model_config(model_steady, Cs=cs, dt_s=1.0)
        self.assertIsNotNone(st)
        self.assertIn("theta", st)
        self.assertTrue(np.all(np.asarray(st["theta"], dtype=float) >= 0.0))


if __name__ == "__main__":
    unittest.main()
