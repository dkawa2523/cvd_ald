from __future__ import annotations

from pathlib import Path
import unittest

from deposim_schema import resolve_config_root

from .schema import load_opt_run_spec
from .transforms import inverse_transform_value, transform_value


class TestOptScaffold(unittest.TestCase):
    def test_opt_config_root_is_separate_from_sim(self) -> None:
        root = Path.cwd()
        sim_root = resolve_config_root("sim", project_root=root)
        opt_root = resolve_config_root("opt", project_root=root)
        self.assertNotEqual(sim_root, opt_root)
        self.assertTrue(str(opt_root).endswith("configs/opt"))

    def test_opt_config_can_be_loaded(self) -> None:
        spec = load_opt_run_spec(Path("configs/opt/base.yaml"))
        self.assertEqual(spec.run_name, "opt_baseline")
        self.assertGreaterEqual(len(spec.parameters), 1)

    def test_parameter_transform_roundtrip(self) -> None:
        initial = 0.25
        unconstrained = transform_value(initial, "positive")
        recovered = inverse_transform_value(unconstrained, "positive")
        self.assertAlmostEqual(initial, recovered, places=12)


if __name__ == "__main__":
    unittest.main()
