from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from deposim_schema import compose_and_save_sim_config, compose_opt_config, compose_sim_config


class TestSimConfigCompose(unittest.TestCase):
    def test_compose_example_cvd(self) -> None:
        run_spec = compose_sim_config("cvd_steady_min")
        self.assertEqual(run_spec.model.name, "aib_ode")
        self.assertIn(run_spec.time_mode, {"steady", "transient"})
        self.assertTrue(run_spec.output.run_name)
        self.assertEqual(run_spec.inputs.fluent.io_loader_name, "")

    def test_compose_and_save_resolved_yaml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "resolved.yaml"
            run_spec = compose_and_save_sim_config(output_path, "cvd_steady_min")

            self.assertTrue(output_path.exists())
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("sim:", text)
            self.assertIn("time_mode:", text)
            self.assertIn("output:", text)
            self.assertIn(f"run_name: {run_spec.output.run_name}", text)

    def test_time_mode_aliases_are_normalized(self) -> None:
        run_spec = compose_sim_config("cvd_steady_min", overrides=["sim.time_mode=transient", "sim.inputs.fluent.mode=transient"])
        self.assertEqual(run_spec.time_mode, "transient")
        self.assertEqual(run_spec.inputs.fluent.mode, "transient")

    def test_unknown_time_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            compose_sim_config("cvd_steady_min", overrides=["sim.time_mode=unknown_mode"])

    def test_compose_opt_config(self) -> None:
        spec = compose_opt_config("fit_cvd_steady_min")
        self.assertEqual(spec.sim.model.name, "aib_ode")
        self.assertEqual(spec.opt.task, "fit_roles_and_params")

    def test_explicit_loader_names_are_composed(self) -> None:
        run_spec = compose_sim_config(
            "cvd_steady_min",
            overrides=[
                "sim.inputs.fluent.io_loader_name=csv",
                "sim.measurement.io_loader_name=npz",
            ],
        )
        self.assertEqual(run_spec.inputs.fluent.io_loader_name, "csv")
        self.assertEqual(run_spec.measurement.io_loader_name, "npz")


if __name__ == "__main__":
    unittest.main()
