from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from deposim_schema import compose_and_save_sim_config, compose_sim_config


class TestSimConfigCompose(unittest.TestCase):
    def test_compose_example_cvd(self) -> None:
        run_spec = compose_sim_config("example_cvd")
        self.assertTrue(run_spec.run_name)
        self.assertIn(run_spec.time.mode, {"cvd_steady", "cvd_transient", "ald_cycle"})
        self.assertTrue(run_spec.output.run_dir_name)

    def test_compose_and_save_resolved_yaml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "resolved.yaml"
            run_spec = compose_and_save_sim_config(output_path, "example_cvd")

            self.assertTrue(output_path.exists())
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("run_name:", text)
            self.assertIn("time:", text)
            self.assertIn("mode:", text)
            self.assertIn("output:", text)
            self.assertIn("run_dir_name:", text)
            self.assertIn(f"run_name: {run_spec.run_name}", text)
            self.assertIn(f"mode: {run_spec.time.mode}", text)
            self.assertIn(f"run_dir_name: {run_spec.output.run_dir_name}", text)

    def test_time_mode_aliases_are_normalized(self) -> None:
        aliases = [
            ("steady", "cvd_steady"),
            ("cvd_steady", "cvd_steady"),
            ("transient", "cvd_transient"),
            ("cvd_transient", "cvd_transient"),
            ("phases", "ald_cycle"),
            ("ald_cycle", "ald_cycle"),
        ]
        for mode_input, expected in aliases:
            with self.subTest(mode_input=mode_input):
                run_spec = compose_sim_config("example_cvd", overrides=[f"time.mode={mode_input}"])
                self.assertEqual(run_spec.time.mode, expected)

    def test_unknown_time_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            compose_sim_config("example_cvd", overrides=["time.mode=unknown_mode"])


if __name__ == "__main__":
    unittest.main()
