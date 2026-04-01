from __future__ import annotations

from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest

from deposim_schema import compose_sim_config

from .compute_engine import build_engine_context, is_jax_available
from .smoke import main as smoke_main


@unittest.skip("legacy compute.engine path is isolated pending schema-aligned rewrite")
class TestJaxOptional(unittest.TestCase):
    def test_numpy_remains_baseline(self) -> None:
        run_spec = compose_sim_config("smoke", overrides=["compute.engine=numpy"])
        ctx = build_engine_context(run_spec.compute.engine)
        self.assertEqual(ctx["requested_engine"], "numpy")
        self.assertEqual(ctx["selected_engine"], "numpy")
        self.assertEqual(ctx["execution_backend"], "numpy")

    def test_engine_selection_respects_yaml_choice(self) -> None:
        run_spec = compose_sim_config("smoke", overrides=["compute.engine=jax"])
        if is_jax_available():
            ctx = build_engine_context(run_spec.compute.engine)
            self.assertEqual(ctx["selected_engine"], "jax")
            self.assertEqual(ctx["execution_backend"], "numpy")
        else:
            with self.assertRaisesRegex(RuntimeError, "deposim\\[jax\\]"):
                build_engine_context(run_spec.compute.engine)

    def test_smoke_path_for_selected_engine(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            argv = [
                "--config-name",
                "smoke",
                "domain.nr=6",
                "domain.ntheta=12",
                f"output.project_dir={out_dir}",
                "output.run_dir_name=smoke_jax_optional",
            ]
            if is_jax_available():
                rc = smoke_main(argv + ["compute.engine=jax"])
                self.assertEqual(rc, 0)
                runs = sorted([p for p in (out_dir / "runs").iterdir() if p.is_dir()])
                latest = runs[-1]
                summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["engine_selected"], "jax")
                self.assertEqual(summary["engine_execution_backend"], "numpy")
            else:
                with self.assertRaisesRegex(RuntimeError, "deposim\\[jax\\]"):
                    smoke_main(argv + ["compute.engine=jax"])


if __name__ == "__main__":
    unittest.main()
