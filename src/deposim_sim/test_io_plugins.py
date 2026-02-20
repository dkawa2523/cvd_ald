from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deposim_schema import compose_sim_config

from .input_builder import build_field_bundle
from .smoke import main as smoke_main
from .domain import build_domain_grid
from .io_plugins import available_io_loaders, load_inputs_from_run_spec, load_with_io_loader


class TestIOPlugins(unittest.TestCase):
    def test_builtin_loader_registry_contains_npz_and_csv(self) -> None:
        loaders = available_io_loaders()
        self.assertIn("npz", loaders)
        self.assertIn("csv", loaders)

    def test_npz_loader_reads_simple_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            np.savez(path, a=np.array([1.0, 2.0]), b=np.array([[3.0, 4.0]]))
            loaded = load_with_io_loader("npz", path)
            self.assertIn("a", loaded)
            self.assertIn("b", loaded)
            np.testing.assert_allclose(loaded["a"], np.array([1.0, 2.0]))

    def test_csv_loader_is_selectable_from_config(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            np.savetxt(path, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")
            run_spec = compose_sim_config("smoke", overrides=["inputs.io_loader_name=csv"])
            loaded = load_inputs_from_run_spec(run_spec, path)
            self.assertIn("array", loaded)
            np.testing.assert_allclose(loaded["array"], np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_file_input_smoke_path_works_with_csv_loader(self) -> None:
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "fields.csv"
            csv_path.write_text("C_ref__precursor,T\n1.8,710.0\n", encoding="utf-8")
            project_dir = Path(tmp) / "out"
            rc = smoke_main(
                [
                    "--config-name",
                    "smoke",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                    "inputs.source_kind=file",
                    "inputs.io_loader_name=csv",
                    f"inputs.field_path={csv_path}",
                    f"output.project_dir={project_dir}",
                    "output.run_dir_name=io_file_smoke",
                ]
            )
            self.assertEqual(rc, 0)
            runs = sorted([p for p in (project_dir / "runs").iterdir() if p.is_dir()])
            self.assertGreaterEqual(len(runs), 1)
            latest = runs[-1]
            self.assertTrue((latest / "outputs" / "thickness.npz").exists())
            self.assertTrue((latest / "report.html").exists())

    def test_invalid_file_input_contract_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "bad.npz"
            np.savez(npz_path, T=np.array([700.0]))
            run_spec = compose_sim_config(
                "smoke",
                overrides=[
                    "inputs.source_kind=file",
                    "inputs.io_loader_name=npz",
                    f"inputs.field_path={npz_path}",
                ],
            )
            grid = build_domain_grid(run_spec.domain)
            with self.assertRaisesRegex(ValueError, "C_ref__<species>"):
                build_field_bundle(run_spec, grid)


if __name__ == "__main__":
    unittest.main()
