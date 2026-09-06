from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .doe import run_doe
from .smoke import main as smoke_main
from .zarr_output import is_h5py_available, is_zarr_available, save_array_store


class TestZarrOutput(unittest.TestCase):
    def _write_fluent(self, tmp: str) -> Path:
        fluent_path = Path(tmp) / "fluent.npz"
        xy = np.array([[-15.0, -10.0], [0.0, 0.0], [20.0, 8.0], [30.0, -12.0]], dtype=float)
        cref = np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.9, 0.3, 0.1, 0.0],
                [0.8, 0.3, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=float,
        )
        np.savez(fluent_path, xy=xy, cref=cref)
        return fluent_path

    def _latest_run_dir(self, project_root: Path) -> Path:
        runs = sorted(path for path in project_root.glob("*/runs/*") if path.is_dir())
        if not runs:
            self.fail(f"no smoke run directory found under {project_root}")
        return runs[-1]

    def test_npz_is_default_array_store(self) -> None:
        with TemporaryDirectory() as tmp:
            info = save_array_store(
                base_path=Path(tmp) / "sample",
                arrays={"x": np.array([1.0, 2.0])},
                store="npz",
            )
            self.assertEqual(info["store_used"], "npz")
            self.assertTrue(Path(info["path"]).exists())

    def test_zarr_optional_with_npz_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            info = save_array_store(
                base_path=Path(tmp) / "sample",
                arrays={"x": np.array([1.0, 2.0])},
                store="zarr",
            )
            if is_zarr_available():
                self.assertEqual(info["store_used"], "zarr")
                self.assertTrue(Path(info["path"]).suffix == ".zarr")
            else:
                self.assertEqual(info["store_used"], "npz_fallback")
                self.assertTrue(Path(info["path"]).suffix == ".npz")
            self.assertTrue(Path(info["path"]).exists())

    def test_unknown_store_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_array_store(
                    base_path=Path(tmp) / "sample",
                    arrays={"x": np.array([1.0])},
                    store="unknown",
                )

    def test_hdf5_optional_with_npz_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            info = save_array_store(
                base_path=Path(tmp) / "sample",
                arrays={"x": np.array([1.0, 2.0])},
                store="hdf5",
            )
            if is_h5py_available():
                self.assertEqual(info["store_used"], "hdf5")
                self.assertTrue(Path(info["path"]).suffix == ".h5")
            else:
                self.assertEqual(info["store_used"], "npz_fallback")
                self.assertTrue(Path(info["path"]).suffix == ".npz")
            self.assertTrue(Path(info["path"]).exists())

    def test_doe_store_selection_is_reflected_in_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = self._write_fluent(tmp)
            result = run_doe(
                config_name="cvd_steady_min",
                base_overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    f"output.project_dir={tmp}",
                    "output.array_store=zarr",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                ],
                sweep={"inputs.c_ref_mol_m3": [1.2, 1.8]},
                sampling="grid",
            )
            summary_path = result.run_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn("doe_cases_store_used", summary_text)
            if is_zarr_available():
                self.assertTrue((result.run_dir / "outputs" / "doe_cases.zarr").exists())
            else:
                self.assertTrue((result.run_dir / "outputs" / "doe_cases.npz").exists())

    def test_smoke_store_selection_is_reflected_in_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = self._write_fluent(tmp)
            project_dir = Path(tmp) / "out"
            rc = smoke_main(
                [
                    "--config-name",
                    "cvd_steady_min",
                    f"sim.inputs.fluent.file={fluent_path}",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                    "output.array_store=zarr",
                    f"output.project_dir={project_dir}",
                    "output.run_dir_name=zarr_smoke",
                ]
            )
            self.assertEqual(rc, 0)
            latest = self._latest_run_dir(project_dir)
            summary_text = (latest / "summary.json").read_text(encoding="utf-8")
            self.assertIn("fields_store_used", summary_text)
            if is_zarr_available():
                self.assertTrue((latest / "outputs" / "fields.zarr").exists())
            else:
                self.assertTrue((latest / "outputs" / "fields.npz").exists())

    def test_doe_hdf5_store_selection_is_reflected_in_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = self._write_fluent(tmp)
            result = run_doe(
                config_name="cvd_steady_min",
                base_overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    f"output.project_dir={tmp}",
                    "output.array_store=hdf5",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                ],
                sweep={"inputs.c_ref_mol_m3": [1.2, 1.8]},
                sampling="grid",
            )
            if is_h5py_available():
                self.assertTrue((result.run_dir / "outputs" / "doe_cases.h5").exists())
            else:
                self.assertTrue((result.run_dir / "outputs" / "doe_cases.npz").exists())

    def test_smoke_hdf5_store_selection_is_reflected_in_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = self._write_fluent(tmp)
            project_dir = Path(tmp) / "out"
            rc = smoke_main(
                [
                    "--config-name",
                    "cvd_steady_min",
                    f"sim.inputs.fluent.file={fluent_path}",
                    "domain.nr=4",
                    "domain.ntheta=8",
                    "time.process_time_s=1.0",
                    "output.array_store=hdf5",
                    f"output.project_dir={project_dir}",
                    "output.run_dir_name=hdf5_smoke",
                ]
            )
            self.assertEqual(rc, 0)
            latest = self._latest_run_dir(project_dir)
            summary_text = (latest / "summary.json").read_text(encoding="utf-8")
            self.assertIn("fields_store_used", summary_text)
            if is_h5py_available():
                self.assertTrue((latest / "outputs" / "fields.h5").exists())
            else:
                self.assertTrue((latest / "outputs" / "fields.npz").exists())


if __name__ == "__main__":
    unittest.main()
