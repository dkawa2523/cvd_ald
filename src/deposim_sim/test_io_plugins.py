from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .io_plugins import (
    available_io_loaders,
    load_fluent_from_run_spec,
    load_fluent_input,
    load_measurement_from_run_spec,
    load_measurement_input,
)


@unittest.skipIf(np is None, "NumPy is required for io plugin tests")
class TestIOPlugins(unittest.TestCase):
    def test_loader_registry_contains_npz_and_csv(self) -> None:
        loaders = available_io_loaders()
        self.assertIn("npz", loaders)
        self.assertIn("csv", loaders)

    def test_npz_fluent_loader_reads_aib_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fluent.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
            cref = np.array([[1.0, 0.1], [0.8, 0.2]], dtype=float)
            np.savez(path, xy=xy, cref=cref)

            out = load_fluent_input(
                loader_name="npz",
                path=path,
                mode="steady",
                species=["s0", "s1"],
                keys={"xy": "xy", "cref": "cref"},
            )
            self.assertEqual(out.mode, "steady")
            np.testing.assert_allclose(out.xy, xy)
            np.testing.assert_allclose(out.cref, cref)

    def test_csv_fluent_loader_reads_aib_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fluent.csv"
            path.write_text("x,y,s0,s1\n0.0,0.0,1.0,0.1\n10.0,0.0,0.8,0.2\n", encoding="utf-8")
            out = load_fluent_input(
                loader_name="csv",
                path=path,
                mode="steady",
                species=["s0", "s1"],
                keys={"x": "x", "y": "y"},
            )
            self.assertEqual(out.cref.shape, (2, 2))
            self.assertEqual(out.xy.shape, (2, 2))

    def test_measurement_loader_supports_npz_and_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "meas.npz"
            csv_path = Path(tmp) / "meas.csv"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
            h = np.array([1.2, 1.0], dtype=float)
            np.savez(npz_path, xy=xy, h_nm=h)
            csv_path.write_text("x,y,h_nm\n0.0,0.0,1.2\n10.0,0.0,1.0\n", encoding="utf-8")

            npz_out = load_measurement_input(loader_name="npz", path=npz_path, keys={"xy": "xy", "h": "h_nm"})
            csv_out = load_measurement_input(loader_name="csv", path=csv_path, keys={"x": "x", "y": "y", "h": "h_nm"})
            np.testing.assert_allclose(npz_out.xy, xy)
            np.testing.assert_allclose(npz_out.h, h)
            np.testing.assert_allclose(csv_out.xy, xy)
            np.testing.assert_allclose(csv_out.h, h)

    def test_measurement_npz_loads_configured_state_histories(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_meas.npz"
            xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
            history = np.array([[1.0, 0.9], [0.6, 0.5]], dtype=float)
            np.savez(
                path,
                xy=xy,
                h_nm=np.array([0.2, 0.3]),
                time=np.array([0.0, 1.0]),
                chi=history,
                chi_sigma=np.full(history.shape, 0.02),
            )
            out = load_measurement_input(
                loader_name="npz",
                path=path,
                keys={
                    "xy": "xy",
                    "h": "h_nm",
                    "time": "time",
                    "oxidized_fraction_history": "chi",
                    "oxidized_fraction_history_sigma": "chi_sigma",
                },
            )
            np.testing.assert_allclose(out.extra["time"], [0.0, 1.0])
            np.testing.assert_allclose(
                out.extra["oxidized_fraction_history"], history
            )

    def test_run_spec_helpers_resolve_loader_from_suffix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent.csv"
            meas_path = root / "meas.csv"
            fluent_path.write_text("x,y,s0,s1,s2,s3\n0,0,1.0,0.2,0.1,0.0\n10,0,0.8,0.2,0.1,0.0\n", encoding="utf-8")
            meas_path.write_text("x,y,h_nm\n0,0,1.0\n10,0,0.8\n", encoding="utf-8")

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.measurement.enabled=true",
                    f"sim.measurement.file={meas_path}",
                ],
            )
            fluent = load_fluent_from_run_spec(spec)
            meas = load_measurement_from_run_spec(spec)
            self.assertEqual(fluent.xy.shape, (2, 2))
            self.assertEqual(fluent.cref.shape, (2, 4))
            self.assertEqual(meas.xy.shape, (2, 2))
            self.assertEqual(meas.h.shape, (2,))

    def test_explicit_loader_name_overrides_suffix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fluent_path = root / "fluent_payload.data"
            meas_path = root / "meas_payload.bin"
            fluent_path.write_text("x,y,s0,s1,s2,s3\n0,0,1.0,0.2,0.1,0.0\n10,0,0.8,0.2,0.1,0.0\n", encoding="utf-8")
            meas_path.write_text("x,y,h_nm\n0,0,1.0\n10,0,0.8\n", encoding="utf-8")

            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.inputs.fluent.io_loader_name=csv",
                    "sim.measurement.enabled=true",
                    f"sim.measurement.file={meas_path}",
                    "sim.measurement.io_loader_name=csv",
                ],
            )
            fluent = load_fluent_from_run_spec(spec)
            meas = load_measurement_from_run_spec(spec)
            self.assertEqual(fluent.cref.shape, (2, 4))
            self.assertEqual(meas.h.shape, (2,))

    def test_measurement_npz_missing_key_raises_with_context(self) -> None:
        with TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "meas.npz"
            np.savez(npz_path, xy=np.array([[0.0, 0.0]], dtype=float))
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                load_measurement_input(loader_name="npz", path=npz_path, keys={"xy": "xy", "h": "h_nm"})


if __name__ == "__main__":
    unittest.main()
