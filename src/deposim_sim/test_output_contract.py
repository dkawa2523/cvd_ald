from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from deposim_schema import compose_sim_config

from .benchmark_wafer2d import run_wafer2d_benchmark
from .output_manifest import ManifestError, SCHEMA_VERSION, validate_manifest, validate_manifest_files
from .pipeline import run_aib_from_spec
from .run_manager import save_run_outputs


@unittest.skipIf(np is None, "NumPy is required")
class TestOutputContract(unittest.TestCase):
    def _write_fluent(self, path: Path) -> None:
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
        np.savez(path, xy=xy, cref=cref)

    def test_validate_manifest_rejects_missing_required_keys(self) -> None:
        with self.assertRaises(ManifestError):
            validate_manifest({"schema_version": SCHEMA_VERSION, "run_id": "x", "mode": "simulation"})

    def test_validate_manifest_rejects_invalid_artifact_path(self) -> None:
        with self.assertRaises(ManifestError):
            validate_manifest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "x",
                    "mode": "simulation",
                    "created_at_utc": "2026-02-23T00:00:00Z",
                    "artifacts": [
                        {"id": "summary", "path": "../summary.json", "kind": "json", "required": True},
                    ],
                    "plots": [],
                }
            )

    def test_validate_manifest_rejects_duplicate_plot_id(self) -> None:
        with self.assertRaises(ManifestError):
            validate_manifest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "x",
                    "mode": "simulation",
                    "created_at_utc": "2026-02-23T00:00:00Z",
                    "artifacts": [{"id": "summary", "path": "summary.json", "kind": "json", "required": True}],
                    "plots": [
                        {"plot_id": "p0", "path": "plots/p0.png", "source_key": "h_nm"},
                        {"plot_id": "p0", "path": "plots/p1.png", "source_key": "h_nm"},
                    ],
                }
            )

    def test_validate_manifest_files_checks_required_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": "x",
                "mode": "simulation",
                "created_at_utc": "2026-02-23T00:00:00Z",
                "artifacts": [{"id": "summary", "path": "summary.json", "kind": "json", "required": True}],
                "plots": [],
            }
            with self.assertRaises(ManifestError):
                validate_manifest_files(Path(tmp), payload)

    def test_simulation_manifest_written_and_valid(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent = Path(tmp) / "fluent.npz"
            self._write_fluent(fluent)
            spec = compose_sim_config(
                "cvd_steady_min",
                overrides=[
                    f"sim.inputs.fluent.file={fluent}",
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=manifest_test",
                    "sim.output.run_name=manifest_test",
                ],
            )
            result = run_aib_from_spec(spec)
            run_dir = save_run_outputs(
                run_spec=spec,
                config_name="cvd_steady_min",
                config_overrides=[
                    f"sim.inputs.fluent.file={fluent}",
                    f"sim.output.root_dir={tmp}",
                    "sim.output.project=manifest_test",
                    "sim.output.run_name=manifest_test",
                ],
                result=result,
            )
            manifest_path = run_dir / "outputs" / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest(payload)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)

    def test_benchmark_manifest_written_and_valid(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_wafer2d_benchmark(
                config_name="cvd_steady_min",
                overrides=[f"sim.output.root_dir={tmp}", "sim.output.project=manifest_bench"],
                with_physviz=True,
                physviz_fast=True,
            )
            run_dir = Path(out["run_dir"])
            payload = json.loads((run_dir / "outputs" / "manifest.json").read_text(encoding="utf-8"))
            validate_manifest(payload)
            artifact_ids = {row["id"] for row in payload["artifacts"]}
            self.assertIn("ranking", artifact_ids)
            self.assertIn("class_compare", artifact_ids)


if __name__ == "__main__":
    unittest.main()
