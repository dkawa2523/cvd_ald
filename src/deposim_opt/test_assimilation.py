from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .assimilate import run_synthetic_assimilation


class TestAssimilation(unittest.TestCase):
    def test_synthetic_assimilation_reduces_loss_and_saves_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            fluent_path = Path(tmp) / "fluent.npz"
            xy = np.array([[-10.0, -10.0], [0.0, 0.0], [15.0, 12.0], [25.0, -8.0]], dtype=float)
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

            out = run_synthetic_assimilation(
                sim_config_name="cvd_steady_min",
                output_dir=tmp,
                target_k0=0.018,
                initial_k0=0.004,
                max_iters=8,
                sim_overrides=[
                    f"sim.inputs.fluent.file={fluent_path}",
                    "sim.time.t_proc_s=3.0",
                ],
            )
            self.assertLessEqual(out["final_loss"], out["initial_loss"])

            out_dir = Path(tmp)
            self.assertTrue((out_dir / "fitted_params.json").exists())
            self.assertTrue((out_dir / "assimilation_summary.json").exists())
            self.assertTrue((out_dir / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
