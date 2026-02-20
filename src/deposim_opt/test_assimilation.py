from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .assimilate import run_synthetic_assimilation


class TestAssimilation(unittest.TestCase):
    def test_synthetic_assimilation_reduces_loss_and_saves_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            out = run_synthetic_assimilation(
                sim_config_name="smoke",
                output_dir=tmp,
                target_k0=1.7,
                initial_k0=0.5,
                max_iters=10,
                sim_overrides=[
                    "domain.nr=6",
                    "domain.ntheta=12",
                    "time.process_time_s=2.0",
                ],
            )
            self.assertLess(out["final_loss"], out["initial_loss"])

            out_dir = Path(tmp)
            self.assertTrue((out_dir / "fitted_params.json").exists())
            self.assertTrue((out_dir / "assimilation_summary.json").exists())
            self.assertTrue((out_dir / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
