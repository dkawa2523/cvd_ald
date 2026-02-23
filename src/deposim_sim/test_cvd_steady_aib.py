from __future__ import annotations

import unittest

from .physics.cvd_steady import run_cvd_steady


class TestLegacyCVDSteadyRetired(unittest.TestCase):
    def test_legacy_path_is_retired(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "legacy path was retired"):
            run_cvd_steady()


if __name__ == "__main__":
    unittest.main()
