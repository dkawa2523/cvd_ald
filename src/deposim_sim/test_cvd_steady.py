from __future__ import annotations

import unittest


@unittest.skip("legacy cvd_steady tests are isolated; active path is pipeline AIB tests")
class TestLegacyCVDSteady(unittest.TestCase):
    def test_legacy_placeholder(self) -> None:
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
