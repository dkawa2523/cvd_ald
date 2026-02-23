from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .input_builder import build_domain_from_fluent_xy, load_fluent_npz_v2


@unittest.skipIf(np is None, "NumPy is required")
class TestFluentLoader(unittest.TestCase):
    def test_load_steady(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "steady.npz"
            xy = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=float)
            cref = np.array([[1.0, 0.5], [0.2, 0.1]], dtype=float)
            np.savez(path, xy=xy, cref=cref)

            keys = type("Keys", (), {"cref": "cref", "xy": "xy", "time": "time"})()
            out = load_fluent_npz_v2(path=path, mode="steady", keys=keys, species=["s0", "s1"])
            self.assertEqual(out.cref.shape, (2, 2))
            grid = build_domain_from_fluent_xy(xy=out.xy, xy_unit="mm", wafer_radius_mm=10.0)
            self.assertEqual(grid.kind, "from_fluent_xy")
            self.assertEqual(grid.shape, (2,))

    def test_load_transient(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "transient.npz"
            xy = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=float)
            cref = np.array(
                [
                    [[1.0, 0.5], [0.2, 0.1]],
                    [[0.9, 0.4], [0.3, 0.2]],
                ],
                dtype=float,
            )
            time = np.array([0.0, 1.0], dtype=float)
            np.savez(path, xy=xy, cref=cref, time=time)

            keys = type("Keys", (), {"cref": "cref", "xy": "xy", "time": "time"})()
            out = load_fluent_npz_v2(path=path, mode="transient", keys=keys, species=["s0", "s1"])
            self.assertEqual(out.cref.shape, (2, 2, 2))
            self.assertEqual(out.time.shape, (2,))


if __name__ == "__main__":
    unittest.main()
