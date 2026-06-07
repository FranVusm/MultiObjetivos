from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import hypervolume_2d, igd, load_reference_front


class TestMetrics(unittest.TestCase):
    def test_hypervolume_2d_positive(self) -> None:
        points = [(1, 5), (2, 3), (4, 1)]
        self.assertGreater(hypervolume_2d(points, (5, 6)), 0)

    def test_igd_zero_when_same_front(self) -> None:
        points = [(1, 5), (2, 3), (4, 1)]
        self.assertEqual(igd(points, points), 0)

    def test_load_reference_front(self) -> None:
        path = PROJECT_ROOT / "tests" / "tmp_reference_front.txt"
        path.write_text("F1,F2\n930,141\n870,148\n", encoding="utf-8")
        try:
            self.assertEqual(load_reference_front(str(path)), [(930.0, 141.0), (870.0, 148.0)])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
