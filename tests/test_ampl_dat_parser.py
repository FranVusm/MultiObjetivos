from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ampl_dat_parser import DataConsistencyWarning, load_ampl_dat


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


class TestAmplDatParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_read_sets(self) -> None:
        self.assertEqual(self.data.N, [1, 2, 3, 4, 5, 6, 7, 20])
        self.assertEqual(self.data.P, [1, 2, 3])
        self.assertEqual(self.data.K, [1, 2])
        self.assertEqual(self.data.V, [1, 2])

    def test_read_scalar_params(self) -> None:
        self.assertEqual(self.data.O, 1)
        self.assertEqual(self.data.q, 1500.0)

    def test_read_demand(self) -> None:
        self.assertEqual(self.data.d[1], 0.0)
        self.assertEqual(self.data.d[2], 200.0)
        self.assertEqual(self.data.d[7], 200.0)
        self.assertEqual(self.data.d[20], 0.0)

    def test_read_period_bounds(self) -> None:
        self.assertEqual(self.data.LI[1], 0.0)
        self.assertEqual(self.data.LI[3], 8.0)
        self.assertAlmostEqual(self.data.LS[1], 3.99)
        self.assertAlmostEqual(self.data.LS[3], 12.0)

    def test_read_sigma(self) -> None:
        self.assertAlmostEqual(self.data.sigma[(1, 1)], 0.1)
        self.assertAlmostEqual(self.data.sigma[(1, 2)], 0.9)
        self.assertAlmostEqual(self.data.sigma[(9, 1)], 0.9)
        self.assertAlmostEqual(self.data.sigma[(9, 2)], 0.1)

    def test_read_3d_matrix_param_e(self) -> None:
        self.assertEqual(self.data.e[(1, 1, 1)], 0.0)
        self.assertEqual(self.data.e[(1, 2, 1)], 150.0)
        self.assertEqual(self.data.e[(1, 20, 1)], 0.0)
        self.assertEqual(self.data.e[(20, 20, 1)], 0.0)
        self.assertIn((7, 20, 3), self.data.e)

    def test_detect_nodes_in_params_not_in_N(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataConsistencyWarning)
            load_ampl_dat(str(DATA_PATH))

        messages = [str(item.message) for item in caught]
        self.assertFalse(any("uses nodes not present in set N" in message for message in messages))

    def test_customers_are_positive_demand_nodes(self) -> None:
        self.assertEqual(self.data.customers, [2, 3, 4, 5, 6, 7])

    def test_rejects_positive_demand_nodes_not_declared_in_N(self) -> None:
        invalid_path = PROJECT_ROOT / "Modelo.dat"
        with self.assertRaisesRegex(ValueError, "nodes not present in set N"):
            load_ampl_dat(str(invalid_path))


if __name__ == "__main__":
    unittest.main()
