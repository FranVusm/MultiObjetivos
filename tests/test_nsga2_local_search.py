from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ampl_dat_parser import DataConsistencyWarning, load_ampl_dat
from nsga2 import NSGA2Config, run_nsga2


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


class TestNSGA2LocalSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_run_nsga2_with_local_search_returns_population_and_front(self) -> None:
        config = NSGA2Config(
            population_size=10,
            generations=5,
            seed=42,
            verbose=False,
            initial_departure_max=12.0,
            max_wait_between_trips=12.0,
            local_search_method="vnd",
            local_search_rate=0.2,
            local_search_generations_interval=2,
            local_search_max_iterations=2,
            local_search_neighbors=3,
        )
        population, front, history = run_nsga2(self.data, config)

        self.assertEqual(len(population), 10)
        self.assertGreaterEqual(len(front), 1)
        self.assertEqual(len(history), 5)
        self.assertTrue(all(math.isfinite(ind.objectives[0]) for ind in population))
        self.assertTrue(all(math.isfinite(ind.objectives[1]) for ind in population))


if __name__ == "__main__":
    unittest.main()
