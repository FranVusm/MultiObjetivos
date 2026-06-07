from __future__ import annotations

import random
import sys
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ampl_dat_parser import DataConsistencyWarning, load_ampl_dat
from chromosome import create_random_individual, validate_individual
from local_search import alns_improve, evaluate_candidate, lns_improve, vnd_improve
from nsga2 import NSGA2Config


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


class TestLocalSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def _config(self) -> NSGA2Config:
        return NSGA2Config(
            initial_departure_max=12.0,
            max_wait_between_trips=12.0,
            local_search_max_iterations=3,
            local_search_neighbors=5,
            verbose=False,
        )

    def _individual(self):
        ind = create_random_individual(
            self.data,
            rng=42,
            initial_departure_max=12.0,
            max_wait_between_trips=12.0,
        )
        return evaluate_candidate(ind, self.data, self._config())

    def test_vnd_does_not_break_feasibility(self) -> None:
        improved = vnd_improve(
            self._individual(),
            self.data,
            "balanced",
            rng=random.Random(1),
            max_iterations=3,
            max_neighbors_per_operator=5,
            config=self._config(),
        )
        self.assertTrue(improved.feasible)

    def test_vnd_improvement_not_worse_for_f1(self) -> None:
        ind = self._individual()
        improved = vnd_improve(
            ind,
            self.data,
            "f1",
            rng=random.Random(2),
            max_iterations=3,
            max_neighbors_per_operator=5,
            config=self._config(),
        )
        if ind.feasible and improved.feasible:
            self.assertLessEqual(improved.objectives[0], ind.objectives[0])

    def test_vnd_improvement_not_worse_for_f2(self) -> None:
        ind = self._individual()
        improved = vnd_improve(
            ind,
            self.data,
            "f2",
            rng=random.Random(3),
            max_iterations=3,
            max_neighbors_per_operator=5,
            config=self._config(),
        )
        if ind.feasible and improved.feasible:
            self.assertLessEqual(improved.objectives[1], ind.objectives[1])

    def test_lns_does_not_break_structure(self) -> None:
        improved = lns_improve(
            self._individual(),
            self.data,
            "balanced",
            rng=random.Random(4),
            attempts=3,
            config=self._config(),
        )
        validate_individual(improved, self.data, check_capacity=False)

    def test_alns_does_not_break_structure(self) -> None:
        improved = alns_improve(
            self._individual(),
            self.data,
            "balanced",
            rng=random.Random(5),
            iterations=3,
            config=self._config(),
        )
        validate_individual(improved, self.data, check_capacity=False)


if __name__ == "__main__":
    unittest.main()
