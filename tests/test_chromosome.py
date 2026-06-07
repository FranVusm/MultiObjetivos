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
from chromosome import (
    Individual,
    create_random_individual,
    get_trip_slots,
    random_capacity_feasible_cuts,
    split_by_cuts,
    validate_individual,
)


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


class TestChromosome(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_trip_slots_intermediate_case(self) -> None:
        self.assertEqual(get_trip_slots(self.data), [(1, 1), (2, 1), (1, 2), (2, 2)])

    def test_random_individual_contains_all_customers_once(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        self.assertEqual(sorted(ind.perm), sorted(self.data.customers))

    def test_random_individual_cuts_shape(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        self.assertEqual(ind.cuts[0], 0)
        self.assertEqual(ind.cuts[-1], len(ind.perm))
        self.assertEqual(len(ind.cuts), len(get_trip_slots(self.data)) + 1)

    def test_split_by_cuts_returns_all_slots(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        self.assertEqual(list(split_by_cuts(ind, self.data)), get_trip_slots(self.data))

    def test_alpha_has_all_slots(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        self.assertEqual(set(ind.alpha), set(get_trip_slots(self.data)))

    def test_validate_individual_accepts_valid_individual(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 2, 4, 5, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        validate_individual(ind, self.data)

    def test_validate_individual_rejects_repeated_customer(self) -> None:
        ind = Individual(
            perm=[2, 3, 3, 5, 6, 7],
            cuts=[0, 2, 4, 5, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        with self.assertRaises(ValueError):
            validate_individual(ind, self.data)

    def test_validate_individual_rejects_missing_customer(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6],
            cuts=[0, 2, 4, 5, 5],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        with self.assertRaises(ValueError):
            validate_individual(ind, self.data)

    def test_validate_individual_rejects_bad_cuts(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 3, 2, 6, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        with self.assertRaises(ValueError):
            validate_individual(ind, self.data)

    def test_validate_individual_rejects_capacity_violation(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 6, 6, 6, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        with self.assertRaises(ValueError):
            validate_individual(ind, self.data, check_capacity=True)

    def test_random_capacity_feasible_cuts_shape(self) -> None:
        perm = list(self.data.customers)
        cuts = random_capacity_feasible_cuts(perm, self.data, rng=42)
        self.assertEqual(cuts[0], 0)
        self.assertEqual(cuts[-1], len(perm))
        self.assertEqual(len(cuts), len(get_trip_slots(self.data)) + 1)

    def test_random_capacity_feasible_cuts_respects_capacity(self) -> None:
        perm = list(self.data.customers)
        cuts = random_capacity_feasible_cuts(perm, self.data, rng=42)
        for start, end in zip(cuts, cuts[1:]):
            demand = sum(self.data.d[customer] for customer in perm[start:end])
            self.assertLessEqual(demand, self.data.q)

    def test_random_capacity_feasible_cuts_can_generate_diversity(self) -> None:
        perm = list(self.data.customers)
        structures = {
            tuple(random_capacity_feasible_cuts(perm, self.data, rng=seed))
            for seed in range(30)
        }
        self.assertGreaterEqual(len(structures), 2)


if __name__ == "__main__":
    unittest.main()
