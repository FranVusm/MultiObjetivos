from __future__ import annotations

import math
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
from chromosome import Individual, create_random_individual, get_trip_slots, validate_individual
from nsga2 import (
    NSGA2Config,
    assign_crowding_distance,
    dominates,
    fast_non_dominated_sort,
    mutate,
    ordered_crossover,
    repair_cuts_capacity_aware,
    run_nsga2,
)


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


def make_ind(objectives: tuple[float, float], feasible: bool = True) -> Individual:
    ind = Individual(perm=[], cuts=[], alpha={})
    ind.objectives = objectives
    ind.feasible = feasible
    return ind


class TestNSGA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_dominates_minimization(self) -> None:
        a = make_ind((100, 200))
        b = make_ind((120, 250))
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_dominates_tradeoff(self) -> None:
        a = make_ind((100, 300))
        b = make_ind((120, 200))
        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_fast_non_dominated_sort_assigns_ranks(self) -> None:
        best = make_ind((100, 100))
        dominated_one = make_ind((120, 120))
        dominated_two = make_ind((140, 140))
        tradeoff = make_ind((90, 130))

        fronts = fast_non_dominated_sort([best, dominated_one, dominated_two, tradeoff])

        self.assertEqual(best.rank, 0)
        self.assertEqual(tradeoff.rank, 0)
        self.assertEqual(dominated_one.rank, 1)
        self.assertEqual(dominated_two.rank, 2)
        self.assertEqual(len(fronts[0]), 2)

    def test_crowding_distance_assigns_infinity_to_extremes(self) -> None:
        front = [make_ind((100, 300)), make_ind((150, 200)), make_ind((200, 100))]
        assign_crowding_distance(front)
        self.assertEqual(sum(math.isinf(ind.crowding_distance) for ind in front), 2)

    def test_ordered_crossover_preserves_customers(self) -> None:
        parent1 = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 2, 4, 5, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.5, (1, 2): 1.0, (2, 2): 0.25},
        )
        parent2 = Individual(
            perm=[5, 2, 7, 3, 6, 4],
            cuts=[0, 3, 5, 6, 6],
            alpha={(1, 1): 0.2, (2, 1): 0.3, (1, 2): 0.4, (2, 2): 0.5},
        )

        child1, child2 = ordered_crossover(parent1, parent2, self.data, random.Random(42))

        for child in (child1, child2):
            self.assertEqual(sorted(child.perm), sorted(self.data.customers))
            self.assertEqual(len(set(child.perm)), len(self.data.customers))
            validate_individual(child, self.data, check_capacity=False)
            self.assertEqual(set(child.alpha), set(get_trip_slots(self.data)))

    def test_mutation_preserves_valid_structure(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        config = NSGA2Config(mutation_probability=1.0)
        mutated = mutate(ind, self.data, random.Random(7), config)
        validate_individual(mutated, self.data, check_capacity=False)

    def test_cut_reset_mutation_preserves_validity(self) -> None:
        ind = create_random_individual(self.data, rng=42)
        config = NSGA2Config(mutation_probability=0.0, cut_reset_probability=1.0)
        mutated = mutate(ind, self.data, random.Random(7), config)
        validate_individual(mutated, self.data, check_capacity=True)

    def test_repair_cuts_capacity_aware_shape(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 5, 2],
            alpha={(1, 1): 0.0, (2, 1): 0.0, (1, 2): 0.0, (2, 2): 0.0},
        )
        repaired = repair_cuts_capacity_aware(ind, self.data)
        self.assertEqual(repaired.cuts[0], 0)
        self.assertEqual(repaired.cuts[-1], len(repaired.perm))
        self.assertEqual(len(repaired.cuts), len(get_trip_slots(self.data)) + 1)
        self.assertTrue(all(left <= right for left, right in zip(repaired.cuts, repaired.cuts[1:])))

    def test_run_nsga2_returns_population_and_front(self) -> None:
        config = NSGA2Config(
            population_size=10,
            generations=3,
            seed=42,
            verbose=False,
        )
        population, front, history = run_nsga2(self.data, config)

        self.assertEqual(len(population), 10)
        self.assertGreaterEqual(len(front), 1)
        self.assertEqual(len(history), 3)
        self.assertTrue(all(math.isfinite(ind.objectives[0]) for ind in population))
        self.assertTrue(all(math.isfinite(ind.objectives[1]) for ind in population))
        self.assertTrue(all(ind.rank is not None for ind in population))
        self.assertTrue(all("front0_size" in row for row in history))

    def test_run_nsga2_writes_progress_log_when_configured(self) -> None:
        output_path = PROJECT_ROOT / "results" / "nsga2_progress_test.log"
        if output_path.exists():
            output_path.unlink()
        config = NSGA2Config(
            population_size=10,
            generations=2,
            seed=42,
            verbose=False,
            progress_log_path=str(output_path),
        )
        try:
            run_nsga2(self.data, config)

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("NSGA-II execution log", content)
            self.assertIn("Generation 1/2", content)
            self.assertIn("Finished | elapsed_seconds=", content)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_random_alpha_uses_small_waits_after_first_trip(self) -> None:
        rng = random.Random(42)
        ind = create_random_individual(
            self.data,
            rng=rng,
            initial_departure_max=1.0,
            max_wait_between_trips=0.25,
        )
        first_trip = min(self.data.V)
        for (trip, _), value in ind.alpha.items():
            if trip == first_trip:
                self.assertLessEqual(value, 1.0)
            else:
                self.assertLessEqual(value, 0.25)

    def test_boundary_alpha_mutation_can_create_long_wait(self) -> None:
        ind = Individual(
            perm=list(self.data.customers),
            cuts=[0, 2, 4, 5, 6],
            alpha={slot: 0.0 for slot in get_trip_slots(self.data)},
        )
        config = NSGA2Config(
            mutation_probability=1.0,
            initial_departure_max=12.0,
            max_wait_between_trips=12.0,
            alpha_mutation_step=1.0,
            alpha_boundary_mutation_probability=1.0,
            alpha_random_reset_probability=0.0,
            cut_reset_probability=0.0,
        )
        mutated = mutate(ind, self.data, random.Random(3), config)

        validate_individual(
            mutated,
            self.data,
            check_capacity=False,
            initial_departure_max=12.0,
            max_wait_between_trips=12.0,
        )
        self.assertTrue(any(value > 0.25 for value in mutated.alpha.values()))


if __name__ == "__main__":
    unittest.main()
