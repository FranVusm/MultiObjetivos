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
from chromosome import Individual
from decoder import build_routes, decode_individual, locate_period


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"


class TestDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def fixed_individual(self) -> Individual:
        return Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 2, 4, 5, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.0, (1, 2): 0.0, (2, 2): 0.0},
        )

    def test_locate_period(self) -> None:
        self.assertEqual(locate_period(0.0, self.data), 1)
        self.assertEqual(locate_period(3.99, self.data), 1)
        self.assertEqual(locate_period(4.0, self.data), 2)
        self.assertEqual(locate_period(7.99, self.data), 2)
        self.assertEqual(locate_period(8.0, self.data), 3)
        self.assertEqual(locate_period(12.0, self.data), 3)
        self.assertIsNone(locate_period(12.1, self.data))

    def test_build_routes_from_fixed_individual(self) -> None:
        routes = build_routes(self.fixed_individual(), self.data)
        self.assertEqual(routes[(1, 1)], [1, 2, 3, 20])
        self.assertEqual(routes[(2, 1)], [1, 4, 5, 20])
        self.assertEqual(routes[(1, 2)], [1, 6, 20])
        self.assertEqual(routes[(2, 2)], [1, 7, 20])

    def test_decode_builds_Y(self) -> None:
        decoded = decode_individual(self.fixed_individual(), self.data)
        self.assertEqual(decoded.Y[(2, 1, 1)], 1)
        self.assertEqual(decoded.Y[(3, 1, 1)], 1)
        self.assertEqual(decoded.Y[(4, 2, 1)], 1)
        self.assertEqual(decoded.Y[(5, 2, 1)], 1)
        self.assertEqual(decoded.Y[(6, 1, 2)], 1)
        self.assertEqual(decoded.Y[(7, 2, 2)], 1)

    def test_decode_builds_sparse_X(self) -> None:
        decoded = decode_individual(self.fixed_individual(), self.data)
        self.assertEqual(len(decoded.X), 10)

    def test_decode_times_are_non_decreasing_inside_routes(self) -> None:
        decoded = decode_individual(self.fixed_individual(), self.data)
        for row in decoded.arc_periods:
            self.assertGreaterEqual(row["arrival_or_departure_next"], row["depart_time"])

    def test_decode_objectives_match_sparse_X_sum(self) -> None:
        decoded = decode_individual(self.fixed_individual(), self.data)
        manual_F1 = sum(
            self.data.e[(i, j, p)] + self.data.ee[(j, p)]
            for i, j, p, _, _ in decoded.X
        )
        manual_F2 = sum(
            self.data.g[(i, j, p)] + self.data.gg[(j, p)]
            for i, j, p, _, _ in decoded.X
        )
        self.assertEqual(decoded.F1, manual_F1)
        self.assertEqual(decoded.F2, manual_F2)

    def test_decode_updates_individual_objectives(self) -> None:
        ind = self.fixed_individual()
        decoded = decode_individual(ind, self.data)
        self.assertEqual(ind.objectives, (decoded.F1, decoded.F2))
        self.assertEqual(ind.feasible, decoded.feasible)

    def test_decode_capacity_violation(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 6, 6, 6, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.0, (1, 2): 0.0, (2, 2): 0.0},
        )
        decoded = decode_individual(ind, self.data)
        self.assertFalse(decoded.feasible)
        self.assertTrue(any("Capacity violation" in item for item in decoded.violations))

    def test_decode_rejects_active_trip_after_empty_trip(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 0, 3, 6, 6],
            alpha={(1, 1): 0.0, (2, 1): 0.0, (1, 2): 0.0, (2, 2): 0.0},
        )
        decoded = decode_individual(ind, self.data)
        self.assertFalse(decoded.feasible)
        self.assertTrue(any("Trip prefix violation" in item for item in decoded.violations))

    def test_decode_time_outside_periods_marks_infeasible(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 2, 4, 5, 6],
            alpha={(1, 1): 12.0, (2, 1): 12.0, (1, 2): 12.0, (2, 2): 12.0},
        )
        decoded = decode_individual(ind, self.data)
        self.assertFalse(decoded.feasible)
        self.assertTrue(
            any("Time outside periods" in item or "outside planning horizon" in item for item in decoded.violations)
        )

    def test_decode_rejects_service_completion_outside_selected_period(self) -> None:
        ind = Individual(
            perm=[2, 3, 4, 5, 6, 7],
            cuts=[0, 1, 3, 5, 6],
            alpha={(1, 1): 3.0, (2, 1): 0.0, (1, 2): 0.0, (2, 2): 0.0},
        )
        decoded = decode_individual(ind, self.data)
        self.assertFalse(decoded.feasible)
        self.assertTrue(
            any("Service completion outside selected period" in item for item in decoded.violations)
        )


if __name__ == "__main__":
    unittest.main()
