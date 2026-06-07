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
from chromosome import Individual, get_trip_slots
from decoder import decode_individual
from nsga2 import repair_cuts_capacity_aware


DATA_PATH = PROJECT_ROOT / "data" / "Modelo(1).dat"


class TestDecoderFeasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_zero_alpha_diagnostic_has_no_capacity_violations(self) -> None:
        slots = get_trip_slots(self.data)
        ind = Individual(
            perm=list(self.data.customers),
            cuts=[0, len(self.data.customers)],
            alpha={slot: 0.0 for slot in slots},
        )
        ind = repair_cuts_capacity_aware(ind, self.data)
        decoded = decode_individual(ind, self.data)
        capacity_violations = [
            violation for violation in decoded.violations if "Capacity violation" in violation
        ]
        total_demand = sum(self.data.d[customer] for customer in self.data.customers)
        total_capacity = len(slots) * self.data.q
        if total_demand <= total_capacity:
            self.assertEqual(
                capacity_violations,
                [],
                msg="Zero-alpha diagnostic capacity violations: " + "; ".join(capacity_violations),
            )
        else:
            self.assertTrue(
                capacity_violations,
                msg=(
                    "Expected capacity violations because total demand exceeds total capacity: "
                    f"demand={total_demand}, capacity={total_capacity}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
