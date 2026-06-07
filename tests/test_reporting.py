from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ampl_dat_parser import DataConsistencyWarning, load_ampl_dat
from chromosome import Individual
from nsga2 import NSGA2Config, run_nsga2
from reporting import (
    generate_run_report,
    select_extreme_solutions,
    unique_by_objectives,
    write_sigma_pareto_solutions_txt,
)


DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "Modelo_intermedio.dat"
TEMP_ROOT = Path("C:/tmp")


def make_ind(objectives: tuple[float, float]) -> Individual:
    ind = Individual(perm=[], cuts=[], alpha={})
    ind.objectives = objectives
    ind.feasible = True
    return ind


class TestReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataConsistencyWarning)
            cls.data = load_ampl_dat(str(DATA_PATH))

    def test_unique_by_objectives_removes_duplicates(self) -> None:
        front = [make_ind((100, 200)), make_ind((100, 200)), make_ind((120, 180))]
        self.assertEqual(len(unique_by_objectives(front)), 2)

    def test_select_extreme_solutions(self) -> None:
        a = make_ind((100, 300))
        b = make_ind((200, 100))
        c = make_ind((150, 150))
        extremes = select_extreme_solutions([a, b, c])
        self.assertIs(extremes["min_f1"], a)
        self.assertIs(extremes["min_f2"], b)
        self.assertIsNotNone(extremes["compromise"])

    def test_generate_run_report_creates_files(self) -> None:
        config = NSGA2Config(population_size=10, generations=3, seed=42, verbose=False)
        population, front, history = run_nsga2(self.data, config)
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp_dir:
            report_paths = generate_run_report(tmp_dir, self.data, config, population, front, history)
            expected_names = {
                "summary.txt",
                "pareto_points.txt",
                "generation_history.txt",
                "pareto_front.png",
                "best_f1_history.png",
                "best_f2_history.png",
                "front0_size_history.png",
                "feasible_count_history.png",
            }
            self.assertEqual({Path(path).name for path in report_paths.values()}, expected_names)
            self.assertTrue(all(Path(path).exists() for path in report_paths.values()))

    def test_generation_history_txt_has_header(self) -> None:
        config = NSGA2Config(population_size=10, generations=3, seed=42, verbose=False)
        population, front, history = run_nsga2(self.data, config)
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp_dir:
            report_paths = generate_run_report(tmp_dir, self.data, config, population, front, history)
            content = Path(report_paths["generation_history"]).read_text(encoding="utf-8")
            self.assertIn("generation,front0_size,best_f1,best_f2", content)

    def test_summary_contains_key_sections(self) -> None:
        config = NSGA2Config(population_size=10, generations=3, seed=42, verbose=False)
        population, front, history = run_nsga2(self.data, config)
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp_dir:
            report_paths = generate_run_report(tmp_dir, self.data, config, population, front, history)
            content = Path(report_paths["summary"]).read_text(encoding="utf-8")
            self.assertIn("TD-MT-GVRP NSGA-II Run Summary", content)
            self.assertIn("Problem data", content)
            self.assertIn("NSGA-II configuration", content)
            self.assertIn("Run summary", content)
            self.assertIn("Unique Pareto points", content)

    def test_sigma_pareto_solutions_txt_deduplicates_final_representations(self) -> None:
        config = NSGA2Config(population_size=10, generations=3, seed=42, verbose=False)
        _population, front, _history = run_nsga2(self.data, config)
        candidate = {
            "sigma": 1,
            "weights": (0.5, 0.5),
            "individual": front[0],
        }
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp_dir:
            output_path = write_sigma_pareto_solutions_txt(
                tmp_dir,
                self.data,
                [candidate, dict(candidate)],
            )
            content = Path(output_path).read_text(encoding="utf-8")

            self.assertIn("TD-MT-GVRP Sigma Sweep Pareto Solutions", content)
            self.assertIn("Best environmental solution (min F1)", content)
            self.assertIn("Best time/cost solution (min F2)", content)
            self.assertIn("Most balanced solution", content)
            self.assertIn("Global Pareto front - non-dominated final representations", content)
            self.assertIn("Unique final representations: 1", content)
            self.assertEqual(content.count("--- Pareto solution"), 1)


if __name__ == "__main__":
    unittest.main()
