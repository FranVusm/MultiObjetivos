from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

from ampl_dat_parser import load_ampl_dat
from chromosome import Individual, get_trip_slots
from nsga2 import NSGA2Config, run_nsga2
from reporting import (
    generate_run_report,
    plot_sigma_pareto_points,
    unique_by_objectives,
    write_sigma_pareto_solutions_txt,
)


DEFAULT_DAT_PATH = "data/Modelo(1).dat"


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = _parse_args(argv)
    data_path = _resolve_dat_path(args.dat_file, project_root)
    print(f"Loading data from: {data_path}")
    data = load_ampl_dat(str(data_path))
    total_demand = sum(data.d[customer] for customer in data.customers)
    total_capacity = len(get_trip_slots(data)) * data.q
    if total_demand > total_capacity:
        print("Instance is infeasible before running NSGA-II.")
        print(f"Total customer demand: {total_demand}")
        print(f"Total available trip capacity: {total_capacity}")
        print(
            f"Increase vehicles K, trips V, or capacity q; or reduce the customers/demand in {data_path}."
        )
        return

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = project_root / "results" / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    rows = list(_sigma_rows(data))
    if not rows:
        rows = [(None, None)]

    summary_rows = []
    sigma_solution_candidates = []
    sweep_start = perf_counter()
    objective_normalization = _estimate_objective_normalization(data, output_root)
    for sigma_index, weights in rows:
        sigma_start = perf_counter()
        label = f"sigma_{sigma_index:02d}" if sigma_index is not None else "pareto"
        print(f"\nRunning {label} with weights={weights}")

        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        config = _make_config(data, sigma_index, weights, objective_normalization)
        config.progress_log_path = str(output_dir / "execution.log")
        population, pareto_front, history = run_nsga2(data, config)
        selected = _select_sigma_solution(pareto_front, weights, objective_normalization)
        report_paths = generate_run_report(
            output_dir=str(output_dir),
            data=data,
            config=config,
            population=population,
            pareto_front=pareto_front,
            history=history,
        )

        unique_front = unique_by_objectives(pareto_front)
        unique_count = len(unique_front)
        sigma_solution_candidates.extend(
            {
                "sigma": sigma_index,
                "weights": weights,
                "individual": ind,
            }
            for ind in unique_front
        )
        feasible_count = sum(1 for ind in population if ind.feasible)
        sigma_elapsed_seconds = perf_counter() - sigma_start
        summary_rows.append(
            {
                "sigma": sigma_index,
                "beta_f1": "" if weights is None else weights[0],
                "beta_f2": "" if weights is None else weights[1],
                "selected_f1": selected.objectives[0] if selected else "",
                "selected_f2": selected.objectives[1] if selected else "",
                "feasible_count": feasible_count,
                "unique_points": unique_count,
                "nsga2_elapsed_seconds": config.runtime_stats.get("elapsed_seconds", 0.0),
                "sigma_elapsed_seconds": sigma_elapsed_seconds,
                "label": label,
                "summary": report_paths["summary"],
            }
        )
        if selected is not None:
            print(
                f"{label}: selected F1={selected.objectives[0]} "
                f"F2={selected.objectives[1]} feasible={selected.feasible} "
                f"front_unique={unique_count} feasible_count={feasible_count} "
                f"elapsed_seconds={sigma_elapsed_seconds:.3f}"
            )

    total_elapsed_seconds = perf_counter() - sweep_start
    summary_path = _write_sigma_summary(output_root, summary_rows, total_elapsed_seconds)
    solutions_path = write_sigma_pareto_solutions_txt(
        output_dir=str(output_root),
        data=data,
        sigma_candidates=sigma_solution_candidates,
        sigma_timings=summary_rows,
        total_elapsed_seconds=total_elapsed_seconds,
        objective_normalization=objective_normalization,
    )
    sigma_plot_path = plot_sigma_pareto_points(
        output_dir=str(output_root),
        data=data,
        sigma_candidates=sigma_solution_candidates,
    )
    print(f"\nSigma sweep summary: {summary_path}")
    print(f"Sigma Pareto solutions: {solutions_path}")
    print(f"Sigma Pareto points plot: {sigma_plot_path}")
    print(f"Total elapsed seconds: {total_elapsed_seconds:.3f}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NSGA-II for a TD-MT-GVRP AMPL .dat instance.",
    )
    parser.add_argument(
        "dat_file",
        nargs="?",
        default=None,
        help=(
            "Path to the AMPL .dat file. Bare filenames are searched first in data/ "
            f"and then in the project root. Default: {DEFAULT_DAT_PATH}"
        ),
    )
    parser.add_argument(
        "--data",
        dest="data_option",
        default=None,
        help="Alternative way to pass the AMPL .dat file path.",
    )
    args = parser.parse_args(argv)
    if args.dat_file and args.data_option:
        parser.error("Use either the positional dat_file argument or --data, not both.")
    args.dat_file = args.data_option or args.dat_file or DEFAULT_DAT_PATH
    return args


def _resolve_dat_path(dat_file: str, project_root: Path) -> Path:
    path = Path(dat_file)
    if path.is_absolute():
        return path

    if len(path.parts) == 1:
        data_candidate = project_root / "data" / path
        if data_candidate.exists():
            return data_candidate

    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    if len(path.parts) == 1:
        return project_root / "data" / path
    return project_candidate


def _make_config(
    data,
    sigma_index: int | None,
    weights: tuple[float, float] | None,
    objective_normalization: dict | None = None,
) -> NSGA2Config:
    planning_end = max(data.LS.values())
    seed = 12 if sigma_index is None else 12 + sigma_index
    initial_departure_max = min(6.0, planning_end)
    max_wait_between_trips = min(0.5, planning_end)
    return NSGA2Config(
        population_size=40,
        generations=500,
        crossover_probability=0.9,
        mutation_probability=0.2,
        seed=seed,
        verbose=False,
        initial_departure_max=initial_departure_max,
        max_wait_between_trips=max_wait_between_trips,
        alpha_mutation_step=1.0,
        feasible_initialization_attempts=50,
        cut_reset_probability=0.35,
        cut_diversity_repair_probability=0.75,
        alpha_boundary_mutation_probability=0.45,
        alpha_random_reset_probability=0.1,
        local_search_method="vnd",
        local_search_rate=0.2,
        local_search_generations_interval=5,
        local_search_max_individuals=10,
        local_search_max_iterations=50,
        local_search_neighbors=30,
        local_search_evaluation_budget=4000,
        evaluation_cache_enabled=True,
        alpha_signature_precision=4,
        deduplicate_environmental_selection=True,
        max_evaluations=None,
        early_stopping_generations=100,
        early_stopping_min_delta=1e-6,
        objective_weights=weights,
        objective_normalization=objective_normalization,
        sigma_index=sigma_index,
    )


def _estimate_objective_normalization(data, output_root: Path) -> dict:
    probes = [
        ("normalization_f1", "min_f1", (1.0, 0.0), -101),
        ("normalization_f2", "min_f2", (0.0, 1.0), -102),
    ]
    records = {}
    for label, key, weights, sigma_index in probes:
        probe_start = perf_counter()
        print(f"\nRunning {label} with weights={weights}")
        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        config = _make_config(data, sigma_index, weights)
        config.progress_log_path = str(output_dir / "execution.log")
        population, pareto_front, history = run_nsga2(data, config)
        report_paths = generate_run_report(
            output_dir=str(output_dir),
            data=data,
            config=config,
            population=population,
            pareto_front=pareto_front,
            history=history,
        )
        selected = _select_extreme_for_normalization(population + pareto_front, key)
        elapsed = perf_counter() - probe_start
        records[key] = {
            "label": label,
            "weights": weights,
            "f1": selected.objectives[0],
            "f2": selected.objectives[1],
            "feasible": selected.feasible,
            "nsga2_elapsed_seconds": config.runtime_stats.get("elapsed_seconds", 0.0),
            "elapsed_seconds": elapsed,
            "summary": report_paths["summary"],
        }
        print(
            f"{label}: selected F1={selected.objectives[0]} F2={selected.objectives[1]} "
            f"feasible={selected.feasible} elapsed_seconds={elapsed:.3f}"
        )

    min_f1 = records["min_f1"]
    min_f2 = records["min_f2"]
    normalization = {
        "ideal_f1": min_f1["f1"],
        "anti_f1": min_f2["f1"],
        "ideal_f2": min_f2["f2"],
        "anti_f2": min_f1["f2"],
        "source": "normalization_f1_and_normalization_f2",
        "min_f1_run": min_f1,
        "min_f2_run": min_f2,
    }
    _repair_degenerate_normalization(normalization, [min_f1, min_f2])
    _write_objective_normalization(output_root, normalization)
    print(
        "Objective normalization: "
        f"ideal_f1={normalization['ideal_f1']} anti_f1={normalization['anti_f1']} "
        f"ideal_f2={normalization['ideal_f2']} anti_f2={normalization['anti_f2']}"
    )
    return normalization


def _select_extreme_for_normalization(candidates: list[Individual], objective_key: str) -> Individual:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise ValueError(f"Cannot estimate objective normalization for {objective_key}: no feasible candidates")
    if objective_key == "min_f1":
        return min(feasible, key=lambda ind: (ind.objectives[0], ind.objectives[1]))
    if objective_key == "min_f2":
        return min(feasible, key=lambda ind: (ind.objectives[1], ind.objectives[0]))
    raise ValueError(f"Unknown normalization objective key: {objective_key}")


def _repair_degenerate_normalization(normalization: dict, records: list[dict]) -> None:
    if normalization["anti_f1"] <= normalization["ideal_f1"]:
        normalization["anti_f1"] = max(record["f1"] for record in records)
    if normalization["anti_f2"] <= normalization["ideal_f2"]:
        normalization["anti_f2"] = max(record["f2"] for record in records)
    if normalization["anti_f1"] <= normalization["ideal_f1"]:
        normalization["anti_f1"] = normalization["ideal_f1"] + 1.0
    if normalization["anti_f2"] <= normalization["ideal_f2"]:
        normalization["anti_f2"] = normalization["ideal_f2"] + 1.0


def _write_objective_normalization(output_root: Path, normalization: dict) -> str:
    output_path = output_root / "objective_normalization.json"
    output_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    return str(output_path)


def _sigma_rows(data) -> Iterable[tuple[int, tuple[float, float]]]:
    rows = sorted({row for row, column in data.sigma if column in (1, 2)})
    for row in rows:
        if (row, 1) in data.sigma and (row, 2) in data.sigma:
            yield row, (float(data.sigma[(row, 1)]), float(data.sigma[(row, 2)]))


def _select_sigma_solution(
    pareto_front: list[Individual],
    weights: tuple[float, float] | None,
    objective_normalization: dict | None = None,
) -> Individual | None:
    unique_front = unique_by_objectives(pareto_front)
    if not unique_front:
        return None
    if weights is None:
        return min(unique_front, key=lambda ind: (ind.objectives[0], ind.objectives[1]))

    if objective_normalization:
        min_f1 = float(objective_normalization["ideal_f1"])
        max_f1 = float(objective_normalization["anti_f1"])
        min_f2 = float(objective_normalization["ideal_f2"])
        max_f2 = float(objective_normalization["anti_f2"])
    else:
        min_f1 = min(ind.objectives[0] for ind in unique_front)
        max_f1 = max(ind.objectives[0] for ind in unique_front)
        min_f2 = min(ind.objectives[1] for ind in unique_front)
        max_f2 = max(ind.objectives[1] for ind in unique_front)
    range_f1 = max_f1 - min_f1
    range_f2 = max_f2 - min_f2

    def score(ind: Individual) -> float:
        norm_f1 = 0.0 if range_f1 == 0 else (ind.objectives[0] - min_f1) / range_f1
        norm_f2 = 0.0 if range_f2 == 0 else (ind.objectives[1] - min_f2) / range_f2
        return weights[0] * norm_f1 + weights[1] * norm_f2

    return min(unique_front, key=lambda ind: (score(ind), ind.objectives[0], ind.objectives[1]))


def _write_sigma_summary(output_root: Path, rows: list[dict], total_elapsed_seconds: float) -> str:
    output_path = output_root / "sigma_sweep_summary.csv"
    header = (
        "sigma,beta_f1,beta_f2,selected_f1,selected_f2,"
        "feasible_count,unique_points,nsga2_elapsed_seconds,"
        "sigma_elapsed_seconds,total_elapsed_seconds,summary"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                str(row.get(key, ""))
                for key in (
                    "sigma",
                    "beta_f1",
                    "beta_f2",
                    "selected_f1",
                    "selected_f2",
                    "feasible_count",
                    "unique_points",
                    "nsga2_elapsed_seconds",
                    "sigma_elapsed_seconds",
                    "total_elapsed_seconds",
                    "summary",
                )
            )
        )
    lines.append(
        ",".join(
            str(value)
            for value in (
                "TOTAL",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                total_elapsed_seconds,
                "",
            )
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)


if __name__ == "__main__":
    main()
