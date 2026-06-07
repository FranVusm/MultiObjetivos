from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import List

try:
    from .chromosome import Individual, get_trip_slots
    from .decoder import decode_individual
    from .metrics import (
        hypervolume_2d,
        igd,
        load_reference_front,
        normalize_points,
        objective_points,
    )
except ImportError:  # Allows running src/main.py directly.
    from chromosome import Individual, get_trip_slots
    from decoder import decode_individual
    from metrics import (
        hypervolume_2d,
        igd,
        load_reference_front,
        normalize_points,
        objective_points,
    )


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def unique_by_objectives(front: List[Individual]) -> List[Individual]:
    """Deduplicate a front by objective pair."""

    unique = {}
    for ind in front:
        unique.setdefault((ind.objectives[0], ind.objectives[1]), ind)
    return [unique[key] for key in sorted(unique)]


def select_extreme_solutions(unique_front: List[Individual], objective_normalization: dict | None = None) -> dict:
    """Select min-emission, min-cost and normalized compromise solutions."""

    if not unique_front:
        raise ValueError("Cannot select extreme solutions from an empty front")

    min_f1 = min(unique_front, key=lambda ind: (ind.objectives[0], ind.objectives[1]))
    min_f2 = min(unique_front, key=lambda ind: (ind.objectives[1], ind.objectives[0]))

    if objective_normalization:
        ideal_f1 = float(objective_normalization["ideal_f1"])
        ideal_f2 = float(objective_normalization["ideal_f2"])
        nadir_f1 = float(objective_normalization["anti_f1"])
        nadir_f2 = float(objective_normalization["anti_f2"])
    else:
        ideal_f1 = min(ind.objectives[0] for ind in unique_front)
        ideal_f2 = min(ind.objectives[1] for ind in unique_front)
        nadir_f1 = max(ind.objectives[0] for ind in unique_front)
        nadir_f2 = max(ind.objectives[1] for ind in unique_front)
    denom_f1 = nadir_f1 - ideal_f1
    denom_f2 = nadir_f2 - ideal_f2

    def normalized_distance(ind: Individual) -> float:
        norm_f1 = 0.0 if denom_f1 == 0 else (ind.objectives[0] - ideal_f1) / denom_f1
        norm_f2 = 0.0 if denom_f2 == 0 else (ind.objectives[1] - ideal_f2) / denom_f2
        return math.sqrt(norm_f1**2 + norm_f2**2)

    return {
        "min_f1": min_f1,
        "min_f2": min_f2,
        "compromise": min(unique_front, key=normalized_distance),
    }


def write_run_summary_txt(
    output_dir: str,
    data,
    config,
    population: List[Individual],
    pareto_front: List[Individual],
    history: List[dict],
    reference_front_path: str | None = None,
) -> str:
    """Write a human-readable run summary."""

    ensure_dir(output_dir)
    output_path = Path(output_dir) / "summary.txt"
    unique_front = unique_by_objectives(pareto_front)
    initial_stats = history[0] if history else {}
    final_stats = history[-1] if history else {}
    initial_best_f1 = initial_stats.get("best_f1", math.inf)
    initial_best_f2 = initial_stats.get("best_f2", math.inf)
    final_best_f1 = final_stats.get("best_f1", math.inf)
    final_best_f2 = final_stats.get("best_f2", math.inf)
    quality = _quality_metrics(pareto_front, reference_front_path or _default_reference_front_path(output_dir))

    lines = [
        "TD-MT-GVRP NSGA-II Run Summary",
        "",
        "Problem data",
        f"Nodes: {data.N}",
        f"Customers: {data.customers}",
        f"Periods: {data.P}",
        f"Vehicles: {data.K}",
        f"Trips: {data.V}",
        f"Depot: {data.depot}",
        f"Dummy depot: {data.dummy_depot}",
        f"Capacity: {data.q}",
        "",
        "NSGA-II configuration",
        f"Population size: {config.population_size}",
        f"Generations: {config.generations}",
        f"Crossover probability: {config.crossover_probability}",
        f"Mutation probability: {config.mutation_probability}",
        f"Seed: {config.seed}",
        f"Sigma index: {getattr(config, 'sigma_index', None)}",
        f"Objective weights (F1,F2): {getattr(config, 'objective_weights', None)}",
        f"Initial departure max: {getattr(config, 'initial_departure_max', None)}",
        f"Max wait between trips: {getattr(config, 'max_wait_between_trips', None)}",
        f"Alpha mutation step: {getattr(config, 'alpha_mutation_step', None)}",
        f"Cut reset probability: {getattr(config, 'cut_reset_probability', None)}",
        f"Cut diversity repair probability: {getattr(config, 'cut_diversity_repair_probability', None)}",
        "Alpha boundary mutation probability: "
        f"{getattr(config, 'alpha_boundary_mutation_probability', None)}",
        f"Evaluation cache enabled: {getattr(config, 'evaluation_cache_enabled', None)}",
        f"Alpha signature precision: {getattr(config, 'alpha_signature_precision', None)}",
        f"Deduplicate environmental selection: {getattr(config, 'deduplicate_environmental_selection', None)}",
        f"Max evaluations: {getattr(config, 'max_evaluations', None)}",
        f"Early stopping generations: {getattr(config, 'early_stopping_generations', None)}",
        f"Local search evaluation budget: {getattr(config, 'local_search_evaluation_budget', None)}",
        "",
        "Runtime stats",
        f"Elapsed seconds: {getattr(config, 'runtime_stats', {}).get('elapsed_seconds', 0.0)}",
        f"Evaluation requests: {getattr(config, 'runtime_stats', {}).get('evaluation_requests', 0)}",
        f"Real decoder evaluations: {getattr(config, 'runtime_stats', {}).get('evaluations', 0)}",
        f"Cache hits: {getattr(config, 'runtime_stats', {}).get('cache_hits', 0)}",
        f"Decoder time seconds: {getattr(config, 'runtime_stats', {}).get('decoder_time_seconds', 0.0)}",
        f"Local search evaluations: {getattr(config, 'runtime_stats', {}).get('local_search_evaluations', 0)}",
        f"Local search improvements: {getattr(config, 'runtime_stats', {}).get('local_search_improvements', 0)}",
        f"Duplicate candidates removed: {getattr(config, 'runtime_stats', {}).get('duplicate_candidates_removed', 0)}",
        f"Stopped early: {getattr(config, 'runtime_stats', {}).get('stopped_early', False)}",
        f"Stop reason: {getattr(config, 'runtime_stats', {}).get('stop_reason', '')}",
        "",
        "Run summary",
        f"Initial best F1: {initial_best_f1}",
        f"Initial best F2: {initial_best_f2}",
        f"Final best F1: {final_best_f1}",
        f"Final best F2: {final_best_f2}",
        f"Final feasible count: {final_stats.get('feasible_count', 0)}",
        f"Final Front0 raw size: {len(pareto_front)}",
        f"Final Front0 unique objective points: {len(unique_front)}",
        f"Unique cuts structures in final Pareto front: {len(_cuts_counts(pareto_front))}",
        "",
        "Quality metrics",
        f"Hypervolume: {quality['hypervolume']}",
        f"IGD: {quality['igd']}",
        f"Reference point: {quality['reference_point']}",
        f"Reference front file: {quality['reference_front_path']}",
        "",
    ]
    if quality["reference_warning"]:
        lines.extend([quality["reference_warning"], ""])
    if final_stats.get("feasible_count", 0) == 0:
        lines.extend(
            [
                "ERROR/WARNING: No feasible solution was found. The reported Pareto front is not valid for decision making.",
                "",
            ]
        )

    lines.extend([
        "Improvements",
        f"Best F1 improvement: {_format_improvement(initial_best_f1, final_best_f1)}",
        f"Best F2 improvement: {_format_improvement(initial_best_f2, final_best_f2)}",
        "",
    ])

    if unique_front:
        extremes = select_extreme_solutions(unique_front, getattr(config, "objective_normalization", None))
        lines.extend(_solution_section("Minimum emissions solution", extremes["min_f1"], data))
        lines.extend(_solution_section("Minimum cost solution", extremes["min_f2"], data))
        lines.extend(_solution_section("Compromise solution", extremes["compromise"], data))

    lines.extend(["Unique Pareto points"])
    for index, ind in enumerate(unique_front, start=1):
        lines.append(f"{index}. F1={ind.objectives[0]}, F2={ind.objectives[1]}")

    lines.extend(["", "Cuts structures"])
    for index, (cuts, count) in enumerate(_cuts_counts(pareto_front).items(), start=1):
        lines.append(f"{index}. {list(cuts)} count={count}")

    lines.extend(["", "Automatic notes"])
    lines.extend(_automatic_notes(config, pareto_front, unique_front, data))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)


def write_pareto_points_txt(output_dir: str, pareto_front: List[Individual]) -> str:
    """Write unique objective points and the raw front."""

    ensure_dir(output_dir)
    output_path = Path(output_dir) / "pareto_points.txt"
    lines = [
        "UNIQUE OBJECTIVE POINTS",
        "index,F1,F2,feasible,perm,cuts,alpha",
    ]
    for index, ind in enumerate(unique_by_objectives(pareto_front), start=1):
        lines.append(_individual_csv_line(index, ind))

    lines.extend(["", "RAW FRONT", "index,F1,F2,feasible,perm,cuts,alpha"])
    for index, ind in enumerate(pareto_front, start=1):
        lines.append(_individual_csv_line(index, ind))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)


def write_generation_history_txt(output_dir: str, history: List[dict]) -> str:
    """Write generation history as CSV-like text."""

    ensure_dir(output_dir)
    output_path = Path(output_dir) / "generation_history.txt"
    header_keys = (
        "generation",
        "front0_size",
        "best_f1",
        "best_f2",
        "avg_f1",
        "avg_f2",
        "feasible_count",
        "population_size",
        "front0_unique_objectives",
        "evaluations",
        "evaluation_requests",
        "cache_hits",
        "decoder_time_seconds",
        "elapsed_seconds",
        "hypervolume",
        "stagnation_generations",
    )
    header = ",".join(header_keys)
    lines = [header]
    for row in history:
        lines.append(",".join(str(row.get(key, "")) for key in header_keys))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)


def plot_pareto_front(output_dir: str, pareto_front: List[Individual]) -> str:
    """Plot raw and unique Pareto objective points."""

    plt = _get_pyplot()
    ensure_dir(output_dir)
    output_path = Path(output_dir) / "pareto_front.png"
    unique_front = unique_by_objectives(pareto_front)

    plt.figure(figsize=(7, 5))
    if pareto_front:
        plt.scatter(
            [ind.objectives[0] for ind in pareto_front],
            [ind.objectives[1] for ind in pareto_front],
            color="lightgray",
            alpha=0.55,
            label="Raw front",
        )
    if unique_front:
        plt.scatter(
            [ind.objectives[0] for ind in unique_front],
            [ind.objectives[1] for ind in unique_front],
            color="#0B6E69",
            label="Unique points",
        )
    plt.xlabel("F1 emissions")
    plt.ylabel("F2 cost")
    plt.title("Final Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_best_f1_history(output_dir: str, history: List[dict]) -> str:
    """Plot best F1 by generation."""

    return _plot_history(output_dir, history, "best_f1", "best_f1_history.png", "Best F1 History", "best F1")


def plot_best_f2_history(output_dir: str, history: List[dict]) -> str:
    """Plot best F2 by generation."""

    return _plot_history(output_dir, history, "best_f2", "best_f2_history.png", "Best F2 History", "best F2")


def plot_front0_size_history(output_dir: str, history: List[dict]) -> str:
    """Plot first-front size by generation."""

    return _plot_history(output_dir, history, "front0_size", "front0_size_history.png", "Front0 Size History", "Front0 size")


def plot_feasible_count_history(output_dir: str, history: List[dict]) -> str:
    """Plot feasible count by generation."""

    return _plot_history(output_dir, history, "feasible_count", "feasible_count_history.png", "Feasible Count History", "Feasible count")


def plot_hypervolume_history(output_dir: str, history: List[dict]) -> str:
    """Plot approximate hypervolume by generation."""

    return _plot_history(output_dir, history, "hypervolume", "hypervolume_history.png", "Hypervolume History", "hypervolume")


def write_config_json(output_dir: str, config) -> str:
    """Persist reproducible configuration and runtime stats."""

    ensure_dir(output_dir)
    output_path = Path(output_dir) / "config.json"
    payload = {
        "config": _config_to_dict(config),
        "runtime_stats": getattr(config, "runtime_stats", {}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def plot_pareto_with_reference(
    output_dir: str,
    pareto_front: List[Individual],
    reference_points: List[tuple[float, float]],
) -> str:
    """Plot NSGA-II Pareto points against an optional exact/reference front."""

    plt = _get_pyplot()
    ensure_dir(output_dir)
    output_path = Path(output_dir) / "pareto_front_with_reference.png"
    unique_front = unique_by_objectives(pareto_front)

    plt.figure(figsize=(7, 5))
    if unique_front:
        plt.scatter(
            [ind.objectives[0] for ind in unique_front],
            [ind.objectives[1] for ind in unique_front],
            color="#0B6E69",
            label="NSGA-II unique points",
        )
    if reference_points:
        plt.scatter(
            [point[0] for point in reference_points],
            [point[1] for point in reference_points],
            color="#C43E1C",
            marker="x",
            s=70,
            label="Reference front",
        )
    plt.xlabel("F1 emissions")
    plt.ylabel("F2 cost")
    plt.title("Pareto Front vs Reference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def generate_run_report(
    output_dir: str,
    data,
    config,
    population: List[Individual],
    pareto_front: List[Individual],
    history: List[dict],
    reference_front_path: str | None = None,
) -> dict:
    """Generate all text and plot reports for one run."""

    ensure_dir(output_dir)
    reference_path = reference_front_path or _default_reference_front_path(output_dir)
    reference_points = load_reference_front(reference_path)
    paths = {
        "summary": write_run_summary_txt(
            output_dir,
            data,
            config,
            population,
            pareto_front,
            history,
            reference_front_path=reference_path,
        ),
        "pareto_points": write_pareto_points_txt(output_dir, pareto_front),
        "generation_history": write_generation_history_txt(output_dir, history),
        "pareto_plot": plot_pareto_front(output_dir, pareto_front),
        "best_f1_plot": plot_best_f1_history(output_dir, history),
        "best_f2_plot": plot_best_f2_history(output_dir, history),
        "front0_size_plot": plot_front0_size_history(output_dir, history),
        "feasible_count_plot": plot_feasible_count_history(output_dir, history),
    }
    plot_hypervolume_history(output_dir, history)
    write_config_json(output_dir, config)
    if reference_points:
        paths["pareto_reference_plot"] = plot_pareto_with_reference(
            output_dir,
            pareto_front,
            reference_points,
        )
    return paths


def write_sigma_pareto_solutions_txt(
    output_dir: str,
    data,
    sigma_candidates: List[dict],
    sigma_timings: List[dict] | None = None,
    total_elapsed_seconds: float | None = None,
    objective_normalization: dict | None = None,
    filename: str = "sigma_pareto_solutions.txt",
    title: str = "TD-MT-GVRP Sigma Sweep Pareto Solutions",
) -> str:
    """Write the global non-dominated solutions found across all sigma runs."""

    ensure_dir(output_dir)
    output_path = Path(output_dir) / filename
    feasible_candidates = [
        candidate for candidate in sigma_candidates if candidate["individual"].feasible
    ]
    unique_candidates = _unique_sigma_candidates(feasible_candidates, data)
    pareto_candidates = _nondominated_sigma_candidates(unique_candidates)

    lines = [
        title,
        "",
        "Aggregate summary",
        f"Stored candidate solutions: {len(sigma_candidates)}",
        f"Feasible candidate solutions: {len(feasible_candidates)}",
        f"Unique final representations: {len(unique_candidates)}",
        f"Global non-dominated solutions: {len(pareto_candidates)}",
    ]
    if total_elapsed_seconds is not None:
        lines.append(f"Total elapsed seconds: {total_elapsed_seconds}")
    if objective_normalization:
        lines.extend(
            [
                "",
                "Objective normalization",
                f"Ideal F1: {objective_normalization['ideal_f1']}",
                f"Anti-ideal F1: {objective_normalization['anti_f1']}",
                f"Ideal F2: {objective_normalization['ideal_f2']}",
                f"Anti-ideal F2: {objective_normalization['anti_f2']}",
            ]
        )
    if sigma_timings:
        lines.extend(["", "Sigma runtime summary"])
        lines.append("sigma,beta_f1,beta_f2,nsga2_elapsed_seconds,sigma_elapsed_seconds")
        for row in sigma_timings:
            lines.append(
                ",".join(
                    str(row.get(key, ""))
                    for key in (
                        "sigma",
                        "beta_f1",
                        "beta_f2",
                        "nsga2_elapsed_seconds",
                        "sigma_elapsed_seconds",
                    )
                )
            )
    lines.append("")

    if not pareto_candidates:
        lines.append("No non-dominated solution was available.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(output_path)

    selected = _select_sigma_pareto_representatives(pareto_candidates, objective_normalization)
    lines.extend(["Selected solutions", ""])
    lines.extend(
        _sigma_solution_section(
            "Best environmental solution (min F1)",
            selected["min_f1"],
            data,
        )
    )
    lines.extend(
        _sigma_solution_section(
            "Best time/cost solution (min F2)",
            selected["min_f2"],
            data,
        )
    )
    lines.extend(
        _sigma_solution_section(
            "Most balanced solution",
            selected["compromise"],
            data,
        )
    )

    lines.extend(["Global Pareto front - non-dominated final representations", ""])
    for index, candidate in enumerate(pareto_candidates, start=1):
        lines.extend(_sigma_solution_section(f"Pareto solution {index}", candidate, data))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)


def plot_sigma_pareto_points(
    output_dir: str,
    data,
    sigma_candidates: List[dict],
) -> str:
    """Plot dominated and non-dominated objective points from sigma runs."""

    plt = _get_pyplot()
    ensure_dir(output_dir)
    output_path = Path(output_dir) / "sigma_pareto_points.png"
    feasible_candidates = [
        candidate for candidate in sigma_candidates if candidate["individual"].feasible
    ]
    unique_candidates = _unique_sigma_candidates(feasible_candidates, data)
    pareto_candidates = _nondominated_sigma_candidates(unique_candidates)
    pareto_ids = {id(candidate) for candidate in pareto_candidates}
    dominated_candidates = [
        candidate for candidate in unique_candidates if id(candidate) not in pareto_ids
    ]

    plt.figure(figsize=(7, 5))
    if dominated_candidates:
        plt.scatter(
            [candidate["individual"].objectives[0] for candidate in dominated_candidates],
            [candidate["individual"].objectives[1] for candidate in dominated_candidates],
            color="#549A96",
            alpha=0.75,
            s=36,
            label="Dominated solutions",
        )
    if pareto_candidates:
        plt.scatter(
            [candidate["individual"].objectives[0] for candidate in pareto_candidates],
            [candidate["individual"].objectives[1] for candidate in pareto_candidates],
            color="#0B6E69",
            s=44,
            label="Non-dominated solutions",
        )
    plt.xlabel("F1 emissions")
    plt.ylabel("F2 cost")
    plt.title("Sigma Sweep Pareto Points")
    plt.grid(True, alpha=0.3)
    if dominated_candidates or pareto_candidates:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def _quality_metrics(pareto_front: List[Individual], reference_front_path: str) -> dict:
    points = objective_points(unique_by_objectives(pareto_front), use_raw=True)
    reference_points = load_reference_front(reference_front_path)
    combined = points + reference_points
    if not points:
        return {
            "hypervolume": 0.0,
            "igd": None,
            "reference_point": None,
            "reference_front_path": reference_front_path,
            "reference_warning": "WARNING: Pareto front is empty; quality metrics are unavailable.",
        }

    ref_f1 = max(point[0] for point in combined) * 1.1
    ref_f2 = max(point[1] for point in combined) * 1.1
    reference_point = (ref_f1, ref_f2)
    hypervolume = hypervolume_2d(points, reference_point)

    reference_warning = ""
    igd_value = None
    if reference_points:
        ideal = (
            min(point[0] for point in combined),
            min(point[1] for point in combined),
        )
        nadir = (
            max(point[0] for point in combined),
            max(point[1] for point in combined),
        )
        igd_value = igd(
            normalize_points(points, ideal=ideal, nadir=nadir),
            normalize_points(reference_points, ideal=ideal, nadir=nadir),
        )
    else:
        reference_warning = (
            "WARNING: Reference front file not found or empty. IGD was not calculated."
        )

    return {
        "hypervolume": hypervolume,
        "igd": igd_value,
        "reference_point": reference_point,
        "reference_front_path": reference_front_path,
        "reference_warning": reference_warning,
    }


def _default_reference_front_path(output_dir: str) -> str:
    output_path = Path(output_dir)
    candidates = [output_path] + list(output_path.parents)
    for parent in candidates:
        candidate = parent / "data" / "gurobi_reference_front.txt"
        if candidate.exists():
            return str(candidate)
    if len(output_path.parents) >= 2:
        return str(output_path.parents[1] / "data" / "gurobi_reference_front.txt")
    return str(output_path / "data" / "gurobi_reference_front.txt")


def _get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for reports. Install it with: pip install matplotlib") from exc
    return plt


def _plot_history(
    output_dir: str,
    history: List[dict],
    key: str,
    filename: str,
    title: str,
    ylabel: str,
) -> str:
    plt = _get_pyplot()
    ensure_dir(output_dir)
    output_path = Path(output_dir) / filename
    if not history:
        output_path.write_bytes(b"")
        return str(output_path)
    generations = [row["generation"] for row in history]
    values = [row.get(key, 0.0) for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(generations, values, marker="o", linewidth=1.5)
    plt.xlabel("generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def _solution_section(title: str, ind: Individual, data) -> List[str]:
    decoded = decode_individual(ind, data, update_individual=False)
    raw_f1, raw_f2 = ind.metadata.get("raw_objectives", (decoded.F1, decoded.F2))
    violations = ind.metadata.get("violations", decoded.violations)
    lines = [
        f"--- {title} ---",
        f"Raw F1: {raw_f1}",
        f"Raw F2: {raw_f2}",
        f"Penalized F1: {ind.objectives[0]}",
        f"Penalized F2: {ind.objectives[1]}",
        f"Feasible: {ind.feasible}",
        "Violations:",
    ]
    if violations:
        lines.extend(f"  - {violation}" for violation in violations[:10])
    else:
        lines.append("  none")

    active_trips = {
        slot: customers for slot, customers in decoded.trips.items() if customers
    }
    empty_slots = [slot for slot, customers in decoded.trips.items() if not customers]

    lines.extend([
        f"Permutation: {ind.perm}",
        f"Cuts: {ind.cuts}",
        f"Alpha: {ind.alpha}",
        "Active trips:",
    ])
    for slot, customers in active_trips.items():
        demand = sum(data.d[customer] for customer in customers)
        lines.append(f"  {slot}: {customers} demand={demand}")
    if empty_slots:
        lines.append(f"Empty slots: {empty_slots}")
    lines.append("Active routes:")
    for slot, route in decoded.routes.items():
        if not route:
            continue
        lines.append(f"  {slot}: {route}")
    lines.append("")
    return lines


def _format_improvement(initial_value: float, final_value: float) -> str:
    if not math.isfinite(initial_value) or not math.isfinite(final_value):
        return "n/a"
    improvement = initial_value - final_value
    if initial_value == 0:
        return f"{improvement}"
    percentage = (improvement / initial_value) * 100.0
    return f"{improvement} ({percentage:.2f}%)"


def _automatic_notes(config, pareto_front: List[Individual], unique_front: List[Individual], data=None) -> List[str]:
    notes = []
    if data is not None:
        total_demand = sum(data.d[customer] for customer in data.customers)
        total_capacity = len(get_trip_slots(data)) * data.q
        if total_demand > total_capacity:
            notes.append(
                "WARNING: Total customer demand exceeds total available trip capacity. "
                f"Demand={total_demand}, capacity={total_capacity}. No feasible solution can exist without more trips, vehicles, or capacity."
            )
    if len(pareto_front) == config.population_size:
        notes.append(
            "WARNING: The first front contains the whole population. This may indicate weak selection pressure, too many trade-off solutions, or duplicated individuals."
        )
    if len(unique_front) < len(pareto_front):
        notes.append(
            "WARNING: Raw Pareto front has duplicated objective values. Use unique Pareto points for interpretation."
        )
    if pareto_front and len({tuple(ind.cuts) for ind in pareto_front}) == 1:
        notes.append(
            "WARNING: All Pareto individuals share the same cuts structure. The algorithm may require stronger cut mutation or repair diversity."
        )
    runtime = getattr(config, "runtime_stats", {})
    if runtime.get("stopped_early"):
        notes.append(f"INFO: Run stopped early: {runtime.get('stop_reason', '')}")
    if getattr(config, "evaluation_cache_enabled", False):
        requests = runtime.get("evaluation_requests", 0)
        hits = runtime.get("cache_hits", 0)
        if requests:
            notes.append(f"INFO: Evaluation cache hit rate: {(hits / requests) * 100.0:.2f}%.")
    if not notes:
        notes.append("No automatic warnings.")
    return notes


def _cuts_counts(front: List[Individual]) -> dict:
    counts = {}
    for ind in front:
        key = tuple(ind.cuts)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _individual_csv_line(index: int, ind: Individual) -> str:
    return f"{index},{ind.objectives[0]},{ind.objectives[1]},{ind.feasible},{ind.perm},{ind.cuts},{ind.alpha}"


def _unique_sigma_candidates(sigma_candidates: List[dict], data) -> List[dict]:
    unique = {}
    for candidate in sigma_candidates:
        ind = candidate["individual"]
        key = _operational_solution_signature(ind, data)
        source = _format_sigma_source(candidate)
        if key not in unique:
            stored = dict(candidate)
            stored["sources"] = [source]
            unique[key] = stored
        elif source not in unique[key]["sources"]:
            unique[key]["sources"].append(source)
    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate["individual"].objectives[0],
            candidate["individual"].objectives[1],
            str(candidate.get("sigma", "")),
        ),
    )


def _nondominated_sigma_candidates(candidates: List[dict]) -> List[dict]:
    front = []
    for index, candidate in enumerate(candidates):
        ind = candidate["individual"]
        if any(
            other_index != index
            and _dominates_objectives(other["individual"], ind)
            for other_index, other in enumerate(candidates)
        ):
            continue
        front.append(candidate)
    return sorted(
        front,
        key=lambda candidate: (
            candidate["individual"].objectives[0],
            candidate["individual"].objectives[1],
            str(candidate.get("sigma", "")),
        ),
    )


def _dominates_objectives(a: Individual, b: Individual) -> bool:
    return (
        a.objectives[0] <= b.objectives[0]
        and a.objectives[1] <= b.objectives[1]
        and (a.objectives[0] < b.objectives[0] or a.objectives[1] < b.objectives[1])
    )


def _select_sigma_pareto_representatives(candidates: List[dict], objective_normalization: dict | None = None) -> dict:
    selected_individuals = select_extreme_solutions(
        [candidate["individual"] for candidate in candidates],
        objective_normalization,
    )
    return {
        key: _candidate_for_individual(candidates, selected_individual)
        for key, selected_individual in selected_individuals.items()
    }


def _candidate_for_individual(candidates: List[dict], selected: Individual) -> dict:
    for candidate in candidates:
        if candidate["individual"] is selected:
            return candidate
    raise ValueError("Selected individual was not found in sigma candidates")


def _sigma_solution_section(title: str, candidate: dict, data) -> List[str]:
    section = _solution_section(title, candidate["individual"], data)
    section.insert(1, f"Source sigma runs: {'; '.join(candidate.get('sources', []))}")
    section.insert(2, "Final representation:")
    return section


def _format_sigma_source(candidate: dict) -> str:
    seed = candidate.get("seed")
    sigma = candidate.get("sigma")
    weights = candidate.get("weights")
    label = "pareto" if sigma is None else f"sigma_{sigma}"
    if seed not in (None, ""):
        label = f"seed_{int(seed):02d}/{label}"
    return f"{label}, weights={weights}"


def _operational_solution_signature(ind: Individual, data) -> tuple:
    decoded = decode_individual(ind, data, update_individual=False)
    active_routes = tuple(
        (slot, tuple(route))
        for slot, route in sorted(decoded.routes.items())
        if route
    )
    arc_timing = tuple(
        (
            row["slot"],
            row["arc"],
            row["period"],
            round(float(row["depart_time"]), 8),
        )
        for row in decoded.arc_periods
    )
    raw_f1, raw_f2 = ind.metadata.get("raw_objectives", ind.objectives)
    return (
        active_routes,
        arc_timing,
        round(float(raw_f1), 8),
        round(float(raw_f2), 8),
    )


def _config_to_dict(config) -> dict:
    if not is_dataclass(config):
        return dict(getattr(config, "__dict__", {}))
    output = {}
    for item in fields(config):
        if item.name.startswith("_") or item.name == "runtime_stats":
            continue
        output[item.name] = getattr(config, item.name)
    return output
