from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .chromosome import (
        Individual,
        create_random_individual,
        get_trip_slots,
        planning_horizon,
        random_capacity_feasible_cuts,
        validate_individual,
    )
    from .decoder import decode_individual
    from .evaluation import (
        clone_individual_fast,
        evaluate_candidate,
        evaluation_budget_reached,
        individual_signature,
        initialize_runtime,
        reset_runtime,
    )
    from .local_search import intensify_individual
except ImportError:  # Allows running src/main.py directly.
    from chromosome import (
        Individual,
        create_random_individual,
        get_trip_slots,
        planning_horizon,
        random_capacity_feasible_cuts,
        validate_individual,
    )
    from decoder import decode_individual
    from evaluation import (
        clone_individual_fast,
        evaluate_candidate,
        evaluation_budget_reached,
        individual_signature,
        initialize_runtime,
        reset_runtime,
    )
    from local_search import intensify_individual


@dataclass
class NSGA2Config:
    population_size: int = 50
    generations: int = 100
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    seed: Optional[int] = None
    tournament_size: int = 2
    capacity_aware_initialization: bool = True
    penalty_value: float = 1_000_000.0
    verbose: bool = True
    initial_departure_max: Optional[float] = None
    max_wait_between_trips: float = 0.25
    alpha_mutation_step: float = 0.25
    feasible_initialization_attempts: int = 50
    cut_reset_probability: float = 0.1
    alpha_boundary_mutation_probability: float = 0.35
    alpha_random_reset_probability: float = 0.1
    local_search_method: str = "vnd"
    local_search_rate: float = 0.2
    local_search_generations_interval: int = 5
    local_search_max_individuals: int = 10
    local_search_max_iterations: int = 50
    local_search_neighbors: int = 30
    local_search_evaluation_budget: Optional[int] = None
    evaluation_cache_enabled: bool = True
    alpha_signature_precision: int = 4
    deduplicate_environmental_selection: bool = True
    max_evaluations: Optional[int] = None
    early_stopping_generations: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    objective_weights: Optional[Tuple[float, float]] = None
    objective_normalization: Optional[dict] = None
    sigma_index: Optional[int] = None
    progress_log_path: Optional[str] = None
    runtime_stats: dict = field(default_factory=dict, init=False, repr=False)
    _evaluation_cache: dict = field(default_factory=dict, init=False, repr=False)


def dominates(a: Individual, b: Individual) -> bool:
    """Return True when a Pareto-dominates b in minimization."""

    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        a_violations = a.metadata.get("violation_count", math.inf)
        b_violations = b.metadata.get("violation_count", math.inf)
        if a_violations < b_violations:
            return True
        if a_violations > b_violations:
            return False

    return (
        a.objectives[0] <= b.objectives[0]
        and a.objectives[1] <= b.objectives[1]
        and (a.objectives[0] < b.objectives[0] or a.objectives[1] < b.objectives[1])
    )


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Assign NSGA-II ranks and return fronts."""

    dominated_by: Dict[int, List[int]] = {}
    domination_count: Dict[int, int] = {}
    fronts_indices: List[List[int]] = [[]]

    for p_index, p in enumerate(population):
        dominated_by[p_index] = []
        domination_count[p_index] = 0
        for q_index, q in enumerate(population):
            if p_index == q_index:
                continue
            if dominates(p, q):
                dominated_by[p_index].append(q_index)
            elif dominates(q, p):
                domination_count[p_index] += 1

        if domination_count[p_index] == 0:
            p.rank = 0
            fronts_indices[0].append(p_index)

    front_number = 0
    while front_number < len(fronts_indices) and fronts_indices[front_number]:
        next_front: List[int] = []
        for p_index in fronts_indices[front_number]:
            for q_index in dominated_by[p_index]:
                domination_count[q_index] -= 1
                if domination_count[q_index] == 0:
                    population[q_index].rank = front_number + 1
                    next_front.append(q_index)
        front_number += 1
        if next_front:
            fronts_indices.append(next_front)

    return [[population[index] for index in front] for front in fronts_indices if front]


def assign_crowding_distance(front: List[Individual]) -> None:
    """Assign crowding distance to one Pareto front."""

    if not front:
        return
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = math.inf
        return

    for ind in front:
        ind.crowding_distance = 0.0

    for objective_index in range(2):
        sorted_front = sorted(front, key=lambda ind: ind.objectives[objective_index])
        sorted_front[0].crowding_distance = math.inf
        sorted_front[-1].crowding_distance = math.inf

        min_obj = sorted_front[0].objectives[objective_index]
        max_obj = sorted_front[-1].objectives[objective_index]
        if max_obj == min_obj:
            continue

        for index in range(1, len(sorted_front) - 1):
            if math.isinf(sorted_front[index].crowding_distance):
                continue
            prev_obj = sorted_front[index - 1].objectives[objective_index]
            next_obj = sorted_front[index + 1].objectives[objective_index]
            sorted_front[index].crowding_distance += (next_obj - prev_obj) / (max_obj - min_obj)


def evaluate_population(
    population: List[Individual],
    data,
    penalty_value: float = 1_000_000.0,
    config: Optional[NSGA2Config] = None,
) -> None:
    """Evaluate every individual with the decoder."""

    for ind in population:
        if config is not None:
            evaluate_candidate(ind, data, config)
        else:
            fallback_config = NSGA2Config(penalty_value=penalty_value)
            fallback_config.evaluation_cache_enabled = False
            evaluate_candidate(ind, data, fallback_config)


def create_initial_population(data, config: NSGA2Config, rng: random.Random) -> List[Individual]:
    """Create and evaluate the initial population."""

    population = []
    for _ in range(config.population_size):
        best_candidate = None
        best_violation_count = math.inf

        for _attempt in range(config.feasible_initialization_attempts):
            candidate = create_random_individual(
                data,
                rng=rng,
                capacity_aware=config.capacity_aware_initialization,
                initial_departure_max=config.initial_departure_max,
                max_wait_between_trips=config.max_wait_between_trips,
                randomize_cuts=True,
            )
            evaluate_candidate(candidate, data, config)

            if candidate.feasible:
                best_candidate = candidate
                break
            violation_count = candidate.metadata.get("violation_count", math.inf)
            if violation_count < best_violation_count:
                best_candidate = candidate
                best_violation_count = violation_count

        if best_candidate is None:
            best_candidate = create_random_individual(
                data,
                rng=rng,
                capacity_aware=config.capacity_aware_initialization,
                initial_departure_max=config.initial_departure_max,
                max_wait_between_trips=config.max_wait_between_trips,
                randomize_cuts=True,
            )
        population.append(best_candidate)

    evaluate_population(population, data, config.penalty_value, config)
    return population


def binary_tournament_selection(
    population: List[Individual],
    rng: random.Random,
    config: Optional[NSGA2Config] = None,
) -> Individual:
    """Select one parent using rank, crowding distance, then randomness."""

    tournament_size = min(2, len(population))
    candidates = rng.sample(population, tournament_size)
    if len(candidates) == 1:
        return candidates[0]

    first, second = candidates
    first_rank = first.rank if first.rank is not None else math.inf
    second_rank = second.rank if second.rank is not None else math.inf

    if first_rank < second_rank:
        return first
    if second_rank < first_rank:
        return second
    if config is not None and config.objective_weights is not None:
        first_score = _weighted_score(
            first,
            population,
            config.objective_weights,
            config.objective_normalization,
        )
        second_score = _weighted_score(
            second,
            population,
            config.objective_weights,
            config.objective_normalization,
        )
        if first_score < second_score:
            return first
        if second_score < first_score:
            return second
    if first.crowding_distance > second.crowding_distance:
        return first
    if second.crowding_distance > first.crowding_distance:
        return second
    return first if rng.random() < 0.5 else second


def ordered_crossover(
    parent1: Individual,
    parent2: Individual,
    data,
    rng: random.Random,
    config: Optional[NSGA2Config] = None,
) -> Tuple[Individual, Individual]:
    """Apply ordered crossover to permutations and blend alpha values."""

    child1_perm, child2_perm = _ordered_crossover_perm(parent1.perm, parent2.perm, rng)
    slots = get_trip_slots(data)
    horizon = planning_horizon(data)
    first_trip = min(data.V)

    child1_alpha = {}
    child2_alpha = {}
    for slot in slots:
        trip, _ = slot
        upper_bound = _alpha_upper_bound(trip, first_trip, horizon, config)
        average = (parent1.alpha[slot] + parent2.alpha[slot]) / 2.0
        child1_alpha[slot] = _clip_alpha(_maybe_perturb_alpha(average, upper_bound, rng), upper_bound)
        child2_alpha[slot] = _clip_alpha(_maybe_perturb_alpha(average, upper_bound, rng), upper_bound)

    if rng.random() < 0.1:
        child1_cuts = list(parent1.cuts if rng.random() < 0.5 else parent2.cuts)
        child2_cuts = list(parent2.cuts if rng.random() < 0.5 else parent1.cuts)
        randomize_repair = False
    else:
        child1_cuts = random_capacity_feasible_cuts(child1_perm, data, rng)
        child2_cuts = random_capacity_feasible_cuts(child2_perm, data, rng)
        randomize_repair = True

    child1 = Individual(
        perm=child1_perm,
        cuts=child1_cuts,
        alpha=child1_alpha,
    )
    child2 = Individual(
        perm=child2_perm,
        cuts=child2_cuts,
        alpha=child2_alpha,
    )
    return (
        repair_cuts_capacity_aware(child1, data, rng=rng, randomize=randomize_repair),
        repair_cuts_capacity_aware(child2, data, rng=rng, randomize=randomize_repair),
    )


def mutate(
    ind: Individual,
    data,
    rng: random.Random,
    config: NSGA2Config,
) -> Individual:
    """Return a mutated copy of an individual."""

    mutated = clone_individual_fast(ind)
    mutated.objectives = (math.inf, math.inf)
    mutated.rank = None
    mutated.crowding_distance = 0.0
    mutated.feasible = False
    mutated.metadata = {}
    reset_cuts = False

    if rng.random() < config.mutation_probability and len(mutated.perm) >= 2:
        i, j = rng.sample(range(len(mutated.perm)), 2)
        mutated.perm[i], mutated.perm[j] = mutated.perm[j], mutated.perm[i]

    if rng.random() < config.mutation_probability and len(mutated.perm) >= 2:
        source = rng.randrange(len(mutated.perm))
        customer = mutated.perm.pop(source)
        destination = rng.randrange(len(mutated.perm) + 1)
        mutated.perm.insert(destination, customer)

    if rng.random() < config.mutation_probability and len(mutated.cuts) > 2:
        cut_index = rng.randrange(1, len(mutated.cuts) - 1)
        mutated.cuts[cut_index] += rng.choice([-1, 1])

    if rng.random() < config.cut_reset_probability:
        mutated.cuts = random_capacity_feasible_cuts(mutated.perm, data, rng)
        reset_cuts = True

    if rng.random() < config.mutation_probability:
        mutated.alpha = _mutate_alpha(mutated, data, rng, config)

    return repair_cuts_capacity_aware(mutated, data, rng=rng, randomize=reset_cuts)


def _mutate_alpha(
    ind: Individual,
    data,
    rng: random.Random,
    config: NSGA2Config,
) -> Dict[Tuple[int, int], float]:
    """Mutate one alpha value, sometimes targeting period boundaries."""

    mutated_alpha = dict(ind.alpha)
    horizon = planning_horizon(data)
    slot = rng.choice(get_trip_slots(data))
    trip, _ = slot
    first_trip = min(data.V)
    upper_bound = _alpha_upper_bound(trip, first_trip, horizon, config)

    if rng.random() < config.alpha_boundary_mutation_probability:
        mutated_alpha[slot] = _boundary_target_alpha(ind, data, slot, upper_bound, rng)
        return mutated_alpha

    if rng.random() < config.alpha_random_reset_probability:
        mutated_alpha[slot] = rng.uniform(0.0, upper_bound)
        return mutated_alpha

    step = config.alpha_mutation_step
    mutated_alpha[slot] = _clip_alpha(
        mutated_alpha[slot] + rng.uniform(-step, step),
        upper_bound,
    )
    return mutated_alpha


def _boundary_target_alpha(
    ind: Individual,
    data,
    slot: Tuple[int, int],
    upper_bound: float,
    rng: random.Random,
) -> float:
    """Return an alpha that moves a departure close to a period boundary."""

    trip, vehicle = slot
    start_time = min(data.LI.values())
    first_trip = min(data.V)
    anchors = _period_time_anchors(data)

    if trip == first_trip:
        candidates = [
            anchor - start_time
            for anchor in anchors
            if 0.0 <= anchor - start_time <= upper_bound
        ]
    else:
        previous_available = _previous_vehicle_available_time(ind, data, trip, vehicle)
        candidates = [
            anchor - previous_available
            for anchor in anchors
            if 0.0 <= anchor - previous_available <= upper_bound
        ]

    if not candidates:
        return rng.uniform(0.0, upper_bound)

    selected = rng.choice(candidates)
    # Small jitter keeps the search near useful boundaries without collapsing.
    return _clip_alpha(selected + rng.uniform(-0.02, 0.02), upper_bound)


def _previous_vehicle_available_time(
    ind: Individual,
    data,
    trip: int,
    vehicle: int,
) -> float:
    """Estimate when the previous trip of a vehicle becomes available."""

    previous_trips = [value for value in data.V if value < trip]
    if not previous_trips:
        return min(data.LI.values())

    try:
        decoded = decode_individual(ind, data, update_individual=False)
    except ValueError:
        return min(data.LI.values())

    for previous_trip in sorted(previous_trips, reverse=True):
        previous_key = (data.dummy_depot, previous_trip, vehicle)
        if previous_key in decoded.t:
            return decoded.t[previous_key]
    return min(data.LI.values())


def _period_time_anchors(data) -> List[float]:
    """Return useful absolute times where objective regimes can change."""

    anchors = {float(min(data.LI.values())), float(max(data.LS.values()))}
    for period in data.P:
        anchors.add(float(data.LI[period]))
        anchors.add(float(data.LS[period]))
    return sorted(anchors)


def repair_cuts_capacity_aware(
    ind: Individual,
    data,
    rng: Optional[random.Random] = None,
    randomize: bool = False,
) -> Individual:
    """Repair cut shape and greedily respect capacity when slots allow it."""

    repaired = clone_individual_fast(ind)
    if randomize:
        repaired.cuts = random_capacity_feasible_cuts(repaired.perm, data, rng)
        return repaired

    num_slots = len(get_trip_slots(data))
    cuts = [0]
    current_demand = 0.0

    for index, customer in enumerate(repaired.perm):
        demand = data.d[customer]
        has_free_slots_after_this_one = len(cuts) < num_slots
        if current_demand > 0 and current_demand + demand > data.q and has_free_slots_after_this_one:
            cuts.append(index)
            current_demand = 0.0
        current_demand += demand

    cuts.append(len(repaired.perm))
    while len(cuts) < num_slots + 1:
        cuts.append(len(repaired.perm))

    repaired.cuts = cuts[: num_slots + 1]
    repaired.cuts[0] = 0
    repaired.cuts[-1] = len(repaired.perm)
    repaired.cuts = _normalize_cuts(repaired.cuts, len(repaired.perm))
    return repaired


def make_offspring(
    population: List[Individual],
    data,
    config: NSGA2Config,
    rng: random.Random,
) -> List[Individual]:
    """Create, repair, validate and evaluate offspring."""

    offspring: List[Individual] = []
    while len(offspring) < config.population_size and not evaluation_budget_reached(config):
        parent1 = binary_tournament_selection(population, rng, config)
        parent2 = binary_tournament_selection(population, rng, config)

        if rng.random() < config.crossover_probability:
            child1, child2 = ordered_crossover(parent1, parent2, data, rng, config)
        else:
            child1, child2 = clone_individual_fast(parent1), clone_individual_fast(parent2)

        child1 = mutate(child1, data, rng, config)
        child2 = mutate(child2, data, rng, config)
        child1 = _ensure_valid_or_repair(child1, data)
        child2 = _ensure_valid_or_repair(child2, data)

        offspring.append(child1)
        if len(offspring) < config.population_size:
            offspring.append(child2)

    evaluate_population(offspring, data, config.penalty_value, config)
    return offspring


def environmental_selection(
    combined: List[Individual],
    population_size: int,
    config: Optional[NSGA2Config] = None,
) -> List[Individual]:
    """Select the next generation from parent and offspring populations."""

    original_combined = list(combined)
    if config is not None and config.deduplicate_environmental_selection:
        original_size = len(combined)
        combined = _deduplicate_candidates(combined, config)
        initialize_runtime(config)
        config.runtime_stats["duplicate_candidates_removed"] += original_size - len(combined)

    fronts = fast_non_dominated_sort(combined)
    next_population: List[Individual] = []

    for front in fronts:
        assign_crowding_distance(front)
        if len(next_population) + len(front) <= population_size:
            next_population.extend(front)
        else:
            remaining = population_size - len(next_population)
            if config is not None and config.objective_weights is not None:
                next_population.extend(_select_weighted_diverse(front, remaining, config))
            else:
                next_population.extend(_select_diverse_by_cuts(front, remaining))
            break

    if len(next_population) < population_size:
        for ind in sorted(original_combined, key=_candidate_sort_key):
            if len(next_population) >= population_size:
                break
            if ind not in next_population:
                next_population.append(ind)

    return next_population


def run_nsga2(data, config: NSGA2Config) -> Tuple[List[Individual], List[Individual], List[dict]]:
    """Run a basic NSGA-II loop and return final population, first front and history."""

    reset_runtime(config)
    start_time = time.perf_counter()
    rng = random.Random(config.seed)
    population = create_initial_population(data, config, rng)
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        assign_crowding_distance(front)
    _reset_progress_log(
        config,
        [
            "NSGA-II execution log",
            f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Population size: {config.population_size}",
            f"Generations: {config.generations}",
            f"Seed: {config.seed}",
            f"Objective weights: {config.objective_weights}",
            (
                "Local search: "
                f"method={config.local_search_method}, "
                f"rate={config.local_search_rate}, "
                f"interval={config.local_search_generations_interval}, "
                f"max_individuals={config.local_search_max_individuals}, "
                f"max_iterations={config.local_search_max_iterations}"
            ),
            "",
            (
                "Initial population created | "
                f"size={len(population)} | "
                f"feasible={sum(1 for ind in population if ind.feasible)}/{len(population)} | "
                f"front0_size={len(fronts[0]) if fronts else 0}"
            ),
            "",
            "Iterations",
        ],
    )

    history: List[dict] = []
    best_f1_seen = math.inf
    best_f2_seen = math.inf
    best_hypervolume_seen = -math.inf
    stagnant_generations = 0
    for generation in range(1, config.generations + 1):
        if evaluation_budget_reached(config):
            config.runtime_stats["stopped_early"] = True
            config.runtime_stats["stop_reason"] = "max_evaluations reached before offspring creation"
            _append_progress_log(config, f"Stopped early: {config.runtime_stats['stop_reason']}")
            break
        offspring = make_offspring(population, data, config, rng)
        population = environmental_selection(population + offspring, config.population_size, config)
        local_search_applied = False
        local_search_candidates = 0
        local_search_improvements_before = config.runtime_stats.get("local_search_improvements", 0)
        if _should_apply_local_search(generation, config):
            local_search_applied = True
            improved = _apply_local_search(population, data, config, rng)
            local_search_candidates = len(improved)
            population = environmental_selection(population + improved, config.population_size, config)
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distance(front)

        config.runtime_stats["elapsed_seconds"] = time.perf_counter() - start_time
        generation_stats = _generation_stats(generation, population, fronts, config)
        generation_stats["offspring_created"] = len(offspring)
        generation_stats["local_search_applied"] = local_search_applied
        generation_stats["local_search_candidates"] = local_search_candidates
        generation_stats["local_search_dominance_improvements"] = (
            config.runtime_stats.get("local_search_improvements", 0) - local_search_improvements_before
        )
        history.append(generation_stats)
        if config.verbose:
            print(_format_generation_status(generation_stats, config.generations))
        improved_f1 = best_f1_seen - generation_stats["best_f1"] > config.early_stopping_min_delta
        improved_f2 = best_f2_seen - generation_stats["best_f2"] > config.early_stopping_min_delta
        improved_hv = generation_stats["hypervolume"] - best_hypervolume_seen > config.early_stopping_min_delta
        if improved_f1 or improved_f2 or improved_hv:
            best_f1_seen = min(best_f1_seen, generation_stats["best_f1"])
            best_f2_seen = min(best_f2_seen, generation_stats["best_f2"])
            best_hypervolume_seen = max(best_hypervolume_seen, generation_stats["hypervolume"])
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        generation_stats["stagnation_generations"] = stagnant_generations
        _append_progress_log(config, _format_generation_log_line(generation_stats, config.generations))

        if config.early_stopping_generations is not None and stagnant_generations >= config.early_stopping_generations:
            config.runtime_stats["stopped_early"] = True
            config.runtime_stats["stop_reason"] = (
                f"no objective or hypervolume improvement for {stagnant_generations} generations"
            )
            _append_progress_log(config, f"Stopped early: {config.runtime_stats['stop_reason']}")
            break
        if evaluation_budget_reached(config):
            config.runtime_stats["stopped_early"] = True
            config.runtime_stats["stop_reason"] = "max_evaluations reached"
            _append_progress_log(config, f"Stopped early: {config.runtime_stats['stop_reason']}")
            break

    config.runtime_stats["elapsed_seconds"] = time.perf_counter() - start_time
    final_fronts = fast_non_dominated_sort(population)
    for front in final_fronts:
        assign_crowding_distance(front)
    _append_progress_log(
        config,
        (
            "Finished | "
            f"elapsed_seconds={config.runtime_stats['elapsed_seconds']:.3f} | "
            f"final_front0_size={len(final_fronts[0]) if final_fronts else 0} | "
            f"evaluations={config.runtime_stats.get('evaluations', 0)} | "
            f"evaluation_requests={config.runtime_stats.get('evaluation_requests', 0)} | "
            f"cache_hits={config.runtime_stats.get('cache_hits', 0)}"
        ),
    )
    return population, final_fronts[0] if final_fronts else [], history


def _reset_progress_log(config: NSGA2Config, lines: List[str]) -> None:
    if not config.progress_log_path:
        return
    output_path = Path(config.progress_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_progress_log(config: NSGA2Config, line: str) -> None:
    if not config.progress_log_path:
        return
    output_path = Path(config.progress_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _format_generation_status(generation_stats: dict, total_generations: int) -> str:
    return (
        f"Generation {generation_stats['generation']}/{total_generations} | "
        f"Front0 size={generation_stats['front0_size']} | "
        f"feasible={generation_stats['feasible_count']}/{generation_stats['population_size']} | "
        f"best F1={generation_stats['best_f1']:.3f} | "
        f"best F2={generation_stats['best_f2']:.3f} | "
        f"evals={generation_stats['evaluations']}"
    )


def _format_generation_log_line(generation_stats: dict, total_generations: int) -> str:
    local_search = "applied" if generation_stats["local_search_applied"] else "skipped"
    return (
        f"{_format_generation_status(generation_stats, total_generations)} | "
        f"avg F1={generation_stats['avg_f1']:.3f} | "
        f"avg F2={generation_stats['avg_f2']:.3f} | "
        f"front0 unique={generation_stats['front0_unique_objectives']} | "
        f"offspring={generation_stats['offspring_created']} | "
        f"local_search={local_search} | "
        f"local_search_candidates={generation_stats['local_search_candidates']} | "
        f"local_search_improvements={generation_stats['local_search_dominance_improvements']} | "
        f"eval_requests={generation_stats['evaluation_requests']} | "
        f"cache_hits={generation_stats['cache_hits']} | "
        f"decoder_time={generation_stats['decoder_time_seconds']:.3f}s | "
        f"elapsed={generation_stats['elapsed_seconds']:.3f}s | "
        f"hypervolume={generation_stats['hypervolume']:.6f} | "
        f"stagnation={generation_stats['stagnation_generations']}"
    )


def _should_apply_local_search(generation: int, config: NSGA2Config) -> bool:
    if config.local_search_method == "none":
        return False
    if config.local_search_rate <= 0:
        return False
    if config.local_search_generations_interval <= 0:
        return False
    return generation % config.local_search_generations_interval == 0


def _apply_local_search(
    population: List[Individual],
    data,
    config: NSGA2Config,
    rng: random.Random,
) -> List[Individual]:
    candidates = _select_individuals_for_local_search(population, config, rng)
    improved = []
    seen = set()
    for index, ind in enumerate(candidates):
        if evaluation_budget_reached(config):
            break
        signature = individual_signature(ind, config.alpha_signature_precision)
        if signature in seen:
            continue
        seen.add(signature)
        direction = _local_search_direction(index, ind, population, config)
        candidate = intensify_individual(
            ind,
            data,
            direction=direction,
            rng=rng,
            method=config.local_search_method,
            config=config,
        )
        if dominates(candidate, ind):
            config.runtime_stats["local_search_improvements"] += 1
        improved.append(candidate)
    return improved


def _select_individuals_for_local_search(
    population: List[Individual],
    config: NSGA2Config,
    rng: random.Random,
) -> List[Individual]:
    fronts = fast_non_dominated_sort(population)
    front0 = list(fronts[0]) if fronts else list(population)
    target_count = max(1, int(round(config.population_size * config.local_search_rate)))
    target_count = min(target_count, config.local_search_max_individuals, len(population))

    selected: List[Individual] = []
    for ind in (
        min(population, key=lambda item: item.objectives[0]),
        min(population, key=lambda item: item.objectives[1]),
    ):
        if ind not in selected:
            selected.append(ind)

    for ind in sorted(front0, key=lambda item: item.crowding_distance, reverse=True):
        if len(selected) >= target_count:
            break
        if ind not in selected:
            selected.append(ind)

    remaining = [ind for ind in population if ind not in selected]
    rng.shuffle(remaining)
    for ind in remaining:
        if len(selected) >= target_count:
            break
        selected.append(ind)
    return selected[:target_count]


def _local_search_direction(
    index: int,
    ind: Individual,
    population: List[Individual],
    config: Optional[NSGA2Config] = None,
) -> str:
    if config is not None and config.objective_weights is not None:
        beta_f1, beta_f2 = config.objective_weights
        if beta_f1 > beta_f2:
            return "f1"
        if beta_f2 > beta_f1:
            return "f2"
        return "balanced"
    best_f1 = min(population, key=lambda item: item.objectives[0])
    best_f2 = min(population, key=lambda item: item.objectives[1])
    if ind is best_f1 or index == 0:
        return "f1"
    if ind is best_f2 or index == 1:
        return "f2"
    return "balanced"


def _ordered_crossover_perm(
    perm1: List[int],
    perm2: List[int],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    if len(perm1) != len(perm2):
        raise ValueError("Parents must have permutations with the same length")
    if len(perm1) < 2:
        return list(perm1), list(perm2)

    start, end = sorted(rng.sample(range(len(perm1)), 2))
    return _make_ox_child(perm1, perm2, start, end), _make_ox_child(perm2, perm1, start, end)


def _make_ox_child(
    segment_parent: List[int],
    order_parent: List[int],
    start: int,
    end: int,
) -> List[int]:
    child: List[Optional[int]] = [None] * len(segment_parent)
    child[start : end + 1] = segment_parent[start : end + 1]
    used = set(segment_parent[start : end + 1])
    fill_values = [customer for customer in order_parent if customer not in used]
    fill_positions = [index for index, value in enumerate(child) if value is None]
    for position, customer in zip(fill_positions, fill_values):
        child[position] = customer
    return [customer for customer in child if customer is not None]


def _maybe_perturb_alpha(value: float, horizon: float, rng: random.Random) -> float:
    if rng.random() < 0.1:
        return value + rng.uniform(-0.5, 0.5)
    return value


def _clip_alpha(value: float, horizon: float) -> float:
    return max(0.0, min(horizon, value))


def _alpha_upper_bound(
    trip: int,
    first_trip: int,
    horizon: float,
    config: Optional[NSGA2Config],
) -> float:
    if config is None:
        return horizon
    if trip == first_trip:
        if config.initial_departure_max is not None:
            return config.initial_departure_max
        return min(1.0, horizon * 0.1)
    return config.max_wait_between_trips


def _normalize_cuts(cuts: List[int], perm_len: int) -> List[int]:
    normalized = [max(0, min(perm_len, cut)) for cut in cuts]
    for index in range(1, len(normalized)):
        if normalized[index] < normalized[index - 1]:
            normalized[index] = normalized[index - 1]
    normalized[-1] = perm_len
    return normalized


def _ensure_valid_or_repair(ind: Individual, data) -> Individual:
    try:
        validate_individual(ind, data, check_capacity=False)
        return ind
    except ValueError:
        repaired = repair_cuts_capacity_aware(ind, data)
        try:
            validate_individual(repaired, data, check_capacity=False)
            return repaired
        except ValueError:
            pass

        repaired.perm = _repair_permutation(repaired.perm, data.customers)
        repaired = repair_cuts_capacity_aware(repaired, data)
        validate_individual(repaired, data, check_capacity=False)
        return repaired


def _repair_permutation(perm: List[int], customers: List[int]) -> List[int]:
    customer_set = set(customers)
    repaired = []
    seen = set()
    for customer in perm:
        if customer in customer_set and customer not in seen:
            repaired.append(customer)
            seen.add(customer)
    repaired.extend(customer for customer in customers if customer not in seen)
    return repaired


def _select_diverse_by_cuts(front: List[Individual], limit: int) -> List[Individual]:
    grouped = {}
    for ind in sorted(front, key=lambda item: item.crowding_distance, reverse=True):
        grouped.setdefault(tuple(ind.cuts), []).append(ind)

    selected = []
    while len(selected) < limit and grouped:
        for cuts in list(grouped):
            if len(selected) >= limit:
                break
            selected.append(grouped[cuts].pop(0))
            if not grouped[cuts]:
                del grouped[cuts]
    return selected


def _select_weighted_diverse(front: List[Individual], limit: int, config: NSGA2Config) -> List[Individual]:
    """Select candidates closest to the active sigma weights while keeping cut diversity."""

    if limit <= 0:
        return []
    weights = config.objective_weights
    if weights is None:
        return _select_diverse_by_cuts(front, limit)

    grouped = {}
    for ind in sorted(
        front,
        key=lambda item: _weighted_score(item, front, weights, config.objective_normalization),
    ):
        grouped.setdefault(tuple(ind.cuts), []).append(ind)

    selected = []
    while len(selected) < limit and grouped:
        for cuts in list(grouped):
            if len(selected) >= limit:
                break
            selected.append(grouped[cuts].pop(0))
            if not grouped[cuts]:
                del grouped[cuts]
    return selected


def _weighted_score(
    ind: Individual,
    reference_population: List[Individual],
    weights: Tuple[float, float],
    normalization: Optional[dict] = None,
) -> float:
    """Return normalized weighted score for minimization."""

    if normalization:
        norm_f1, norm_f2 = _fixed_normalized_objectives(ind, normalization)
        if math.isfinite(norm_f1) and math.isfinite(norm_f2):
            return weights[0] * norm_f1 + weights[1] * norm_f2

    finite = [
        item
        for item in reference_population
        if math.isfinite(item.objectives[0]) and math.isfinite(item.objectives[1])
    ]
    if not finite:
        return math.inf
    min_f1 = min(item.objectives[0] for item in finite)
    max_f1 = max(item.objectives[0] for item in finite)
    min_f2 = min(item.objectives[1] for item in finite)
    max_f2 = max(item.objectives[1] for item in finite)
    range_f1 = max_f1 - min_f1
    range_f2 = max_f2 - min_f2
    norm_f1 = 0.0 if range_f1 == 0 else (ind.objectives[0] - min_f1) / range_f1
    norm_f2 = 0.0 if range_f2 == 0 else (ind.objectives[1] - min_f2) / range_f2
    return weights[0] * norm_f1 + weights[1] * norm_f2


def _fixed_normalized_objectives(ind: Individual, normalization: dict) -> Tuple[float, float]:
    ideal_f1 = float(normalization.get("ideal_f1", math.nan))
    anti_f1 = float(normalization.get("anti_f1", math.nan))
    ideal_f2 = float(normalization.get("ideal_f2", math.nan))
    anti_f2 = float(normalization.get("anti_f2", math.nan))
    denom_f1 = anti_f1 - ideal_f1
    denom_f2 = anti_f2 - ideal_f2
    if denom_f1 <= 0 or denom_f2 <= 0:
        return math.inf, math.inf
    return (
        (ind.objectives[0] - ideal_f1) / denom_f1,
        (ind.objectives[1] - ideal_f2) / denom_f2,
    )


def _generation_stats(
    generation: int,
    population: List[Individual],
    fronts: List[List[Individual]],
    config: Optional[NSGA2Config] = None,
) -> dict:
    finite_f1 = [ind.objectives[0] for ind in population if math.isfinite(ind.objectives[0])]
    finite_f2 = [ind.objectives[1] for ind in population if math.isfinite(ind.objectives[1])]
    front0 = fronts[0] if fronts else []
    unique_front0 = _unique_by_objectives(front0)
    runtime = getattr(config, "runtime_stats", {}) if config is not None else {}
    return {
        "generation": generation,
        "front0_size": len(front0),
        "front0_unique_objectives": len(unique_front0),
        "best_f1": min(finite_f1) if finite_f1 else math.inf,
        "best_f2": min(finite_f2) if finite_f2 else math.inf,
        "avg_f1": sum(finite_f1) / len(finite_f1) if finite_f1 else math.inf,
        "avg_f2": sum(finite_f2) / len(finite_f2) if finite_f2 else math.inf,
        "feasible_count": sum(1 for ind in population if ind.feasible),
        "population_size": len(population),
        "evaluations": runtime.get("evaluations", 0),
        "evaluation_requests": runtime.get("evaluation_requests", 0),
        "cache_hits": runtime.get("cache_hits", 0),
        "decoder_time_seconds": runtime.get("decoder_time_seconds", 0.0),
        "elapsed_seconds": runtime.get("elapsed_seconds", 0.0),
        "hypervolume": _hypervolume_2d_from_front(unique_front0),
    }


def _deduplicate_candidates(population: List[Individual], config: NSGA2Config) -> List[Individual]:
    precision = config.alpha_signature_precision
    unique = {}
    for ind in population:
        key = individual_signature(ind, precision)
        incumbent = unique.get(key)
        if incumbent is None or _candidate_sort_key(ind) < _candidate_sort_key(incumbent):
            unique[key] = ind
    return list(unique.values())


def _candidate_sort_key(ind: Individual) -> tuple:
    rank = ind.rank if ind.rank is not None else math.inf
    violation_count = ind.metadata.get("violation_count", math.inf)
    return (
        not ind.feasible,
        violation_count,
        rank,
        ind.objectives[0],
        ind.objectives[1],
        -ind.crowding_distance,
    )


def _unique_by_objectives(front: List[Individual]) -> List[Individual]:
    unique = {}
    for ind in front:
        unique.setdefault((ind.objectives[0], ind.objectives[1]), ind)
    return [unique[key] for key in sorted(unique)]


def _hypervolume_2d_from_front(front: List[Individual]) -> float:
    points = [
        ind.metadata.get("raw_objectives", ind.objectives)
        for ind in front
        if math.isfinite(ind.objectives[0]) and math.isfinite(ind.objectives[1])
    ]
    if not points:
        return 0.0
    ref_f1 = max(point[0] for point in points) * 1.1
    ref_f2 = max(point[1] for point in points) * 1.1
    filtered = sorted(set((float(f1), float(f2)) for f1, f2 in points if f1 < ref_f1 and f2 < ref_f2))
    volume = 0.0
    previous_f2 = ref_f2
    for f1, f2 in filtered:
        if f2 >= previous_f2:
            continue
        volume += (ref_f1 - f1) * (previous_f2 - f2)
        previous_f2 = f2
    return volume
