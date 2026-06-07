from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple, Union

try:
    from .chromosome import (
        Individual,
        get_trip_slots,
        random_capacity_feasible_cuts,
        validate_individual,
    )
    from .decoder import decode_individual
    from .evaluation import (
        clone_individual_fast,
        evaluate_candidate,
        local_search_budget_reached,
    )
except ImportError:  # Allows running src/main.py directly.
    from chromosome import (
        Individual,
        get_trip_slots,
        random_capacity_feasible_cuts,
        validate_individual,
    )
    from decoder import decode_individual
    from evaluation import clone_individual_fast, evaluate_candidate, local_search_budget_reached


def is_better_for_direction(
    candidate: Individual,
    incumbent: Individual,
    direction: str,
) -> bool:
    """Compare two candidates for a local-search direction."""

    if candidate.feasible and not incumbent.feasible:
        return True
    if not candidate.feasible and incumbent.feasible:
        return False
    if not candidate.feasible and not incumbent.feasible:
        candidate_violations = candidate.metadata.get("violation_count", math.inf)
        incumbent_violations = incumbent.metadata.get("violation_count", math.inf)
        if candidate_violations != incumbent_violations:
            return candidate_violations < incumbent_violations

    if direction == "f1":
        return candidate.objectives < incumbent.objectives
    if direction == "f2":
        return (candidate.objectives[1], candidate.objectives[0]) < (
            incumbent.objectives[1],
            incumbent.objectives[0],
        )
    if direction == "balanced":
        return _balanced_score(candidate, incumbent) < _balanced_score(incumbent, candidate)
    raise ValueError("direction must be one of: f1, f2, balanced")


def generate_swap_neighbors(ind: Individual, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors by swapping two permutation positions."""

    random_generator = _coerce_rng(rng)
    if len(ind.perm) < 2:
        return []
    pairs = [(i, j) for i in range(len(ind.perm)) for j in range(i + 1, len(ind.perm))]
    random_generator.shuffle(pairs)
    neighbors = []
    for i, j in pairs[:max_neighbors]:
        neighbor = clone_individual_fast(ind)
        neighbor.perm[i], neighbor.perm[j] = neighbor.perm[j], neighbor.perm[i]
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
    return neighbors


def generate_insert_neighbors(ind: Individual, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors by moving one customer to another position."""

    random_generator = _coerce_rng(rng)
    moves = [(i, j) for i in range(len(ind.perm)) for j in range(len(ind.perm)) if i != j]
    random_generator.shuffle(moves)
    neighbors = []
    for source, destination in moves[:max_neighbors]:
        neighbor = clone_individual_fast(ind)
        customer = neighbor.perm.pop(source)
        neighbor.perm.insert(destination, customer)
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
    return neighbors


def generate_reverse_segment_neighbors(ind: Individual, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors by reversing one segment of the permutation."""

    random_generator = _coerce_rng(rng)
    if len(ind.perm) < 2:
        return []
    segments = [(i, j) for i in range(len(ind.perm)) for j in range(i + 1, len(ind.perm))]
    random_generator.shuffle(segments)
    neighbors = []
    for start, end in segments[:max_neighbors]:
        neighbor = clone_individual_fast(ind)
        neighbor.perm[start : end + 1] = reversed(neighbor.perm[start : end + 1])
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
    return neighbors


def generate_cut_shift_neighbors(ind: Individual, data, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors by shifting one internal cut while preserving cut shape."""

    random_generator = _coerce_rng(rng)
    neighbors = []
    internal_indices = list(range(1, len(ind.cuts) - 1))
    moves = [(index, delta) for index in internal_indices for delta in (-1, 1)]
    random_generator.shuffle(moves)
    for index, delta in moves[: max_neighbors * 2]:
        neighbor = clone_individual_fast(ind)
        neighbor.cuts[index] += delta
        neighbor.cuts = _normalize_cuts(neighbor.cuts, len(neighbor.perm))
        if len(neighbors) >= max_neighbors:
            break
        try:
            validate_individual(neighbor, data, check_capacity=True)
        except ValueError:
            continue
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
    return neighbors


def generate_cut_reset_neighbors(ind: Individual, data, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors with randomized capacity-feasible cut structures."""

    random_generator = _coerce_rng(rng)
    neighbors = []
    seen = {tuple(ind.cuts)}
    attempts = max(max_neighbors * 4, max_neighbors)
    for _ in range(attempts):
        neighbor = clone_individual_fast(ind)
        neighbor.cuts = random_capacity_feasible_cuts(neighbor.perm, data, random_generator)
        key = tuple(neighbor.cuts)
        if key in seen:
            continue
        seen.add(key)
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def generate_alpha_shift_neighbors(ind: Individual, data, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors by shifting alpha values up or down."""

    random_generator = _coerce_rng(rng)
    upper = max(data.LS.values())
    shifts = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
    candidates = [(slot, shift) for slot in get_trip_slots(data) for shift in shifts]
    random_generator.shuffle(candidates)
    neighbors = []
    for slot, shift in candidates[: max_neighbors * 2]:
        neighbor = clone_individual_fast(ind)
        neighbor.alpha[slot] = _clip(neighbor.alpha[slot] + shift, 0.0, upper)
        if neighbor.alpha[slot] == ind.alpha[slot]:
            continue
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def generate_alpha_boundary_neighbors(ind: Individual, data, max_neighbors: int, rng) -> List[Individual]:
    """Generate neighbors that move departures near period boundaries."""

    random_generator = _coerce_rng(rng)
    anchors = _period_anchors(data)
    upper = max(data.LS.values())
    decoded = decode_individual(ind, data, update_individual=False)
    start_time = min(data.LI.values())
    first_trip = min(data.V)
    candidates = []

    for slot in get_trip_slots(data):
        trip, vehicle = slot
        base_time = start_time
        if trip != first_trip:
            base_time = _previous_available_time(decoded, data, trip, vehicle, start_time)
        for anchor in anchors:
            value = anchor - base_time
            if 0.0 <= value <= upper:
                candidates.append((slot, value))
        candidates.append((slot, 0.0))

    random_generator.shuffle(candidates)
    neighbors = []
    seen = set()
    for slot, value in candidates:
        rounded_key = (slot, round(value, 4))
        if rounded_key in seen:
            continue
        seen.add(rounded_key)
        neighbor = clone_individual_fast(ind)
        neighbor.alpha[slot] = _clip(value + random_generator.uniform(-0.01, 0.01), 0.0, upper)
        _reset_search_state(neighbor)
        neighbors.append(neighbor)
        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def vnd_improve(
    ind: Individual,
    data,
    direction: str,
    rng=None,
    max_iterations: int = 50,
    max_neighbors_per_operator: int = 30,
    config=None,
) -> Individual:
    """Improve an individual with variable-neighborhood descent."""

    random_generator = _coerce_rng(rng)
    incumbent = evaluate_candidate(clone_individual_fast(ind), data, config, source="local_search")
    neighborhoods = [
        lambda current: generate_alpha_boundary_neighbors(current, data, max_neighbors_per_operator, random_generator),
        lambda current: generate_alpha_shift_neighbors(current, data, max_neighbors_per_operator, random_generator),
        lambda current: generate_swap_neighbors(current, max_neighbors_per_operator, random_generator),
        lambda current: generate_insert_neighbors(current, max_neighbors_per_operator, random_generator),
        lambda current: generate_reverse_segment_neighbors(current, max_neighbors_per_operator, random_generator),
        lambda current: generate_cut_shift_neighbors(current, data, max_neighbors_per_operator, random_generator),
        lambda current: generate_cut_reset_neighbors(current, data, max_neighbors_per_operator, random_generator),
    ]

    iterations = 0
    neighborhood_index = 0
    while iterations < max_iterations and neighborhood_index < len(neighborhoods):
        best_neighbor = incumbent
        for neighbor in neighborhoods[neighborhood_index](incumbent):
            if local_search_budget_reached(config):
                break
            neighbor = evaluate_candidate(neighbor, data, config, source="local_search")
            if is_better_for_direction(neighbor, best_neighbor, direction):
                best_neighbor = neighbor

        iterations += 1
        if best_neighbor is not incumbent and is_better_for_direction(best_neighbor, incumbent, direction):
            incumbent = best_neighbor
            neighborhood_index = 0
        else:
            neighborhood_index += 1
    return incumbent


def lns_improve(
    ind: Individual,
    data,
    direction: str,
    rng=None,
    destroy_fraction: float = 0.3,
    attempts: int = 30,
    config=None,
) -> Individual:
    """Apply a small destroy/repair search over the permutation."""

    random_generator = _coerce_rng(rng)
    incumbent = evaluate_candidate(clone_individual_fast(ind), data, config, source="local_search")
    remove_count = max(1, int(round(len(ind.perm) * destroy_fraction)))

    for _ in range(attempts):
        if local_search_budget_reached(config):
            break
        candidate = clone_individual_fast(incumbent)
        indices = sorted(random_generator.sample(range(len(candidate.perm)), remove_count), reverse=True)
        removed = [candidate.perm.pop(index) for index in indices]
        random_generator.shuffle(removed)
        for customer in removed:
            candidate.perm.insert(random_generator.randrange(len(candidate.perm) + 1), customer)
        candidate.cuts = random_capacity_feasible_cuts(candidate.perm, data, random_generator)
        _reset_search_state(candidate)
        candidate = evaluate_candidate(candidate, data, config, source="local_search")
        if is_better_for_direction(candidate, incumbent, direction):
            incumbent = candidate
    return incumbent


def alns_improve(
    ind: Individual,
    data,
    direction: str,
    rng=None,
    iterations: int = 50,
    config=None,
) -> Individual:
    """Apply a simple ALNS-style loop with lightweight destroy/repair choices."""

    random_generator = _coerce_rng(rng)
    incumbent = evaluate_candidate(clone_individual_fast(ind), data, config, source="local_search")
    destroy_methods = ("random", "sequence", "worst")
    repair_methods = ("random", "greedy", "cuts")

    for _ in range(iterations):
        if local_search_budget_reached(config):
            break
        destroy = random_generator.choice(destroy_methods)
        repair = random_generator.choice(repair_methods)
        candidate = _destroy_repair(incumbent, data, destroy, repair, random_generator, config)
        if is_better_for_direction(candidate, incumbent, direction):
            incumbent = candidate
    return incumbent


def intensify_individual(
    ind: Individual,
    data,
    direction: str,
    rng=None,
    method: str = "vnd",
    config=None,
) -> Individual:
    """Run the requested local-search method over one individual."""

    if method == "none":
        return evaluate_candidate(clone_individual_fast(ind), data, config, source="local_search")
    if method == "vnd":
        return vnd_improve(
            ind,
            data,
            direction,
            rng=rng,
            max_iterations=getattr(config, "local_search_max_iterations", 50),
            max_neighbors_per_operator=getattr(config, "local_search_neighbors", 30),
            config=config,
        )
    if method == "lns":
        return lns_improve(ind, data, direction, rng=rng, config=config)
    if method == "alns":
        return alns_improve(ind, data, direction, rng=rng, config=config)
    if method == "vnd_lns":
        improved = vnd_improve(
            ind,
            data,
            direction,
            rng=rng,
            max_iterations=getattr(config, "local_search_max_iterations", 50),
            max_neighbors_per_operator=getattr(config, "local_search_neighbors", 30),
            config=config,
        )
        return lns_improve(improved, data, direction, rng=rng, config=config)
    if method == "vnd_alns":
        improved = vnd_improve(
            ind,
            data,
            direction,
            rng=rng,
            max_iterations=getattr(config, "local_search_max_iterations", 50),
            max_neighbors_per_operator=getattr(config, "local_search_neighbors", 30),
            config=config,
        )
        return alns_improve(improved, data, direction, rng=rng, config=config)
    raise ValueError(f"Unknown local search method: {method}")


def _destroy_repair(
    ind: Individual,
    data,
    destroy: str,
    repair: str,
    rng: random.Random,
    config=None,
) -> Individual:
    candidate = clone_individual_fast(ind)
    remove_count = max(1, int(round(len(candidate.perm) * 0.3)))
    removed = _remove_customers(candidate, data, destroy, remove_count, rng, config)
    if repair == "greedy":
        _greedy_reinsert(candidate, removed, data, rng, config)
    else:
        rng.shuffle(removed)
        for customer in removed:
            candidate.perm.insert(rng.randrange(len(candidate.perm) + 1), customer)
    candidate.cuts = random_capacity_feasible_cuts(candidate.perm, data, rng)
    _reset_search_state(candidate)
    return evaluate_candidate(candidate, data, config, source="local_search")


def _remove_customers(
    candidate: Individual,
    data,
    destroy: str,
    remove_count: int,
    rng: random.Random,
    config=None,
) -> List[int]:
    if destroy == "sequence" and len(candidate.perm) > remove_count:
        start = rng.randrange(len(candidate.perm) - remove_count + 1)
        removed = candidate.perm[start : start + remove_count]
        del candidate.perm[start : start + remove_count]
        return removed
    if destroy == "worst":
        scores = _customer_contribution_scores(candidate, data, config)
        to_remove = {customer for customer, _ in scores[:remove_count]}
        remaining = []
        removed = []
        for customer in candidate.perm:
            if customer in to_remove:
                removed.append(customer)
            else:
                remaining.append(customer)
        candidate.perm = remaining
        return removed

    indices = sorted(rng.sample(range(len(candidate.perm)), remove_count), reverse=True)
    return [candidate.perm.pop(index) for index in indices]


def _greedy_reinsert(candidate: Individual, removed: List[int], data, rng: random.Random, config=None) -> None:
    rng.shuffle(removed)
    for customer in removed:
        best_perm = None
        best_score = math.inf
        for position in range(len(candidate.perm) + 1):
            if local_search_budget_reached(config):
                break
            trial = clone_individual_fast(candidate)
            trial.perm.insert(position, customer)
            trial.cuts = random_capacity_feasible_cuts(trial.perm, data, rng)
            trial = evaluate_candidate(trial, data, config, source="local_search")
            score = sum(trial.objectives)
            if score < best_score:
                best_score = score
                best_perm = trial.perm
        candidate.perm = best_perm if best_perm is not None else candidate.perm + [customer]


def _customer_contribution_scores(ind: Individual, data, config=None) -> List[Tuple[int, float]]:
    evaluated = evaluate_candidate(clone_individual_fast(ind), data, config, source="local_search")
    try:
        decoded = decode_individual(evaluated, data, update_individual=False)
    except (ValueError, KeyError):
        return [(customer, 0.0) for customer in ind.perm]
    scores = {customer: 0.0 for customer in ind.perm}
    for row in decoded.arc_periods:
        i, j = row["arc"]
        contribution = row["emission"] + row["cost"]
        if i in scores:
            scores[i] += contribution / 2.0
        if j in scores:
            scores[j] += contribution / 2.0
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _balanced_score(ind: Individual, other: Individual) -> float:
    values = [ind.objectives, other.objectives]
    ideal_f1 = min(value[0] for value in values)
    ideal_f2 = min(value[1] for value in values)
    nadir_f1 = max(value[0] for value in values)
    nadir_f2 = max(value[1] for value in values)
    denom_f1 = nadir_f1 - ideal_f1
    denom_f2 = nadir_f2 - ideal_f2
    norm_f1 = 0.0 if denom_f1 == 0 else (ind.objectives[0] - ideal_f1) / denom_f1
    norm_f2 = 0.0 if denom_f2 == 0 else (ind.objectives[1] - ideal_f2) / denom_f2
    score = math.sqrt(norm_f1**2 + norm_f2**2)
    if score == math.sqrt(1.0):
        return sum(ind.objectives)
    return score


def _previous_available_time(decoded, data, trip: int, vehicle: int, fallback: float) -> float:
    previous_trips = [value for value in data.V if value < trip]
    for previous_trip in sorted(previous_trips, reverse=True):
        key = (data.dummy_depot, previous_trip, vehicle)
        if key in decoded.t:
            return decoded.t[key]
    return fallback


def _period_anchors(data) -> List[float]:
    anchors = {0.0, float(min(data.LI.values())), float(max(data.LS.values()))}
    for period in data.P:
        anchors.add(float(data.LI[period]))
        anchors.add(float(data.LS[period]))
    return sorted(anchors)


def _normalize_cuts(cuts: List[int], perm_len: int) -> List[int]:
    normalized = [max(0, min(perm_len, cut)) for cut in cuts]
    for index in range(1, len(normalized)):
        if normalized[index] < normalized[index - 1]:
            normalized[index] = normalized[index - 1]
    normalized[0] = 0
    normalized[-1] = perm_len
    return normalized


def _reset_search_state(ind: Individual) -> None:
    ind.objectives = (math.inf, math.inf)
    ind.rank = None
    ind.crowding_distance = 0.0
    ind.feasible = False
    ind.metadata = {}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_rng(rng: Union[int, random.Random, None]) -> random.Random:
    if rng is None:
        return random.Random()
    if isinstance(rng, int):
        return random.Random(rng)
    if isinstance(rng, random.Random):
        return rng
    raise TypeError("rng must be None, an int seed, or random.Random")
