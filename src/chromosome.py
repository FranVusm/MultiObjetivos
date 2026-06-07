from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Individual:
    """Chromosome representation used before decoding routes."""

    perm: List[int]
    cuts: List[int]
    alpha: Dict[Tuple[int, int], float]

    objectives: Tuple[float, float] = (math.inf, math.inf)
    rank: Optional[int] = None
    crowding_distance: float = 0.0
    feasible: bool = False
    metadata: dict = field(default_factory=dict)


def get_trip_slots(data) -> List[Tuple[int, int]]:
    """Return trip slots in the same vehicle/turn order used by the model."""

    return [(v, k) for k in sorted(data.K) for v in sorted(data.V)]


def planning_horizon(data) -> float:
    """Return the planning horizon length in hours."""

    if not data.LI or not data.LS:
        raise ValueError("Cannot calculate planning horizon without LI and LS values")
    return max(data.LS.values()) - min(data.LI.values())


def split_by_cuts(ind: Individual, data) -> Dict[Tuple[int, int], List[int]]:
    """Split a permutation into one customer list per trip slot."""

    slots = get_trip_slots(data)
    if len(ind.cuts) != len(slots) + 1:
        raise ValueError(
            f"cuts must have length {len(slots) + 1}, got {len(ind.cuts)}"
        )

    trips: Dict[Tuple[int, int], List[int]] = {}
    for index, slot in enumerate(slots):
        start = ind.cuts[index]
        end = ind.cuts[index + 1]
        trips[slot] = ind.perm[start:end]
    return trips


def validate_individual(
    ind: Individual,
    data,
    check_capacity: bool = True,
    check_active_trip_prefix: bool = True,
    initial_departure_max: Optional[float] = None,
    max_wait_between_trips: Optional[float] = None,
) -> None:
    """Validate chromosome structure and optional per-trip capacity."""

    customers = list(data.customers)
    customer_set = set(customers)
    perm_set = set(ind.perm)

    if len(ind.perm) != len(customers):
        raise ValueError(
            f"perm must contain {len(customers)} customers, got {len(ind.perm)}"
        )
    if len(perm_set) != len(ind.perm):
        repeated = sorted({customer for customer in ind.perm if ind.perm.count(customer) > 1})
        raise ValueError(f"perm contains repeated customers: {repeated}")

    unknown = sorted(perm_set - customer_set)
    if unknown:
        raise ValueError(f"perm contains customers not present in data.customers: {unknown}")

    missing = sorted(customer_set - perm_set)
    if missing:
        raise ValueError(f"perm is missing customers: {missing}")

    slots = get_trip_slots(data)
    expected_cuts_len = len(slots) + 1
    if len(ind.cuts) != expected_cuts_len:
        raise ValueError(f"cuts must have length {expected_cuts_len}, got {len(ind.cuts)}")
    if not ind.cuts:
        raise ValueError("cuts must not be empty")
    if ind.cuts[0] != 0:
        raise ValueError(f"cuts[0] must be 0, got {ind.cuts[0]}")
    if ind.cuts[-1] != len(ind.perm):
        raise ValueError(f"cuts[-1] must be len(perm)={len(ind.perm)}, got {ind.cuts[-1]}")
    if any(cut < 0 or cut > len(ind.perm) for cut in ind.cuts):
        raise ValueError("cuts must stay within the permutation bounds")
    if any(left > right for left, right in zip(ind.cuts, ind.cuts[1:])):
        raise ValueError("cuts must be sorted in nondecreasing order")
    if check_active_trip_prefix:
        _validate_active_trip_prefix(ind, data)

    slot_set = set(slots)
    alpha_set = set(ind.alpha)
    missing_alpha = sorted(slot_set - alpha_set)
    if missing_alpha:
        raise ValueError(f"alpha is missing trip slots: {missing_alpha}")

    extra_alpha = sorted(alpha_set - slot_set)
    if extra_alpha:
        raise ValueError(f"alpha contains unknown trip slots: {extra_alpha}")

    horizon = planning_horizon(data)
    first_trip = min(data.V)
    for slot, value in ind.alpha.items():
        trip, _ = slot
        upper_bound = horizon
        if trip == first_trip and initial_departure_max is not None:
            upper_bound = initial_departure_max
        elif trip != first_trip and max_wait_between_trips is not None:
            upper_bound = max_wait_between_trips
        if value < 0:
            raise ValueError(f"alpha[{slot}] must be >= 0, got {value}")
        if value > upper_bound:
            raise ValueError(f"alpha[{slot}] must be <= {upper_bound}, got {value}")

    if check_capacity:
        for slot, route in split_by_cuts(ind, data).items():
            demand = sum(data.d[customer] for customer in route)
            if demand > data.q:
                raise ValueError(
                    f"Trip slot {slot} exceeds capacity: demand {demand} > q {data.q}"
                )


def _validate_active_trip_prefix(ind: Individual, data) -> None:
    slots = get_trip_slots(data)
    slot_sizes = {
        slot: ind.cuts[index + 1] - ind.cuts[index]
        for index, slot in enumerate(slots)
    }
    for vehicle in sorted(data.K):
        empty_seen = False
        for trip in sorted(data.V):
            slot = (trip, vehicle)
            active = slot_sizes.get(slot, 0) > 0
            if active and empty_seen:
                raise ValueError(
                    f"Trip slot {slot} is active after an empty trip for vehicle {vehicle}"
                )
            if not active:
                empty_seen = True


def create_random_individual(
    data,
    rng: Union[int, random.Random, None] = None,
    capacity_aware: bool = True,
    initial_departure_max: Optional[float] = None,
    max_wait_between_trips: float = 0.25,
    randomize_cuts: bool = True,
) -> Individual:
    """Create a random structurally valid chromosome when capacity allows it."""

    random_generator = _coerce_rng(rng)
    perm = list(data.customers)
    random_generator.shuffle(perm)

    slots = get_trip_slots(data)
    if capacity_aware:
        cuts = (
            random_capacity_feasible_cuts(perm, data, random_generator)
            if randomize_cuts
            else _capacity_aware_cuts(perm, data, len(slots))
        )
    else:
        cuts = _random_cuts(len(perm), len(slots), random_generator)

    horizon = planning_horizon(data)
    if initial_departure_max is None:
        initial_departure_max = min(1.0, horizon * 0.1)

    first_trip = min(data.V)
    alpha = {}
    for trip, vehicle in slots:
        upper_bound = initial_departure_max if trip == first_trip else max_wait_between_trips
        alpha[(trip, vehicle)] = _sample_alpha(upper_bound, random_generator)
    return Individual(perm=perm, cuts=cuts, alpha=alpha)


def random_capacity_feasible_cuts(
    perm: List[int],
    data,
    rng: Union[int, random.Random, None] = None,
) -> List[int]:
    """Create randomized cuts that try to keep every slot within capacity."""

    random_generator = _coerce_rng(rng)
    num_slots = len(get_trip_slots(data))
    if num_slots <= 0:
        raise ValueError("Cannot build cuts without trip slots")

    cuts = [0]
    position = 0
    for slot_index in range(num_slots):
        remaining_customers = len(perm) - position
        remaining_slots = num_slots - slot_index
        if remaining_customers <= 0:
            cuts.append(len(perm))
            continue

        feasible_sizes = _feasible_block_sizes(
            perm=perm,
            start=position,
            remaining_slots=remaining_slots,
            data=data,
        )
        if not feasible_sizes:
            # If capacity cannot be respected, put the rest in the last available slots.
            take = max(1, remaining_customers - (remaining_slots - 1))
        else:
            take = random_generator.choice(feasible_sizes)

        position = min(len(perm), position + take)
        cuts.append(position)

    cuts[-1] = len(perm)
    return cuts[: num_slots + 1]


def clone_individual(ind: Individual) -> Individual:
    """Return a deep copy of an individual."""

    return copy.deepcopy(ind)


def _coerce_rng(rng: Union[int, random.Random, None]) -> random.Random:
    if rng is None:
        return random.Random()
    if isinstance(rng, int):
        return random.Random(rng)
    if isinstance(rng, random.Random):
        return rng
    raise TypeError("rng must be None, an int seed, or random.Random")


def _capacity_aware_cuts(perm: List[int], data, num_slots: int) -> List[int]:
    cuts = [0]
    current_demand = 0.0

    for index, customer in enumerate(perm):
        demand = data.d[customer]
        can_open_new_slot = len(cuts) < num_slots
        would_exceed_capacity = current_demand + demand > data.q

        if would_exceed_capacity and current_demand > 0 and can_open_new_slot:
            cuts.append(index)
            current_demand = 0.0

        current_demand += demand

    cuts.append(len(perm))
    while len(cuts) < num_slots + 1:
        cuts.append(len(perm))
    return cuts[: num_slots + 1]


def _feasible_block_sizes(
    perm: List[int],
    start: int,
    remaining_slots: int,
    data,
) -> List[int]:
    remaining_customers = len(perm) - start
    if remaining_customers <= 0:
        return [0]

    min_take = 1
    max_take = remaining_customers
    feasible_sizes = []
    demand = 0.0

    for take in range(1, max_take + 1):
        demand += data.d[perm[start + take - 1]]
        if demand > data.q:
            break

        customers_left = remaining_customers - take
        slots_left = remaining_slots - 1
        if slots_left == 0:
            if customers_left == 0:
                feasible_sizes.append(take)
            continue

        if _can_pack_remaining(perm[start + take :], slots_left, data):
            feasible_sizes.append(take)

    return [size for size in feasible_sizes if size >= min_take]


def _can_pack_remaining(remaining_perm: List[int], slots_left: int, data) -> bool:
    if not remaining_perm:
        return True
    if slots_left <= 0:
        return False

    used_slots = 1
    current_demand = 0.0
    for customer in remaining_perm:
        demand = data.d[customer]
        if demand > data.q:
            return False
        if current_demand > 0 and current_demand + demand > data.q:
            used_slots += 1
            current_demand = 0.0
        current_demand += demand
    return used_slots <= slots_left


def _random_cuts(
    perm_len: int,
    num_slots: int,
    random_generator: random.Random,
) -> List[int]:
    inner_cuts = [random_generator.randint(0, perm_len) for _ in range(num_slots - 1)]
    return [0] + sorted(inner_cuts) + [perm_len]


def _sample_alpha(upper_bound: float, random_generator: random.Random) -> float:
    """Sample alpha with mostly safe waits but occasional long exploration."""

    if upper_bound <= 0:
        return 0.0
    if upper_bound <= 0.25:
        return random_generator.uniform(0.0, upper_bound)

    roll = random_generator.random()
    if roll < 0.65:
        return random_generator.uniform(0.0, min(0.25, upper_bound))
    if roll < 0.85:
        return random_generator.uniform(0.0, min(1.0, upper_bound))
    return random_generator.uniform(0.0, upper_bound)
