from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .chromosome import Individual, split_by_cuts, validate_individual
except ImportError:  # Allows running src/main.py directly.
    from chromosome import Individual, split_by_cuts, validate_individual


@dataclass
class DecodedSolution:
    """Evaluated solution obtained from a chromosome."""

    trips: Dict[Tuple[int, int], List[int]]
    routes: Dict[Tuple[int, int], List[int]]
    t: Dict[Tuple[int, int, int], float]
    X: Dict[Tuple[int, int, int, int, int], int]
    Y: Dict[Tuple[int, int, int], int]
    F1: float
    F2: float
    feasible: bool
    violations: List[str] = field(default_factory=list)
    arc_periods: List[dict] = field(default_factory=list)


def locate_period(time_value: float, data) -> Optional[int]:
    """Return the period containing time_value, or None if no period matches."""

    for period in sorted(data.P):
        if data.LI[period] <= time_value <= data.LS[period]:
            return period
    return None


def closest_period(time_value: float, data) -> int:
    """Return a valid period even when time_value is outside all intervals."""

    periods = sorted(data.P)
    if not periods:
        raise ValueError("Cannot locate a closest period because data.P is empty")

    located = locate_period(time_value, data)
    if located is not None:
        return located

    first_period = min(periods, key=lambda period: data.LI[period])
    last_period = max(periods, key=lambda period: data.LS[period])
    if time_value < data.LI[first_period]:
        return first_period
    if time_value > data.LS[last_period]:
        return last_period

    return min(periods, key=lambda period: _distance_to_period(time_value, period, data))


def build_routes(ind: Individual, data) -> Dict[Tuple[int, int], List[int]]:
    """Build explicit routes [origin, customers..., dummy] for non-empty trips."""

    trips = split_by_cuts(ind, data)
    routes: Dict[Tuple[int, int], List[int]] = {}
    for slot, customers in trips.items():
        routes[slot] = [data.O] + customers + [data.dummy_depot] if customers else []
    return routes


def decode_individual(
    ind: Individual,
    data,
    update_individual: bool = True,
) -> DecodedSolution:
    """Decode and evaluate one chromosome without changing its route order."""

    validate_individual(ind, data, check_capacity=False, check_active_trip_prefix=False)

    trips = split_by_cuts(ind, data)
    routes = build_routes(ind, data)
    t: Dict[Tuple[int, int, int], float] = {}
    X: Dict[Tuple[int, int, int, int, int], int] = {}
    Y: Dict[Tuple[int, int, int], int] = {}
    arc_periods: List[dict] = []
    violations: List[str] = []
    F1 = 0.0
    F2 = 0.0

    violations.extend(_active_trip_prefix_violations(trips, data))

    for (v, k), customers in trips.items():
        for customer in customers:
            Y[(customer, v, k)] = 1

        load = sum(data.d[customer] for customer in customers)
        if load > data.q:
            violations.append(f"Capacity violation in slot {(v, k)}: load={load}, q={data.q}")

    start_time = min(data.LI.values())
    planning_end = max(data.LS.values())

    for k in sorted(data.K):
        previous_vehicle_available_time = start_time
        for v in sorted(data.V):
            slot = (v, k)
            route = routes[slot]
            if not route:
                continue

            current_time = (
                start_time + ind.alpha[slot]
                if v == min(data.V)
                else previous_vehicle_available_time + ind.alpha[slot]
            )
            t[(data.O, v, k)] = current_time

            for i, j in zip(route, route[1:]):
                depart_time = current_time
                period = locate_period(depart_time, data)
                if period is None:
                    violations.append(
                        f"Time outside periods at arc {(i, j)}, slot {(v, k)}, time={depart_time}"
                    )
                    break

                X[(i, j, period, v, k)] = 1

                emission = _require(data.e, (i, j, period), "e") + _require(
                    data.ee, (j, period), "ee"
                )
                cost = _require(data.g, (i, j, period), "g") + _require(
                    data.gg, (j, period), "gg"
                )
                travel_time = _require(data.T, (i, j, period), "T")
                service_time = _require(data.tt, (j, period), "tt")

                F1 += emission
                F2 += cost
                current_time = current_time + travel_time + service_time
                if current_time > data.LS[period]:
                    violations.append(
                        "Service completion outside selected period at "
                        f"arc {(i, j)}, slot {(v, k)}, period={period}, "
                        f"depart={depart_time}, completion={current_time}, LS={data.LS[period]}"
                    )
                t[(j, v, k)] = current_time

                arc_periods.append(
                    {
                        "slot": slot,
                        "arc": (i, j),
                        "period": period,
                        "depart_time": depart_time,
                        "travel_time": travel_time,
                        "service_time": service_time,
                        "arrival_or_departure_next": current_time,
                        "emission": emission,
                        "cost": cost,
                    }
                )

            previous_key = (data.dummy_depot, v, k)
            if previous_key not in t:
                continue
            previous_vehicle_available_time = t[previous_key]
            if previous_vehicle_available_time > planning_end:
                violations.append(
                    f"Vehicle {k} finishes slot {(v, k)} outside planning horizon: "
                    f"finish={previous_vehicle_available_time}"
                )

    feasible = not violations
    decoded = DecodedSolution(
        trips=trips,
        routes=routes,
        t=t,
        X=X,
        Y=Y,
        F1=F1,
        F2=F2,
        feasible=feasible,
        violations=violations,
        arc_periods=arc_periods,
    )

    if update_individual:
        ind.objectives = (F1, F2)
        ind.feasible = feasible
        ind.metadata["F1"] = F1
        ind.metadata["F2"] = F2
        ind.metadata["violations"] = list(violations)
        ind.metadata["violation_count"] = len(violations)

    return decoded


def _active_trip_prefix_violations(
    trips: Dict[Tuple[int, int], List[int]],
    data,
) -> List[str]:
    violations = []
    for vehicle in sorted(data.K):
        empty_seen = False
        for trip in sorted(data.V):
            slot = (trip, vehicle)
            active = bool(trips.get(slot))
            if active and empty_seen:
                violations.append(
                    f"Trip prefix violation for vehicle {vehicle}: slot {slot} is active after an empty trip"
                )
            if not active:
                empty_seen = True
    return violations


def _distance_to_period(time_value: float, period: int, data) -> float:
    if time_value < data.LI[period]:
        return data.LI[period] - time_value
    if time_value > data.LS[period]:
        return time_value - data.LS[period]
    return 0.0


def _require(values: dict, key: tuple, name: str) -> float:
    try:
        return values[key]
    except KeyError as exc:
        raise KeyError(f"Missing required parameter {name}{key}") from exc
