from __future__ import annotations

import math
import time
from typing import Any

try:
    from .chromosome import Individual
    from .decoder import decode_individual
except ImportError:  # Allows running src/main.py directly.
    from chromosome import Individual
    from decoder import decode_individual


def initialize_runtime(config: Any) -> None:
    """Ensure a config object has mutable runtime counters and cache."""

    if config is None:
        return
    if not hasattr(config, "runtime_stats") or not isinstance(config.runtime_stats, dict):
        config.runtime_stats = {}
    defaults = {
        "evaluation_requests": 0,
        "evaluations": 0,
        "cache_hits": 0,
        "decoder_time_seconds": 0.0,
        "local_search_evaluations": 0,
        "local_search_improvements": 0,
        "local_search_budget_skips": 0,
        "duplicate_candidates_removed": 0,
        "stopped_early": False,
        "stop_reason": "",
        "elapsed_seconds": 0.0,
    }
    for key, value in defaults.items():
        config.runtime_stats.setdefault(key, value)
    if not hasattr(config, "_evaluation_cache") or not isinstance(config._evaluation_cache, dict):
        config._evaluation_cache = {}


def reset_runtime(config: Any) -> None:
    """Clear runtime counters and cache before a full run."""

    if config is None:
        return
    config.runtime_stats = {}
    config._evaluation_cache = {}
    initialize_runtime(config)


def clone_individual_fast(ind: Individual) -> Individual:
    """Copy an individual without deepcopying immutable scalar state."""

    clone = Individual(
        perm=list(ind.perm),
        cuts=list(ind.cuts),
        alpha=dict(ind.alpha),
        objectives=tuple(ind.objectives),
        rank=ind.rank,
        crowding_distance=ind.crowding_distance,
        feasible=ind.feasible,
        metadata=dict(ind.metadata),
    )
    return clone


def individual_signature(ind: Individual, precision: int = 4) -> tuple:
    """Return a stable signature for duplicate detection and evaluation caching."""

    return (
        tuple(ind.perm),
        tuple(ind.cuts),
        tuple((slot, round(value, precision)) for slot, value in sorted(ind.alpha.items())),
    )


def evaluate_candidate(
    ind: Individual,
    data,
    config=None,
    source: str = "decoder",
) -> Individual:
    """Decode a candidate, apply penalties, and reuse cached evaluations when possible."""

    penalty_value = getattr(config, "penalty_value", 1_000_000.0)
    cache_enabled = bool(getattr(config, "evaluation_cache_enabled", False))
    precision = int(getattr(config, "alpha_signature_precision", 4))
    initialize_runtime(config)

    cache_key = individual_signature(ind, precision) if cache_enabled else None
    if config is not None:
        config.runtime_stats["evaluation_requests"] += 1
        if source == "local_search":
            config.runtime_stats["local_search_evaluations"] += 1
        if cache_enabled and cache_key in config._evaluation_cache:
            config.runtime_stats["cache_hits"] += 1
            _apply_record(ind, config._evaluation_cache[cache_key])
            return ind

    start = time.perf_counter()
    try:
        decoded = decode_individual(ind, data, update_individual=True)
    except (ValueError, KeyError) as exc:
        record = {
            "objectives": (penalty_value, penalty_value),
            "feasible": False,
            "metadata": {
                "raw_objectives": (math.inf, math.inf),
                "violations": [str(exc)],
                "violation_count": 1,
            },
        }
        _apply_record(ind, record)
    else:
        record = {
            "objectives": (decoded.F1, decoded.F2),
            "feasible": decoded.feasible,
            "metadata": {
                "F1": decoded.F1,
                "F2": decoded.F2,
                "raw_objectives": (decoded.F1, decoded.F2),
                "violations": list(decoded.violations),
                "violation_count": len(decoded.violations),
            },
        }
        if not math.isfinite(record["objectives"][0]) or not math.isfinite(record["objectives"][1]):
            record["objectives"] = (penalty_value, penalty_value)
            record["feasible"] = False
        elif not decoded.feasible:
            record["objectives"] = (decoded.F1 + penalty_value, decoded.F2 + penalty_value)
            record["feasible"] = False
        _apply_record(ind, record)

    if config is not None:
        config.runtime_stats["evaluations"] += 1
        config.runtime_stats["decoder_time_seconds"] += time.perf_counter() - start
        if cache_enabled and cache_key is not None:
            config._evaluation_cache[cache_key] = _record_from_individual(ind)
    return ind


def evaluation_budget_reached(config: Any) -> bool:
    initialize_runtime(config)
    max_evaluations = getattr(config, "max_evaluations", None)
    return max_evaluations is not None and config.runtime_stats["evaluations"] >= max_evaluations


def local_search_budget_reached(config: Any) -> bool:
    initialize_runtime(config)
    budget = getattr(config, "local_search_evaluation_budget", None)
    return budget is not None and config.runtime_stats["local_search_evaluations"] >= budget


def _apply_record(ind: Individual, record: dict) -> None:
    ind.objectives = tuple(record["objectives"])
    ind.feasible = bool(record["feasible"])
    ind.metadata = dict(record.get("metadata", {}))
    ind.rank = None
    ind.crowding_distance = 0.0


def _record_from_individual(ind: Individual) -> dict:
    return {
        "objectives": tuple(ind.objectives),
        "feasible": ind.feasible,
        "metadata": dict(ind.metadata),
    }
