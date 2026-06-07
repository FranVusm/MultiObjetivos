from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .chromosome import Individual
except ImportError:  # Allows running src/main.py directly.
    from chromosome import Individual


Point = Tuple[float, float]


def objective_points(front: List[Individual], use_raw: bool = True) -> List[Point]:
    """Extract finite objective pairs from a front."""

    points = []
    for ind in front:
        if use_raw and "raw_objectives" in ind.metadata:
            point = ind.metadata["raw_objectives"]
        else:
            point = ind.objectives
        if math.isfinite(point[0]) and math.isfinite(point[1]):
            points.append((float(point[0]), float(point[1])))
    return points


def hypervolume_2d(points: List[Point], reference_point: Point) -> float:
    """Calculate dominated 2D hypervolume for a minimization front."""

    nondominated = _nondominated_points(points)
    ref_f1, ref_f2 = reference_point
    filtered = [(f1, f2) for f1, f2 in nondominated if f1 < ref_f1 and f2 < ref_f2]
    if not filtered:
        return 0.0

    volume = 0.0
    previous_f2 = ref_f2
    for f1, f2 in sorted(filtered):
        if f2 >= previous_f2:
            continue
        volume += (ref_f1 - f1) * (previous_f2 - f2)
        previous_f2 = f2
    return volume


def igd(
    approximate_points: List[Point],
    reference_points: List[Point],
) -> Optional[float]:
    """Return inverted generational distance from reference to approximation."""

    if not reference_points:
        return None
    if not approximate_points:
        return math.inf

    distances = []
    for ref in reference_points:
        distances.append(min(_distance(ref, point) for point in approximate_points))
    return sum(distances) / len(distances)


def normalize_points(
    points: List[Point],
    ideal: Optional[Point] = None,
    nadir: Optional[Point] = None,
) -> List[Point]:
    """Normalize points to [0, 1] using supplied or inferred ideal/nadir."""

    if not points:
        return []
    if ideal is None:
        ideal = (min(point[0] for point in points), min(point[1] for point in points))
    if nadir is None:
        nadir = (max(point[0] for point in points), max(point[1] for point in points))

    range_f1 = nadir[0] - ideal[0]
    range_f2 = nadir[1] - ideal[1]
    normalized = []
    for f1, f2 in points:
        norm_f1 = 0.0 if range_f1 == 0 else (f1 - ideal[0]) / range_f1
        norm_f2 = 0.0 if range_f2 == 0 else (f2 - ideal[1]) / range_f2
        normalized.append((norm_f1, norm_f2))
    return normalized


def load_reference_front(path: str) -> List[Point]:
    """Load a CSV reference front with F1,F2 columns; missing file is allowed."""

    input_path = Path(path)
    if not input_path.exists():
        return []

    points = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((float(row["F1"]), float(row["F2"])))
    return points


def _nondominated_points(points: List[Point]) -> List[Point]:
    unique_points = sorted(set(points))
    nondominated = []
    best_f2 = math.inf
    for f1, f2 in unique_points:
        if f2 < best_f2:
            nondominated.append((f1, f2))
            best_f2 = f2
    return nondominated


def _distance(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
