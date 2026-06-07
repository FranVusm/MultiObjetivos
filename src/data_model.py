from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemData:
    N: list[int]
    P: list[int]
    K: list[int]
    V: list[int]
    O: int
    q: float
    d: dict[int, float]
    LI: dict[int, float]
    LS: dict[int, float]
    sigma: dict[tuple[int, int], float]
    e: dict[tuple[int, int, int], float]
    ee: dict[tuple[int, int], float]
    g: dict[tuple[int, int, int], float]
    gg: dict[tuple[int, int], float]
    T: dict[tuple[int, int, int], float]
    tt: dict[tuple[int, int], float]

    @property
    def depot(self) -> int:
        return self.O

    @property
    def dummy_depot(self) -> int:
        return self.N[-1]

    @property
    def customers(self) -> list[int]:
        return sorted(node for node, demand in self.d.items() if demand > 0)

    @property
    def max_vehicle_capacity(self) -> float:
        return self.q
