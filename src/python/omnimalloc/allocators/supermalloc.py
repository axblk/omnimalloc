#
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum

from omnimalloc._cpp import Partition, SearchOptions, solve_many
from omnimalloc.allocators.base import BaseAllocator
from omnimalloc.allocators.greedy_cpp import GreedyBySizeAllocatorCpp
from omnimalloc.primitives.allocation import Allocation

logger = logging.getLogger(__name__)


class SortKey(str, Enum):
    """Static-sort alphabet; each value is the char code C++ `static_sort` parses."""

    AREA = "A"
    SECTIONS = "C"
    LOWER = "L"
    OVERLAPS = "O"
    SECTION_TOTAL = "T"
    UPPER = "U"
    WIDTH = "W"
    SIZE = "Z"


Heuristic = tuple[SortKey, ...]

_W, _A, _T = SortKey.WIDTH, SortKey.AREA, SortKey.SECTION_TOTAL

DEFAULT_HEURISTICS: tuple[Heuristic, ...] = (
    (_W, _A, _T),
    (_T, _A, _W),
    (_T, _W, _A),
)

_DEEPEN_NODE_BUDGET = 1_000_000_000


@dataclass
class SupermallocConfig:
    timeout: float = 10.0
    heuristics: tuple[Heuristic, ...] = DEFAULT_HEURISTICS
    initial_node_budget: int = 2000
    multicore: int | None = None  # None = auto-detect
    canonical: bool = True
    dominance: bool = True
    floor_inference: bool = True
    monotonic_floor: bool = True
    decompose: bool = True

    def num_threads(self) -> int:
        if self.multicore is None:
            return os.cpu_count() or 1
        return max(1, self.multicore)

    def search_options(self) -> SearchOptions:
        options = SearchOptions()
        options.canonical = self.canonical
        options.dominance = self.dominance
        options.floor_inference = self.floor_inference
        options.monotonic_floor = self.monotonic_floor
        options.decompose = self.decompose
        return options


def _greedy_pre_bound(
    allocations: tuple[Allocation, ...],
) -> tuple[int, tuple[Allocation, ...]]:
    best_allocs = GreedyBySizeAllocatorCpp().allocate(allocations)
    best_height = max((alloc.offset or 0) + alloc.size for alloc in best_allocs)
    return best_height, best_allocs


@dataclass(frozen=True)
class _Portfolio:
    """The search invariants for one allocate() run, shared by every round."""

    partitions: list[Partition]
    options: SearchOptions
    threads: int
    heuristics: list[str]
    deadline: float

    def solve(
        self, target: int, budget: int, height: int, allocs: tuple[Allocation, ...]
    ) -> tuple[int, tuple[Allocation, ...], bool]:
        """Solve the portfolio at `target` in parallel, keeping any improvement."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return height, allocs, False
        result = solve_many(
            self.partitions,
            budget,
            remaining,
            min(target, height),
            self.options,
            self.threads,
            self.heuristics,
        )
        if result is not None and result.height < height:
            allocs = tuple(
                a.with_offset(result.allocated_offset(a)) for a in result.allocations
            )
            return result.height, allocs, True
        return height, allocs, False


class SupermallocAllocator(BaseAllocator):
    def __init__(self, config: SupermallocConfig | None = None) -> None:
        super().__init__()
        self._config = config or SupermallocConfig()

    def allocate(self, allocations: tuple[Allocation, ...]) -> tuple[Allocation, ...]:
        if not allocations:
            return allocations

        deadline = time.monotonic() + self._config.timeout
        height, allocs = _greedy_pre_bound(allocations)

        base = Partition.from_allocations(allocations)
        low = base.lower_bound
        heuristics = self._config.heuristics or DEFAULT_HEURISTICS
        heuristic_codes = ["".join(h) for h in heuristics]
        partitions = [base.static_sort(code) for code in heuristic_codes]
        budget = self._config.initial_node_budget
        portfolio = _Portfolio(
            partitions,
            self._config.search_options(),
            self._config.num_threads(),
            heuristic_codes,
            deadline,
        )

        # Probe at LB+1: LB-feasible instances solve here immediately.
        height, allocs, _ = portfolio.solve(low + 1, budget, height, allocs)

        # Ratchet the bound down from the current best toward LB+1.
        target, stalls = height, 0
        while height > low and stalls < 2 and time.monotonic() < deadline:
            height, allocs, improved = portfolio.solve(target, budget, height, allocs)
            if improved:
                target, stalls = height, 0
            else:
                budget *= 2
                target = max(low + 1, (target + low) // 2)
                stalls += 1

        # Pin at LB+1 with a growing budget; any hit is LB-optimal.
        stalls = 0
        while height > low and stalls < 3 and time.monotonic() < deadline:
            height, allocs, improved = portfolio.solve(low + 1, budget, height, allocs)
            if improved:
                break
            budget *= 2
            stalls += 1

        while height > low and time.monotonic() < deadline:
            height, allocs, improved = portfolio.solve(
                height, _DEEPEN_NODE_BUDGET, height, allocs
            )
            if not improved:
                break

        if height > low:
            logger.info("Supermalloc timed out")

        return allocs
