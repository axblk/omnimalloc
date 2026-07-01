#
# SPDX-License-Identifier: Apache-2.0
#

import random
from abc import abstractmethod
from dataclasses import dataclass

from omnimalloc.primitives import Allocation, Pool

from .base import BaseSource


@dataclass(frozen=True)
class _Tile:
    """A leaf rectangle in the (time x memory) plane.

    ``offset`` is the leaf's memory position in the ground-truth packing; it is
    discarded when handed to an allocator but kept for validation/reference.
    """

    start: int
    end: int
    offset: int
    size: int


class TilingBase(BaseSource):
    """Reverse-constructed packing problems with a known, tight optimum.

    A ``capacity x makespan`` rectangle is recursively split into leaf tiles
    until ``num_allocations`` is reached; each leaf becomes one allocation whose
    temporal extent is its time span and whose size is its memory height.
    Because the leaves perfectly tile the rectangle, peak pressure equals
    ``capacity`` exactly, so ``capacity`` is both a lower bound and a provably
    achievable optimum. Subclasses supply the cut geometry via :meth:`_can_split`
    and :meth:`_split`; sweeping ``num_allocations`` yields a difficulty ladder.
    """

    def __init__(
        self,
        num_allocations: int,
        capacity: int,
        makespan: int,
        min_size: int,
        min_duration: int,
        seed: int | None,
    ) -> None:
        super().__init__(num_allocations=num_allocations)
        self.capacity = capacity
        self.makespan = makespan
        self.min_size = min_size
        self.min_duration = min_duration
        self.seed = seed

    @abstractmethod
    def _can_split(self, tile: _Tile) -> bool: ...

    @abstractmethod
    def _split(self, tile: _Tile, rng: random.Random) -> list[_Tile]:
        """Split a tile into children; ``children[0]`` replaces it in place."""

    def _build_tiles(self, num: int, rng: random.Random) -> list[_Tile]:
        tiles = [_Tile(0, self.makespan, 0, self.capacity)]
        while len(tiles) < num:
            splittable = [i for i, t in enumerate(tiles) if self._can_split(t)]
            if not splittable:
                break
            # Weight by area so larger tiles split first, yielding balanced
            # layouts rather than a few dominant blocks plus many slivers.
            weights = [
                (tiles[i].end - tiles[i].start) * tiles[i].size for i in splittable
            ]
            idx = rng.choices(splittable, weights=weights, k=1)[0]
            children = self._split(tiles[idx], rng)
            tiles[idx] = children[0]
            tiles.extend(children[1:])
        return tiles

    def _rng(self, skip: int, num: int) -> random.Random:
        """Seed an independent instance per pool slice (see ``get_pools``)."""
        if self.seed is None:
            return random.Random()
        return random.Random(self.seed + (skip // num if num else 0))

    def get_allocations(
        self, num_allocations: int | None = None, skip: int = 0
    ) -> tuple[Allocation, ...]:
        num = num_allocations if num_allocations is not None else self.num_allocations
        tiles = self._build_tiles(num, self._rng(skip, num))
        return tuple(
            Allocation(id=i, size=t.size, start=t.start, end=t.end)
            for i, t in enumerate(tiles)
        )

    def get_ground_truth_pool(self, num_allocations: int | None = None) -> Pool:
        """Return the pool with the construction (zero-fragmentation) offsets."""
        num = num_allocations if num_allocations is not None else self.num_allocations
        tiles = self._build_tiles(num, self._rng(0, num))
        allocations = tuple(
            Allocation(id=i, size=t.size, start=t.start, end=t.end, offset=t.offset)
            for i, t in enumerate(tiles)
        )
        return Pool(id=f"{self.name()}_ground_truth", allocations=allocations)
