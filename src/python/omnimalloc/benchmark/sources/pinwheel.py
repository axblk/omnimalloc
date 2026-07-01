#
# SPDX-License-Identifier: Apache-2.0
#

import random

from .tiling_base import TilingBase, _Tile


class PinwheelSource(TilingBase):
    """Generate non-guillotine packing problems with a known, tight optimum.

    The adversarial sibling of :class:`TilingSource`: it splits each tile into a
    **pinwheel** — a central rectangle ringed by four blades in 90-degree
    rotational symmetry. No straight line crosses the whole rectangle without
    slicing a blade, so the packing is *non-guillotine*, and because the blades
    overlap across every interior time instant there is no clean time split
    either. Allocators that lean on decomposition or canonical guillotine search
    (supermalloc's ``decompose``, minimalloc's canonical recursion) cannot
    shortcut to the optimum and must actually search. See
    :class:`~.tiling_base.TilingBase` for the construction and optimality
    guarantee.

    Each split turns one tile into five (net ``+4``), so ``num_allocations`` is
    rounded up to the nearest ``1 + 4k``. Sweeping it (via integer variants)
    yields a graded difficulty ladder of nested pinwheels.
    """

    def __init__(
        self,
        num_allocations: int = 129,
        capacity: int = 1024 * 1024,
        makespan: int = 1024 * 1024,
        min_size: int = 1024,
        min_duration: int = 1,
        seed: int | None = 42,
    ) -> None:
        if min_size <= 0:
            raise ValueError("min_size must be positive")
        if min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if capacity < 3 * min_size:
            raise ValueError("capacity must be >= 3 * min_size to seat a pinwheel")
        if makespan < 3 * min_duration:
            raise ValueError("makespan must be >= 3 * min_duration to seat a pinwheel")
        super().__init__(
            num_allocations, capacity, makespan, min_size, min_duration, seed
        )

    def _can_split(self, tile: _Tile) -> bool:
        return (
            tile.end - tile.start >= 3 * self.min_duration
            and tile.size >= 3 * self.min_size
        )

    def _split(self, tile: _Tile, rng: random.Random) -> list[_Tile]:
        """Split a tile into a five-piece pinwheel (center + four blades)."""
        t0, t1, m0 = tile.start, tile.end, tile.offset
        width, height = t1 - t0, tile.size
        m1 = m0 + height
        # Blade thicknesses; bounds keep all five children >= the minima.
        p = rng.randint(self.min_duration, (width - self.min_duration) // 2)
        q = rng.randint(self.min_size, (height - self.min_size) // 2)
        return [
            _Tile(t0, t1 - p, m0, q),  # bottom
            _Tile(t1 - p, t1, m0, height - q),  # right
            _Tile(t0 + p, t1, m1 - q, q),  # top
            _Tile(t0, t0 + p, m0 + q, height - q),  # left
            _Tile(t0 + p, t1 - p, m0 + q, height - 2 * q),  # center
        ]
