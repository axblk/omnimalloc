#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from omnimalloc import run_allocation, validate_allocation
from omnimalloc.benchmark.sources import BaseSource
from omnimalloc.benchmark.sources.tiling import TilingSource
from omnimalloc.primitives.utils import get_pressure


def test_tiling_source_is_registered() -> None:
    assert "tiling_source" in BaseSource.registry()
    assert BaseSource.get("tiling_source") is TilingSource


def test_tiling_source_produces_requested_count() -> None:
    source = TilingSource(num_allocations=128)
    allocations = source.get_allocations()
    assert len(allocations) == 128


@pytest.mark.parametrize("num", [1, 16, 64, 256, 512])
def test_tiling_optimum_is_tight(num: int) -> None:
    capacity = 1024 * 1024
    source = TilingSource(num_allocations=num, capacity=capacity)
    allocations = source.get_allocations()
    assert get_pressure(allocations) == capacity


def test_tiling_allocations_fit_within_makespan() -> None:
    makespan = 4096
    source = TilingSource(num_allocations=64, makespan=makespan, min_size=1)
    for alloc in source.get_allocations():
        assert 0 <= alloc.start < alloc.end <= makespan


def test_tiling_respects_min_size() -> None:
    source = TilingSource(num_allocations=256, min_size=2048)
    assert all(a.size >= 2048 for a in source.get_allocations())


def test_tiling_ground_truth_is_valid_and_optimal() -> None:
    capacity = 1024 * 1024
    source = TilingSource(num_allocations=200, capacity=capacity)
    pool = source.get_ground_truth_pool()

    validate_allocation(pool)
    assert pool.is_allocated
    assert pool.size == capacity
    assert pool.pressure == capacity


def test_tiling_ground_truth_matches_get_allocations() -> None:
    source = TilingSource(num_allocations=64)
    truth = source.get_ground_truth_pool()
    allocs = source.get_allocations()
    assert len(truth.allocations) == len(allocs)
    for t, a in zip(truth.allocations, allocs, strict=True):
        assert (t.start, t.end, t.size) == (a.start, a.end, a.size)


def test_tiling_is_deterministic_per_seed() -> None:
    a = TilingSource(num_allocations=128, seed=7).get_allocations()
    b = TilingSource(num_allocations=128, seed=7).get_allocations()
    c = TilingSource(num_allocations=128, seed=8).get_allocations()
    sig = lambda allocs: [(x.start, x.end, x.size) for x in allocs]  # noqa: E731
    assert sig(a) == sig(b)
    assert sig(a) != sig(c)


def test_tiling_distinct_pools_differ() -> None:
    source = TilingSource(num_allocations=32)
    pools = source.get_pools(num_pools=2)
    assert len(pools) == 2
    sig = lambda p: [(a.start, a.end, a.size) for a in p.allocations]  # noqa: E731
    assert sig(pools[0]) != sig(pools[1])


def test_tiling_variant_sweep_builds_ladder() -> None:
    source = TilingSource(capacity=1024 * 1024)
    for num in (64, 128, 256):
        pool = source.get_variant(num)
        assert len(pool.allocations) == num
        assert get_pressure(pool.allocations) == 1024 * 1024


def test_tiling_no_allocator_beats_the_optimum() -> None:
    capacity = 1024 * 1024
    source = TilingSource(num_allocations=150, capacity=capacity)
    pool = source.get_pool()
    allocated = run_allocation(pool, "greedy_by_size_allocator_cpp", validate=True)
    assert allocated.size >= capacity


def test_tiling_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="mem_cut_prob"):
        TilingSource(mem_cut_prob=1.5)
    with pytest.raises(ValueError, match="capacity"):
        TilingSource(capacity=10, min_size=1024)
    with pytest.raises(ValueError, match="min_size"):
        TilingSource(min_size=0)
