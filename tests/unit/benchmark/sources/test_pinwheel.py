#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from omnimalloc import run_allocation, validate_allocation
from omnimalloc.benchmark.sources import BaseSource
from omnimalloc.benchmark.sources.pinwheel import PinwheelSource
from omnimalloc.primitives import Pool
from omnimalloc.primitives.utils import get_pressure


def _has_guillotine_cut(pool: Pool) -> bool:
    allocs = pool.allocations
    times = {a.start for a in allocs} | {a.end for a in allocs}
    mems = {a.offset for a in allocs} | {a.offset + a.size for a in allocs}
    for t in times:
        if min(a.start for a in allocs) < t < max(a.end for a in allocs) and not any(
            a.start < t < a.end for a in allocs
        ):
            return True
    for m in mems:
        if min(a.offset for a in allocs) < m < max(
            a.offset + a.size for a in allocs
        ) and not any(a.offset < m < a.offset + a.size for a in allocs):
            return True
    return False


def test_pinwheel_source_is_registered() -> None:
    assert "pinwheel_source" in BaseSource.registry()
    assert BaseSource.get("pinwheel_source") is PinwheelSource


def test_pinwheel_count_rounds_up_to_pinwheel_size() -> None:
    allocations = PinwheelSource(num_allocations=64).get_allocations()
    assert len(allocations) == 65
    assert (len(allocations) - 1) % 4 == 0


@pytest.mark.parametrize("num", [5, 17, 65, 257, 513])
def test_pinwheel_optimum_is_tight(num: int) -> None:
    capacity = 1024 * 1024
    source = PinwheelSource(num_allocations=num, capacity=capacity)
    allocations = source.get_allocations()
    assert get_pressure(allocations) == capacity


def test_pinwheel_allocations_fit_within_makespan() -> None:
    makespan = 4096
    source = PinwheelSource(num_allocations=64, makespan=makespan, min_size=1)
    for alloc in source.get_allocations():
        assert 0 <= alloc.start < alloc.end <= makespan


def test_pinwheel_respects_min_size() -> None:
    source = PinwheelSource(num_allocations=256, min_size=2048)
    assert all(a.size >= 2048 for a in source.get_allocations())


def test_pinwheel_ground_truth_is_valid_and_optimal() -> None:
    capacity = 1024 * 1024
    source = PinwheelSource(num_allocations=200, capacity=capacity)
    pool = source.get_ground_truth_pool()

    validate_allocation(pool)
    assert pool.is_allocated
    assert pool.size == capacity
    assert pool.pressure == capacity


def test_pinwheel_packing_is_non_guillotine() -> None:
    pool = PinwheelSource(num_allocations=64).get_ground_truth_pool()
    assert not _has_guillotine_cut(pool)


def test_pinwheel_ground_truth_matches_get_allocations() -> None:
    source = PinwheelSource(num_allocations=64)
    truth = source.get_ground_truth_pool()
    allocs = source.get_allocations()
    assert len(truth.allocations) == len(allocs)
    for t, a in zip(truth.allocations, allocs, strict=True):
        assert (t.start, t.end, t.size) == (a.start, a.end, a.size)


def test_pinwheel_is_deterministic_per_seed() -> None:
    a = PinwheelSource(num_allocations=129, seed=7).get_allocations()
    b = PinwheelSource(num_allocations=129, seed=7).get_allocations()
    c = PinwheelSource(num_allocations=129, seed=8).get_allocations()
    sig = lambda allocs: [(x.start, x.end, x.size) for x in allocs]  # noqa: E731
    assert sig(a) == sig(b)
    assert sig(a) != sig(c)


def test_pinwheel_distinct_pools_differ() -> None:
    source = PinwheelSource(num_allocations=33)
    pools = source.get_pools(num_pools=2)
    assert len(pools) == 2
    sig = lambda p: [(a.start, a.end, a.size) for a in p.allocations]  # noqa: E731
    assert sig(pools[0]) != sig(pools[1])


def test_pinwheel_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PinwheelSource(capacity=2048, min_size=1024)
    with pytest.raises(ValueError, match="makespan"):
        PinwheelSource(makespan=2, min_duration=1)
    with pytest.raises(ValueError, match="min_size"):
        PinwheelSource(min_size=0)


def test_pinwheel_no_allocator_beats_the_optimum() -> None:
    capacity = 1024 * 1024
    source = PinwheelSource(num_allocations=150, capacity=capacity)
    pool = source.get_pool()
    allocated = run_allocation(pool, "greedy_by_size_allocator_cpp", validate=True)
    assert allocated.size >= capacity
