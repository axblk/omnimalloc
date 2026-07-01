from omnimalloc.benchmark import (
    plot_benchmark,
    run_benchmark,
    save_benchmark,
)


def test_minimalloc_benchmark() -> None:
    allocators = (
        "greedy_by_size_allocator_cpp",
        "greedy_by_all_allocator_cpp",
        "minimalloc_allocator",
        "supermalloc_allocator",
    )

    sources = ("minimalloc_source",)

    variants = (
        "A.1048576",
        "B.1048576",
        "C.1048576",
        "D.1048576",
        "E.1048576",
        "F.1048576",
        "G.1048576",
        "H.1048576",
        "I.1048576",
        "J.1048576",
        "K.1048576",
    )

    campaign = run_benchmark(
        allocators=allocators,
        sources=sources,
        variants=variants,
        validate=True,
    )
    plot_benchmark(campaign)
    save_benchmark(campaign, "benchmark_results_minimalloc")


def test_tiling_benchmark() -> None:
    allocators = (
        "greedy_by_size_allocator_cpp",
        "greedy_by_all_allocator_cpp",
        "minimalloc_allocator",
        "supermalloc_allocator",
    )

    sources = ("tiling_source",)

    variants = (
        64,
        128,
        256,
        500,
        512,
        750,
        1024,
        1500,
    )

    campaign = run_benchmark(
        allocators=allocators,
        sources=sources,
        variants=variants,
        validate=True,
    )
    plot_benchmark(campaign)
    save_benchmark(campaign, "benchmark_results_tiling")


def test_pinwheel_benchmark() -> None:
    allocators = (
        "greedy_by_size_allocator_cpp",
        "greedy_by_all_allocator_cpp",
        "minimalloc_allocator",
        "supermalloc_allocator",
    )

    sources = ("pinwheel_source",)

    variants = (
        64,
        128,
        256,
        500,
        512,
        750,
        1024,
        1500,
    )

    campaign = run_benchmark(
        allocators=allocators,
        sources=sources,
        variants=variants,
        validate=True,
    )
    plot_benchmark(campaign)
    save_benchmark(campaign, "benchmark_results_pinwheel")
