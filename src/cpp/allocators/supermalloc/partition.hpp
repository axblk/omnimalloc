//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "primitives/allocation.hpp"

namespace omnimalloc {

// Toggles for the branch-and-bound pruning rules, so each can be ablated
// independently (see supermalloc.md "Kept"). All default on; turning one
// off removes exactly that rule from the search.
struct SearchOptions {
  bool canonical = true;
  bool dominance = true;
  bool floor_inference = true;
  bool monotonic_floor = true;
  bool decompose = true;
};

// Holds the structural data for a temporal allocation problem (sections,
// overlaps, section spans) plus the search state (offsets, floors, totals,
// best_height). All state-mutating operations return a new Partition
// with shared structural data, mirroring the immutable dataclass design on
// the Python side.
class Partition {
 public:
  // Build a fresh partition from `allocations` via a sweep-line scan over
  // (start, end) events.
  static Partition from_allocations(std::vector<Allocation> allocations);

  // Merge fully-solved sub-partitions into a single result partition. The
  // returned partition has empty per-section structural data; only the
  // `offsets()` and `best_height()` are meaningful for callers reading the
  // result.
  static Partition merge(const std::vector<Partition>& partitions);

  // Structural accessors (immutable, shared across copies).
  const std::vector<Allocation>& allocations() const noexcept {
    return data_->allocations;
  }
  // For each allocation, the largest lower index that is interchangeable with
  // it (identical span and size), or -1. Used for symmetry breaking.
  const std::vector<int>& sym_predecessor() const noexcept {
    return data_->sym_predecessor;
  }
  // Throws std::out_of_range if `a` is not in this partition.
  int index_of(const Allocation& a) const;

  // Indices of unallocated allocations in (min_offset, idx) order.
  // Used by the search loop to iterate candidates without per-step hash
  // lookups.
  std::vector<int> unallocated_sorted_indices() const;

  // Mutable state accessors.
  const std::vector<int64_t>& min_offsets() const noexcept {
    return min_offsets_;
  }
  // Per-allocation offsets; -1 indicates the allocation is unplaced.
  const std::vector<int64_t>& offsets() const noexcept { return offsets_; }
  int64_t best_height() const noexcept { return best_height_; }

  // Computed properties.
  bool is_allocated() const noexcept {
    return num_allocated_ == static_cast<int>(data_->allocations.size());
  }
  int64_t height() const;
  int64_t min_height() const;
  int64_t lower_bound() const;

  // Per-allocation queries.
  int64_t allocated_offset(const Allocation& a) const;
  // With `monotonic_floor`, every section the candidate does not span has its
  // floor raised to the placement offset (the canonical order places nothing
  // below it); the rule can be ablated independently.
  bool can_allocate_at(int idx, bool monotonic_floor = true) const;
  // `offset` is the placement offset of the parent allocation; with
  // `monotonic_floor` it raises every section's effective floor to that offset.
  bool is_feasible(int64_t offset = 0, bool monotonic_floor = true) const;

  Partition with_best_height(int64_t h) const;

  // Diff recorded by `apply_at` so `revert` can restore the exact prior state.
  struct PlacementUndo {
    int idx;
    int first;
    int last;
    int64_t alloc_size;
    std::vector<std::pair<int, int64_t>> floor_changes;
    std::vector<std::pair<int, int64_t>> min_offset_changes;
  };

  // In-place placement for the B&B hot loop: `apply_at` places allocation `idx`
  // at its `min_offset` and returns the diff; `revert` undoes it in O(touched),
  // avoiding a per-node copy of the state vectors. `floor_inference` skips the
  // unallocated-floor inference (the `affected[]` pass) when false, so that
  // rule can be ablated.
  PlacementUndo apply_at(int idx, bool floor_inference = true);
  void revert(const PlacementUndo& undo);
  void set_best_height(int64_t h) noexcept { best_height_ = h; }

  // Reorder allocations by `heuristic`. Each character defines a sort key
  // in descending order; the original index is the final tiebreaker.
  // Recognised characters: A, C, L, O, T, U, W, Z.
  Partition static_sort(const std::string& heuristic) const;

  // Like `static_sort` but preserves the existing section grid (floors,
  // totals, cuts) instead of re-sweeping it, so it is valid on a decomposed
  // sub-partition. Re-derives the heuristic order on the partition's own live
  // section totals, matching minimalloc's per-sub-partition SubSolve preorder.
  Partition reorder(const std::string& heuristic) const;

  // Split the partition at zero-cut boundaries into independent sub-parts, or
  // nullopt when no live boundary has a zero cut (the partition is monolithic).
  std::optional<std::vector<Partition>> decompose() const;

 private:
  // Immutable structural data shared across copies via shared_ptr.
  struct SharedData {
    std::vector<Allocation> allocations;
    std::unordered_map<Allocation, int> index;
    std::vector<int64_t> alloc_sizes;
    std::vector<std::vector<int>> sections;  // section_idx -> alloc indices
    std::vector<std::vector<int>> overlaps;  // alloc_idx -> overlapping indices
    std::vector<std::pair<int, int>>
        section_spans;                 // alloc_idx -> (first, last)
    std::vector<int> sym_predecessor;  // alloc_idx -> interchangeable pred, -1
  };

  Partition(std::shared_ptr<const SharedData> data,
            std::vector<int64_t> min_offsets,
            std::vector<int64_t> section_floors,
            std::vector<int64_t> section_totals, std::vector<int64_t> offsets,
            int64_t best_height, int num_allocated);

  // Sweep-line construction shared between `from_allocations` and
  // `static_sort`. Section totals are returned via `out_section_totals`.
  static std::shared_ptr<SharedData> build_shared_data(
      std::vector<Allocation> allocations,
      std::vector<int64_t>& out_section_totals);

  // Assemble a SharedData from its primary fields, deriving the `index` map and
  // `sym_predecessor` (both pure functions of `allocations`) so no call site
  // can build a SharedData with a stale or omitted derived field.
  static std::shared_ptr<SharedData> make_shared_data(
      std::vector<Allocation> allocations, std::vector<int64_t> alloc_sizes,
      std::vector<std::vector<int>> sections,
      std::vector<std::vector<int>> overlaps,
      std::vector<std::pair<int, int>> section_spans);

  // Build the sub-partition for the section band [start, end), or nullopt when
  // the band contains no allocations. Used by `decompose`.
  std::optional<Partition> build_sub_partition(int start, int end) const;

  // Permutation that orders allocations by `heuristic` (descending keys,
  // original index as tiebreak). Shared by `static_sort` and `reorder`.
  std::vector<int> sort_order(const std::string& heuristic) const;

  // Per-boundary count of still-unplaced allocations crossing each section
  // boundary; a zero marks a safe split point for `decompose`.
  std::vector<int64_t> live_cuts() const;

  std::shared_ptr<const SharedData> data_;
  std::vector<int64_t> min_offsets_;
  std::vector<int64_t> section_floors_;
  std::vector<int64_t> section_totals_;
  std::vector<int64_t> offsets_;
  int64_t best_height_;
  int num_allocated_;
};

// Run `partitions` (typically the same problem under different heuristic
// orderings) as an independent-search portfolio across `num_threads`, sharing
// one atomic best bound so a solution found by any search prunes the others.
// Returns the lowest partition found, or nullopt if none beats `best_bound`.
// `num_threads <= 1` runs them sequentially.
std::optional<Partition> solve_many(
    const std::vector<Partition>& partitions, int64_t node_limit,
    double max_seconds, int64_t best_bound, SearchOptions options,
    int num_threads, const std::vector<std::string>& heuristics = {});

}  // namespace omnimalloc
