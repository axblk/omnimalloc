//
// SPDX-License-Identifier: Apache-2.0
//

#include "partition.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace omnimalloc {

namespace {

constexpr int kExitEvent = 0;
constexpr int kEnterEvent = 1;

// For each allocation, the largest lower index that is interchangeable with it
// (identical start, end, and size), or -1. Interchangeable buffers are fully
// swappable, so the search may force them into index order (symmetry breaking).
// Sorting by (start, end, size, idx) groups interchangeable buffers in
// ascending index order, so each one's predecessor is simply the previous group
// member.
std::vector<int> compute_sym_predecessor(
    const std::vector<Allocation>& allocations) {
  const int n = static_cast<int>(allocations.size());
  auto key = [&](int i) {
    const Allocation& a = allocations[i];
    return std::make_tuple(a.start(), a.end(), a.size(), i);
  };
  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return key(a) < key(b); });

  std::vector<int> pred(static_cast<size_t>(n), -1);
  for (int k = 1; k < n; ++k) {
    const Allocation& cur = allocations[order[k]];
    const Allocation& prev = allocations[order[k - 1]];
    if (cur.start() == prev.start() && cur.end() == prev.end() &&
        cur.size() == prev.size()) {
      pred[order[k]] = order[k - 1];
    }
  }
  return pred;
}

}  // namespace

std::shared_ptr<Partition::SharedData> Partition::make_shared_data(
    std::vector<Allocation> allocations, std::vector<int64_t> alloc_sizes,
    std::vector<std::vector<int>> sections,
    std::vector<std::vector<int>> overlaps,
    std::vector<std::pair<int, int>> section_spans) {
  std::unordered_map<Allocation, int> index;
  index.reserve(allocations.size());
  for (int i = 0; i < static_cast<int>(allocations.size()); ++i) {
    index.emplace(allocations[i], i);
  }
  std::vector<int> sym_predecessor = compute_sym_predecessor(allocations);

  return std::make_shared<SharedData>(SharedData{
      std::move(allocations),
      std::move(index),
      std::move(alloc_sizes),
      std::move(sections),
      std::move(overlaps),
      std::move(section_spans),
      std::move(sym_predecessor),
  });
}

std::shared_ptr<Partition::SharedData> Partition::build_shared_data(
    std::vector<Allocation> allocations,
    std::vector<int64_t>& out_section_totals) {
  const int n = static_cast<int>(allocations.size());

  // (time, event, idx) tuples; std::tuple ordering puts EXIT (0) before
  // ENTER (1) at equal timestamps.
  std::vector<std::tuple<int64_t, int, int>> events;
  events.reserve(static_cast<size_t>(n) * 2);
  for (int i = 0; i < n; ++i) {
    events.emplace_back(allocations[i].start(), kEnterEvent, i);
    events.emplace_back(allocations[i].end(), kExitEvent, i);
  }
  std::sort(events.begin(), events.end());

  std::vector<std::vector<int>> sections;
  std::vector<std::vector<int>> overlaps(static_cast<size_t>(n));
  std::vector<int> span_start(static_cast<size_t>(n));
  std::vector<int> span_end(static_cast<size_t>(n));

  std::unordered_set<int> alive;

  bool has_prev = false;
  int64_t prev_time = 0;
  int prev_event = 0;

  for (const auto& [time, event, idx] : events) {
    // A section is a maximal span over which the alive set is constant, so
    // snapshot it (when non-empty) right before it changes.
    const bool time_changed = !has_prev || time != prev_time;
    const bool exit_to_enter =
        has_prev && prev_event == kExitEvent && event == kEnterEvent;
    if (!alive.empty() && (time_changed || exit_to_enter)) {
      sections.emplace_back(alive.begin(), alive.end());
    }

    has_prev = true;
    prev_time = time;
    prev_event = event;

    if (event == kExitEvent) {
      span_end[idx] = static_cast<int>(sections.size());
      alive.erase(idx);
    } else {
      for (int other : alive) {
        overlaps[idx].push_back(other);
        overlaps[other].push_back(idx);
      }
      span_start[idx] = static_cast<int>(sections.size());
      alive.insert(idx);
    }
  }

  std::vector<std::pair<int, int>> section_spans(static_cast<size_t>(n));
  std::vector<int64_t> alloc_sizes(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    section_spans[i] = {span_start[i], span_end[i]};
    alloc_sizes[i] = allocations[i].size();
  }

  out_section_totals.assign(sections.size(), 0);
  for (size_t s = 0; s < sections.size(); ++s) {
    int64_t total = 0;
    for (int idx : sections[s]) total += alloc_sizes[idx];
    out_section_totals[s] = total;
  }

  return make_shared_data(std::move(allocations), std::move(alloc_sizes),
                          std::move(sections), std::move(overlaps),
                          std::move(section_spans));
}

Partition::Partition(std::shared_ptr<const SharedData> data,
                     std::vector<int64_t> min_offsets,
                     std::vector<int64_t> section_floors,
                     std::vector<int64_t> section_totals,
                     std::vector<int64_t> offsets, int64_t best_height,
                     int num_allocated)
    : data_(std::move(data)),
      min_offsets_(std::move(min_offsets)),
      section_floors_(std::move(section_floors)),
      section_totals_(std::move(section_totals)),
      offsets_(std::move(offsets)),
      best_height_(best_height),
      num_allocated_(num_allocated) {}

Partition Partition::from_allocations(std::vector<Allocation> allocations) {
  std::vector<int64_t> section_totals;
  auto data = build_shared_data(std::move(allocations), section_totals);

  const int n = static_cast<int>(data->allocations.size());
  const int num_sections = static_cast<int>(data->sections.size());
  std::vector<int64_t> min_offsets(static_cast<size_t>(n), 0);
  std::vector<int64_t> section_floors(static_cast<size_t>(num_sections), 0);
  std::vector<int64_t> offsets(static_cast<size_t>(n), -1);

  return Partition(std::move(data), std::move(min_offsets),
                   std::move(section_floors), std::move(section_totals),
                   std::move(offsets), INT64_MAX, 0);
}

int Partition::index_of(const Allocation& a) const {
  auto it = data_->index.find(a);
  if (it == data_->index.end()) {
    throw std::out_of_range("Allocation not found in partition");
  }
  return it->second;
}

std::vector<int> Partition::unallocated_sorted_indices() const {
  const int n = static_cast<int>(data_->allocations.size());
  std::vector<int> indices;
  indices.reserve(static_cast<size_t>(n - num_allocated_));
  for (int i = 0; i < n; ++i) {
    if (offsets_[i] < 0) indices.push_back(i);
  }
  std::sort(indices.begin(), indices.end(), [this](int a, int b) {
    return std::tie(min_offsets_[a], a) < std::tie(min_offsets_[b], b);
  });
  return indices;
}

int64_t Partition::height() const {
  int64_t h = 0;
  const int n = static_cast<int>(data_->allocations.size());
  for (int i = 0; i < n; ++i) {
    if (offsets_[i] >= 0) h = std::max(h, offsets_[i] + data_->alloc_sizes[i]);
  }
  return h;
}

int64_t Partition::min_height() const {
  int64_t mh = INT64_MAX;
  const int n = static_cast<int>(data_->allocations.size());
  for (int i = 0; i < n; ++i) {
    if (offsets_[i] < 0)
      mh = std::min(mh, min_offsets_[i] + data_->alloc_sizes[i]);
  }
  return mh;
}

int64_t Partition::lower_bound() const {
  if (section_floors_.empty()) return 0;
  int64_t lb = 0;
  for (size_t s = 0; s < section_floors_.size(); ++s) {
    lb = std::max(lb, section_floors_[s] + section_totals_[s]);
  }
  return lb;
}

std::vector<int64_t> Partition::live_cuts() const {
  const int num_sections = static_cast<int>(data_->sections.size());
  std::vector<int64_t> cuts(static_cast<size_t>(std::max(num_sections - 1, 0)),
                            0);
  const int n = static_cast<int>(data_->allocations.size());
  for (int i = 0; i < n; ++i) {
    if (offsets_[i] >= 0) continue;  // placed buffers no longer block a split
    const auto [first, last] = data_->section_spans[i];
    for (int b = first; b < last - 1; ++b) cuts[b] += 1;
  }
  return cuts;
}

int64_t Partition::allocated_offset(const Allocation& a) const {
  const int idx = index_of(a);
  if (offsets_[idx] < 0) {
    throw std::out_of_range("Allocation is not allocated");
  }
  return offsets_[idx];
}

bool Partition::can_allocate_at(int idx, bool monotonic_floor) const {
  // Pre-allocation feasibility (section_inference): spanned sections take the
  // `top` bump and shed `size`. With `monotonic_floor`, every other section's
  // floor is additionally raised to the placement offset.
  const int64_t offset = min_offsets_[idx];
  const int64_t alloc_size = data_->alloc_sizes[idx];
  const auto [first, last] = data_->section_spans[idx];
  const int64_t top = offset + alloc_size;
  const int num_sections = static_cast<int>(data_->sections.size());

  for (int s = 0; s < num_sections; ++s) {
    int64_t floor = section_floors_[s];
    int64_t total = section_totals_[s];
    if (s >= first && s < last) {
      floor = std::max(floor, top);
      total -= alloc_size;
    } else if (monotonic_floor) {
      floor = std::max(floor, offset);
    }
    if (floor + total >= best_height_) return false;
  }
  return true;
}

bool Partition::is_feasible(int64_t offset, bool monotonic_floor) const {
  // Post-allocation feasibility (section_inference). With `monotonic_floor`,
  // propagate the parent's placement `offset` as a floor to every section
  // (its effective floor is at least `offset`).
  for (size_t s = 0; s < section_floors_.size(); ++s) {
    const int64_t floor = monotonic_floor ? std::max(section_floors_[s], offset)
                                          : section_floors_[s];
    if (floor + section_totals_[s] >= best_height_) {
      return false;
    }
  }
  return true;
}

Partition Partition::with_best_height(int64_t h) const {
  return Partition(data_, min_offsets_, section_floors_, section_totals_,
                   offsets_, h, num_allocated_);
}

Partition::PlacementUndo Partition::apply_at(int idx, bool floor_inference) {
  const int64_t offset = min_offsets_[idx];
  const int64_t alloc_size = data_->alloc_sizes[idx];
  const int64_t top = offset + alloc_size;
  const auto [first, last] = data_->section_spans[idx];

  PlacementUndo undo{idx, first, last, alloc_size, {}, {}};
  offsets_[idx] = offset;

  for (int s = first; s < last; ++s) {
    undo.floor_changes.emplace_back(s, section_floors_[s]);
    section_floors_[s] = std::max(section_floors_[s], top);
    section_totals_[s] -= alloc_size;
  }

  // Propagate the placed allocation's top to overlapping unplaced
  // allocations, tracking sections that gained a raised min_offset for the
  // floor inference below.
  std::vector<bool> affected(data_->sections.size(), false);
  for (int j : data_->overlaps[idx]) {
    if (offsets_[j] >= 0) continue;
    if (top <= min_offsets_[j]) continue;
    undo.min_offset_changes.emplace_back(j, min_offsets_[j]);
    min_offsets_[j] = top;
    if (!floor_inference) continue;
    const auto [jf, jl] = data_->section_spans[j];
    for (int s = jf; s < jl; ++s) affected[s] = true;
  }

  if (floor_inference) {
    // Sections in [first, last) already have floor >= top and host the
    // just-placed allocation, so their floor cannot be raised further.
    for (int s = first; s < last; ++s) affected[s] = false;

    // Raise section floors where every remaining unplaced allocation has
    // been pushed above the current floor (unallocated-floor inference).
    const int num_sections = static_cast<int>(data_->sections.size());
    for (int s = 0; s < num_sections; ++s) {
      if (!affected[s]) continue;
      const int64_t floor_s = section_floors_[s];
      int64_t s_min = INT64_MAX;
      bool broke = false;
      for (int b : data_->sections[s]) {
        if (offsets_[b] >= 0) continue;
        const int64_t off = min_offsets_[b];
        if (off <= floor_s) {
          broke = true;
          break;
        }
        if (off < s_min) s_min = off;
      }
      if (!broke && s_min < INT64_MAX) {
        undo.floor_changes.emplace_back(s, section_floors_[s]);
        section_floors_[s] = s_min;
      }
    }
  }

  ++num_allocated_;
  return undo;
}

void Partition::revert(const PlacementUndo& undo) {
  for (const auto& [s, old_floor] : undo.floor_changes) {
    section_floors_[s] = old_floor;
  }
  for (const auto& [j, old_min] : undo.min_offset_changes) {
    min_offsets_[j] = old_min;
  }
  for (int s = undo.first; s < undo.last; ++s) {
    section_totals_[s] += undo.alloc_size;
  }
  offsets_[undo.idx] = -1;
  --num_allocated_;
}

std::vector<int> Partition::sort_order(const std::string& heuristic) const {
  const int n = static_cast<int>(data_->allocations.size());
  const size_t key_len = heuristic.size() + 1;

  // Per-allocation sort keys; the original index is the final tiebreaker.
  std::vector<std::vector<int64_t>> keys(static_cast<size_t>(n),
                                         std::vector<int64_t>(key_len, 0));
  for (int i = 0; i < n; ++i) {
    const Allocation& a = data_->allocations[i];
    const auto [first, last] = data_->section_spans[i];
    for (size_t k = 0; k < heuristic.size(); ++k) {
      int64_t key = 0;
      switch (heuristic[k]) {
        case 'A':
          key = -a.area();
          break;
        case 'C':
          key = -static_cast<int64_t>(last - first);
          break;
        case 'L':
          key = -a.start();
          break;
        case 'O':
          key = -static_cast<int64_t>(data_->overlaps[i].size());
          break;
        case 'T': {
          int64_t max_total = 0;
          for (int s = first; s < last; ++s) {
            if (section_totals_[s] > max_total) max_total = section_totals_[s];
          }
          key = -max_total;
          break;
        }
        case 'U':
          key = -a.end();
          break;
        case 'W':
          key = -a.duration();
          break;
        case 'Z':
          key = -a.size();
          break;
        default:
          break;  // Unknown character contributes a zero key.
      }
      keys[i][k] = key;
    }
    keys[i][key_len - 1] = i;
  }

  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&keys](int a, int b) { return keys[a] < keys[b]; });
  return order;
}

Partition Partition::reorder(const std::string& heuristic) const {
  const int n = static_cast<int>(data_->allocations.size());
  const std::vector<int> order = sort_order(heuristic);

  // Inverse permutation: old index -> new index. Used to renumber every
  // index-bearing array while keeping the section grid (floors, totals, cuts)
  // untouched — unlike static_sort, which re-sweeps the grid and is therefore
  // only valid on a freshly built partition, not a decomposed slice.
  std::vector<int> inv(static_cast<size_t>(n));
  for (int new_idx = 0; new_idx < n; ++new_idx) inv[order[new_idx]] = new_idx;

  std::vector<Allocation> new_allocations;
  std::vector<int64_t> new_alloc_sizes(static_cast<size_t>(n));
  std::vector<std::pair<int, int>> new_section_spans(static_cast<size_t>(n));
  std::vector<std::vector<int>> new_overlaps(static_cast<size_t>(n));
  std::vector<int64_t> new_min_offsets(static_cast<size_t>(n));
  std::vector<int64_t> new_offsets(static_cast<size_t>(n));
  new_allocations.reserve(static_cast<size_t>(n));
  for (int new_idx = 0; new_idx < n; ++new_idx) {
    const int old_idx = order[new_idx];
    new_allocations.push_back(data_->allocations[old_idx]);
    new_alloc_sizes[new_idx] = data_->alloc_sizes[old_idx];
    new_section_spans[new_idx] = data_->section_spans[old_idx];
    new_min_offsets[new_idx] = min_offsets_[old_idx];
    new_offsets[new_idx] = offsets_[old_idx];
    for (int j : data_->overlaps[old_idx])
      new_overlaps[new_idx].push_back(inv[j]);
  }

  std::vector<std::vector<int>> new_sections(data_->sections.size());
  for (size_t s = 0; s < data_->sections.size(); ++s) {
    for (int idx : data_->sections[s]) new_sections[s].push_back(inv[idx]);
  }

  auto new_data =
      make_shared_data(std::move(new_allocations), std::move(new_alloc_sizes),
                       std::move(new_sections), std::move(new_overlaps),
                       std::move(new_section_spans));

  return Partition(std::move(new_data), std::move(new_min_offsets),
                   section_floors_, section_totals_, std::move(new_offsets),
                   best_height_, num_allocated_);
}

Partition Partition::static_sort(const std::string& heuristic) const {
  const int n = static_cast<int>(data_->allocations.size());
  const std::vector<int> order = sort_order(heuristic);

  // Rebuild SharedData under the new allocation ordering. The sweep-line
  // produces a temporally identical section structure (only the alloc
  // indices inside each section / overlap list / span are renumbered), so
  // per-section state can carry over unchanged.
  std::vector<Allocation> new_allocations;
  new_allocations.reserve(static_cast<size_t>(n));
  for (int i : order) new_allocations.push_back(data_->allocations[i]);

  std::vector<int64_t> rebuilt_totals;
  auto new_data = build_shared_data(std::move(new_allocations), rebuilt_totals);

  std::vector<int64_t> new_min_offsets(static_cast<size_t>(n));
  std::vector<int64_t> new_offsets(static_cast<size_t>(n));
  for (int new_idx = 0; new_idx < n; ++new_idx) {
    const int old_idx = order[new_idx];
    new_min_offsets[new_idx] = min_offsets_[old_idx];
    new_offsets[new_idx] = offsets_[old_idx];
  }

  return Partition(std::move(new_data), std::move(new_min_offsets),
                   section_floors_, section_totals_, std::move(new_offsets),
                   best_height_, num_allocated_);
}

std::optional<Partition> Partition::build_sub_partition(int start,
                                                        int end) const {
  const int n = static_cast<int>(data_->allocations.size());

  std::vector<int> sub_old_indices;
  sub_old_indices.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const auto [first, last] = data_->section_spans[i];
    const bool placed = offsets_[i] >= 0;
    const bool include = placed ? (start <= first && first < end)
                                : (first < end && last > start);
    if (include) sub_old_indices.push_back(i);
  }
  if (sub_old_indices.empty()) return std::nullopt;

  const int sub_n = static_cast<int>(sub_old_indices.size());
  const int sub_num_sections = end - start;

  std::unordered_map<int, int> old_to_new;
  old_to_new.reserve(static_cast<size_t>(sub_n));
  std::vector<Allocation> sub_allocs;
  sub_allocs.reserve(static_cast<size_t>(sub_n));
  std::vector<int64_t> sub_alloc_sizes;
  sub_alloc_sizes.reserve(static_cast<size_t>(sub_n));
  for (int new_idx = 0; new_idx < sub_n; ++new_idx) {
    const int old_idx = sub_old_indices[new_idx];
    old_to_new.emplace(old_idx, new_idx);
    sub_allocs.push_back(data_->allocations[old_idx]);
    sub_alloc_sizes.push_back(data_->alloc_sizes[old_idx]);
  }

  std::vector<std::pair<int, int>> sub_section_spans(
      static_cast<size_t>(sub_n));
  std::vector<int64_t> sub_min_offsets(static_cast<size_t>(sub_n));
  std::vector<int64_t> sub_offsets(static_cast<size_t>(sub_n));
  int sub_num_allocated = 0;
  for (int new_idx = 0; new_idx < sub_n; ++new_idx) {
    const int old_idx = sub_old_indices[new_idx];
    const auto [first, last] = data_->section_spans[old_idx];
    sub_section_spans[new_idx] = {std::max(first, start) - start,
                                  std::min(last, end) - start};
    sub_min_offsets[new_idx] = min_offsets_[old_idx];
    sub_offsets[new_idx] = offsets_[old_idx];
    if (sub_offsets[new_idx] >= 0) ++sub_num_allocated;
  }

  std::vector<std::vector<int>> sub_sections(
      static_cast<size_t>(sub_num_sections));
  for (int s = start; s < end; ++s) {
    auto& bucket = sub_sections[s - start];
    for (int idx : data_->sections[s]) {
      auto it = old_to_new.find(idx);
      if (it != old_to_new.end()) bucket.push_back(it->second);
    }
  }

  std::vector<std::vector<int>> sub_overlaps(static_cast<size_t>(sub_n));
  for (int new_idx = 0; new_idx < sub_n; ++new_idx) {
    const int old_idx = sub_old_indices[new_idx];
    auto& bucket = sub_overlaps[new_idx];
    for (int j_old : data_->overlaps[old_idx]) {
      auto it = old_to_new.find(j_old);
      if (it != old_to_new.end()) bucket.push_back(it->second);
    }
  }

  std::vector<int64_t> sub_section_floors(section_floors_.begin() + start,
                                          section_floors_.begin() + end);
  std::vector<int64_t> sub_section_totals(section_totals_.begin() + start,
                                          section_totals_.begin() + end);

  auto sub_data =
      make_shared_data(std::move(sub_allocs), std::move(sub_alloc_sizes),
                       std::move(sub_sections), std::move(sub_overlaps),
                       std::move(sub_section_spans));

  return Partition(std::move(sub_data), std::move(sub_min_offsets),
                   std::move(sub_section_floors), std::move(sub_section_totals),
                   std::move(sub_offsets), best_height_, sub_num_allocated);
}

std::optional<std::vector<Partition>> Partition::decompose() const {
  // A single sweep of the live per-boundary cut counts; a zero marks a safe
  // split. nullopt means no boundary is fully cut, so the problem is
  // monolithic.
  const std::vector<int64_t> cuts = live_cuts();
  std::vector<int> boundaries{0};
  for (size_t i = 0; i < cuts.size(); ++i) {
    if (cuts[i] == 0) boundaries.push_back(static_cast<int>(i + 1));
  }
  if (boundaries.size() == 1) return std::nullopt;
  boundaries.push_back(static_cast<int>(data_->sections.size()));

  std::vector<Partition> sub_parts;
  std::unordered_set<Allocation> seen;

  for (size_t b = 0; b + 1 < boundaries.size(); ++b) {
    std::optional<Partition> sub =
        build_sub_partition(boundaries[b], boundaries[b + 1]);
    if (!sub) continue;
    for (const Allocation& a : sub->allocations()) {
      if (!seen.insert(a).second) {
        throw std::invalid_argument(
            "Allocation appears in multiple sub-parts.");
      }
    }
    sub_parts.push_back(std::move(*sub));
  }

  return sub_parts;
}

namespace {

// Lower `shared` to `h` if it is smaller (lock-free min), so a solution found
// by one portfolio thread tightens the bound the others prune against.
void lower_shared(std::atomic<int64_t>* shared, int64_t h) {
  if (!shared) return;
  int64_t cur = shared->load(std::memory_order_relaxed);
  while (h < cur && !shared->compare_exchange_weak(cur, h)) {
  }
}

// Recursive branch-and-bound descent driving `solve_many`. `node` is
// mutated in place via `apply_at`/`revert` (mutate-and-undo) so each child
// costs O(touched) rather than a full copy of the state vectors. Its
// `best_height` is the live pruning bound for this mutate chain: lowered on
// every improvement and never restored, so a better solution found anywhere
// below immediately tightens pruning for its siblings and ancestors. The best
// complete solution is written to `best` (an output parameter) rather than
// returned, so propagating it up never collides with the bound it just
// lowered. `decompose` breaks the chain: each independent sub-part is solved
// with its own bound and accumulator, then merged. `shared` (when non-null) is
// a portfolio-wide atomic bound: solutions are published to it and it is read
// back per node, so sibling searches prune each other across threads.
void solve_dfs(Partition& node, int64_t min_offset, int min_idx,
               int64_t node_limit, int64_t& nodes,
               std::chrono::steady_clock::time_point deadline,
               const SearchOptions& opts, std::optional<Partition>& best,
               std::atomic<int64_t>* shared, const std::string& heuristic,
               bool first_leaf) {
  ++nodes;

  if (node.is_allocated()) {
    const int64_t h = node.height();
    if (h < node.best_height()) {
      best = node;
      node.set_best_height(h);
      lower_shared(shared, h);
    }
    return;
  }
  if (nodes >= node_limit) return;
  if (std::chrono::steady_clock::now() > deadline) return;

  // Pull in any improvement a sibling portfolio thread published.
  if (shared) {
    const int64_t gb = shared->load(std::memory_order_relaxed);
    if (gb < node.best_height()) node.set_best_height(gb);
  }

  if (opts.decompose) {
    if (std::optional<std::vector<Partition>> sub_parts = node.decompose()) {
      for (Partition& sub : *sub_parts) {
        // Re-derive the preordering on the sub-part's own (live) section
        // totals, exactly as minimalloc's SubSolve does: the heuristic's
        // section-total key shifts once the problem shrinks, and the resulting
        // canonical tiebreak order is what lets each independent sub-dive reach
        // a leaf cheaply instead of thrashing under the parent's stale order.
        if (!heuristic.empty()) sub = sub.reorder(heuristic);
        // A sub-part's solution height bounds only that subset, not the whole
        // problem, so it must not publish to (or read) the portfolio bound; it
        // inherits the bound captured above and the merged result publishes.
        // Solve each sub feasibility-first: it only has to fit under the
        // inherited bound, not reach its own optimum (minimalloc's SubSolve
        // returns at the first leaf). The global ratchet drives optimality.
        std::optional<Partition> sub_best;
        solve_dfs(sub, 0, 0, node_limit, nodes, deadline, opts, sub_best,
                  nullptr, heuristic, /*first_leaf=*/true);
        if (!sub_best) return;  // a sub-part has no solution below its bound
        sub = std::move(*sub_best);
      }
      Partition merged = Partition::merge(*sub_parts);
      const int64_t mh = merged.height();
      if (mh < node.best_height()) {
        node.set_best_height(mh);
        best = std::move(merged);
        lower_shared(shared, mh);
      }
      return;
    }
  }

  const std::vector<int> candidates = node.unallocated_sorted_indices();
  const std::vector<int64_t>& min_offsets = node.min_offsets();
  const std::vector<int>& sym_pred = node.sym_predecessor();
  const std::vector<int64_t>& offsets = node.offsets();
  const int64_t min_height = node.min_height();
  // Floors/totals are restored by `revert` before each loop iteration's break
  // check, so the lower bound is invariant across the candidate loop.
  const int64_t lower_bound = node.lower_bound();

  // Candidates are sorted by (min_offset, idx); skip the head that lex-precedes
  // the canonical floor in one binary search (matches the Python bisect_left).
  auto start = candidates.begin();
  if (opts.canonical) {
    start = std::lower_bound(candidates.begin(), candidates.end(),
                             std::pair<int64_t, int>{min_offset, min_idx},
                             [&](int cand, const std::pair<int64_t, int>& key) {
                               return std::pair<int64_t, int>{min_offsets[cand],
                                                              cand} < key;
                             });
  }

  for (auto it = start; it != candidates.end(); ++it) {
    const int alloc_idx = *it;

    // Symmetry breaking: interchangeable buffers (identical span + size) are
    // forced into index order, so skip one whose earlier-indexed twin is still
    // unplaced. This collapses the k! permutations of k identical buffers to a
    // single ordering — decisive on high-symmetry inputs (repeated buffers,
    // perfect tilings). A strengthening of the canonical form, so gated on it.
    if (opts.canonical) {
      const int pred = sym_pred[alloc_idx];
      if (pred >= 0 && offsets[pred] < 0) continue;
    }

    const int64_t alloc_offset = min_offsets[alloc_idx];

    // Dominance: every later candidate sits at or above the lowest unplaced
    // buffer's top, so it is dominated too.
    if (opts.dominance && alloc_offset >= min_height) break;

    if (!node.can_allocate_at(alloc_idx, opts.monotonic_floor)) continue;

    auto undo = node.apply_at(alloc_idx, opts.floor_inference);
    if (!node.is_feasible(alloc_offset, opts.monotonic_floor)) {
      node.revert(undo);
      continue;
    }

    solve_dfs(node, alloc_offset, alloc_idx, node_limit, nodes, deadline, opts,
              best, shared, heuristic, first_leaf);
    node.revert(undo);

    // Feasibility mode: the first complete packing under the bound is enough,
    // so stop exploring siblings the moment one is found.
    if (first_leaf && best) return;
    if (node.best_height() <= lower_bound) break;
  }
}

}  // namespace

std::optional<Partition> solve_many(
    const std::vector<Partition>& partitions, int64_t node_limit,
    double max_seconds, int64_t best_bound, SearchOptions options,
    int num_threads, const std::vector<std::string>& heuristics) {
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(max_seconds));

  std::atomic<int64_t> shared_best{best_bound};
  std::vector<std::optional<Partition>> results(partitions.size());
  std::atomic<size_t> next{0};

  auto worker = [&] {
    for (size_t i = next.fetch_add(1); i < partitions.size();
         i = next.fetch_add(1)) {
      const Partition& p = partitions[i];
      Partition root = p.with_best_height(std::min(
          p.best_height(), shared_best.load(std::memory_order_relaxed)));
      int64_t nodes = 0;
      const std::string& heuristic = i < heuristics.size() ? heuristics[i] : "";
      solve_dfs(root, 0, 0, node_limit, nodes, deadline, options, results[i],
                &shared_best, heuristic, /*first_leaf=*/false);
    }
  };

  const int n_workers =
      std::clamp(num_threads, 1, static_cast<int>(partitions.size()));
  if (n_workers <= 1) {
    worker();
  } else {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_workers));
    for (int t = 0; t < n_workers; ++t) threads.emplace_back(worker);
    for (std::thread& t : threads) t.join();
  }

  std::optional<Partition> best;
  for (const auto& r : results) {
    if (r && (!best || r->height() < best->height())) best = r;
  }
  return best;
}

Partition Partition::merge(const std::vector<Partition>& partitions) {
  std::vector<Allocation> all_allocs;
  std::vector<int64_t> alloc_sizes;
  std::vector<int64_t> offsets;
  std::unordered_set<Allocation> seen;
  int64_t best_height = 0;
  int num_allocated = 0;

  for (const Partition& p : partitions) {
    if (p.best_height_ > best_height) best_height = p.best_height_;
    const int pn = static_cast<int>(p.data_->allocations.size());
    for (int i = 0; i < pn; ++i) {
      const Allocation& a = p.data_->allocations[i];
      if (!seen.insert(a).second) {
        throw std::invalid_argument("Duplicate allocation in merge.");
      }
      all_allocs.push_back(a);
      alloc_sizes.push_back(p.data_->alloc_sizes[i]);
      offsets.push_back(p.offsets_[i]);
      if (p.offsets_[i] >= 0) ++num_allocated;
    }
  }

  const int n = static_cast<int>(all_allocs.size());

  std::unordered_map<Allocation, int> index;
  index.reserve(all_allocs.size());
  for (int i = 0; i < n; ++i) index.emplace(all_allocs[i], i);

  // Merged partitions are read-only result containers: only offsets() and
  // best_height() are meaningful, so the per-section structure (and the unused
  // sym_predecessor) stays empty rather than paying make_shared_data's derive.
  auto data = std::make_shared<SharedData>(SharedData{
      std::move(all_allocs),
      std::move(index),
      std::move(alloc_sizes),
      {},
      std::vector<std::vector<int>>(static_cast<size_t>(n)),
      std::vector<std::pair<int, int>>(static_cast<size_t>(n), {0, 0}),
  });

  std::vector<int64_t> min_offsets(static_cast<size_t>(n), 0);
  return Partition(std::move(data), std::move(min_offsets), {}, {},
                   std::move(offsets), best_height, num_allocated);
}

}  // namespace omnimalloc
