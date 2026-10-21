#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>

#include "slotio/pool.hpp"
#include "slotio/prefetch.hpp"

namespace slotio {

namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

Ensurer::Ensurer(Pool& pool, Reader& reader, const File& experts)
    : pool_(pool), reader_(reader), file_(experts), clock_(pool.config().max_slots), pins_(pool.config().max_slots) {
  pool_.set_shrink_hook([this](int n) {
    pins_.drop_above(n);
    clock_.drop_above(n);
  });
}

void Ensurer::refresh_pins_locked() {
  if (!progress_) return;
  pins_.retire_below(__atomic_load_n(progress_, __ATOMIC_ACQUIRE));
}

void Ensurer::resize(int n) {
  std::lock_guard<std::mutex> g(mu_);
  if (prefetch_) prefetch_->expire_locked(-1);  // a reserved slot above the new size must not be left to a lane
  pool_.resize_slots(n);
}

std::vector<int32_t> Ensurer::ensure(int layer, const int32_t* experts, size_t n, uint64_t gen, EnsureTiming* timing) {
  std::lock_guard<std::mutex> g(mu_);
  const int64_t t_start = now_ns();
  const PoolConfig& cfg = pool_.config();
  // (1) validate and dedupe, keeping first-seen order
  if (layer < 0 || layer >= cfg.layers) throw std::invalid_argument("layer out of range");
  std::vector<int32_t> uniq;
  std::unordered_map<int32_t, int> seen;
  for (size_t i = 0; i < n; ++i) {
    int32_t e = experts[i];
    if (e < 0 || e >= cfg.experts) throw std::invalid_argument("expert id out of range");
    if (seen.emplace(e, 0).second) uniq.push_back(e);
  }
  if (progress_) {
    uint64_t p = __atomic_load_n(progress_, __ATOMIC_ACQUIRE);
    pins_.retire_below(p);
  }
  int32_t* map = pool_.map_host();
  int adopted = 0;
  if (prefetch_)
    for (int32_t e : uniq)
      if (__atomic_load_n(&map[layer * cfg.experts + e], __ATOMIC_ACQUIRE) < 0 && prefetch_->claim_locked(layer, e)) ++adopted;
  const int slots = pool_.slots();
  std::vector<int32_t> hit_slot(uniq.size(), -1);
  std::vector<size_t> miss;
  int unpinned_hits = 0;
  for (size_t i = 0; i < uniq.size(); ++i) {
    int32_t s = __atomic_load_n(&map[layer * cfg.experts + uniq[i]], __ATOMIC_ACQUIRE);
    if (s >= 0) {
      hit_slot[i] = s;
      if (!pins_.pinned(s)) ++unpinned_hits;
    } else {
      miss.push_back(i);
    }
  }
  // (2) capacity check, no state change on failure
  if (static_cast<long>(miss.size()) > static_cast<long>(slots) - pins_.count() - unpinned_hits)
    throw Exhausted("ensure needs " + std::to_string(miss.size()) + " slots, " +
                    std::to_string(slots - pins_.count() - unpinned_hits) + " unpinned");
  Pins::Snapshot snap = pins_.snapshot();
  // (3) pin the hits
  for (size_t i = 0; i < uniq.size(); ++i)
    if (hit_slot[i] >= 0) {
      pins_.pin(hit_slot[i], gen);
      clock_.set_ref(hit_slot[i]);
    }
  // (4) choose and pin every victim
  std::vector<int32_t> victims;
  try {
    for (size_t k = 0; k < miss.size(); ++k) {
      int v = clock_.victim(slots, [&](int s) { return pins_.pinned(s); });
      pins_.pin(v, gen);
      victims.push_back(v);
    }
  } catch (...) {
    pins_.restore(snap);
    throw;
  }
  // (5) unmap each victim with a release store before any write
  for (int32_t v : victims) {
    int32_t o = pool_.owner(v);
    if (o >= 0) {
      pool_.unmap(o);
      pool_.set_owner(v, -1);
    }
    clock_.clear_ref(v);
  }
  // (6) Demand jobs into the victim slots
  std::vector<Job> jobs;
  for (size_t k = 0; k < miss.size(); ++k) {
    uint64_t off = (static_cast<uint64_t>(layer) * cfg.experts + uniq[miss[k]]) * cfg.record_bytes;
    jobs.push_back(Job{&file_, off, static_cast<uint32_t>(cfg.record_bytes), pool_.slot_host(victims[k]), Priority::Demand});
  }
  BatchResult res;
  const int64_t t_sub = now_ns();
  res.ok = true;
  if (!jobs.empty()) res = reader_.join(reader_.submit(jobs.data(), jobs.size()));  // (7)
  const int64_t t_joined = now_ns();
  // (8) error: restore pins, leave victims unmapped, publish nothing
  if (!res.ok) {
    pins_.restore(snap);
    throw IoError("ensure read failed (layer " + std::to_string(layer) + ")", res.err);
  }
  // (9) publish
  for (size_t k = 0; k < miss.size(); ++k) {
    int32_t idx = layer * cfg.experts + uniq[miss[k]];
    pool_.set_owner(victims[k], idx);
    pool_.publish(idx, victims[k]);
    clock_.set_ref(victims[k]);
    hit_slot[miss[k]] = victims[k];
  }
  for (size_t i = 0; i < uniq.size(); ++i) seen[uniq[i]] = hit_slot[i];
  std::vector<int32_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = seen[experts[i]];
  if (timing) {
    timing->hits = static_cast<int>(uniq.size() - miss.size());
    timing->misses = static_cast<int>(miss.size());
    timing->adopted = adopted;
    timing->bytes = static_cast<uint64_t>(miss.size()) * cfg.record_bytes;
    timing->read_wall_ns = t_joined - t_sub;
    timing->copy_tail_ns = jobs.empty() ? 0 : std::max<int64_t>(0, res.last_copy_done_ns - res.last_read_done_ns);
    timing->total_ns = now_ns() - t_start;
  }
  return out;
}

}  // namespace slotio
