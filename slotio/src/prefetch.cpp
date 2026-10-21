#include "slotio/prefetch.hpp"

#include <cerrno>
#include <chrono>
#include <string>

namespace slotio {

namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
constexpr int64_t kClaimTimeoutNs = 30'000'000'000;
}  // namespace

Prefetch::Prefetch(Ensurer& en, Reader& reader, const File& experts, const PrefetchConfig& cfg)
    : en_(en), rd_(reader), file_(experts), cfg_(cfg) {
  std::lock_guard<std::mutex> g(en_.mutex());
  en_.set_prefetch(this);
}

Prefetch::~Prefetch() {
  std::lock_guard<std::mutex> g(en_.mutex());
  expire_locked(-1);
  en_.set_prefetch(nullptr);
}

void Prefetch::release_locked(const Ticket& t) {
  Pool& pool = en_.pool();
  if (pool.owner(t.slot) == t.idx) pool.set_owner(t.slot, -1);
  en_.clock().set_reserved(t.slot, false);
  en_.clock().clear_ref(t.slot);
}

int Prefetch::reserve_locked(int layer, int32_t idx) {
  Pool& pool = en_.pool();
  const PoolConfig& cfg = pool.config();
  if (static_cast<int>(tickets_.size()) >= cfg_.slot_cap) {
    ++st_.refused_cap;
    return -1;
  }
  en_.refresh_pins_locked();
  const long evictable = static_cast<long>(pool.slots()) - en_.pins().count() - static_cast<long>(tickets_.size());
  if (evictable <= cfg_.min_evictable) {
    ++st_.refused_evictable;
    return -1;
  }
  const int slots = pool.slots();
  int v;
  try {  // prefer a victim of the same layer (or an empty slot)
    v = en_.clock().victim(slots, [&](int s) {
      if (en_.pins().pinned(s)) return true;
      const int32_t o = pool.owner(s);
      return o >= 0 && o / cfg.experts != layer;
    });
  } catch (const Exhausted&) {
    try {
      v = en_.clock().victim(slots, [&](int s) { return en_.pins().pinned(s); });
    } catch (const Exhausted&) {
      ++st_.refused_evictable;
      return -1;
    }
  }
  const int32_t o = pool.owner(v);
  if (o >= 0) {
    pool.unmap(o);
    pool.set_owner(v, -1);
  }
  pool.set_owner(v, idx);
  en_.clock().clear_ref(v);
  en_.clock().set_reserved(v, true);
  return v;
}

int Prefetch::issue(int layer, const int32_t* ids, size_t n) {
  std::lock_guard<std::mutex> g(en_.mutex());
  Pool& pool = en_.pool();
  const PoolConfig& cfg = pool.config();
  if (layer < 0 || layer >= cfg.layers) throw std::invalid_argument("layer out of range");
  int issued = 0;
  for (size_t i = 0; i < n; ++i) {
    const int32_t e = ids[i];
    if (e < 0 || e >= cfg.experts) throw std::invalid_argument("expert id out of range");
    const int32_t idx = layer * cfg.experts + e;
    if (__atomic_load_n(&pool.map_host()[idx], __ATOMIC_ACQUIRE) >= 0 || tickets_.count(idx)) {
      ++st_.skipped;
      continue;
    }
    const int v = reserve_locked(layer, idx);
    if (v < 0) continue;
    Job job{&file_, static_cast<uint64_t>(idx) * cfg.record_bytes, static_cast<uint32_t>(cfg.record_bytes),
            pool.slot_host(v), Priority::Speculative};
    BatchId id;
    try {
      id = rd_.submit(&job, 1);
    } catch (...) {
      pool.set_owner(v, -1);
      en_.clock().set_reserved(v, false);
      throw;
    }
    tickets_.emplace(idx, Ticket{layer, idx, v, id, TicketState::InFlight});
    ++st_.issued;
    ++issued;
  }
  return issued;
}

bool Prefetch::claim_locked(int layer, int32_t e) {
  Pool& pool = en_.pool();
  const int32_t idx = layer * pool.config().experts + e;
  auto it = tickets_.find(idx);
  if (it == tickets_.end()) return false;
  Ticket t = it->second;
  BatchResult r = rd_.join(t.batch, 0);
  bool late = false;
  if (!r.ok && r.err == ETIMEDOUT) {
    late = true;
    const int64_t t0 = now_ns();
    r = rd_.join(t.batch, kClaimTimeoutNs);
    st_.wait_ns += static_cast<uint64_t>(now_ns() - t0);
    if (!r.ok && r.err == ETIMEDOUT)
      throw IoError("speculative read did not finish (layer " + std::to_string(layer) + ")", ETIMEDOUT);
  }
  tickets_.erase(it);
  if (!r.ok) {
    release_locked(t);
    ++st_.failed;
    return false;
  }
  en_.clock().set_reserved(t.slot, false);
  pool.publish(t.idx, t.slot);
  en_.clock().set_ref(t.slot);
  ++st_.adopted;
  ++(late ? st_.adopted_late : st_.adopted_ready);
  return true;
}

int Prefetch::expire_locked(int layer) {
  int n = 0;
  for (auto it = tickets_.begin(); it != tickets_.end();) {
    if (layer >= 0 && it->second.layer > layer) {
      ++it;
      continue;
    }
    BatchResult r = rd_.join(it->second.batch, -1);  // the lane may still write into the slot: wait for it
    release_locked(it->second);
    ++(r.ok ? st_.discarded : st_.failed);
    ++n;
    it = tickets_.erase(it);
  }
  return n;
}

int Prefetch::expire_layer(int layer) {
  std::lock_guard<std::mutex> g(en_.mutex());
  return expire_locked(layer);
}

int Prefetch::expire_all() {
  std::lock_guard<std::mutex> g(en_.mutex());
  return expire_locked(-1);
}

int Prefetch::outstanding() const {
  std::lock_guard<std::mutex> g(en_.mutex());
  return static_cast<int>(tickets_.size());
}

PrefetchStats Prefetch::stats() const {
  std::lock_guard<std::mutex> g(en_.mutex());
  return st_;
}

}  // namespace slotio
