// Expert slot pool, CLOCK victim selection, generation pins and the transactional ensure.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "slotio/reader.hpp"

namespace slotio {

class Prefetch;

enum class MemType { HostAlloc, VmmHostNuma, AnonThp, HostMallocForTests };

struct Exhausted : std::runtime_error {
  explicit Exhausted(const std::string& w) : std::runtime_error(w) {}
};

struct PoolConfig {
  uint64_t record_bytes = 2'764'800;
  int slots_per_chunk = 388;
  uint64_t chunk_bytes = 1ull << 30;
  int layers = 49;
  int experts = 512;
  int max_slots = 25'088;
};

struct PoolStats {
  int slots = 0, chunks = 0;
  uint64_t owned_bytes = 0;
  std::vector<int64_t> free_ns;  // one entry per chunk freed so far
  uint64_t alloc_ns = 0;
};

class Pool {
 public:
  Pool(MemType type, const PoolConfig& cfg);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  // Grow: allocate and pre-touch new chunks. Shrink: map=-1 for every slot >= n, then free whole chunks.
  // The caller must not run ensure concurrently (Ensurer::resize takes its lock).
  void resize_slots(int n);
  int slots() const { return slots_; }
  int chunks() const { return chunks_; }
  const PoolConfig& config() const { return cfg_; }
  MemType type() const { return type_; }

  uint8_t* slot_host(int s) const;
  uint64_t slot_dev(int s) const;
  uint64_t owned_bytes() const { return static_cast<uint64_t>(chunks_) * cfg_.chunk_bytes; }
  PoolStats stats() const;

  int32_t* map_host() const { return map_; }
  uint64_t map_dev() const { return map_dev_; }
  size_t map_len() const { return static_cast<size_t>(cfg_.layers) * cfg_.experts; }
  uint64_t* slot_ptr_host() const { return slot_ptr_; }
  uint64_t slot_ptr_dev() const { return slot_ptr_dev_; }

  // Owner bookkeeping (index layer*experts+e, or -1), used by shrink and by the ensurer.
  int32_t owner(int s) const { return owner_[s]; }
  void set_owner(int s, int32_t idx) { owner_[s] = idx; }
  void publish(int32_t idx, int32_t slot);  // map[idx] = slot, release store
  void unmap(int32_t idx);                  // map[idx] = -1, release store
  void set_shrink_hook(std::function<void(int)> h) { shrink_hook_ = std::move(h); }

 private:
  void alloc_chunk(int c);
  void free_chunk(int c);
  void cuda_init();
  MemType type_;
  PoolConfig cfg_;
  int slots_ = 0, chunks_ = 0;
  int32_t* map_ = nullptr;
  uint64_t* slot_ptr_ = nullptr;
  uint64_t map_dev_ = 0, slot_ptr_dev_ = 0;
  std::vector<int32_t> owner_;
  std::vector<uint8_t*> chunk_host_;
  std::vector<uint64_t> chunk_dev_;
  std::vector<uint64_t> chunk_handle_;  // CUmemGenericAllocationHandle for VmmHostNuma
  uint64_t va_ = 0, va_bytes_ = 0;
  uint64_t gran_ = 0;
  PoolStats st_;
  std::function<void(int)> shrink_hook_;
  bool cuda_ = false;
  int device_ = 0;
};

class Clock {
 public:
  explicit Clock(int max_slots) : ref_(max_slots, 0), reserved_(max_slots, 0) {}
  void set_ref(int s) { ref_[s] = 1; }
  void clear_ref(int s) { ref_[s] = 0; }
  void set_reserved(int s, bool r) { reserved_[s] = r; }
  bool reserved(int s) const { return reserved_[s] != 0; }
  // Global hand with reference bits; skips pinned and reserved slots. Throws Exhausted after 3*slots scans.
  int victim(int slots, const std::function<bool(int)>& pinned);
  void drop_above(int n);
  int hand() const { return hand_; }

 private:
  std::vector<uint8_t> ref_, reserved_;
  int hand_ = 0;
};

class Pins {
 public:
  struct Snapshot {
    std::vector<int64_t> gen;
    int count = 0;
  };
  explicit Pins(int max_slots) : gen_(max_slots, -1) {}
  void pin(int s, uint64_t gen);
  bool pinned(int s) const { return gen_[s] >= 0; }
  int count() const { return count_; }
  void retire_below(uint64_t progress);
  Snapshot snapshot() const { return Snapshot{gen_, count_}; }
  void restore(const Snapshot& s) {
    gen_ = s.gen;
    count_ = s.count;
  }
  void drop_above(int n);

 private:
  std::vector<int64_t> gen_;
  int count_ = 0;
};

struct EnsureTiming {
  int64_t total_ns = 0, read_wall_ns = 0, copy_tail_ns = 0;
  int hits = 0, misses = 0;
  int adopted = 0;  // of the hits, speculative tickets adopted by this call
  uint64_t bytes = 0;
};

class Ensurer {
 public:
  Ensurer(Pool& pool, Reader& reader, const File& experts);
  // Returns the slot of every requested expert (input order, duplicates allowed). Transactional.
  std::vector<int32_t> ensure(int layer, const int32_t* experts, size_t n, uint64_t gen, EnsureTiming* t = nullptr);
  void set_progress_ptr(const uint64_t* p) { progress_ = p; }
  void resize(int n);
  Pins& pins() { return pins_; }
  Clock& clock() { return clock_; }
  Pool& pool() { return pool_; }
  std::mutex& mutex() { return mu_; }  // the sweep's pool operations take the same lock as ensure
  // Speculative prefetch (prefetch.hpp) attaches here; ensure claims its tickets before the hit scan. Both calls need mu_.
  void set_prefetch(Prefetch* p) { prefetch_ = p; }
  void refresh_pins_locked();  // retire pins below the progress word

 private:
  Pool& pool_;
  Reader& reader_;
  const File& file_;
  Clock clock_;
  Pins pins_;
  const uint64_t* progress_ = nullptr;
  Prefetch* prefetch_ = nullptr;
  std::mutex mu_;
};

}  // namespace slotio
