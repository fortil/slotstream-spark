// Sweep reader: group reads into a 2x32 staging double buffer, plus the pool operations the sweep needs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "slotio/pool.hpp"

namespace slotio {

// Two groups of up to 32 expert records, allocated once in the platform pool memory type. Outside CLOCK and the slot table.
class Staging {
 public:
  Staging(MemType type, int groups = 2, int records = 32, uint64_t record_bytes = 2'764'800);
  ~Staging();
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;
  int groups() const { return groups_; }
  int records() const { return records_; }
  uint64_t record_bytes() const { return rec_; }
  uint64_t bytes() const { return static_cast<uint64_t>(groups_) * records_ * rec_; }  // 176,947,200 for 2x32 records
  uint64_t mapped_bytes() const { return mapped_; }                                     // bytes() rounded up to the allocation granularity
  uint8_t* host(int buf, int i) const;
  uint64_t dev(int buf, int i) const;

 private:
  MemType type_;
  int groups_, records_;
  uint64_t rec_, mapped_ = 0;
  uint8_t* host_ = nullptr;
  uint64_t dev_ = 0, handle_ = 0;
  bool cuda_ = false;
  int device_ = 0;
};

struct SweepTicket {
  BatchId id = 0;
  int buf = -1;
  int n = 0;
  int layer = -1;
  int admitted = 0;        // set by wait(): experts published into the pool from this read
  int admit_refused = 0;   // set by read_group(): admit ids that found no unpinned victim
};

class Sweep {
 public:
  Sweep(Ensurer& ensurer, Reader& reader, const File& experts, Staging& staging);
  // Queues one Sweep-priority read per expert into staging buffer `buf` (<= staging.records() experts). Safe path:
  // the reader preads O_DIRECT into a lane scratch and memcpys into staging. Throws std::logic_error when `buf`
  // still has a read in flight (a ready buffer may be reused: the caller owns the CUDA-event rule).
  // `admit` (ids that must be in `experts`): each non-resident one reserves a CLOCK victim slot now, the lane copies the
  // record into it as a second destination, and wait() publishes it. A pinned-out pool refuses (admit_refused).
  SweepTicket read_group(int layer, const int32_t* experts, size_t n, int buf, const int32_t* admit = nullptr, size_t n_admit = 0);
  // true: every record of the group is in staging. false: timed out (the ticket stays joinable). A failed read
  // throws IoError and leaves nothing handed out: record_* keep refusing this buffer until a new read succeeds.
  // On success also publishes the reserved admission slots and sets t.admitted. On IoError the reserved slots are
  // released unpublished.
  bool wait(SweepTicket& t, int64_t timeout_ms);
  uint64_t record_dev(const SweepTicket& t, int i) const;
  const uint8_t* record_host(const SweepTicket& t, int i) const;

  void set_generation(uint64_t gen) { gen_ = gen; }
  // Pins the slot for the current generation without setting the CLOCK reference bit; nullopt when not resident.
  std::optional<uint64_t> resident_ptr_nopin_touch(int layer, int e);
  // Final pass only: an unpinned CLOCK victim takes the record at host address `src` (one record_bytes copy) and the
  // expert is published. true if already resident or admitted, false when every slot is pinned.
  bool admit_from_staging(int layer, int e, uint64_t src);
  Staging& staging() { return st_; }

 private:
  enum class State : uint8_t { Idle, InFlight, Ready, Failed };
  Ensurer& en_;
  Reader& rd_;
  const File& file_;
  Staging& st_;
  uint64_t gen_ = 0;
  std::vector<State> state_;
  std::vector<int> count_;
  std::vector<std::vector<std::pair<int, int32_t>>> reserved_;  // per buffer: (slot, map index) awaiting publish
};

}  // namespace slotio
