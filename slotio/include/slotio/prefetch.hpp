// Speculative prefetch: tickets for forecast experts, read at Speculative priority into reserved CLOCK victim slots.
#pragma once
#include <cstdint>
#include <map>
#include <vector>

#include "slotio/pool.hpp"

namespace slotio {

// Queued and Reading are one state (the reader does not say which): InFlight.
enum class TicketState : uint8_t { InFlight, Ready, Adopted, Discarded, Failed };

struct PrefetchConfig {
  int slot_cap = 64;        // outstanding tickets
  int min_evictable = 256;  // a reservation needs slots - pinned - in_flight > min_evictable
};

struct PrefetchStats {
  uint64_t issued = 0;
  uint64_t adopted = 0;         // map entry published at claim (no copy)
  uint64_t adopted_ready = 0;   // of those, the read had already finished
  uint64_t adopted_late = 0;    // of those, claim had to wait for the read
  uint64_t discarded = 0;       // expired unadopted, slot returned
  uint64_t failed = 0;          // read failed; slot returned, demand re-reads
  uint64_t refused_evictable = 0;
  uint64_t refused_cap = 0;
  uint64_t skipped = 0;         // resident or already ticketed
  uint64_t wait_ns = 0;         // time claim spent waiting for reading tickets
};

class Prefetch {
 public:
  Prefetch(Ensurer& en, Reader& reader, const File& experts, const PrefetchConfig& cfg = PrefetchConfig{});
  ~Prefetch();
  Prefetch(const Prefetch&) = delete;
  Prefetch& operator=(const Prefetch&) = delete;

  // Tickets for (layer, ids). Returns how many were issued. Resident keys, ticketed keys and keys beyond the caps are skipped.
  int issue(int layer, const int32_t* ids, size_t n);
  // Join and return every unadopted ticket of layers <= layer. Returns how many were returned.
  int expire_layer(int layer);
  int expire_all();
  int outstanding() const;
  PrefetchStats stats() const;

  // For the Ensurer, mutex held. true: the ticket for the key finished and its slot is now published (hit).
  // A failed ticket is discarded (false: demand reads). A reading ticket is waited for (bounded, throws IoError ETIMEDOUT).
  bool claim_locked(int layer, int32_t e);
  int expire_locked(int layer);  // layer < 0: all

 private:
  struct Ticket {
    int layer;
    int32_t idx;
    int slot;
    BatchId batch;
    TicketState state;
  };
  void release_locked(const Ticket& t);
  int reserve_locked(int layer, int32_t idx);  // slot or -1 (counts the refusal)
  Ensurer& en_;
  Reader& rd_;
  const File& file_;
  PrefetchConfig cfg_;
  std::map<int32_t, Ticket> tickets_;
  PrefetchStats st_;
};

}  // namespace slotio
