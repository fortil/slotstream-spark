#include <algorithm>

#include "slotio/pool.hpp"

namespace slotio {

int Clock::victim(int slots, const std::function<bool(int)>& pinned) {
  if (slots <= 0) throw Exhausted("pool has no slots");
  const long limit = 3L * slots;
  for (long i = 0; i < limit; ++i) {
    int s = hand_;
    hand_ = (hand_ + 1) % slots;
    if (reserved_[s] || pinned(s)) continue;
    if (ref_[s]) {
      ref_[s] = 0;
      continue;
    }
    return s;
  }
  throw Exhausted("CLOCK found no victim (all slots pinned or reserved)");
}

void Clock::drop_above(int n) {
  for (size_t s = static_cast<size_t>(n); s < ref_.size(); ++s) ref_[s] = reserved_[s] = 0;
  if (hand_ >= n) hand_ = 0;
}

void Pins::pin(int s, uint64_t gen) {
  if (gen_[s] < 0) ++count_;
  gen_[s] = std::max<int64_t>(gen_[s], static_cast<int64_t>(gen));
}

void Pins::retire_below(uint64_t progress) {
  for (size_t s = 0; s < gen_.size(); ++s) {
    if (gen_[s] >= 0 && static_cast<uint64_t>(gen_[s]) < progress) {
      gen_[s] = -1;
      --count_;
    }
  }
}

void Pins::drop_above(int n) {
  for (size_t s = static_cast<size_t>(n); s < gen_.size(); ++s) {
    if (gen_[s] >= 0) {
      gen_[s] = -1;
      --count_;
    }
  }
}

}  // namespace slotio
