#include <sched.h>
#include <time.h>

#include "slotio/rows.hpp"

namespace slotio {

namespace {
int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}
}  // namespace

void wait_seq(const uint64_t* p, uint64_t target, int64_t timeout_ns) {
  const int64_t deadline = mono_ns() + timeout_ns;
  for (unsigned spin = 0;; ++spin) {
    if (__atomic_load_n(p, __ATOMIC_ACQUIRE) >= target) return;
    if (timeout_ns >= 0 && mono_ns() >= deadline) throw Timeout("wait_seq timed out");
    if (spin >= 64) sched_yield();
  }
}

}  // namespace slotio
