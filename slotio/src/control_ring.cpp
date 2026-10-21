#include "slotio/control_ring.hpp"

#include <sched.h>
#include <time.h>

namespace slotio {

namespace {
int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}
}  // namespace

void ControlRing::wait_done(size_t l, uint32_t value, int64_t timeout_ns) const {
  const int64_t deadline = mono_ns() + timeout_ns;
  for (unsigned spin = 0;; ++spin) {
    if (done(l) >= value) return;
    if (timeout_ns >= 0 && mono_ns() >= deadline) throw Timeout("ControlRing::wait_done timed out");
    if (spin >= 64) sched_yield();
  }
}

}  // namespace slotio
