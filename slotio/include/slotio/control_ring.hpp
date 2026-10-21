#pragma once
// Control ring of the per-token graph (plan step 91): two arrays of u32 words in pinned host memory that the Python side
// allocates (torch pinned tensors, so the graph can also copy device values into them) and this class accesses with the
// right memory order.
//   ready[l]  written by the host (release store) once ensure(l) published the experts of layer l; the graph waits on it
//             with a stream memory operation (cuStreamWaitValue32) and re-arms it to 0 with cuStreamWriteValue32
//   done[l]   written by the graph at the end of the segment that runs MoE(l) with the layer's sequence number
// The class does not own the memory and never calls CUDA.
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "slotio/rows.hpp"

namespace slotio {

class ControlRing {
 public:
  ControlRing(uint32_t* ready, uint32_t* done, size_t n) : ready_(ready), done_(done), n_(n) {
    if (!ready || !done || n == 0) throw std::invalid_argument("ControlRing needs both arrays and n > 0");
  }
  size_t size() const { return n_; }
  // Release store of 1 into ready[l]: the ensure of layer l is complete and visible to the device.
  void publish(size_t l, uint32_t value = 1) { __atomic_store_n(ready_ + check(l), value, __ATOMIC_RELEASE); }
  uint32_t ready(size_t l) const { return __atomic_load_n(ready_ + check(l), __ATOMIC_ACQUIRE); }
  uint32_t done(size_t l) const { return __atomic_load_n(done_ + check(l), __ATOMIC_ACQUIRE); }
  // Number of ready words that are not 0 (the graph re-arms each one after its wait).
  size_t armed() const {
    size_t c = 0;
    for (size_t i = 0; i < n_; ++i) c += __atomic_load_n(ready_ + i, __ATOMIC_ACQUIRE) != 0;
    return c;
  }
  // Publish every layer in [from, n): drains a graph that is waiting (error paths).
  void publish_from(size_t from) {
    for (size_t i = from; i < n_; ++i) publish(i);
  }
  void clear() {
    for (size_t i = 0; i < n_; ++i) {
      __atomic_store_n(ready_ + i, 0u, __ATOMIC_RELEASE);
      __atomic_store_n(done_ + i, 0u, __ATOMIC_RELEASE);
    }
  }
  // Spins (acquire load, sched_yield) until done[l] >= value; throws Timeout after timeout_ns.
  void wait_done(size_t l, uint32_t value, int64_t timeout_ns) const;

 private:
  size_t check(size_t l) const {
    if (l >= n_) throw std::out_of_range("ControlRing index");
    return l;
  }
  uint32_t* ready_;
  uint32_t* done_;
  size_t n_;
};

}  // namespace slotio
