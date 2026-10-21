#include <atomic>
#include <chrono>
#include <thread>

#include "doctest.h"
#include "slotio/control_ring.hpp"
#include "slotio/rows.hpp"

using namespace slotio;

TEST_CASE("control ring: wait_done returns at once when done[l] already reached the value") {
  uint32_t ready[4] = {0, 0, 0, 0}, done[4] = {0, 3, 0, 0};
  ControlRing r(ready, done, 4);
  r.wait_done(1, 3, 1'000'000'000);
  r.wait_done(1, 2, 1'000'000'000);
}

TEST_CASE("control ring: wait_done times out with Timeout when done[l] stays below the value") {
  uint32_t ready[2] = {0, 0}, done[2] = {0, 0};
  ControlRing r(ready, done, 2);
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_THROWS_AS(r.wait_done(0, 1, 20'000'000), Timeout);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  CHECK(ms >= 15);
  CHECK(ms < 2000);
}

TEST_CASE("control ring: wait_done wakes when another thread writes done[l] with release") {
  uint32_t ready[2] = {0, 0}, done[2] = {0, 0};
  ControlRing r(ready, done, 2);
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    __atomic_store_n(done + 1, 5u, __ATOMIC_RELEASE);
  });
  r.wait_done(1, 5, 5'000'000'000);
  CHECK(r.done(1) == 5u);
  t.join();
}

TEST_CASE("control ring: a layer outside the ring is refused") {
  uint32_t ready[2] = {0, 0}, done[2] = {0, 0};
  ControlRing r(ready, done, 2);
  CHECK_THROWS_AS(r.wait_done(2, 1, 1'000'000), std::out_of_range);
}
