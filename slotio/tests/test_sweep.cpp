#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "doctest.h"
#include "slotio/sweep.hpp"

using namespace slotio;

namespace {
constexpr uint32_t kRec = 4096;
constexpr int kN = 64;

struct FileFixture {
  std::string path = std::string(SLOTIO_BUILD_DIR) + "/t_sweep.bin";
  std::vector<uint8_t> bytes;
  File file;
  FileFixture() {
    bytes.resize(size_t(kN) * kRec);
    uint32_t x = 777;
    for (auto& b : bytes) {
      x = x * 1664525u + 1013904223u;
      b = uint8_t(x >> 24);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    f.close();
    file = File::open_direct(path);
  }
  ~FileFixture() { ::unlink(path.c_str()); }
};

struct Fixture : FileFixture {
  Reader reader;
  Pool pool;
  Staging staging;
  Ensurer ensurer;
  Sweep sweep;
  explicit Fixture(int slots = 8)
      : pool(MemType::HostMallocForTests, cfg()),
        staging(MemType::HostMallocForTests, 2, 32, kRec),
        ensurer(pool, reader, file),
        sweep(ensurer, reader, file, staging) {
    pool.resize_slots(slots);
  }
  static PoolConfig cfg() {
    PoolConfig c;
    c.record_bytes = kRec;
    c.chunk_bytes = 1ull << 21;
    c.max_slots = 64;
    return c;
  }
  std::vector<int32_t> ids(int lo, int n) {
    std::vector<int32_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = lo + i;
    return v;
  }
  // buffered reference read of expert e (plain pread, no O_DIRECT)
  std::vector<uint8_t> reference(int e) { return std::vector<uint8_t>(bytes.begin() + size_t(e) * kRec, bytes.begin() + size_t(e + 1) * kRec); }
};
}  // namespace

TEST_CASE("staging allocation is 2 x 32 records and is logged") {
  Staging st(MemType::HostMallocForTests);  // default geometry: 2 groups x 32 records x 2,764,800 B
  std::printf("staging allocation size check: %llu B\n", static_cast<unsigned long long>(st.bytes()));
  CHECK(st.bytes() == 176'947'200ull);
  CHECK(st.groups() == 2);
  CHECK(st.records() == 32);
  CHECK(st.dev(1, 31) - st.dev(0, 0) == (32ull + 31ull) * 2'764'800ull);
}

TEST_CASE("64 experts through the sweep equal a buffered reference read, in-flight stays <= 12") {
  Fixture fx;
  for (int buf = 0; buf < 2; ++buf) {
    auto ids = fx.ids(buf * 32, 32);
    SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), buf);
    CHECK(fx.sweep.wait(t, 5000));
    for (int i = 0; i < 32; ++i)
      CHECK(std::memcmp(fx.sweep.record_host(t, i), fx.reference(ids[i]).data(), kRec) == 0);
  }
  CHECK(fx.reader.stats().max_inflight <= 12);
  CHECK(fx.reader.stats().jobs == 64);
}

TEST_CASE("an injected ReadFault fails the ticket and hands nothing out") {
  Fixture fx;
  fx.reader.set_fault(std::make_shared<ReadFault>(5));
  auto ids = fx.ids(0, 32);
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0);
  CHECK_THROWS_AS(fx.sweep.wait(t, 5000), IoError);
  CHECK_THROWS_AS(fx.sweep.record_host(t, 0), std::logic_error);
  CHECK_THROWS_AS(fx.sweep.record_dev(t, 0), std::logic_error);
  // the one-shot fault is spent: the same buffer reads cleanly again
  SweepTicket t2 = fx.sweep.read_group(0, ids.data(), ids.size(), 0);
  CHECK(fx.sweep.wait(t2, 5000));
  CHECK(std::memcmp(fx.sweep.record_host(t2, 7), fx.reference(7).data(), kRec) == 0);
}

TEST_CASE("a buffer with a read in flight refuses a second read, and wait is bounded") {
  Fixture fx;
  fx.reader.set_paused(true);
  auto ids = fx.ids(0, 4);
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 1);
  CHECK_THROWS_AS(fx.sweep.read_group(0, ids.data(), ids.size(), 1), std::logic_error);
  CHECK_FALSE(fx.sweep.wait(t, 20));
  fx.reader.set_paused(false);
  CHECK(fx.sweep.wait(t, 5000));
}

TEST_CASE("resident_ptr_nopin_touch pins without the reference bit; admit_from_staging publishes") {
  Fixture fx(4);
  fx.sweep.set_generation(9);
  CHECK_FALSE(fx.sweep.resident_ptr_nopin_touch(0, 5).has_value());
  auto ids = fx.ids(0, 2);
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0);
  REQUIRE(fx.sweep.wait(t, 5000));
  CHECK(fx.sweep.admit_from_staging(0, 1, reinterpret_cast<uint64_t>(fx.sweep.record_host(t, 1))));
  int32_t s = fx.pool.map_host()[1];
  REQUIRE(s >= 0);
  CHECK(std::memcmp(fx.pool.slot_host(s), fx.reference(1).data(), kRec) == 0);
  CHECK(fx.sweep.admit_from_staging(0, 1, reinterpret_cast<uint64_t>(fx.sweep.record_host(t, 1))));  // already resident
  const int pinned_before = fx.ensurer.pins().count();
  auto p = fx.sweep.resident_ptr_nopin_touch(0, 1);
  REQUIRE(p.has_value());
  CHECK(*p == fx.pool.slot_dev(s));
  CHECK(fx.ensurer.pins().count() == pinned_before + 1);
}

TEST_CASE("admit_from_staging returns false when every slot is pinned") {
  Fixture fx(2);
  fx.sweep.set_generation(3);
  int32_t a[] = {0, 1};
  fx.ensurer.ensure(0, a, 2, 3);  // both slots pinned for generation 3
  auto ids = fx.ids(10, 1);
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0);
  REQUIRE(fx.sweep.wait(t, 5000));
  CHECK_FALSE(fx.sweep.admit_from_staging(0, 10, reinterpret_cast<uint64_t>(fx.sweep.record_host(t, 0))));
  CHECK(fx.pool.map_host()[10] == -1);
}

TEST_CASE("admit ids reserve a slot at issue time; the lane copies; wait publishes") {
  Fixture fx(4);
  fx.sweep.set_generation(5);
  auto ids = fx.ids(0, 32);
  int32_t adm[] = {1, 5, 9};
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0, adm, 3);
  CHECK(t.admit_refused == 0);
  REQUIRE(fx.sweep.wait(t, 5000));
  CHECK(t.admitted == 3);
  for (int e : {1, 5, 9}) {
    int32_t s = fx.pool.map_host()[e];
    REQUIRE(s >= 0);
    CHECK(std::memcmp(fx.pool.slot_host(s), fx.reference(e).data(), kRec) == 0);
  }
  CHECK(fx.reader.stats().jobs == 32);
}

TEST_CASE("a ReadFault during an admit read publishes nothing and frees the reserved slots") {
  Fixture fx(4);
  fx.sweep.set_generation(5);
  fx.reader.set_fault(std::make_shared<ReadFault>(3));
  auto ids = fx.ids(0, 32);
  int32_t adm[] = {1, 5, 9};
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0, adm, 3);
  CHECK_THROWS_AS(fx.sweep.wait(t, 5000), IoError);
  for (int e : {1, 5, 9}) CHECK(fx.pool.map_host()[e] == -1);
  for (int s = 0; s < fx.pool.slots(); ++s) CHECK(fx.pool.owner(s) == -1);
  CHECK(t.admitted == 0);
}

TEST_CASE("admit with every slot pinned is refused, the read itself succeeds") {
  Fixture fx(2);
  fx.sweep.set_generation(3);
  int32_t a[] = {0, 1};
  fx.ensurer.ensure(0, a, 2, 3);
  auto ids = fx.ids(10, 4);
  int32_t adm[] = {10, 11};
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 0, adm, 2);
  CHECK(t.admit_refused == 2);
  REQUIRE(fx.sweep.wait(t, 5000));
  CHECK(t.admitted == 0);
  CHECK(fx.pool.map_host()[10] == -1);
  CHECK(std::memcmp(fx.sweep.record_host(t, 1), fx.reference(11).data(), kRec) == 0);
}

TEST_CASE("an admit id absent from the group throws invalid_argument") {
  Fixture fx(4);
  auto ids = fx.ids(0, 4);
  int32_t adm[] = {40};
  CHECK_THROWS_AS(fx.sweep.read_group(0, ids.data(), ids.size(), 0, adm, 1), std::invalid_argument);
}

TEST_CASE("an admit id that is already resident is skipped, and a duplicate admit id reserves once") {
  Fixture fx(4);
  fx.sweep.set_generation(5);
  auto ids = fx.ids(0, 8);
  SweepTicket t0 = fx.sweep.read_group(0, ids.data(), ids.size(), 0);
  REQUIRE(fx.sweep.wait(t0, 5000));
  REQUIRE(fx.sweep.admit_from_staging(0, 2, reinterpret_cast<uint64_t>(fx.sweep.record_host(t0, 2))));
  const int32_t resident_slot = fx.pool.map_host()[2];
  REQUIRE(resident_slot >= 0);
  int32_t adm[] = {2, 4, 4};
  SweepTicket t = fx.sweep.read_group(0, ids.data(), ids.size(), 1, adm, 3);
  CHECK(t.admit_refused == 0);
  REQUIRE(fx.sweep.wait(t, 5000));
  CHECK(t.admitted == 1);  // 2 is resident, the second 4 is a duplicate
  CHECK(fx.pool.map_host()[2] == resident_slot);
  const int32_t s4 = fx.pool.map_host()[4];
  REQUIRE(s4 >= 0);
  CHECK(s4 != resident_slot);
  CHECK(std::memcmp(fx.pool.slot_host(s4), fx.reference(4).data(), kRec) == 0);
  int owners = 0;
  for (int s = 0; s < fx.pool.slots(); ++s) owners += fx.pool.owner(s) == 4 ? 1 : 0;
  CHECK(owners == 1);
}
