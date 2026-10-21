#include <unistd.h>

#include <cstring>
#include <fstream>
#include <vector>

#include "doctest.h"
#include "slotio/pool.hpp"

using namespace slotio;

namespace {
constexpr uint32_t kRec = 4096;
constexpr int kN = 64;

struct Fixture {
  std::string path = std::string(SLOTIO_BUILD_DIR) + "/t_ensure.bin";
  std::vector<uint8_t> bytes;
  File file;
  Reader reader;
  Pool pool;
  Fixture(int slots, int max_slots = 600) : pool(MemType::HostMallocForTests, cfg(max_slots)) {
    bytes.resize(size_t(kN) * kRec);
    uint32_t x = 12345;
    for (auto& b : bytes) {
      x = x * 1664525u + 1013904223u;
      b = uint8_t(x >> 24);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    f.close();
    file = File::open_direct(path);
    pool.resize_slots(slots);
  }
  ~Fixture() { ::unlink(path.c_str()); }
  static PoolConfig cfg(int max_slots) {
    PoolConfig c;
    c.record_bytes = kRec;
    c.chunk_bytes = 1ull << 21;
    c.max_slots = max_slots;
    return c;
  }
  bool slot_matches(int slot, int expert) {
    return std::memcmp(pool.slot_host(slot), bytes.data() + size_t(expert) * kRec, kRec) == 0;
  }
};
}  // namespace

TEST_CASE("ensure loads misses, reports hits, and the bytes equal the file") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t ids[] = {5, 9, 5, 12};
  EnsureTiming t;
  auto slots = en.ensure(0, ids, 4, 1, &t);
  CHECK(t.misses == 3);
  CHECK(t.hits == 0);
  CHECK(t.bytes == 3ull * kRec);
  CHECK(slots[0] == slots[2]);
  for (int i = 0; i < 4; ++i) {
    CHECK(fx.pool.map_host()[ids[i]] == slots[i]);
    CHECK(fx.slot_matches(slots[i], ids[i]));
  }
  auto again = en.ensure(0, ids, 4, 2, &t);
  CHECK(t.hits == 3);
  CHECK(t.misses == 0);
  CHECK(again == slots);
}

TEST_CASE("CLOCK Exhausted when all slots are pinned, and nothing changes") {
  Fixture fx(4);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t a[] = {0, 1, 2, 3};
  en.ensure(0, a, 4, 10);
  std::vector<int32_t> map_before(fx.pool.map_host(), fx.pool.map_host() + fx.pool.map_len());
  auto pins_before = en.pins().snapshot();
  int32_t b[] = {7};
  CHECK_THROWS_AS(en.ensure(0, b, 1, 11), Exhausted);
  CHECK(std::vector<int32_t>(fx.pool.map_host(), fx.pool.map_host() + fx.pool.map_len()) == map_before);
  CHECK(en.pins().snapshot().gen == pins_before.gen);
  CHECK(en.pins().count() == 4);
}

TEST_CASE("an injected fault publishes nothing, restores the pins, and a retry succeeds") {
  Fixture fx(8);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t warm[] = {1, 2};
  en.ensure(0, warm, 2, 1);
  std::vector<int32_t> map_before(fx.pool.map_host(), fx.pool.map_host() + fx.pool.map_len());
  auto pins_before = en.pins().snapshot();
  auto flt = std::make_shared<ReadFault>(1);  // second job of the batch fails
  fx.reader.set_fault(flt);
  int32_t ids[] = {1, 20, 21, 22};
  CHECK_THROWS_AS(en.ensure(0, ids, 4, 2), IoError);
  CHECK(flt->fired());
  CHECK(en.pins().snapshot().gen == pins_before.gen);
  CHECK(en.pins().count() == pins_before.count);
  for (int e : {20, 21, 22}) CHECK(fx.pool.map_host()[e] == -1);
  CHECK(fx.pool.map_host()[1] == map_before[1]);
  CHECK(fx.pool.map_host()[2] == map_before[2]);
  auto slots = en.ensure(0, ids, 4, 2);  // the fault is one-shot
  for (int i = 0; i < 4; ++i) CHECK(fx.slot_matches(slots[i], ids[i]));
}

TEST_CASE("pins of generation g are freed only when progress > g") {
  Fixture fx(4);
  Ensurer en(fx.pool, fx.reader, fx.file);
  uint64_t progress = 0;
  en.set_progress_ptr(&progress);
  int32_t a[] = {0, 1, 2, 3};
  en.ensure(0, a, 4, 5);
  int32_t b[] = {10};
  progress = 5;
  CHECK_THROWS_AS(en.ensure(0, b, 1, 6), Exhausted);  // progress == g does not retire
  progress = 6;
  auto s = en.ensure(0, b, 1, 7);
  CHECK(fx.slot_matches(s[0], 10));
}

TEST_CASE("input validation") {
  Fixture fx(4);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t bad[] = {512};
  CHECK_THROWS_AS(en.ensure(0, bad, 1, 0), std::invalid_argument);
  int32_t ok[] = {1};
  CHECK_THROWS_AS(en.ensure(49, ok, 1, 0), std::invalid_argument);
}

TEST_CASE("resize through the ensurer drops pins above the new size") {
  Fixture fx(8);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  en.ensure(0, a, 8, 1);
  CHECK(en.pins().count() == 8);
  en.resize(4);
  CHECK(en.pins().count() == 4);
  int e_gone = 0;
  for (int e = 0; e < 8; ++e) e_gone += fx.pool.map_host()[e] == -1;
  CHECK(e_gone == 4);
}
