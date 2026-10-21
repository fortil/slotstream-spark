#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "doctest.h"
#include "slotio/prefetch.hpp"

using namespace slotio;

namespace {
constexpr uint32_t kRec = 4096;
constexpr int kN = 64;

struct Fixture {
  std::string path = std::string(SLOTIO_BUILD_DIR) + "/t_prefetch.bin";
  std::vector<uint8_t> bytes;
  File file;
  Reader reader;
  Pool pool;
  explicit Fixture(int slots) : pool(MemType::HostMallocForTests, cfg()) {
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
    pool.resize_slots(slots);
  }
  ~Fixture() { ::unlink(path.c_str()); }
  static PoolConfig cfg() {
    PoolConfig c;
    c.record_bytes = kRec;
    c.chunk_bytes = 1ull << 21;
    c.max_slots = 600;
    return c;
  }
  bool slot_matches(int slot, int expert) {
    return std::memcmp(pool.slot_host(slot), bytes.data() + size_t(expert) * kRec, kRec) == 0;
  }
};

int count_mapped(Fixture& fx) {
  int n = 0;
  for (size_t i = 0; i < fx.pool.map_len(); ++i) n += fx.pool.map_host()[i] >= 0;
  return n;
}
int count_reserved(Fixture& fx, Ensurer& en) {
  int n = 0;
  for (int s = 0; s < fx.pool.slots(); ++s) n += en.clock().reserved(s);
  return n;
}
}  // namespace

TEST_CASE("the default reservation floor is 256 evictable slots and the cap is 64 tickets") {
  PrefetchConfig c;
  CHECK(c.min_evictable == 256);
  CHECK(c.slot_cap == 64);
}

TEST_CASE("a reservation needs more than min_evictable evictable slots") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 10;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t ids[] = {1, 2, 3, 4, 5, 6, 7, 8};
  // slots 16, nothing pinned: reservations while 16 - 0 - in_flight > 10, i.e. 6 tickets
  CHECK(pf.issue(0, ids, 8) == 6);
  CHECK(pf.stats().refused_evictable == 2);
  CHECK(pf.expire_all() == 6);
  Fixture fy(16);
  Ensurer ey(fy.pool, fy.reader, fy.file);
  Prefetch pd(ey, fy.reader, fy.file);  // default floor 256 > 16 slots: nothing is ever reserved
  CHECK(pd.issue(0, ids, 8) == 0);
  CHECK(pd.outstanding() == 0);
  CHECK(count_reserved(fy, ey) == 0);
}

TEST_CASE("pinned slots do not count as evictable") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  int32_t held[] = {40, 41, 42, 43, 44, 45, 46, 47};
  en.ensure(0, held, 8, 5);  // 8 pinned at gen 5
  PrefetchConfig c;
  c.min_evictable = 4;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t ids[] = {1, 2, 3, 4, 5};
  CHECK(pf.issue(0, ids, 5) == 4);  // 16 - 8 - k > 4 -> k < 4 ... k = 0..3 reserve, the fourth needs 16-8-3=5 > 4
  for (int e : held) CHECK(fx.pool.map_host()[e] >= 0);
  pf.expire_all();
}

TEST_CASE("a finished ticket is adopted by ensure without a second read, bytes equal the file") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t ids[] = {3, 9, 11};
  REQUIRE(pf.issue(0, ids, 3) == 3);
  for (int e : ids) CHECK(fx.pool.map_host()[e] == -1);  // nothing is visible before the claim
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EnsureTiming t;
  auto slots = en.ensure(0, ids, 3, 1, &t);
  CHECK(t.adopted == 3);
  CHECK(t.misses == 0);
  CHECK(t.hits == 3);
  for (int i = 0; i < 3; ++i) CHECK(fx.slot_matches(slots[i], ids[i]));
  auto st = pf.stats();
  CHECK(st.adopted == 3);
  CHECK(pf.outstanding() == 0);
  CHECK(count_reserved(fx, en) == 0);
}

TEST_CASE("a reading ticket is waited for, never read twice, and published only after the copy") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  Prefetch pf(en, fx.reader, fx.file, c);
  fx.reader.set_paused(true);
  int32_t ids[] = {5, 6};
  REQUIRE(pf.issue(0, ids, 2) == 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(fx.pool.map_host()[5] == -1);  // reserved slot, read queued: not published
  CHECK(fx.pool.map_host()[6] == -1);
  std::thread rel([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fx.reader.set_paused(false);
  });
  uint64_t jobs_before = fx.reader.stats().jobs;
  EnsureTiming t;
  auto slots = en.ensure(0, ids, 2, 1, &t);
  rel.join();
  CHECK(t.adopted == 2);
  CHECK(t.misses == 0);
  CHECK(fx.reader.stats().jobs - jobs_before <= 2);  // the two speculative jobs, no demand re-read
  CHECK(fx.reader.stats().jobs == 2);
  for (int i = 0; i < 2; ++i) CHECK(fx.slot_matches(slots[i], ids[i]));
  CHECK(pf.stats().adopted == 2);
  CHECK(pf.stats().adopted_late >= 1);  // the first claim waited; the second ticket may already be done
}

TEST_CASE("expire_layer returns unadopted slots and nothing leaks") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t ids[] = {1, 2, 3, 4};
  REQUIRE(pf.issue(0, ids, 4) == 4);
  CHECK(count_reserved(fx, en) == 4);
  int32_t one[] = {2};
  en.ensure(0, one, 1, 1);  // adopts 2
  CHECK(pf.expire_layer(0) == 3);
  CHECK(pf.outstanding() == 0);
  CHECK(count_reserved(fx, en) == 0);
  CHECK(count_mapped(fx) == 1);  // only the adopted expert is visible
  auto st = pf.stats();
  CHECK(st.issued == 4);
  CHECK(st.issued == st.adopted + st.discarded + st.failed);
  // the returned slots are usable by demand
  int32_t more[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  auto s = en.ensure(0, more, 15, 2);
  for (int i = 0; i < 15; ++i) CHECK(fx.slot_matches(s[i], more[i]));
}

TEST_CASE("a failed speculative read is discarded and demand re-reads the same bytes") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  Prefetch pf(en, fx.reader, fx.file, c);
  auto flt = std::make_shared<ReadFault>(0);
  fx.reader.set_fault(flt);
  int32_t ids[] = {7};
  REQUIRE(pf.issue(0, ids, 1) == 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(flt->fired());
  EnsureTiming t;
  auto s = en.ensure(0, ids, 1, 1, &t);
  CHECK(t.adopted == 0);
  CHECK(t.misses == 1);
  CHECK(fx.slot_matches(s[0], 7));
  CHECK(pf.stats().failed == 1);
  CHECK(count_reserved(fx, en) == 0);
}

TEST_CASE("resident and ticketed keys are skipped and the ticket cap holds") {
  Fixture fx(40);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  c.slot_cap = 5;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t res[] = {0};
  en.ensure(0, res, 1, 1);
  int32_t ids[] = {0, 1, 1, 2, 3, 4, 5, 6, 7};
  CHECK(pf.issue(0, ids, 9) == 5);  // 0 resident, second 1 ticketed, cap 5
  auto st = pf.stats();
  CHECK(st.skipped == 2);
  CHECK(st.refused_cap >= 1);
  CHECK(pf.expire_all() == 5);
}

TEST_CASE("resize waits for in-flight tickets before slots go away") {
  Fixture fx(16);
  Ensurer en(fx.pool, fx.reader, fx.file);
  PrefetchConfig c;
  c.min_evictable = 2;
  Prefetch pf(en, fx.reader, fx.file, c);
  int32_t ids[] = {1, 2, 3};
  REQUIRE(pf.issue(0, ids, 3) == 3);
  en.resize(8);
  CHECK(pf.outstanding() == 0);
  CHECK(count_reserved(fx, en) == 0);
}
