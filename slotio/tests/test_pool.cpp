#include <cstring>

#include "doctest.h"
#include "slotio/pool.hpp"

using namespace slotio;

namespace {
PoolConfig small_cfg() {
  PoolConfig c;
  c.record_bytes = 4096;
  c.slots_per_chunk = 388;
  c.chunk_bytes = 1ull << 21;  // 2 MiB holds 388 * 4096
  return c;
}
}  // namespace

TEST_CASE("slot 387 is in chunk 0 and slot 388 in chunk 1") {
  Pool p(MemType::HostMallocForTests, small_cfg());
  p.resize_slots(624);
  CHECK(p.chunks() == 2);
  uint8_t* s387 = p.slot_host(387);
  uint8_t* s388 = p.slot_host(388);
  uint8_t* s0 = p.slot_host(0);
  CHECK(s387 == s0 + 387ull * 4096);
  CHECK(s388 != s387 + 4096);  // a new chunk starts, not the next record
  CHECK(p.slot_host(389) == s388 + 4096);
  CHECK(p.slot_ptr_host()[388] == p.slot_dev(388));
  CHECK(p.owned_bytes() == 2ull * (1ull << 21));
}

TEST_CASE("resize 624 -> 19012 -> 1440 -> 2400 keeps the surviving slots") {
  Pool p(MemType::HostMallocForTests, small_cfg());
  p.resize_slots(624);
  for (int s : {0, 100, 387, 388, 623}) {
    std::memset(p.slot_host(s), 0x40 + s % 31, 4096);
    p.set_owner(s, s);
    p.publish(s, s);
  }
  uint8_t* keep0 = p.slot_host(0);
  p.resize_slots(19012);
  CHECK(p.chunks() == 49);
  CHECK(p.slot_host(0) == keep0);  // growing never moves old chunks
  for (int s : {0, 100, 387, 388, 623}) CHECK(p.slot_host(s)[17] == uint8_t(0x40 + s % 31));
  p.resize_slots(1440);
  CHECK(p.chunks() == 4);
  CHECK(p.stats().free_ns.size() == 45);
  // slots 0,100,387,388 survive; 623 < 1440 survives too
  for (int s : {0, 100, 387, 388, 623}) {
    CHECK(p.slot_host(s)[17] == uint8_t(0x40 + s % 31));
    CHECK(p.map_host()[s] == s);
  }
  p.resize_slots(2400);
  CHECK(p.chunks() == 7);
  for (int s : {0, 100, 387, 388, 623}) CHECK(p.slot_host(s)[17] == uint8_t(0x40 + s % 31));
}

TEST_CASE("shrinking unmaps the slots that go away and keeps the others") {
  Pool p(MemType::HostMallocForTests, small_cfg());
  p.resize_slots(1000);
  p.set_owner(10, 10);
  p.publish(10, 10);
  p.set_owner(900, 77);
  p.publish(77, 900);
  p.resize_slots(500);
  CHECK(p.map_host()[10] == 10);
  CHECK(p.map_host()[77] == -1);
  CHECK(p.slot_ptr_host()[900] == 0);
  CHECK(p.chunks() == 2);
}

TEST_CASE("CLOCK gives second chances and throws Exhausted when everything is pinned") {
  Clock c(8);
  Pins pins(8);
  for (int s = 0; s < 8; ++s) c.set_ref(s);
  int v = c.victim(8, [&](int s) { return pins.pinned(s); });
  CHECK(v == 0);  // second pass: the first sweep cleared every bit
  c.set_ref(1);
  CHECK(c.victim(8, [&](int s) { return pins.pinned(s); }) == 2);
  c.set_reserved(3, true);
  CHECK(c.victim(8, [&](int s) { return pins.pinned(s); }) == 4);
  for (int s = 0; s < 8; ++s) pins.pin(s, 5);
  CHECK_THROWS_AS(c.victim(8, [&](int s) { return pins.pinned(s); }), Exhausted);
}

TEST_CASE("pins retire only when progress passes the generation, snapshot/restore round-trips") {
  Pins p(16);
  p.pin(3, 7);
  p.pin(4, 8);
  CHECK(p.count() == 2);
  p.retire_below(7);
  CHECK(p.count() == 2);
  p.retire_below(8);  // gen 7 < 8
  CHECK(p.count() == 1);
  CHECK_FALSE(p.pinned(3));
  auto snap = p.snapshot();
  p.pin(9, 20);
  p.retire_below(100);
  CHECK(p.count() == 0);
  p.restore(snap);
  CHECK(p.count() == 1);
  CHECK(p.pinned(4));
}
