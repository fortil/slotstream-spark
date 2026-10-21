#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "doctest.h"
#include "slotio/rows.hpp"

using namespace slotio;

namespace {
std::vector<uint8_t> noise(size_t n, uint32_t seed) {
  std::vector<uint8_t> v(n);
  uint32_t x = seed;
  for (auto& b : v) {
    x = x * 1664525u + 1013904223u;
    b = uint8_t(x >> 24);
  }
  return v;
}
struct Tmp {
  std::string path;
  std::vector<uint8_t> bytes;
  Tmp(const std::string& name, size_t n) : path(std::string(SLOTIO_BUILD_DIR) + "/" + name), bytes(noise(n, 99)) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
  }
  ~Tmp() { ::unlink(path.c_str()); }
};
}  // namespace

TEST_CASE("n-gram rows: 100-byte rows at unaligned offsets, cache hits on repeat") {
  Tmp tf("t_rows_ngram.bin", 100 * 5000);
  File f = File::open_direct(tf.path);
  Reader r;
  NgramRows ng(r, f, 100, 1000);
  CHECK(ng.rows() == 5000);
  int64_t gids[16];
  for (int i = 0; i < 16; ++i) gids[i] = (i * 313 + 7) % 5000;
  std::vector<uint8_t> out(16 * 100);
  ng.read(gids, 16, out.data());
  for (int i = 0; i < 16; ++i) CHECK(std::memcmp(out.data() + i * 100, tf.bytes.data() + gids[i] * 100, 100) == 0);
  CHECK(ng.cache().misses() == 16);
  std::vector<uint8_t> out2(16 * 100);
  ng.read(gids, 16, out2.data());
  CHECK(out2 == out);
  CHECK(ng.cache().hits() == 16);
  int64_t bad[1] = {5000};
  CHECK_THROWS_AS(ng.read(bad, 1, out.data()), std::invalid_argument);
}

TEST_CASE("n-gram prefetch/wait: bytes equal a direct pread, hits rise on repeats, tickets are single-use") {
  Tmp tf("t_rows_prefetch.bin", 100 * 5000);
  File f = File::open_direct(tf.path);
  Reader r;
  NgramRows ng(r, f, 100, 1000);
  int64_t gids[16];
  for (int i = 0; i < 16; ++i) gids[i] = (i * 211 + 3) % 5000;
  gids[5] = gids[2];  // a duplicate inside one ticket
  NgramTicket t = ng.prefetch(gids, 16);
  CHECK(t.size() == 16);
  CHECK(t.misses() == 16);
  CHECK(t.pending());
  std::vector<uint8_t> out(16 * 100);
  ng.wait(t, out.data());
  CHECK(!t.pending());
  for (int i = 0; i < 16; ++i) CHECK(std::memcmp(out.data() + i * 100, tf.bytes.data() + gids[i] * 100, 100) == 0);
  CHECK_THROWS_AS(ng.wait(t, out.data()), std::logic_error);
  uint64_t h = ng.cache().hits();
  NgramTicket t2 = ng.prefetch(gids, 16);
  CHECK(t2.misses() == 0);  // all answered from the cache at prefetch time
  CHECK(ng.cache().hits() == h + 16);
  std::vector<uint8_t> out2(16 * 100);
  ng.wait(t2, out2.data());
  CHECK(out2 == out);
  // a different gid never returns another row's bytes, also after eviction pressure
  int64_t bad[1] = {5000};
  CHECK_THROWS_AS(ng.prefetch(bad, 1), std::invalid_argument);
  {
    NgramTicket dropped = ng.prefetch(gids + 8, 4);  // destroyed without wait: joins, no crash
  }
  std::vector<uint8_t> out3(16 * 100);
  ng.read(gids, 16, out3.data());
  CHECK(out3 == out);
}

TEST_CASE("n-gram cache is FIFO: the oldest row is dropped when full") {
  Tmp tf("t_rows_fifo.bin", 100 * 100);
  File f = File::open_direct(tf.path);
  Reader r;
  NgramRows ng(r, f, 100, 4);
  std::vector<uint8_t> out(100);
  for (int64_t g = 0; g < 5; ++g) ng.read(&g, 1, out.data());
  CHECK(ng.cache().size() == 4);
  int64_t g0 = 0, g4 = 4;
  uint64_t h = ng.cache().hits();
  ng.read(&g4, 1, out.data());
  CHECK(ng.cache().hits() == h + 1);
  ng.read(&g0, 1, out.data());  // evicted
  CHECK(ng.cache().hits() == h + 1);
}

TEST_CASE("embedding rows: weight, scales and biases from three tensors, 1440 bytes") {
  const uint64_t V = 300;
  const uint64_t w_off = 8192, s_off = w_off + V * 1280 + 128, b_off = s_off + V * 80 + 44;
  Tmp tf("t_rows_embed.bin", b_off + V * 80 + 4096);
  File f = File::open_direct(tf.path);
  Reader r;
  EmbedLayout l;
  l.w_off = w_off;
  l.s_off = s_off;
  l.b_off = b_off;
  l.vocab = V;
  EmbedRows er(r, f, l, 8);
  CHECK(er.row_bytes() == 1440);
  int64_t ids[3] = {0, 17, 299};
  std::vector<uint8_t> out(3 * 1440);
  er.read(ids, 3, out.data());
  for (int i = 0; i < 3; ++i) {
    const uint8_t* o = out.data() + i * 1440;
    CHECK(std::memcmp(o, tf.bytes.data() + w_off + ids[i] * 1280, 1280) == 0);
    CHECK(std::memcmp(o + 1280, tf.bytes.data() + s_off + ids[i] * 80, 80) == 0);
    CHECK(std::memcmp(o + 1360, tf.bytes.data() + b_off + ids[i] * 80, 80) == 0);
  }
  int64_t bad[1] = {300};
  CHECK_THROWS_AS(er.read(bad, 1, out.data()), std::invalid_argument);
}

TEST_CASE("wait_seq returns after another thread stores the target, and times out otherwise") {
  alignas(64) uint64_t seq = 0;
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    __atomic_store_n(&seq, uint64_t(3), __ATOMIC_RELEASE);
  });
  wait_seq(&seq, 3, 2'000'000'000LL);
  CHECK(seq >= 3);
  t.join();
  auto t0 = std::chrono::steady_clock::now();
  CHECK_THROWS_AS(wait_seq(&seq, 4, 100'000'000LL), Timeout);
  auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  CHECK(dt >= 100'000'000LL);
  CHECK(dt <= 100'000'000LL + 10'000'000LL);
}
