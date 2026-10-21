#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "doctest.h"
#include "slotio/reader.hpp"

using namespace slotio;

namespace {
constexpr uint32_t kRec = 8192;
constexpr int kN = 64;

std::vector<uint8_t> pattern(size_t n, uint32_t seed) {
  std::vector<uint8_t> v(n);
  uint32_t x = seed * 2654435761u + 1;
  for (auto& b : v) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<uint8_t>(x >> 24);
  }
  return v;
}

struct TmpFile {
  std::string path;
  std::vector<uint8_t> bytes;
  explicit TmpFile(const std::string& name, size_t records = kN, size_t extra = 0) {
    path = std::string(SLOTIO_BUILD_DIR) + "/" + name;
    bytes = pattern(records * kRec + extra, 7);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  ~TmpFile() { ::unlink(path.c_str()); }
};
}  // namespace

TEST_CASE("every byte of 64 records round-trips") {
  TmpFile tf("t_reader_rt.bin");
  File f = File::open_direct(tf.path);
  Reader r;
  std::vector<uint8_t> out(kN * kRec, 0);
  std::vector<Job> jobs;
  for (int i = 0; i < kN; ++i) jobs.push_back({&f, uint64_t(i) * kRec, kRec, out.data() + size_t(i) * kRec, Priority::Demand});
  BatchResult res = r.join(r.submit(jobs.data(), jobs.size()));
  CHECK(res.ok);
  CHECK(res.bytes == uint64_t(kN) * kRec);
  CHECK(out == tf.bytes);
  CHECK(res.last_copy_done_ns >= res.last_read_done_ns);
}

TEST_CASE("unaligned jobs are widened to the alignment") {
  TmpFile tf("t_reader_un.bin");
  File f = File::open_direct(tf.path);
  Reader r;
  uint8_t a[100], b[1440];
  Job jobs[2] = {{&f, 4096 + 37, 100, a, Priority::Demand}, {&f, 3 * kRec - 700, 1440, b, Priority::Demand}};
  BatchResult res = r.join(r.submit(jobs, 2));
  CHECK(res.ok);
  CHECK(std::memcmp(a, tf.bytes.data() + 4096 + 37, 100) == 0);
  CHECK(std::memcmp(b, tf.bytes.data() + 3 * kRec - 700, 1440) == 0);
}

TEST_CASE("a read widened past the end of an unaligned file stops at EOF") {
  TmpFile tf("t_reader_tail.bin", 4, 1000);  // size = 4*8192 + 1000, not a multiple of 4096
  File f = File::open_direct(tf.path);
  Reader r;
  uint8_t a[200];
  Job j{&f, 4 * kRec + 800, 200, a, Priority::Demand};
  BatchResult res = r.join(r.submit(&j, 1));
  CHECK(res.ok);
  CHECK(std::memcmp(a, tf.bytes.data() + 4 * kRec + 800, 200) == 0);
}

TEST_CASE("in flight never exceeds the cap") {
  TmpFile tf("t_reader_cap.bin");
  File f = File::open_direct(tf.path);
  std::vector<uint8_t> out(kN * kRec);
  std::vector<Job> jobs;
  for (int rep = 0; rep < 8; ++rep)
    for (int i = 0; i < kN; ++i) jobs.push_back({&f, uint64_t(i) * kRec, kRec, out.data() + size_t(i) * kRec, Priority::Sweep});
  {
    Reader r;
    CHECK(r.join(r.submit(jobs.data(), jobs.size())).ok);
    CHECK(r.stats().max_inflight <= 12);
  }
  {
    Reader::Config c;
    c.total_cap = 3;
    Reader r(c);
    CHECK(r.join(r.submit(jobs.data(), jobs.size())).ok);
    CHECK(r.stats().max_inflight <= 3);
  }
  {
    Reader r;
    r.set_caps(2, 1);
    CHECK(r.join(r.submit(jobs.data(), jobs.size())).ok);
    CHECK(r.stats().max_inflight <= 2);
  }
}

TEST_CASE("Demand submitted after 100 Sweep jobs finishes first") {
  TmpFile tf("t_reader_prio.bin");
  File f = File::open_direct(tf.path);
  Reader::Config c;
  c.lanes = 1;
  c.total_cap = 1;
  Reader r(c);
  r.set_paused(true);
  std::vector<uint8_t> out(100 * kRec), d(kRec);
  std::vector<Job> sweep;
  for (int i = 0; i < 100; ++i) sweep.push_back({&f, uint64_t(i % kN) * kRec, kRec, out.data() + size_t(i) * kRec, Priority::Sweep});
  BatchId bs = r.submit(sweep.data(), sweep.size());
  Job dj{&f, 5 * kRec, kRec, d.data(), Priority::Demand};
  BatchId bd = r.submit(&dj, 1);
  r.set_paused(false);
  BatchResult rd = r.join(bd);
  CHECK(rd.ok);
  // 1 lane: when the Demand batch is done at most the one job that was already dispatched finished before it
  CHECK(r.stats().jobs <= 2);
  BatchResult rs = r.join(bs);
  CHECK(rs.ok);
  CHECK(rd.last_copy_done_ns < rs.last_copy_done_ns);
  CHECK(std::memcmp(d.data(), tf.bytes.data() + 5 * kRec, kRec) == 0);
}

TEST_CASE("a fault after 37 jobs fails the join, fires once, the next batch succeeds") {
  TmpFile tf("t_reader_fault.bin");
  File f = File::open_direct(tf.path);
  Reader::Config c;
  c.lanes = 1;
  c.total_cap = 1;
  Reader r(c);
  auto flt = std::make_shared<ReadFault>(37);
  r.set_fault(flt);
  std::vector<uint8_t> out(kN * kRec);
  std::vector<Job> jobs;
  for (int i = 0; i < kN; ++i) jobs.push_back({&f, uint64_t(i) * kRec, kRec, out.data() + size_t(i) * kRec, Priority::Demand});
  BatchResult a = r.join(r.submit(jobs.data(), jobs.size()));
  CHECK_FALSE(a.ok);
  CHECK(a.err == EIO);
  CHECK(a.bytes == 0);  // never partial
  CHECK(flt->fired());
  BatchResult b = r.join(r.submit(jobs.data(), jobs.size()));
  CHECK(b.ok);
  CHECK(out == tf.bytes);
}

TEST_CASE("reading past the end is an error") {
  TmpFile tf("t_reader_eof.bin");
  File f = File::open_direct(tf.path);
  Reader r;
  std::vector<uint8_t> d(kRec);
  Job j{&f, uint64_t(kN) * kRec, kRec, d.data(), Priority::Demand};
  BatchResult res = r.join(r.submit(&j, 1));
  CHECK_FALSE(res.ok);
  CHECK(res.err != 0);
}

TEST_CASE("truncation after open gives ESTALE") {
  TmpFile tf("t_reader_stale.bin");
  File f = File::open_direct(tf.path);
  CHECK(f.check_seal() == 0);
  FileSeal wrong = f.seal();
  wrong.size += 1;
  CHECK_THROWS_AS(File::open_direct(tf.path, &wrong), IoError);
  Reader::Config c;
  c.seal_every = 2;
  Reader r(c);
  std::vector<uint8_t> d(kRec);
  Job j{&f, 0, kRec, d.data(), Priority::Demand};
  CHECK(r.join(r.submit(&j, 1)).ok);  // batch 1: no check
  REQUIRE(::truncate(tf.path.c_str(), 10 * kRec) == 0);
  CHECK(f.check_seal() == ESTALE);
  BatchResult res = r.join(r.submit(&j, 1));  // batch 2: sealed check fires
  CHECK_FALSE(res.ok);
  CHECK(res.err == ESTALE);
}

TEST_CASE("the read syscall appears only inside exact_transfer") {
  std::string dir = SLOTIO_SRC_DIR;
  const char* files[] = {"reader.cpp", "pool.cpp", "clock.cpp", "ensure.cpp", "rows.cpp", "embed.cpp", "wait.cpp", "bindings.cpp"};
  int total = 0;
  for (const char* name : files) {
    std::ifstream in(dir + "/" + name);
    if (!in) continue;
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    size_t pos = 0;
    while ((pos = s.find("pread(", pos)) != std::string::npos) {
      if (pos == 0 || (!isalnum(static_cast<unsigned char>(s[pos - 1])) && s[pos - 1] != '_')) {
        ++total;
        CHECK(std::string(name) == "reader.cpp");
        size_t fn = s.find("int64_t exact_transfer(");
        REQUIRE(fn != std::string::npos);
        size_t end = s.find("\n}\n", fn);
        CHECK(pos > fn);
        CHECK(pos < end);
      }
      pos += 6;
    }
  }
  CHECK(total == 1);
}

TEST_CASE("a real-size record fills the default scratch exactly and round-trips") {
  constexpr uint32_t kReal = 2'764'800;  // 675 pages
  constexpr int kRecs = 3;
  std::string path = std::string(SLOTIO_BUILD_DIR) + "/t_reader_real.bin";
  std::vector<uint8_t> bytes = pattern(size_t(kRecs) * kReal, 11);
  {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  {
    File f = File::open_direct(path);
    Reader::Config c;
    CHECK(c.scratch_bytes == kReal);
    Reader r(c);
    std::vector<uint8_t> out(size_t(kRecs) * kReal, 0);
    std::vector<Job> jobs;
    for (int i = 0; i < kRecs; ++i) jobs.push_back({&f, uint64_t(i) * kReal, kReal, out.data() + size_t(i) * kReal, Priority::Demand});
    BatchResult res = r.join(r.submit(jobs.data(), jobs.size()));
    CHECK(res.ok);
    CHECK(res.bytes == uint64_t(kRecs) * kReal);
    CHECK(out == bytes);
    // one byte wider than the scratch after widening is rejected
    Job bad{&f, 4096 + 1, kReal, out.data(), Priority::Demand};
    CHECK_THROWS_AS(r.submit(&bad, 1), std::invalid_argument);
  }
  ::unlink(path.c_str());
}

TEST_CASE("join with a timeout reports ETIMEDOUT and the batch stays joinable") {
  TmpFile tf("t_reader_to.bin");
  File f = File::open_direct(tf.path);
  Reader r;
  r.set_paused(true);
  std::vector<uint8_t> d(kRec);
  Job j{&f, 2 * kRec, kRec, d.data(), Priority::Demand};
  BatchId id = r.submit(&j, 1);
  BatchResult t = r.join(id, 20'000'000);
  CHECK_FALSE(t.ok);
  CHECK(t.err == ETIMEDOUT);
  r.set_paused(false);
  BatchResult ok = r.join(id, -1);
  CHECK(ok.ok);
  CHECK(ok.bytes == kRec);
  CHECK(std::memcmp(d.data(), tf.bytes.data() + 2 * kRec, kRec) == 0);
}
