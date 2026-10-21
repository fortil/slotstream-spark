#include <fstream>
#include <string>

#include "doctest.h"
#include "slotio/ledger.hpp"

using namespace slotio;

namespace {
std::string tmp(const std::string& name) { return std::string(SLOTIO_BUILD_DIR) + "/" + name; }
void write(const std::string& p, const std::string& s) {
  std::ofstream f(p, std::ios::trunc);
  f << s;
}
void diskstats(const std::string& p, uint64_t bytes) {
  write(p, "   7 0 loop0 1 0 8 0 0 0 0 0 0 0 0\n 259 0 nvme0n1 100 0 " + std::to_string(bytes / 512) + " 0 0 0 0 0 0 0 0\n");
}
}  // namespace

TEST_CASE("arbiter: foreign reads above 200 MB/s on 2 samples close the speculative cap") {
  std::string ds = tmp("arb_diskstats"), hw = tmp("arb_temp");
  write(hw, "50000\n");
  uint64_t own = 0;
  Arbiter::Config c;
  c.diskstats_path = ds;
  c.hwmon_path = hw;
  Arbiter a(c, [&]() { return own; });
  uint64_t dev = 0;
  diskstats(ds, dev);
  a.step(0.0);
  // 100 MB in 0.25 s = 400 MB/s foreign, own reads zero
  dev += 100'000'000;
  diskstats(ds, dev);
  CHECK(a.step(0.25).spec == 12);  // first sample over: not yet
  dev += 100'000'000;
  diskstats(ds, dev);
  CHECK(a.step(0.5).spec == 0);
  // our own reads do not count: 400 MB/s of own traffic leaves the cap open after calm
  for (double t = 0.75; t < 7.0; t += 0.25) {
    dev += 100'000'000;
    own += 100'000'000;
    diskstats(ds, dev);
    a.step(t);
  }
  CHECK(a.caps().spec == 12);
}

TEST_CASE("arbiter: hwmon at the throttle line caps the total at 6 with 3 C and 5 s of hysteresis") {
  std::string ds = tmp("arb_diskstats2"), hw = tmp("arb_temp2");
  diskstats(ds, 0);
  Arbiter::Config c;
  c.diskstats_path = ds;
  c.hwmon_path = hw;
  c.throttle_c = 80.0;
  Arbiter a(c, nullptr);
  write(hw, "79000\n");
  CHECK(a.step(0.0).total == 12);
  write(hw, "80000\n");
  CHECK(a.step(0.25).total == 6);
  write(hw, "78000\n");  // below the line but inside the 3 C band
  CHECK(a.step(10.0).total == 6);
  write(hw, "76000\n");  // out of the band: needs 5 s
  CHECK(a.step(11.0).total == 6);
  CHECK(a.step(15.9).total == 6);
  CHECK(a.step(16.1).total == 12);
}

TEST_CASE("sampler: registered unified allocations are read at each sample") {
  std::string st = tmp("smp_status"), mi = tmp("smp_meminfo");
  write(st, "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n");
  write(mi, "MemTotal: 999 kB\nMemFree: 4096 kB\nMemAvailable: 8192 kB\n");
  Sampler::Config c;
  c.status_path = st;
  c.meminfo_path = mi;
  c.use_nvml = false;
  Sampler s(c);
  uint64_t pool = 1ull << 30;
  s.register_unified("pool", [&]() { return pool; });
  s.register_unified("reader", []() { return uint64_t(2'764'800); });
  auto x = s.sample_once();
  CHECK(x.vmrss == 1024 * 1024);
  CHECK(x.vmhwm == 2048 * 1024);
  CHECK(x.mem_available == 8192 * 1024);
  CHECK(x.mem_free == 4096 * 1024);
  REQUIRE(x.unified.size() == 2);
  s.start();
  pool = 2ull << 30;
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  s.stop();
  auto ring = s.drain();
  REQUIRE(ring.size() >= 3);
  uint64_t last = 0;
  for (auto& u : ring.back().unified) if (u.first == "pool") last = u.second;
  CHECK(last == (2ull << 30));
  CHECK(s.drain().empty());
}
