// Footprint sampler (20 ms ring of process memory counters) and the NVMe/thermal arbiter of the reader caps.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "slotio/pool.hpp"
#include "slotio/reader.hpp"

namespace slotio {

struct FootprintSample {
  int64_t t_ns = 0;
  uint64_t vmrss = 0, vmhwm = 0, mem_available = 0, mem_free = 0, nvml = 0;
  std::vector<std::pair<std::string, uint64_t>> unified;  // owned bytes per registered allocation
};

// Reads VmRSS/VmHWM, MemAvailable/MemFree, NVML process memory (dlopen libnvidia-ml.so.1, 0 when absent) and the
// owned bytes of every registered unified allocation into a ring buffer every `period_ms`.
class Sampler {
 public:
  struct Config {
    int period_ms = 20;
    size_t ring = 1 << 16;
    std::string status_path = "/proc/self/status";
    std::string meminfo_path = "/proc/meminfo";
    bool use_nvml = true;
  };
  explicit Sampler(const Config& cfg);
  Sampler() : Sampler(Config{}) {}
  ~Sampler();
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  void register_unified(const std::string& name, std::function<uint64_t()> bytes);
  void register_pool(const std::string& name, const Pool* pool);  // owned_bytes() of a pool, read at each sample
  void unregister(const std::string& name);
  FootprintSample sample_once();  // one reading, not stored
  void start();
  void stop();
  std::vector<FootprintSample> drain();  // oldest first; clears the ring
  uint64_t dropped() const { return dropped_.load(); }
  bool nvml_ok() const { return nvml_ok_; }

 private:
  uint64_t read_nvml();
  void main();
  Config cfg_;
  std::mutex mu_;
  std::map<std::string, std::function<uint64_t()>> reg_;
  std::vector<FootprintSample> ring_;
  std::atomic<uint64_t> dropped_{0};
  std::atomic<bool> stop_{true};
  std::thread th_;
  void* nvml_lib_ = nullptr;
  void* nvml_dev_ = nullptr;
  bool nvml_ok_ = false;
};

// Every 250 ms: foreign NVMe reads (device read rate minus our own) above `foreign_mb_s` on 2 samples set
// spec_cap = 0; a hwmon temperature at or above `throttle_c` sets total_cap = `throttled_total`. Release needs
// `hyst_c` degrees / `hyst_s` seconds of calm.
class Arbiter {
 public:
  struct Config {
    std::string diskstats_path = "/proc/diskstats";
    std::string device = "nvme0n1";
    std::string hwmon_path;  // temp1_input in millidegrees; empty = no thermal input
    double foreign_mb_s = 200.0;
    double throttle_c = 80.0;
    double hyst_c = 3.0;
    double hyst_s = 5.0;
    int full_total = 12, full_spec = 12, throttled_total = 6;
  };
  struct Caps {
    int total = 12, spec = 12;
  };
  Arbiter(const Config& cfg, std::function<uint64_t()> own_read_bytes);
  ~Arbiter();
  Caps step(double now_s);  // one 250 ms tick with an injected clock
  Caps caps() const;
  void start(Reader* r);  // thread applying set_caps every 250 ms
  void stop();
  double last_foreign_mb_s() const { return foreign_; }

 private:
  bool read_device(uint64_t* bytes) const;
  bool read_temp(double* c) const;
  Config cfg_;
  std::function<uint64_t()> own_;
  mutable std::mutex mu_;
  bool have_prev_ = false;
  double prev_t_ = 0, foreign_ = 0, calm_since_ = -1, cool_since_ = -1;
  uint64_t prev_dev_ = 0, prev_own_ = 0;
  int over_ = 0;
  bool spec_off_ = false, throttled_ = false;
  Caps caps_;
  std::atomic<bool> stop_{true};
  std::thread th_;
};

}  // namespace slotio
