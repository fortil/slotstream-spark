#include "slotio/ledger.hpp"

#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <unistd.h>

namespace slotio {
namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// "Key:   123 kB" -> bytes
uint64_t kb_field(const std::string& path, const char* key) {
  std::ifstream f(path);
  std::string line;
  size_t n = std::strlen(key);
  while (std::getline(f, line)) {
    if (line.compare(0, n, key) == 0 && line.size() > n && line[n] == ':') {
      return std::strtoull(line.c_str() + n + 1, nullptr, 10) * 1024ull;
    }
  }
  return 0;
}

struct NvmlProc {  // nvmlProcessInfo_v2_t
  unsigned pid;
  unsigned long long used;
  unsigned gpu_instance, compute_instance;
};
using InitFn = int (*)();
using HandleFn = int (*)(unsigned, void**);
using ProcsFn = int (*)(void*, unsigned*, NvmlProc*);
}  // namespace

Sampler::Sampler(const Config& cfg) : cfg_(cfg) {
  ring_.reserve(cfg_.ring);
  if (!cfg_.use_nvml) return;
  nvml_lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!nvml_lib_) return;
  auto init = reinterpret_cast<InitFn>(dlsym(nvml_lib_, "nvmlInit_v2"));
  auto handle = reinterpret_cast<HandleFn>(dlsym(nvml_lib_, "nvmlDeviceGetHandleByIndex_v2"));
  if (init && handle && init() == 0 && handle(0, &nvml_dev_) == 0) nvml_ok_ = true;
}

Sampler::~Sampler() {
  stop();
  if (nvml_lib_) {
    if (nvml_ok_) {
      auto sd = reinterpret_cast<InitFn>(dlsym(nvml_lib_, "nvmlShutdown"));
      if (sd) sd();
    }
    dlclose(nvml_lib_);
  }
}

void Sampler::register_unified(const std::string& name, std::function<uint64_t()> bytes) {
  std::lock_guard<std::mutex> g(mu_);
  reg_[name] = std::move(bytes);
}

void Sampler::register_pool(const std::string& name, const Pool* pool) {
  register_unified(name, [pool]() { return pool->owned_bytes(); });
}

void Sampler::unregister(const std::string& name) {
  std::lock_guard<std::mutex> g(mu_);
  reg_.erase(name);
}

uint64_t Sampler::read_nvml() {
  if (!nvml_ok_) return 0;
  static auto fn = reinterpret_cast<ProcsFn>(dlsym(nvml_lib_, "nvmlDeviceGetComputeRunningProcesses_v3"));
  if (!fn) return 0;
  NvmlProc procs[64];
  unsigned n = 64;
  if (fn(nvml_dev_, &n, procs) != 0) return 0;
  unsigned me = static_cast<unsigned>(getpid());
  for (unsigned i = 0; i < n; i++) {
    if (procs[i].pid == me) return procs[i].used;
  }
  return 0;
}

FootprintSample Sampler::sample_once() {
  FootprintSample s;
  s.t_ns = now_ns();
  s.vmrss = kb_field(cfg_.status_path, "VmRSS");
  s.vmhwm = kb_field(cfg_.status_path, "VmHWM");
  s.mem_available = kb_field(cfg_.meminfo_path, "MemAvailable");
  s.mem_free = kb_field(cfg_.meminfo_path, "MemFree");
  s.nvml = read_nvml();
  std::vector<std::pair<std::string, std::function<uint64_t()>>> regs;
  {
    std::lock_guard<std::mutex> g(mu_);
    regs.assign(reg_.begin(), reg_.end());
  }
  for (auto& r : regs) s.unified.emplace_back(r.first, r.second());
  return s;
}

void Sampler::main() {
  while (!stop_.load()) {
    FootprintSample s = sample_once();
    {
      std::lock_guard<std::mutex> g(mu_);
      if (ring_.size() >= cfg_.ring) {
        ring_.erase(ring_.begin());
        dropped_++;
      }
      ring_.push_back(std::move(s));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.period_ms));
  }
}

void Sampler::start() {
  if (!stop_.exchange(false)) return;
  th_ = std::thread([this]() { main(); });
}

void Sampler::stop() {
  stop_.store(true);
  if (th_.joinable()) th_.join();
}

std::vector<FootprintSample> Sampler::drain() {
  std::lock_guard<std::mutex> g(mu_);
  std::vector<FootprintSample> out;
  out.swap(ring_);
  ring_.reserve(cfg_.ring);
  return out;
}

}  // namespace slotio
