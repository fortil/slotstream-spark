#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "slotio/ledger.hpp"

namespace slotio {

Arbiter::Arbiter(const Config& cfg, std::function<uint64_t()> own_read_bytes)
    : cfg_(cfg), own_(std::move(own_read_bytes)) {
  caps_.total = cfg_.full_total;
  caps_.spec = cfg_.full_spec;
}

Arbiter::~Arbiter() { stop(); }

bool Arbiter::read_device(uint64_t* bytes) const {
  std::ifstream f(cfg_.diskstats_path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::string major, minor, name;
    uint64_t reads, merged, sectors;
    if (!(is >> major >> minor >> name >> reads >> merged >> sectors)) continue;
    if (name == cfg_.device) {
      *bytes = sectors * 512ull;
      return true;
    }
  }
  return false;
}

bool Arbiter::read_temp(double* c) const {
  if (cfg_.hwmon_path.empty()) return false;
  std::ifstream f(cfg_.hwmon_path);
  double v;
  if (!(f >> v)) return false;
  *c = v / 1000.0;
  return true;
}

Arbiter::Caps Arbiter::step(double now) {
  std::lock_guard<std::mutex> g(mu_);
  uint64_t dev = 0;
  uint64_t own = own_ ? own_() : 0;
  if (read_device(&dev)) {
    if (have_prev_ && now > prev_t_) {
      double dt = now - prev_t_;
      double total_rate = double(dev - prev_dev_) / dt;
      double own_rate = double(own - prev_own_) / dt;
      double foreign = total_rate - own_rate;
      if (foreign < 0) foreign = 0;
      foreign_ = foreign / 1e6;
      if (foreign_ > cfg_.foreign_mb_s) {
        if (++over_ >= 2) spec_off_ = true;
        calm_since_ = -1;
      } else {
        over_ = 0;
        if (spec_off_) {
          if (calm_since_ < 0) calm_since_ = now;
          if (now - calm_since_ >= cfg_.hyst_s) spec_off_ = false;
        }
      }
    }
    prev_dev_ = dev;
    prev_own_ = own;
    prev_t_ = now;
    have_prev_ = true;
  }
  double t;
  if (read_temp(&t)) {
    if (t >= cfg_.throttle_c) {
      throttled_ = true;
      cool_since_ = -1;
    } else if (throttled_) {
      if (t <= cfg_.throttle_c - cfg_.hyst_c) {
        if (cool_since_ < 0) cool_since_ = now;
        if (now - cool_since_ >= cfg_.hyst_s) throttled_ = false;
      } else {
        cool_since_ = -1;
      }
    }
  }
  caps_.total = throttled_ ? cfg_.throttled_total : cfg_.full_total;
  caps_.spec = spec_off_ ? 0 : (caps_.total < cfg_.full_spec ? caps_.total : cfg_.full_spec);
  return caps_;
}

Arbiter::Caps Arbiter::caps() const {
  std::lock_guard<std::mutex> g(mu_);
  return caps_;
}

void Arbiter::start(Reader* r) {
  if (!stop_.exchange(false)) return;
  th_ = std::thread([this, r]() {
    auto t0 = std::chrono::steady_clock::now();
    while (!stop_.load()) {
      double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Caps c = step(now);
      if (r) r->set_caps(c.total, c.spec);
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  });
}

void Arbiter::stop() {
  stop_.store(true);
  if (th_.joinable()) th_.join();
}

}  // namespace slotio
