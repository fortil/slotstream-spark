#include "slotio/rows.hpp"

#include <cstring>
#include <string>

namespace slotio {

bool RowCache::get(int64_t key, uint8_t* dst) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) {
    ++misses_;
    return false;
  }
  ++hits_;
  std::memcpy(dst, data_.data() + it->second * row_bytes_, row_bytes_);
  return true;
}

void RowCache::put(int64_t key, const uint8_t* src) {
  if (cap_ == 0) return;
  std::lock_guard<std::mutex> g(mu_);
  if (index_.count(key)) return;
  size_t slot;
  if (keys_.size() < cap_) {
    slot = keys_.size();
    keys_.push_back(key);
    data_.resize((slot + 1) * row_bytes_);
  } else {
    slot = next_;
    next_ = (next_ + 1) % cap_;
    index_.erase(keys_[slot]);
    keys_[slot] = key;
  }
  std::memcpy(data_.data() + slot * row_bytes_, src, row_bytes_);
  index_[key] = slot;
}

NgramRows::NgramRows(Reader& reader, const File& file, uint32_t row_bytes, size_t cache_rows)
    : reader_(reader), file_(file), row_bytes_(row_bytes), rows_(file.size() / row_bytes), cache_(row_bytes, cache_rows) {
  if (row_bytes == 0) throw std::invalid_argument("row_bytes");
}

void NgramRows::read(const int64_t* gids, size_t n, uint8_t* dst) {
  std::vector<Job> jobs;
  std::vector<size_t> idx;
  for (size_t i = 0; i < n; ++i) {
    if (gids[i] < 0 || static_cast<uint64_t>(gids[i]) >= rows_) throw std::invalid_argument("n-gram id out of range");
    uint8_t* d = dst + i * row_bytes_;
    if (cache_.get(gids[i], d)) continue;
    jobs.push_back(Job{&file_, static_cast<uint64_t>(gids[i]) * row_bytes_, row_bytes_, d, Priority::Demand});
    idx.push_back(i);
  }
  if (jobs.empty()) return;
  BatchResult r = reader_.join(reader_.submit(jobs.data(), jobs.size()));
  if (!r.ok) throw IoError("n-gram row read failed", r.err);
  for (size_t k = 0; k < idx.size(); ++k) cache_.put(gids[idx[k]], dst + idx[k] * row_bytes_);
}

NgramTicket::~NgramTicket() {
  if (s_ && s_->live && s_->submitted) s_->reader->join(s_->batch);   // the rows land in our staging buffer
}

NgramTicket NgramRows::prefetch(const int64_t* gids, size_t n) {
  NgramTicket t;
  t.s_ = std::make_unique<NgramTicket::State>();
  auto& s = *t.s_;
  s.reader = &reader_;
  s.gids.assign(gids, gids + n);
  s.stage.resize(n * row_bytes_);
  std::vector<Job> jobs;
  for (size_t i = 0; i < n; ++i) {
    if (gids[i] < 0 || static_cast<uint64_t>(gids[i]) >= rows_) throw std::invalid_argument("n-gram id out of range");
    uint8_t* d = s.stage.data() + i * row_bytes_;
    if (cache_.get(gids[i], d)) continue;
    jobs.push_back(Job{&file_, static_cast<uint64_t>(gids[i]) * row_bytes_, row_bytes_, d, Priority::Demand});
    s.miss.push_back(i);
  }
  if (!jobs.empty()) {
    s.batch = reader_.submit(jobs.data(), jobs.size());
    s.submitted = true;
  }
  s.live = true;
  return t;
}

void NgramRows::wait(NgramTicket& t, uint8_t* dst) {
  if (!t.s_ || !t.s_->live) throw std::logic_error("n-gram ticket already waited on or empty");
  auto& s = *t.s_;
  s.live = false;
  if (s.submitted) {
    BatchResult r = reader_.join(s.batch);
    if (!r.ok) throw IoError("n-gram row read failed", r.err);
    for (size_t i : s.miss) cache_.put(s.gids[i], s.stage.data() + i * row_bytes_);
  }
  std::memcpy(dst, s.stage.data(), s.stage.size());
}

}  // namespace slotio
