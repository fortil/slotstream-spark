// Row readers (n-gram table, token embeddings) and the sequence-word wait.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "slotio/reader.hpp"

namespace slotio {

struct Timeout : std::runtime_error {
  explicit Timeout(const std::string& w) : std::runtime_error(w) {}
};

// Spins with an acquire load and sched_yield until *p >= target; throws Timeout after timeout_ns.
void wait_seq(const uint64_t* p, uint64_t target, int64_t timeout_ns);

// FIFO row cache: when full the oldest row is dropped.
class RowCache {
 public:
  RowCache(uint32_t row_bytes, size_t capacity) : row_bytes_(row_bytes), cap_(capacity) {}
  bool get(int64_t key, uint8_t* dst);
  void put(int64_t key, const uint8_t* src);
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  size_t size() const { return index_.size(); }

 private:
  uint32_t row_bytes_;
  size_t cap_, next_ = 0;
  std::vector<uint8_t> data_;
  std::vector<int64_t> keys_;
  std::unordered_map<int64_t, size_t> index_;
  uint64_t hits_ = 0, misses_ = 0;
  std::mutex mu_;
};

// Handle of a started n-gram prefetch. Owns the staging buffer the misses are read into; destroying it
// before `NgramRows::wait` joins the batch and drops the rows.
class NgramTicket {
 public:
  NgramTicket() = default;
  NgramTicket(NgramTicket&&) noexcept = default;
  NgramTicket& operator=(NgramTicket&&) noexcept = default;
  ~NgramTicket();
  size_t size() const { return s_ ? s_->gids.size() : 0; }
  size_t misses() const { return s_ ? s_->miss.size() : 0; }
  bool pending() const { return s_ && s_->live; }

 private:
  friend class NgramRows;
  struct State {
    Reader* reader = nullptr;
    std::vector<int64_t> gids;
    std::vector<uint8_t> stage;
    std::vector<size_t> miss;
    BatchId batch = 0;
    bool submitted = false, live = false;
  };
  std::unique_ptr<State> s_;
};

class NgramRows {
 public:
  NgramRows(Reader& reader, const File& file, uint32_t row_bytes, size_t cache_rows = 400'000);
  // dst receives n rows of row_bytes each (a pinned 16-row buffer in the engine).
  void read(const int64_t* gids, size_t n, uint8_t* dst);
  // Split form of read(): prefetch() answers the cache hits and submits the misses at Demand priority
  // without blocking; wait() joins the batch, fills the cache and copies the n rows into dst
  // (a pinned buffer in the engine). Each ticket is waited on exactly once.
  NgramTicket prefetch(const int64_t* gids, size_t n);
  void wait(NgramTicket& t, uint8_t* dst);
  uint32_t row_bytes() const { return row_bytes_; }
  uint64_t rows() const { return rows_; }
  RowCache& cache() { return cache_; }

 private:
  Reader& reader_;
  const File& file_;
  uint32_t row_bytes_;
  uint64_t rows_;
  RowCache cache_;
};

struct EmbedLayout {
  uint64_t w_off = 0, s_off = 0, b_off = 0;  // absolute offsets of the three tensors in trunk.safetensors
  uint32_t w_row = 1280, s_row = 80, b_row = 80;
  uint64_t vocab = 248'320;
  uint32_t row_bytes() const { return w_row + s_row + b_row; }
};

class EmbedRows {
 public:
  EmbedRows(Reader& reader, const File& trunk, const EmbedLayout& layout, size_t capacity = 8192);
  // dst receives n rows of [weight | scales | biases] (1,440 bytes for the 4-bit gs64 layout).
  void read(const int64_t* ids, size_t n, uint8_t* dst);
  uint32_t row_bytes() const { return layout_.row_bytes(); }
  RowCache& cache() { return cache_; }

 private:
  Reader& reader_;
  const File& file_;
  EmbedLayout layout_;
  RowCache cache_;
};

}  // namespace slotio
