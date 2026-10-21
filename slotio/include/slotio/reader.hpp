// O_DIRECT reader: priority queue, up to 12 lanes, safe path (pread into a lane scratch, then memcpy).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace slotio {

struct IoError : std::runtime_error {
  int err;
  IoError(const std::string& what, int e) : std::runtime_error(what), err(e) {}
};

enum class Priority : uint8_t { Demand = 0, Speculative = 1, Sweep = 2, Background = 3 };

struct FileSeal {
  uint64_t size = 0;
  int64_t mtime_ns = 0;
  uint64_t ino = 0;
  bool operator==(const FileSeal& o) const { return size == o.size && mtime_ns == o.mtime_ns && ino == o.ino; }
};

class File {
 public:
  // O_RDONLY|O_DIRECT|O_CLOEXEC. Every syscall result is checked. If `expect` is given and differs
  // from fstat, throws IoError(ESTALE). `align` is 512 or 4096 (from the manifest).
  static File open_direct(const std::string& path, const FileSeal* expect = nullptr, uint32_t align = 4096);
  File() = default;
  File(File&& o) noexcept;
  File& operator=(File&& o) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();
  int fd() const { return fd_; }
  uint64_t size() const { return seal_.size; }
  uint32_t align() const { return align_; }
  const FileSeal& seal() const { return seal_; }
  const std::string& path() const { return path_; }
  // 0 if fstat still matches the seal taken at open, ESTALE if not, errno if fstat fails.
  int check_seal() const;

 private:
  int fd_ = -1;
  uint32_t align_ = 4096;
  FileSeal seal_;
  std::string path_;
};

struct Job {
  const File* file;
  uint64_t off;
  uint32_t len;
  uint8_t* dst;
  Priority prio;
  uint8_t* dst2 = nullptr;  // optional second destination, copied by the lane right after dst (pool slot for admission)
};

// One-shot fault: the (after_jobs+1)-th read fails with `err`, exactly once.
class ReadFault {
 public:
  explicit ReadFault(int after_jobs, int err = 5 /*EIO*/) : after_(after_jobs), err_(err) {}
  bool before_read();
  bool fired() const { return fired_.load(); }
  int err() const { return err_; }

 private:
  int after_;
  int err_;
  std::atomic<int> count_{0};
  std::atomic<bool> fired_{false};
};

using BatchId = uint64_t;

struct BatchResult {
  bool ok = false;
  int err = 0;  // errno of the first failure (0 when ok)
  uint32_t jobs = 0;
  uint64_t bytes = 0;
  int64_t last_read_done_ns = 0;  // steady-clock stamps, for the copy tail
  int64_t last_copy_done_ns = 0;
};

struct ReaderStats {
  uint64_t jobs = 0, bytes = 0, read_ns = 0, copy_ns = 0;
  uint32_t max_inflight = 0;
  uint64_t batches = 0, seal_checks = 0;
  std::vector<uint64_t> lane_read_ns, lane_copy_ns;
  std::vector<uint64_t> memcpy_hist;  // bucket b counts copies with floor(log2(ns)) == b
};

// Reads up to `len` bytes at `off` into buf: loops on EINTR, errors on <0, EOF (0) and over-read.
// `min_ok` (<= len) lets a read that was widened past the end of the file stop at EOF.
// This is the only place that calls the read syscall.
int64_t exact_transfer(int fd, uint8_t* buf, uint64_t len, uint64_t off, uint64_t min_ok, int* err);

class Reader {
 public:
  struct Config {
    int lanes = 12;
    int total_cap = 12;
    int spec_cap = 12;
    uint64_t scratch_bytes = 2'764'800;
    int seal_every = 1024;
    bool pin_cores = true;
  };
  explicit Reader(const Config& cfg);
  Reader() : Reader(Config{}) {}
  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  BatchId submit(const Job* jobs, size_t n);
  // All-or-error. timeout_ns < 0 waits forever; on timeout returns err=ETIMEDOUT and the batch stays joinable.
  BatchResult join(BatchId id, int64_t timeout_ns = -1);
  void set_fault(std::shared_ptr<ReadFault> f);
  void set_caps(int total, int spec);
  void set_paused(bool p);  // hold dispatch (used by tests to queue deterministically)
  ReaderStats stats() const;
  const std::vector<int>& lane_cpus() const { return cpus_; }

 private:
  struct Batch;
  struct Item {
    Job job;
    std::shared_ptr<Batch> batch;
  };
  void lane_main(int lane);
  bool pick(Item* out);
  void run_job(int lane, const Item& it);

  Config cfg_;
  std::vector<int> cpus_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Item> q_[4];
  std::map<BatchId, std::shared_ptr<Batch>> batches_;
  BatchId next_id_ = 1;
  uint64_t batch_count_ = 0;
  int inflight_ = 0, spec_inflight_ = 0, total_cap_, spec_cap_;
  bool stop_ = false, paused_ = false;
  std::shared_ptr<ReadFault> fault_;
  std::vector<std::thread> threads_;
  std::vector<uint8_t*> scratch_;
  uint64_t scratch_len_ = 0;
  // stats (guarded by mu_ except lane counters, which are per-lane atomics)
  uint32_t max_inflight_ = 0;
  uint64_t jobs_ = 0, bytes_ = 0, seal_checks_ = 0;
  std::vector<std::atomic<uint64_t>> lane_read_, lane_copy_;
  std::vector<std::atomic<uint64_t>> hist_;
};

}  // namespace slotio
