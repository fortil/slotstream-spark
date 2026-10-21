#include "slotio/reader.hpp"

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <set>

namespace slotio {

namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
FileSeal seal_of(const struct stat& st) {
  FileSeal s;
  s.size = static_cast<uint64_t>(st.st_size);
  s.mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1'000'000'000LL + st.st_mtim.tv_nsec;
  s.ino = static_cast<uint64_t>(st.st_ino);
  return s;
}
}  // namespace

File File::open_direct(const std::string& path, const FileSeal* expect, uint32_t align) {
  if (align != 512 && align != 4096) throw IoError("align must be 512 or 4096", EINVAL);
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) throw IoError("open(O_DIRECT) failed: " + path, errno);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    int e = errno;
    ::close(fd);
    throw IoError("fstat failed: " + path, e);
  }
  int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  if (rc != 0) {
    ::close(fd);
    throw IoError("posix_fadvise(RANDOM) failed: " + path, rc);
  }
  File f;
  f.fd_ = fd;
  f.align_ = align;
  f.seal_ = seal_of(st);
  f.path_ = path;
  if (expect && !(*expect == f.seal_)) {
    throw IoError("file seal mismatch: " + path, ESTALE);  // ~File closes fd
  }
  return f;
}

File::File(File&& o) noexcept : fd_(o.fd_), align_(o.align_), seal_(o.seal_), path_(std::move(o.path_)) { o.fd_ = -1; }
File& File::operator=(File&& o) noexcept {
  if (this != &o) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = o.fd_;
    align_ = o.align_;
    seal_ = o.seal_;
    path_ = std::move(o.path_);
    o.fd_ = -1;
  }
  return *this;
}
File::~File() {
  if (fd_ >= 0) ::close(fd_);
}

int File::check_seal() const {
  struct stat st;
  if (::fstat(fd_, &st) != 0) return errno;
  return seal_of(st) == seal_ ? 0 : ESTALE;
}

bool ReadFault::before_read() {
  int n = count_.fetch_add(1);
  if (n < after_) return false;
  bool expected = false;
  return fired_.compare_exchange_strong(expected, true);
}

int64_t exact_transfer(int fd, uint8_t* buf, uint64_t len, uint64_t off, uint64_t min_ok, int* err) {
  uint64_t done = 0;
  while (done < len) {
    ssize_t r = pread(fd, buf + done, len - done, static_cast<off_t>(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      *err = errno;
      return -1;
    }
    if (r == 0) {
      if (done >= min_ok) break;  // widened read reached the end of the file
      *err = EIO;                 // unexpected EOF
      return -1;
    }
    if (static_cast<uint64_t>(r) > len - done) {
      *err = EIO;  // over-read
      return -1;
    }
    done += static_cast<uint64_t>(r);
  }
  if (done < min_ok) {
    *err = EIO;
    return -1;
  }
  return static_cast<int64_t>(done);
}

struct Reader::Batch {
  std::mutex m;
  std::condition_variable cv;
  uint32_t remaining = 0, jobs = 0;
  uint64_t bytes = 0;
  int err = 0;
  int64_t last_read = 0, last_copy = 0;
  std::vector<const File*> files;
};

static std::vector<int> pick_cpus() {
  std::vector<std::pair<int, long>> caps;
  for (int c = 0; c < 1024; ++c) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity");
    long v;
    if (!(f >> v)) {
      if (c > 0 && access(("/sys/devices/system/cpu/cpu" + std::to_string(c)).c_str(), F_OK) != 0) break;
      continue;
    }
    caps.push_back({c, v});
  }
  std::vector<int> out;
  if (caps.empty()) return out;
  long lo = caps[0].second, hi = caps[0].second;
  for (auto& p : caps) {
    lo = std::min(lo, p.second);
    hi = std::max(hi, p.second);
  }
  if (lo == hi) return out;  // homogeneous: do not pin
  for (auto& p : caps)
    if (2 * p.second < lo + hi) out.push_back(p.first);  // the little (A725) cores
  return out;
}

Reader::Reader(const Config& cfg)
    : cfg_(cfg),
      total_cap_(std::min(cfg.total_cap, cfg.lanes)),
      spec_cap_(cfg.spec_cap),
      lane_read_(cfg.lanes),
      lane_copy_(cfg.lanes),
      hist_(40) {
  if (cfg_.lanes < 1) throw std::invalid_argument("lanes");
  if (cfg_.pin_cores) cpus_ = pick_cpus();
  scratch_len_ = (cfg_.scratch_bytes + 4095) / 4096 * 4096;
  for (int i = 0; i < cfg_.lanes; ++i) {
    void* p = mmap(nullptr, scratch_len_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw IoError("mmap scratch failed", errno);
    scratch_.push_back(static_cast<uint8_t*>(p));
    lane_read_[i] = 0;
    lane_copy_[i] = 0;
  }
  for (auto& h : hist_) h = 0;
  for (int i = 0; i < cfg_.lanes; ++i) threads_.emplace_back([this, i] { lane_main(i); });
}

Reader::~Reader() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
  for (auto* p : scratch_) munmap(p, scratch_len_);
}

BatchId Reader::submit(const Job* jobs, size_t n) {
  auto b = std::make_shared<Batch>();
  b->remaining = b->jobs = static_cast<uint32_t>(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t al = jobs[i].file->align();
    const uint64_t span = ((jobs[i].off % al) + jobs[i].len + al - 1) / al * al;
    if (span > scratch_len_) throw std::invalid_argument("job wider than the lane scratch");
    b->files.push_back(jobs[i].file);
  }
  std::lock_guard<std::mutex> g(mu_);
  BatchId id = next_id_++;
  ++batch_count_;
  if (cfg_.seal_every > 0 && batch_count_ % static_cast<uint64_t>(cfg_.seal_every) == 0) {
    ++seal_checks_;
    std::set<const File*> seen(b->files.begin(), b->files.end());
    for (auto* f : seen) {
      int e = f->check_seal();
      if (e != 0) {
        b->err = e;
        break;
      }
    }
  }
  batches_[id] = b;
  if (n == 0) return id;
  for (size_t i = 0; i < n; ++i) q_[static_cast<int>(jobs[i].prio)].push_back(Item{jobs[i], b});
  cv_.notify_all();
  return id;
}

BatchResult Reader::join(BatchId id, int64_t timeout_ns) {
  std::shared_ptr<Batch> b;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = batches_.find(id);
    if (it == batches_.end()) throw std::invalid_argument("unknown batch");
    b = it->second;
  }
  BatchResult r;
  {
    std::unique_lock<std::mutex> l(b->m);
    auto done = [&] { return b->remaining == 0; };
    if (timeout_ns < 0) {
      b->cv.wait(l, done);
    } else if (!b->cv.wait_for(l, std::chrono::nanoseconds(timeout_ns), done)) {
      r.ok = false;
      r.err = ETIMEDOUT;
      return r;
    }
    r.err = b->err;
    r.ok = (b->err == 0);
    r.jobs = b->jobs;
    r.bytes = r.ok ? b->bytes : 0;
    r.last_read_done_ns = b->last_read;
    r.last_copy_done_ns = b->last_copy;
  }
  std::lock_guard<std::mutex> g(mu_);
  batches_.erase(id);
  return r;
}

void Reader::set_fault(std::shared_ptr<ReadFault> f) {
  std::lock_guard<std::mutex> g(mu_);
  fault_ = std::move(f);
}

void Reader::set_caps(int total, int spec) {
  {
    std::lock_guard<std::mutex> g(mu_);
    total_cap_ = std::max(1, std::min(total, cfg_.lanes));
    spec_cap_ = std::max(0, spec);
  }
  cv_.notify_all();
}

void Reader::set_paused(bool p) {
  {
    std::lock_guard<std::mutex> g(mu_);
    paused_ = p;
  }
  cv_.notify_all();
}

// mu_ held. Highest priority first, FIFO inside a priority.
bool Reader::pick(Item* out) {
  if (paused_ || inflight_ >= total_cap_) return false;
  for (int p = 0; p < 4; ++p) {
    if (q_[p].empty()) continue;
    if (p == static_cast<int>(Priority::Speculative) && spec_inflight_ >= spec_cap_) continue;
    *out = std::move(q_[p].front());
    q_[p].pop_front();
    ++inflight_;
    if (p == static_cast<int>(Priority::Speculative)) ++spec_inflight_;
    max_inflight_ = std::max<uint32_t>(max_inflight_, static_cast<uint32_t>(inflight_));
    return true;
  }
  return false;
}

void Reader::lane_main(int lane) {
  if (!cpus_.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus_[lane % cpus_.size()], &set);
    (void)sched_setaffinity(0, sizeof set, &set);  // best effort: a cpuset may forbid it
  }
  for (;;) {
    Item it;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [&] { return stop_ || pick(&it); });
      if (stop_) return;
    }
    run_job(lane, it);
    {
      std::lock_guard<std::mutex> g(mu_);
      --inflight_;
      if (it.job.prio == Priority::Speculative) --spec_inflight_;
    }
    cv_.notify_all();
  }
}

void Reader::run_job(int lane, const Item& it) {
  const Job& j = it.job;
  Batch& b = *it.batch;
  int err = 0;
  bool skip;
  {
    std::lock_guard<std::mutex> l(b.m);
    skip = b.err != 0;
  }
  int64_t t_read_end = 0, t_copy_end = 0;
  if (!skip) {
    std::shared_ptr<ReadFault> f;
    {
      std::lock_guard<std::mutex> g(mu_);
      f = fault_;
    }
    if (f && f->before_read()) {
      err = f->err();
    } else {
      const uint64_t al = j.file->align();
      uint64_t a_off = j.off / al * al;
      uint64_t a_end = (j.off + j.len + al - 1) / al * al;
      uint64_t need = (j.off + j.len) - a_off;
      uint8_t* scratch = scratch_[lane];
      int64_t t0 = now_ns();
      int64_t got = exact_transfer(j.file->fd(), scratch, a_end - a_off, a_off, need, &err);
      t_read_end = now_ns();
      lane_read_[lane] += static_cast<uint64_t>(t_read_end - t0);
      if (got >= 0) {
        std::memcpy(j.dst, scratch + (j.off - a_off), j.len);
        if (j.dst2) std::memcpy(j.dst2, scratch + (j.off - a_off), j.len);
        t_copy_end = now_ns();
        uint64_t cn = static_cast<uint64_t>(t_copy_end - t_read_end);
        lane_copy_[lane] += cn;
        int bucket = cn ? 63 - __builtin_clzll(cn) : 0;
        hist_[std::min(bucket, 39)]++;
      }
    }
  }
  {
    std::lock_guard<std::mutex> g(mu_);
    if (!skip && err == 0) {
      ++jobs_;
      bytes_ += j.len;
    }
  }
  std::lock_guard<std::mutex> l(b.m);
  if (err != 0 && b.err == 0) b.err = err;
  if (err == 0 && !skip) {
    b.bytes += j.len;
    b.last_read = std::max(b.last_read, t_read_end);
    b.last_copy = std::max(b.last_copy, t_copy_end);
  }
  if (--b.remaining == 0) b.cv.notify_all();
}

ReaderStats Reader::stats() const {
  ReaderStats s;
  {
    std::lock_guard<std::mutex> g(mu_);
    s.jobs = jobs_;
    s.bytes = bytes_;
    s.max_inflight = max_inflight_;
    s.batches = batch_count_;
    s.seal_checks = seal_checks_;
  }
  for (size_t i = 0; i < lane_read_.size(); ++i) {
    s.lane_read_ns.push_back(lane_read_[i]);
    s.lane_copy_ns.push_back(lane_copy_[i]);
    s.read_ns += lane_read_[i];
    s.copy_ns += lane_copy_[i];
  }
  for (auto& h : hist_) s.memcpy_hist.push_back(h);
  return s;
}

}  // namespace slotio
