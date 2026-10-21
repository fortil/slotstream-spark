#include "slotio/sweep.hpp"

#include <cuda.h>
#include <errno.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace slotio {

namespace {
void cu_check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  cuGetErrorString(r, &msg);
  throw IoError(std::string(what) + ": " + (msg ? msg : "cuda error"), static_cast<int>(r));
}
#define CU(x) cu_check((x), #x)
const char* mem_name(MemType t) {
  switch (t) {
    case MemType::HostAlloc: return "host_alloc";
    case MemType::VmmHostNuma: return "vmm_host_numa";
    case MemType::AnonThp: return "anon_thp";
    case MemType::HostMallocForTests: return "host_malloc_for_tests";
  }
  return "?";
}
}  // namespace

Staging::Staging(MemType type, int groups, int records, uint64_t record_bytes)
    : type_(type), groups_(groups), records_(records), rec_(record_bytes) {
  if (groups < 1 || records < 1 || record_bytes == 0) throw std::invalid_argument("bad staging geometry");
  const uint64_t n = bytes();
  mapped_ = n;
  if (type_ == MemType::HostMallocForTests || type_ == MemType::AnonThp) {
    const uint64_t al = type_ == MemType::AnonThp ? (2ull << 20) : 4096;
    mapped_ = (n + al - 1) / al * al;
    void* p = std::aligned_alloc(al, mapped_);
    if (!p) throw IoError("staging aligned_alloc failed", ENOMEM);
    host_ = static_cast<uint8_t*>(p);
    dev_ = reinterpret_cast<uint64_t>(p);
  } else {
    CU(cuInit(0));
    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU(cuCtxSetCurrent(ctx));
    device_ = dev;
    cuda_ = true;
    if (type_ == MemType::HostAlloc) {
      void* p = nullptr;
      CU(cuMemHostAlloc(&p, n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
      CUdeviceptr d = 0;
      CU(cuMemHostGetDevicePointer(&d, p, 0));
      host_ = static_cast<uint8_t*>(p);
      dev_ = d;
    } else {
      CUmemAllocationProp prop;
      std::memset(&prop, 0, sizeof prop);
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      prop.location.id = 0;
      size_t gran = 0;
      CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      mapped_ = (n + gran - 1) / gran * gran;
      CUdeviceptr va = 0;
      CU(cuMemAddressReserve(&va, mapped_, 0, 0, 0));
      CUmemGenericAllocationHandle h;
      CUresult r = cuMemCreate(&h, mapped_, &prop, 0);
      if (r != CUDA_SUCCESS) {
        cuMemAddressFree(va, mapped_);
        cu_check(r, "cuMemCreate(staging)");
      }
      r = cuMemMap(va, mapped_, 0, h, 0);
      if (r == CUDA_SUCCESS) {
        CUmemAccessDesc ad[2];
        std::memset(ad, 0, sizeof ad);
        ad[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad[0].location.id = device_;
        ad[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ad[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        ad[1].location.id = 0;
        ad[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        r = cuMemSetAccess(va, mapped_, ad, 2);
        if (r != CUDA_SUCCESS) cuMemUnmap(va, mapped_);
      }
      if (r != CUDA_SUCCESS) {
        cuMemRelease(h);
        cuMemAddressFree(va, mapped_);
        cu_check(r, "cuMemMap/SetAccess(staging)");
      }
      handle_ = h;
      host_ = reinterpret_cast<uint8_t*>(va);
      dev_ = va;
    }
  }
  std::memset(host_, 0, n);  // pre-touch
  std::fprintf(stderr, "slotio: staging allocation %llu B (%d x %d records of %llu B, mapped %llu B, %s)\n",
               static_cast<unsigned long long>(n), groups_, records_, static_cast<unsigned long long>(rec_),
               static_cast<unsigned long long>(mapped_), mem_name(type_));
  std::fflush(stderr);
}

Staging::~Staging() {
  if (!host_) return;
  if (type_ == MemType::HostMallocForTests || type_ == MemType::AnonThp) {
    std::free(host_);
  } else if (type_ == MemType::HostAlloc) {
    cuMemFreeHost(host_);
  } else {
    cuMemUnmap(dev_, mapped_);
    cuMemRelease(handle_);
    cuMemAddressFree(dev_, mapped_);
  }
  if (cuda_) cuDevicePrimaryCtxRelease(device_);
}

uint8_t* Staging::host(int buf, int i) const {
  if (buf < 0 || buf >= groups_ || i < 0 || i >= records_) throw std::out_of_range("staging index");
  return host_ + (static_cast<uint64_t>(buf) * records_ + i) * rec_;
}
uint64_t Staging::dev(int buf, int i) const {
  if (buf < 0 || buf >= groups_ || i < 0 || i >= records_) throw std::out_of_range("staging index");
  return dev_ + (static_cast<uint64_t>(buf) * records_ + i) * rec_;
}

Sweep::Sweep(Ensurer& ensurer, Reader& reader, const File& experts, Staging& staging)
    : en_(ensurer), rd_(reader), file_(experts), st_(staging), state_(staging.groups(), State::Idle), count_(staging.groups(), 0), reserved_(staging.groups()) {
  if (staging.record_bytes() != ensurer.pool().config().record_bytes) throw std::invalid_argument("staging and pool record sizes differ");
}

SweepTicket Sweep::read_group(int layer, const int32_t* experts, size_t n, int buf, const int32_t* admit, size_t n_admit) {
  const PoolConfig& cfg = en_.pool().config();
  if (buf < 0 || buf >= st_.groups()) throw std::invalid_argument("staging buffer out of range");
  if (layer < 0 || layer >= cfg.layers) throw std::invalid_argument("layer out of range");
  if (n == 0 || n > static_cast<size_t>(st_.records())) throw std::invalid_argument("a group holds 1..records experts");
  for (size_t i = 0; i < n; ++i)
    if (experts[i] < 0 || experts[i] >= cfg.experts) throw std::invalid_argument("expert id out of range");
  if (state_[buf] == State::InFlight) throw std::logic_error("staging buffer has a read in flight");
  for (size_t a = 0; a < n_admit; ++a) {
    bool found = false;
    for (size_t i = 0; i < n && !found; ++i) found = experts[i] == admit[a];
    if (!found) throw std::invalid_argument("admit id is not in the group");
  }
  reserved_[buf].clear();
  std::vector<Job> jobs;
  jobs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t off = (static_cast<uint64_t>(layer) * cfg.experts + experts[i]) * cfg.record_bytes;
    jobs.push_back(Job{&file_, off, static_cast<uint32_t>(cfg.record_bytes), st_.host(buf, static_cast<int>(i)), Priority::Sweep});
  }
  SweepTicket t;
  t.buf = buf;
  t.n = static_cast<int>(n);
  t.layer = layer;
  if (n_admit > 0) {
    std::lock_guard<std::mutex> g(en_.mutex());
    Pool& pool = en_.pool();
    for (size_t a = 0; a < n_admit; ++a) {
      const int32_t idx = layer * cfg.experts + admit[a];
      if (__atomic_load_n(&pool.map_host()[idx], __ATOMIC_ACQUIRE) >= 0) continue;
      bool dup = false;
      for (const auto& r : reserved_[buf]) dup = dup || r.second == idx;
      if (dup) continue;
      int v;
      try {
        v = en_.clock().victim(pool.slots(), [&](int s) { return en_.pins().pinned(s); });
      } catch (const Exhausted&) {
        ++t.admit_refused;
        continue;
      }
      const int32_t o = pool.owner(v);
      if (o >= 0) {
        pool.unmap(o);
        pool.set_owner(v, -1);
      }
      pool.set_owner(v, idx);
      en_.pins().pin(v, gen_);  // a pinned slot is skipped by every victim search until the generation retires
      for (size_t i = 0; i < n; ++i)
        if (experts[i] == admit[a]) jobs[i].dst2 = pool.slot_host(v);
      reserved_[buf].emplace_back(v, idx);
    }
  }
  state_[buf] = State::InFlight;
  count_[buf] = static_cast<int>(n);
  try {
    t.id = rd_.submit(jobs.data(), jobs.size());
  } catch (...) {
    state_[buf] = State::Failed;
    {
      std::lock_guard<std::mutex> g(en_.mutex());
      for (const auto& r : reserved_[buf]) en_.pool().set_owner(r.first, -1);  // nothing was published
      reserved_[buf].clear();
    }
    throw;
  }
  return t;
}

bool Sweep::wait(SweepTicket& t, int64_t timeout_ms) {
  if (t.buf < 0 || t.buf >= st_.groups()) throw std::invalid_argument("bad ticket");
  if (state_[t.buf] == State::Ready) return true;
  if (state_[t.buf] != State::InFlight) throw std::logic_error("ticket has no read in flight");
  BatchResult r = rd_.join(t.id, timeout_ms < 0 ? -1 : timeout_ms * 1'000'000);
  if (!r.ok && r.err == ETIMEDOUT) return false;
  if (!r.ok) {
    state_[t.buf] = State::Failed;
    {
      std::lock_guard<std::mutex> g(en_.mutex());
      for (const auto& r : reserved_[t.buf]) en_.pool().set_owner(r.first, -1);  // nothing was published
      reserved_[t.buf].clear();
    }
    throw IoError("sweep read failed (layer " + std::to_string(t.layer) + ")", r.err);
  }
  {
    std::lock_guard<std::mutex> g(en_.mutex());
    for (const auto& r : reserved_[t.buf]) {
      en_.pool().publish(r.second, r.first);
      en_.clock().set_ref(r.first);
    }
    t.admitted = static_cast<int>(reserved_[t.buf].size());
    reserved_[t.buf].clear();
  }
  state_[t.buf] = State::Ready;
  return true;
}

uint64_t Sweep::record_dev(const SweepTicket& t, int i) const {
  if (t.buf < 0 || t.buf >= st_.groups() || state_[t.buf] != State::Ready) throw std::logic_error("staging buffer is not ready");
  if (i < 0 || i >= count_[t.buf]) throw std::out_of_range("record index");
  return st_.dev(t.buf, i);
}

const uint8_t* Sweep::record_host(const SweepTicket& t, int i) const {
  if (t.buf < 0 || t.buf >= st_.groups() || state_[t.buf] != State::Ready) throw std::logic_error("staging buffer is not ready");
  if (i < 0 || i >= count_[t.buf]) throw std::out_of_range("record index");
  return st_.host(t.buf, i);
}

std::optional<uint64_t> Sweep::resident_ptr_nopin_touch(int layer, int e) {
  std::lock_guard<std::mutex> g(en_.mutex());
  Pool& pool = en_.pool();
  const PoolConfig& cfg = pool.config();
  if (layer < 0 || layer >= cfg.layers || e < 0 || e >= cfg.experts) throw std::invalid_argument("layer or expert out of range");
  const int32_t s = __atomic_load_n(&pool.map_host()[layer * cfg.experts + e], __ATOMIC_ACQUIRE);
  if (s < 0) return std::nullopt;
  en_.pins().pin(s, gen_);  // no clock().set_ref(s): the sweep does not warm the CLOCK
  return pool.slot_dev(s);
}

bool Sweep::admit_from_staging(int layer, int e, uint64_t src) {
  std::lock_guard<std::mutex> g(en_.mutex());
  Pool& pool = en_.pool();
  const PoolConfig& cfg = pool.config();
  if (layer < 0 || layer >= cfg.layers || e < 0 || e >= cfg.experts) throw std::invalid_argument("layer or expert out of range");
  const int32_t idx = layer * cfg.experts + e;
  if (__atomic_load_n(&pool.map_host()[idx], __ATOMIC_ACQUIRE) >= 0) return true;
  int v;
  try {
    v = en_.clock().victim(pool.slots(), [&](int s) { return en_.pins().pinned(s); });
  } catch (const Exhausted&) {
    return false;
  }
  const int32_t o = pool.owner(v);
  if (o >= 0) {
    pool.unmap(o);
    pool.set_owner(v, -1);
  }
  std::memcpy(pool.slot_host(v), reinterpret_cast<const uint8_t*>(src), cfg.record_bytes);
  pool.set_owner(v, idx);
  pool.publish(idx, v);
  en_.clock().set_ref(v);
  return true;
}

}  // namespace slotio
