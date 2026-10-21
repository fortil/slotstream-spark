#include <cuda.h>
#include <errno.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#include "slotio/pool.hpp"

namespace slotio {

namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void cu_check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  cuGetErrorString(r, &msg);
  throw IoError(std::string(what) + ": " + (msg ? msg : "cuda error"), static_cast<int>(r));
}
#define CU(x) cu_check((x), #x)
constexpr uint64_t kHuge = 2ull << 20;
}  // namespace

Pool::Pool(MemType type, const PoolConfig& cfg) : type_(type), cfg_(cfg), owner_(cfg.max_slots, -1) {
  if (static_cast<uint64_t>(cfg_.slots_per_chunk) * cfg_.record_bytes > cfg_.chunk_bytes)
    throw std::invalid_argument("slots_per_chunk * record_bytes exceeds chunk_bytes");
  const size_t map_bytes = map_len() * sizeof(int32_t);
  const size_t ptr_bytes = static_cast<size_t>(cfg_.max_slots) * sizeof(uint64_t);
  if (type_ == MemType::HostMallocForTests) {
    map_ = static_cast<int32_t*>(std::aligned_alloc(4096, (map_bytes + 4095) / 4096 * 4096));
    slot_ptr_ = static_cast<uint64_t*>(std::aligned_alloc(4096, (ptr_bytes + 4095) / 4096 * 4096));
    if (!map_ || !slot_ptr_) throw IoError("table allocation failed", ENOMEM);
    map_dev_ = reinterpret_cast<uint64_t>(map_);
    slot_ptr_dev_ = reinterpret_cast<uint64_t>(slot_ptr_);
  } else {
    cuda_init();
    void* m = nullptr;
    void* p = nullptr;
    CU(cuMemHostAlloc(&m, map_bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
    CU(cuMemHostAlloc(&p, ptr_bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
    map_ = static_cast<int32_t*>(m);
    slot_ptr_ = static_cast<uint64_t*>(p);
    CUdeviceptr d = 0;
    CU(cuMemHostGetDevicePointer(&d, m, 0));
    map_dev_ = d;
    CU(cuMemHostGetDevicePointer(&d, p, 0));
    slot_ptr_dev_ = d;
  }
  for (size_t i = 0; i < map_len(); ++i) map_[i] = -1;
  std::memset(slot_ptr_, 0, ptr_bytes);
  const int max_chunks = (cfg_.max_slots + cfg_.slots_per_chunk - 1) / cfg_.slots_per_chunk;
  chunk_host_.assign(max_chunks, nullptr);
  chunk_dev_.assign(max_chunks, 0);
  chunk_handle_.assign(max_chunks, 0);
  if (type_ == MemType::VmmHostNuma) {
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = 0;
    size_t gran = 0;
    CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    gran_ = gran;
    if (cfg_.chunk_bytes % gran_) throw std::invalid_argument("chunk_bytes not a multiple of the VMM granularity");
    va_bytes_ = static_cast<uint64_t>(max_chunks) * cfg_.chunk_bytes;
    CUdeviceptr va = 0;
    CU(cuMemAddressReserve(&va, va_bytes_, 0, 0, 0));
    va_ = va;
  }
}

Pool::~Pool() {
  for (int c = chunks_ - 1; c >= 0; --c) {
    try {
      free_chunk(c);
    } catch (...) {
    }
  }
  if (va_) cuMemAddressFree(va_, va_bytes_);
  if (type_ == MemType::HostMallocForTests) {
    std::free(map_);
    std::free(slot_ptr_);
  } else {
    if (map_) cuMemFreeHost(map_);
    if (slot_ptr_) cuMemFreeHost(slot_ptr_);
  }
  if (cuda_) cuDevicePrimaryCtxRelease(device_);
}

void Pool::cuda_init() {
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  device_ = dev;
  cuda_ = true;
}

void Pool::alloc_chunk(int c) {
  const uint64_t n = cfg_.chunk_bytes;
  switch (type_) {
    case MemType::HostAlloc: {
      void* p = nullptr;
      CU(cuMemHostAlloc(&p, n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
      CUdeviceptr d = 0;
      CU(cuMemHostGetDevicePointer(&d, p, 0));
      chunk_host_[c] = static_cast<uint8_t*>(p);
      chunk_dev_[c] = d;
      break;
    }
    case MemType::VmmHostNuma: {
      CUmemAllocationProp prop;
      std::memset(&prop, 0, sizeof prop);
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      prop.location.id = 0;
      CUmemGenericAllocationHandle h;
      CU(cuMemCreate(&h, n, &prop, 0));
      CUdeviceptr va = va_ + static_cast<uint64_t>(c) * n;
      CUresult r = cuMemMap(va, n, 0, h, 0);
      if (r != CUDA_SUCCESS) {
        cuMemRelease(h);
        cu_check(r, "cuMemMap");
      }
      CUmemAccessDesc ad[2];
      std::memset(ad, 0, sizeof ad);
      ad[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      ad[0].location.id = device_;
      ad[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      ad[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      ad[1].location.id = 0;
      ad[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      r = cuMemSetAccess(va, n, ad, 2);
      if (r != CUDA_SUCCESS) {
        cuMemUnmap(va, n);
        cuMemRelease(h);
        cu_check(r, "cuMemSetAccess(device+host_numa)");
      }
      chunk_handle_[c] = h;
      chunk_host_[c] = reinterpret_cast<uint8_t*>(va);
      chunk_dev_[c] = va;
      break;
    }
    case MemType::AnonThp: {
      void* p = std::aligned_alloc(kHuge, (n + kHuge - 1) / kHuge * kHuge);
      if (!p) throw IoError("aligned_alloc failed", ENOMEM);
      madvise(p, n, MADV_HUGEPAGE);
      chunk_host_[c] = static_cast<uint8_t*>(p);
      chunk_dev_[c] = reinterpret_cast<uint64_t>(p);  // ATS: the device uses the host VA
      break;
    }
    case MemType::HostMallocForTests: {
      void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
      if (!p) throw IoError("aligned_alloc failed", ENOMEM);
      chunk_host_[c] = static_cast<uint8_t*>(p);
      chunk_dev_[c] = reinterpret_cast<uint64_t>(p);
      break;
    }
  }
  std::memset(chunk_host_[c], 0, n);  // pre-touch
}

void Pool::free_chunk(int c) {
  int64_t t0 = now_ns();
  switch (type_) {
    case MemType::HostAlloc:
      if (chunk_host_[c]) CU(cuMemFreeHost(chunk_host_[c]));
      break;
    case MemType::VmmHostNuma:
      if (chunk_handle_[c]) {
        CU(cuMemUnmap(chunk_dev_[c], cfg_.chunk_bytes));
        CU(cuMemRelease(chunk_handle_[c]));
      }
      break;
    case MemType::AnonThp:
    case MemType::HostMallocForTests:
      std::free(chunk_host_[c]);
      break;
  }
  chunk_host_[c] = nullptr;
  chunk_dev_[c] = 0;
  chunk_handle_[c] = 0;
  st_.free_ns.push_back(now_ns() - t0);
}

void Pool::resize_slots(int n) {
  if (n < 0 || n > cfg_.max_slots) throw std::invalid_argument("resize_slots out of range");
  if (cuda_) {
    CUcontext ctx;
    CU(cuDevicePrimaryCtxRetain(&ctx, device_));
    CU(cuCtxSetCurrent(ctx));
    CU(cuDevicePrimaryCtxRelease(device_));  // the retain made in cuda_init() keeps the context alive
  }
  const int need = (n + cfg_.slots_per_chunk - 1) / cfg_.slots_per_chunk;
  if (n >= slots_) {
    int64_t t0 = now_ns();
    for (int c = chunks_; c < need; ++c) alloc_chunk(c);
    chunks_ = std::max(chunks_, need);
    for (int s = slots_; s < n; ++s) slot_ptr_[s] = slot_dev(s);
    slots_ = n;
    st_.alloc_ns += static_cast<uint64_t>(now_ns() - t0);
    return;
  }
  for (int s = n; s < slots_; ++s) {
    if (owner_[s] >= 0) {
      unmap(owner_[s]);
      owner_[s] = -1;
    }
    slot_ptr_[s] = 0;
  }
  if (shrink_hook_) shrink_hook_(n);
  slots_ = n;
  for (int c = chunks_ - 1; c >= need; --c) free_chunk(c);
  chunks_ = need;
}

uint8_t* Pool::slot_host(int s) const {
  return chunk_host_[s / cfg_.slots_per_chunk] + static_cast<uint64_t>(s % cfg_.slots_per_chunk) * cfg_.record_bytes;
}
uint64_t Pool::slot_dev(int s) const {
  return chunk_dev_[s / cfg_.slots_per_chunk] + static_cast<uint64_t>(s % cfg_.slots_per_chunk) * cfg_.record_bytes;
}

void Pool::publish(int32_t idx, int32_t slot) { __atomic_store_n(&map_[idx], slot, __ATOMIC_RELEASE); }
void Pool::unmap(int32_t idx) { __atomic_store_n(&map_[idx], -1, __ATOMIC_RELEASE); }

PoolStats Pool::stats() const {
  PoolStats s = st_;
  s.slots = slots_;
  s.chunks = chunks_;
  s.owned_bytes = owned_bytes();
  return s;
}

}  // namespace slotio
