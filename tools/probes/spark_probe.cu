// spark_probe: memory-pool types (pool) and host<->GPU wait mechanisms (wait) on the GB10.
//
//   spark_probe pool --file F --slots N --lanes L1,L2 --types host_alloc,vmm_host_numa,anon_thp --tokens T
//   spark_probe wait --iters I --gaps-us 0,200,5000
//
// Prints one JSON object on stdout (with a "decision" block) and, when CAP is set,
// writes $CAP/result-<mode>.json. Files are opened read-only; the only writes are to memory.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

static const size_t SLOT = 2764800;  // one expert record, bytes
static const int A725_CPUS[] = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(r_, &s_); \
  throw std::runtime_error(std::string(#x) + ": " + s_); } } while (0)

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double now_us() { return now_ms() * 1000.0; }

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') o += "\\n"; else o += c; }
  return o + "\"";
}
static std::string jnum(double v) { char b[64]; snprintf(b, sizeof b, "%.4f", v); return b; }
static std::string jbool(bool b) { return b ? "true" : "false"; }

static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, sep)) if (!t.empty()) out.push_back(t);
  return out;
}

// ---- system readings (m6)

static long meminfo_kb(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r"); if (!f) return -1;
  char line[256]; long v = -1; size_t kl = strlen(key);
  while (fgets(line, sizeof line, f)) if (!strncmp(line, key, kl) && line[kl] == ':') { v = atol(line + kl + 1); break; }
  fclose(f); return v;
}
static long cgroup_current() {
  FILE* f = fopen("/proc/self/cgroup", "r"); if (!f) return -1;
  char line[512]; std::string path;
  while (fgets(line, sizeof line, f)) if (!strncmp(line, "0::", 3)) { path = line + 3; break; }
  fclose(f);
  while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
  std::string p = "/sys/fs/cgroup" + path + "/memory.current";
  f = fopen(p.c_str(), "r"); if (!f) return -1;
  long v = -1; if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); return v;
}
static long nvml_own_mib() {
  FILE* p = popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null", "r");
  if (!p) return -1;
  char line[256]; long v = -1; int me = getpid();
  while (fgets(line, sizeof line, p)) { int pid; long mib; if (sscanf(line, "%d, %ld", &pid, &mib) == 2 && pid == me) v = mib; }
  pclose(p); return v;
}
static std::string sysmem_json() {
  std::ostringstream o;
  o << "{\"MemAvailable_kb\":" << meminfo_kb("MemAvailable") << ",\"MemFree_kb\":" << meminfo_kb("MemFree")
    << ",\"AnonPages_kb\":" << meminfo_kb("AnonPages") << ",\"Mlocked_kb\":" << meminfo_kb("Mlocked")
    << ",\"Shmem_kb\":" << meminfo_kb("Shmem") << ",\"cgroup_memory_current\":" << cgroup_current()
    << ",\"nvml_own_mib\":" << nvml_own_mib() << "}";
  return o.str();
}

// ---- GPU kernels

__global__ void gather_kernel(const uint4* __restrict__ base, const int* __restrict__ slots, int nslots,
                              size_t slot_u4, unsigned int* out) {
  unsigned int acc = 0;
  for (int s = 0; s < nslots; s++) {
    const uint4* p = base + (size_t)slots[s] * slot_u4;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < slot_u4; i += (size_t)gridDim.x * blockDim.x) {
      uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1u);
}
__global__ void seq_kernel(const uint4* __restrict__ p, size_t n, unsigned int* out) {
  unsigned int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1u);
}
__global__ void touch_done(volatile unsigned int* done, unsigned int v) {
  *done = v;
  __threadfence_system();
}

// ---- pool memory types

struct Pool {
  std::string type;
  void* host = nullptr;          // CPU-writable view (null when cpu_access is false)
  void* dev = nullptr;           // pointer the kernel reads
  size_t bytes = 0;
  bool cpu_access = true;
  bool triton_addressable = false;
  bool kernel_readable = true;   // false: skip kernel reads (anon without pageable access)
  CUmemGenericAllocationHandle h = 0;
  bool vmm = false;
  CUdeviceptr va = 0;
  std::string note;
};

static void alloc_pool(Pool& p, size_t bytes, int dev_ord) {
  p.bytes = bytes;
  if (p.type == "host_alloc") {
    CK(cudaHostAlloc(&p.host, bytes, cudaHostAllocDefault));
    CUdeviceptr d = 0;
    p.triton_addressable = cuPointerGetAttribute(&d, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)p.host) == CUDA_SUCCESS;
    p.dev = p.triton_addressable ? (void*)d : p.host;
  } else if (p.type == "vmm_host_numa") {
    CUmemAllocationProp prop; memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = 0;
    size_t gran = 0;
    CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t total = (bytes + gran - 1) / gran * gran;
    p.bytes = total;
    CU(cuMemCreate(&p.h, total, &prop, 0));
    CU(cuMemAddressReserve(&p.va, total, 0, 0, 0));
    CU(cuMemMap(p.va, total, 0, p.h, 0));
    CUmemAccessDesc ad[2]; memset(ad, 0, sizeof ad);
    ad[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad[0].location.id = dev_ord; ad[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ad[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; ad[1].location.id = 0; ad[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(p.va, total, ad, 2) == CUDA_SUCCESS) {
      p.cpu_access = true;
    } else {
      CU(cuMemSetAccess(p.va, total, ad, 1));
      p.cpu_access = false;
      p.note = "host access refused; filled via cudaMemcpy";
    }
    p.vmm = true;
    p.dev = (void*)p.va;
    p.host = p.cpu_access ? (void*)p.va : nullptr;
    CUdeviceptr d = 0;
    p.triton_addressable = cuPointerGetAttribute(&d, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, p.va) == CUDA_SUCCESS;
  } else if (p.type == "anon_thp") {
    p.host = aligned_alloc(2u << 20, (bytes + (2u << 20) - 1) / (2u << 20) * (2u << 20));
    if (!p.host) throw std::runtime_error("aligned_alloc failed");
    madvise(p.host, bytes, MADV_HUGEPAGE);
    p.dev = p.host;
    CUdeviceptr d = 0;
    p.triton_addressable = cuPointerGetAttribute(&d, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)p.host) == CUDA_SUCCESS;
    int pma = 0; cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, 0);
    p.kernel_readable = pma != 0;
    if (!p.kernel_readable) p.note = "no pageable memory access; kernel reads skipped";
  } else {
    throw std::runtime_error("unknown type " + p.type);
  }
}

static void free_pool(Pool& p) {
  if (p.type == "host_alloc") cudaFreeHost(p.host);
  else if (p.type == "vmm_host_numa") { cuMemUnmap(p.va, p.bytes); cuMemAddressFree(p.va, p.bytes); cuMemRelease(p.h); }
  else if (p.type == "anon_thp") free(p.host);
}

// Fill N slots with `lanes` threads: O_DIRECT pread into a private 4096-aligned scratch, then memcpy to the slot.
struct FillResult { std::vector<double> pread_ms, memcpy_ms; double wall_ms = 0, gbs = 0; };

static FillResult fill_pool(Pool& p, int fd, size_t fsize, int nslots, int lanes) {
  FillResult r;
  std::mutex mu;
  size_t fsr = fsize / 4096 * 4096;
  if (fsr < SLOT) throw std::runtime_error("file smaller than one record");
  size_t mod = fsr - SLOT + 4096;
  double t0 = now_ms();
  std::vector<std::thread> th;
  for (int l = 0; l < lanes; l++) {
    th.emplace_back([&, l]() {
      cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(A725_CPUS[l % 10], &cs);
      pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
      cudaSetDevice(0);
      void* scratch = aligned_alloc(4096, SLOT);
      std::vector<double> pr, mc;
      for (int rec = l; rec < nslots; rec += lanes) {
        size_t off = ((size_t)rec * SLOT) % mod;
        double a = now_ms();
        size_t got = 0;
        while (got < SLOT) {
          ssize_t n = pread(fd, (char*)scratch + got, SLOT - got, off + got);
          if (n <= 0) break;
          got += n;
        }
        double b = now_ms();
        if (got != SLOT) { fprintf(stderr, "short pread %zu\n", got); continue; }
        if (p.cpu_access && p.host) memcpy((char*)p.host + (size_t)rec * SLOT, scratch, SLOT);
        else cudaMemcpy((char*)p.dev + (size_t)rec * SLOT, scratch, SLOT, cudaMemcpyHostToDevice);
        double c = now_ms();
        pr.push_back(b - a); mc.push_back(c - b);
      }
      free(scratch);
      std::lock_guard<std::mutex> g(mu);
      r.pread_ms.insert(r.pread_ms.end(), pr.begin(), pr.end());
      r.memcpy_ms.insert(r.memcpy_ms.end(), mc.begin(), mc.end());
    });
  }
  for (auto& t : th) t.join();
  r.wall_ms = now_ms() - t0;
  r.gbs = (double)nslots * SLOT / (r.wall_ms * 1e-3) / 1e9;
  return r;
}

static double gather_gbs(Pool& p, int nslots, int tokens, int* ok) {
  const int PER = 10;
  std::vector<int> h(tokens * PER);
  srand(1234);
  for (auto& v : h) v = rand() % nslots;
  int* d = nullptr; unsigned int* out = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(int)));
  CK(cudaMalloc(&out, sizeof(unsigned int)));
  CK(cudaMemset(out, 0, sizeof(unsigned int)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
  size_t u4 = SLOT / 16;
  gather_kernel<<<256, 256>>>((const uint4*)p.dev, d, PER, u4, out);  // warmup
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int t = 0; t < tokens; t++) gather_kernel<<<256, 256>>>((const uint4*)p.dev, d + t * PER, PER, u4, out);
  cudaEventRecord(e1);
  CK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d); cudaFree(out);
  *ok = 1;
  return (double)tokens * PER * SLOT / (ms * 1e-3) / 1e9;
}

static double seq_gbs(Pool& p, size_t bytes) {
  unsigned int* out = nullptr;
  CK(cudaMalloc(&out, 4)); CK(cudaMemset(out, 0, 4));
  size_t n = bytes / 16;
  seq_kernel<<<512, 256>>>((const uint4*)p.dev, n, out);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < 3; i++) seq_kernel<<<512, 256>>>((const uint4*)p.dev, n, out);
  cudaEventRecord(e1);
  CK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(out);
  return 3.0 * n * 16 / (ms * 1e-3) / 1e9;
}

struct TypeResult { std::string type, json; double gather = -1; bool addressable = false; double memcpy8 = -1; };

static TypeResult run_type(const std::string& type, int fd, size_t fsize, int nslots, const std::vector<int>& lanes, int tokens) {
  TypeResult tr; tr.type = type;
  std::ostringstream o;
  Pool p; p.type = type;
  try {
    std::string before = sysmem_json();
    alloc_pool(p, (size_t)nslots * SLOT, 0);
    // touch every page so the memory is really resident
    if (p.cpu_access && p.host) memset(p.host, 0x5a, (size_t)nslots * SLOT);
    else CK(cudaMemset(p.dev, 0x5a, (size_t)nslots * SLOT));
    std::string after = sysmem_json();
    tr.addressable = p.triton_addressable;
    o << "{\"type\":" << jstr(type) << ",\"bytes\":" << p.bytes << ",\"cpu_access\":" << jbool(p.cpu_access)
      << ",\"triton_addressable\":" << jbool(p.triton_addressable) << ",\"note\":" << jstr(p.note)
      << ",\"m6\":{\"before\":" << before << ",\"after\":" << after << "},\"fill\":[";
    bool first = true;
    int max_lane = 0;
    bool seen8 = false;
    for (int L : lanes) {
      FillResult fr = fill_pool(p, fd, fsize, nslots, L);
      if (!first) o << ","; first = false;
      double mp50 = pct(fr.memcpy_ms, 50);
      o << "{\"lanes\":" << L << ",\"pread_ms_p50\":" << jnum(pct(fr.pread_ms, 50)) << ",\"pread_ms_p99\":" << jnum(pct(fr.pread_ms, 99))
        << ",\"memcpy_ms_p50\":" << jnum(mp50) << ",\"memcpy_ms_p99\":" << jnum(pct(fr.memcpy_ms, 99))
        << ",\"aggregate_gbs\":" << jnum(fr.gbs) << ",\"records\":" << fr.pread_ms.size() << "}";
      if (L == 8 || (L > max_lane && !seen8)) tr.memcpy8 = mp50;  // 8 lanes, else the largest lane count run
      if (L == 8) seen8 = true;
      if (L > max_lane) max_lane = L;
    }
    o << "]";
    if (p.kernel_readable) {
      int ok = 0;
      tr.gather = gather_gbs(p, nslots, tokens, &ok);
      double sq = seq_gbs(p, (size_t)nslots * SLOT);
      o << ",\"gather_gbs\":" << jnum(tr.gather) << ",\"seq_read_gbs\":" << jnum(sq) << ",\"tokens\":" << tokens;
    } else {
      o << ",\"gather_gbs\":null,\"seq_read_gbs\":null";
    }
    o << "}";
  } catch (const std::exception& e) {
    o.str(""); o << "{\"type\":" << jstr(type) << ",\"error\":" << jstr(e.what()) << "}";
    tr.gather = -1; tr.addressable = false;
  }
  free_pool(p);
  tr.json = o.str();
  return tr;
}

static std::string arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
  return def;
}

static void emit(const std::string& mode, const std::string& json) {
  printf("%s\n", json.c_str());
  const char* cap = getenv("CAP");
  if (cap && *cap) {
    std::string p = std::string(cap) + "/result-" + mode + ".json";
    FILE* f = fopen(p.c_str(), "w");
    if (f) { fputs(json.c_str(), f); fputc('\n', f); fclose(f); }
  }
}

static int mode_pool(int argc, char** argv) {
  std::string file = arg(argc, argv, "--file", "");
  int nslots = atoi(arg(argc, argv, "--slots", "8").c_str());
  std::vector<int> lanes; for (auto& s : split(arg(argc, argv, "--lanes", "1,8"), ',')) lanes.push_back(atoi(s.c_str()));
  std::vector<std::string> types = split(arg(argc, argv, "--types", "host_alloc,vmm_host_numa,anon_thp"), ',');
  int tokens = atoi(arg(argc, argv, "--tokens", "64").c_str());
  if (file.empty()) { fprintf(stderr, "pool needs --file\n"); return 2; }
  int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open O_DIRECT"); return 2; }
  struct stat st; fstat(fd, &st);
  // anon_thp last: a pageable-memory kernel fault must not poison the other measurements
  std::stable_sort(types.begin(), types.end(), [](const std::string& a, const std::string& b) {
    return (a == "anon_thp") < (b == "anon_thp"); });
  std::vector<TypeResult> res;
  for (auto& t : types) res.push_back(run_type(t, fd, (size_t)st.st_size, nslots, lanes, tokens));
  // decision rules (fixed before measuring)
  const TypeResult* best = nullptr;
  for (auto& r : res) if (r.addressable && r.gather >= 0 && (!best || r.gather > best->gather)) best = &r;
  std::string winner = "none";
  double gwin = -1, mwin = -1;
  if (best) {
    winner = best->type; gwin = best->gather; mwin = best->memcpy8;
    for (auto& r : res) if (r.type == "host_alloc" && r.addressable && r.gather >= 0 && best->gather > 0 &&
                           (best->gather - r.gather) / best->gather < 0.02) { winner = "host_alloc"; gwin = r.gather; mwin = r.memcpy8; }
  }
  std::ostringstream o;
  o << "{\"mode\":\"pool\",\"slot_bytes\":" << SLOT << ",\"slots\":" << nslots << ",\"types\":[";
  for (size_t i = 0; i < res.size(); i++) o << (i ? "," : "") << res[i].json;
  o << "],\"decision\":{\"pool_memory\":" << jstr(winner) << ",\"t_memcpy_ms\":" << jnum(mwin)
    << ",\"pool_gather_gbs\":" << jnum(gwin) << ",\"rule\":\"highest gather GB/s among triton_addressable types; <2% difference -> host_alloc; anon_thp reported only\"}}";
  emit("pool", o.str());
  close(fd);
  return 0;
}

// ---- wait mechanisms

// Spin until *p == v; a stuck GPU side becomes an error after 2 s instead of a hang.
static void spin_eq(volatile unsigned int* p, unsigned int v, const char* where) {
  double t0 = now_us();
  while (*p != v) if (now_us() - t0 > 2e6) throw std::runtime_error(std::string("spin timeout at ") + where);
}

static std::string sm_clock() {
  FILE* p = popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null", "r");
  std::string s; if (!p) return s; char b[64]; if (fgets(b, sizeof b, p)) s = b; pclose(p);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

static std::string stats_json(const std::vector<double>& v) {
  return "{\"p50_us\":" + jnum(pct(v, 50)) + ",\"p99_us\":" + jnum(pct(v, 99)) + ",\"n\":" + std::to_string(v.size()) + "}";
}

static int mode_wait(int argc, char** argv) {
  int iters = atoi(arg(argc, argv, "--iters", "1000").c_str());
  std::vector<int> gaps; for (auto& s : split(arg(argc, argv, "--gaps-us", "0,200,5000"), ',')) gaps.push_back(atoi(s.c_str()));
  std::ostringstream o;
  o << "{\"mode\":\"wait\",\"iters\":" << iters;
  double spin_p50 = -1, evs_p50 = -1;
  std::string memop_in_graph = "unsupported";
  try {
    CK(cudaSetDevice(0));
    CK(cudaFree(0));
    volatile unsigned int* flag = nullptr; volatile unsigned int* done = nullptr;
    CK(cudaHostAlloc((void**)&flag, 4096, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&done, 4096, cudaHostAllocMapped));
    CUdeviceptr dflag = 0, ddone = 0;
    CU(cuPointerGetAttribute(&dflag, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)flag));
    CU(cuPointerGetAttribute(&ddone, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)done));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    *flag = 0; *done = 0;
    unsigned int ctr = 0;
    // warm up: lazy module loading must not happen while a stream wait is pending
    touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, 0);
    CK(cudaStreamSynchronize(st));

    // g2a: stream wait-value, then a kernel sets done
    std::vector<double> g2a;
    std::string g2a_err;
    try {
      for (int i = 0; i < iters; i++) {
        ctr++;
        CU(cuStreamWaitValue32((CUstream)st, dflag, ctr, CU_STREAM_WAIT_VALUE_GEQ));
        touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, ctr);
        double w = now_us(); while (now_us() - w < 50) {}  // let the GPU reach the wait
        double t0 = now_us();
        *flag = ctr; __sync_synchronize();
        spin_eq(done, ctr, "g2a");
        g2a.push_back(now_us() - t0);
      }
      CK(cudaStreamSynchronize(st));
    } catch (const std::exception& e) { g2a_err = e.what(); cudaGetLastError(); }
    o << ",\"g2a\":" << (g2a_err.empty() ? stats_json(g2a) : "{\"error\":" + jstr(g2a_err) + "}");

    // g2b: the same wait + kernel captured into a CUDA graph
    std::vector<double> g2b;
    std::string g2b_err;
    try {
      cudaGraph_t g; cudaGraphExec_t ge;
      // graphs replay fixed parameters: reset the flag each time and wait for equality with 1
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      CU(cuStreamWaitValue32((CUstream)st, dflag, 1, CU_STREAM_WAIT_VALUE_EQ));
      touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, 2);
      CK(cudaStreamEndCapture(st, &g));
      CK(cudaGraphInstantiate(&ge, g, 0));
      cudaGraphDestroy(g);
      for (int i = 0; i < iters; i++) {
        *flag = 0; *done = 0; __sync_synchronize();
        CK(cudaGraphLaunch(ge, st));
        double w = now_us(); while (now_us() - w < 50) {}
        double t0 = now_us();
        *flag = 1; __sync_synchronize();
        spin_eq(done, 2, "g2b");
        g2b.push_back(now_us() - t0);
        CK(cudaStreamSynchronize(st));
      }
      cudaGraphExecDestroy(ge);
      memop_in_graph = "supported";
    } catch (const std::exception& e) { g2b_err = e.what(); cudaGetLastError(); }
    o << ",\"g2b\":" << (g2b_err.empty() ? stats_json(g2b) : "{\"error\":" + jstr(g2b_err) + "}");
    cudaStreamSynchronize(st);

    // g2c: kernel + cudaEventSynchronize vs kernel writing a pinned flag + CPU spin
    cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    std::vector<double> evs, spin;
    for (int i = 0; i < iters; i++) {
      ctr++;
      double t0 = now_us();
      touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, ctr);
      cudaEventRecord(ev, st);
      cudaEventSynchronize(ev);
      evs.push_back(now_us() - t0);
    }
    for (int i = 0; i < iters; i++) {
      ctr++;
      double t0 = now_us();
      touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, ctr);
      spin_eq(done, ctr, "g2c");
      spin.push_back(now_us() - t0);
    }
    CK(cudaStreamSynchronize(st));
    o << ",\"g2c\":{\"event_sync\":" << stats_json(evs) << ",\"spin_flag\":" << stats_json(spin) << "}";
    evs_p50 = pct(evs, 50); spin_p50 = pct(spin, 50);

    // g3: launch->done latency after an idle gap
    o << ",\"g3\":[";
    for (size_t gi = 0; gi < gaps.size(); gi++) {
      std::vector<double> lat;
      for (int i = 0; i < std::max(50, iters / 4); i++) {
        double w = now_us(); while (now_us() - w < gaps[gi]) {}
        ctr++;
        double t0 = now_us();
        touch_done<<<1, 1, 0, st>>>((volatile unsigned int*)ddone, ctr);
        spin_eq(done, ctr, "g3");
        lat.push_back(now_us() - t0);
      }
      std::string clk = sm_clock();
      o << (gi ? "," : "") << "{\"gap_us\":" << gaps[gi] << ",\"latency\":" << stats_json(lat) << ",\"sm_clock_mhz\":" << jstr(clk) << "}";
    }
    o << "]";
  } catch (const std::exception& e) {
    o << ",\"error\":" << jstr(e.what());
  }
  std::string mech = (spin_p50 >= 0 && evs_p50 >= 0) ? (spin_p50 <= evs_p50 ? "spin_flag" : "event_sync") : "unknown";
  o << ",\"decision\":{\"wait_mechanism\":" << jstr(mech) << ",\"memop_in_graph\":" << jstr(memop_in_graph)
    << ",\"rule\":\"spin_flag if its p50 <= event-sync p50\"}}";
  emit("wait", o.str());
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: spark_probe pool|wait ...\n"); return 2; }
  try {
    CK(cudaSetDevice(0));
    CK(cudaFree(0));
    if (!strcmp(argv[1], "pool")) return mode_pool(argc, argv);
    if (!strcmp(argv[1], "wait")) return mode_wait(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
