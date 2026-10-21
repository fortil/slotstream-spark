// Sonda de un solo uso: ¿puede el GB10 leer, desde un kernel, memoria de cada
// origen sin copia previa? Mide GB/s de lectura del kernel sobre:
//   device   cudaMalloc
//   managed  cudaMallocManaged (inicializada en CPU)
//   pinned   cudaHostAlloc(Mapped), puntero host directo
//   anon     malloc() paginable, sin registrar (camino ATS / PageableMemoryAccess)
//   register malloc() + cudaHostRegister
//   file     mmap(PROT_READ, MAP_SHARED) de un archivo existente, sin registrar
//   filereg  mmap del archivo + cudaHostRegister(ReadOnly)
// Archivos: solo lectura (O_RDONLY, PROT_READ). Sin escrituras a disco.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ERR %s -> %d %s\n", #x, (int)e, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned int acc = 0;
  for (; i < n; i += stride) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1ULL);
}

static size_t resident(void* p, size_t bytes) {
  size_t pg = sysconf(_SC_PAGESIZE), n = (bytes + pg - 1) / pg, r = 0;
  unsigned char* v = (unsigned char*)malloc(n);
  if (mincore(p, bytes, v) != 0) { free(v); return (size_t)-1; }
  for (size_t i = 0; i < n; i++) r += v[i] & 1;
  free(v); return r * pg;
}

static int bench(const char* tag, const void* p, size_t bytes, int reps, unsigned long long* dout) {
  int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(a);
    rd<<<sms * 16, 512>>>((const uint4*)p, bytes / 16, dout);
    cudaEventRecord(b);
    cudaError_t e = cudaEventSynchronize(b);
    if (e != cudaSuccess) { printf("%s rep%d ERR %d %s\n", tag, r, (int)e, cudaGetErrorString(e)); return 1; }
    e = cudaGetLastError();
    if (e != cudaSuccess) { printf("%s rep%d ERR %d %s\n", tag, r, (int)e, cudaGetErrorString(e)); return 1; }
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    printf("%-8s rep%d bytes=%zu ms=%.3f GBps=%.1f\n", tag, r, bytes, ms, bytes / (ms * 1e6));
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("uso: %s device|managed|pinned|anon|anonhuge|register [MiB] | file|filereg <ruta> <offset> <MiB>\n", argv[0]); return 2; }
  const char* mode = argv[1];
  unsigned long long* dout; CK(cudaMalloc(&dout, 8)); CK(cudaMemset(dout, 0, 8));
  size_t mib = 1024;
  if (!strcmp(mode, "file") || !strcmp(mode, "filereg")) mib = argc > 4 ? atol(argv[4]) : 256;
  else if (argc > 2) mib = atol(argv[2]);
  if (mib > 1024) { printf("max 1024 MiB\n"); return 2; }
  size_t bytes = mib << 20;
  int reps = 3;

  if (!strcmp(mode, "device")) {
    void* p; CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 1, bytes)); CK(cudaDeviceSynchronize());
    bench(mode, p, bytes, reps, dout); cudaFree(p);
  } else if (!strcmp(mode, "managed")) {
    void* p; CK(cudaMallocManaged(&p, bytes)); memset(p, 1, bytes);
    bench(mode, p, bytes, reps, dout); cudaFree(p);
  } else if (!strcmp(mode, "pinned")) {
    void* p; CK(cudaHostAlloc(&p, bytes, cudaHostAllocMapped)); memset(p, 1, bytes);
    bench(mode, p, bytes, reps, dout); cudaFreeHost(p);
  } else if (!strcmp(mode, "anon")) {
    void* p = aligned_alloc(4096, bytes); memset(p, 1, bytes);
    cudaPointerAttributes at; cudaError_t e = cudaPointerGetAttributes(&at, p);
    printf("anon pointer attrs rc=%d type=%d device=%d devicePointer=%p\n", (int)e, (int)at.type, at.device, at.devicePointer);
    cudaGetLastError();
    bench(mode, p, bytes, reps, dout); free(p);
  } else if (!strcmp(mode, "odirect")) {
    // odirect <pinned|managed|register|device> <ruta> <nrec>: pread O_DIRECT de registros
    // de experto (2 764 800 B) directamente en memoria visible por la GPU, y lectura en kernel.
    if (argc < 5) { printf("uso: odirect <alloc> <ruta> <nrec>\n"); return 2; }
    const char* al = argv[2]; int nrec = atoi(argv[4]); const size_t E = 2764800;
    if (nrec < 1 || nrec > 128) { printf("nrec 1..128\n"); return 2; }
    size_t tot = E * (size_t)nrec; void* p = nullptr; cudaError_t e = cudaSuccess;
    if (!strcmp(al, "pinned")) e = cudaHostAlloc(&p, tot, cudaHostAllocMapped);
    else if (!strcmp(al, "managed")) e = cudaMallocManaged(&p, tot);
    else if (!strcmp(al, "device")) e = cudaMalloc(&p, tot);
    else if (!strcmp(al, "register")) { p = aligned_alloc(4096, tot); memset(p, 0, tot); e = cudaHostRegister(p, tot, cudaHostRegisterDefault); }
    else { printf("alloc desconocido\n"); return 2; }
    if (e != cudaSuccess) { printf("alloc %s ERR %s\n", al, cudaGetErrorString(e)); return 1; }
    int fd = open(argv[3], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t slots = (st.st_size - E) / 4096;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int ok = 0; double worst = 0;
    for (int i = 0; i < nrec; i++) {
      off_t off = (off_t)(((size_t)(i + 1) * 104729ULL * 7919ULL) % slots) * 4096;
      struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
      ssize_t r = pread(fd, (char*)p + (size_t)i * E, E, off);
      clock_gettime(CLOCK_MONOTONIC, &b);
      double us = (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) / 1e3; if (us > worst) worst = us;
      if (r != (ssize_t)E) { printf("pread %s rec%d off=%lld -> %zd errno=%d %s\n", al, i, (long long)off, r, errno, strerror(errno)); break; }
      ok++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("odirect alloc=%s ok=%d/%d bytes=%zu s=%.4f GBps=%.2f us_per_rec=%.0f worst_us=%.0f\n", al, ok, nrec, (size_t)ok * E, s, ok * E / s / 1e9, ok ? s * 1e6 / ok : 0, worst);
    close(fd);
    if (ok == nrec) bench(al, p, tot & ~(size_t)15, 3, dout);
    if (!strcmp(al, "pinned")) cudaFreeHost(p); else if (!strcmp(al, "register")) { cudaHostUnregister(p); free(p); } else cudaFree(p);
  } else if (!strcmp(mode, "anonhuge")) {
    void* p = aligned_alloc(2 << 20, bytes);
    int rc = madvise(p, bytes, MADV_HUGEPAGE);
    memset(p, 1, bytes);
    FILE* f = fopen("/proc/self/smaps_rollup", "r"); char line[256];
    while (f && fgets(line, sizeof line, f)) if (strstr(line, "AnonHugePages")) printf("madvise rc=%d %s", rc, line);
    if (f) fclose(f);
    bench(mode, p, bytes, reps, dout); free(p);
  } else if (!strcmp(mode, "register")) {
    void* p = aligned_alloc(4096, bytes); memset(p, 1, bytes);
    CK(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
    bench(mode, p, bytes, reps, dout); cudaHostUnregister(p); free(p);
  } else if (!strcmp(mode, "file") || !strcmp(mode, "filereg")) {
    if (argc < 4) { printf("falta ruta/offset\n"); return 2; }
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    off_t off = (off_t)atoll(argv[3]) & ~((off_t)4095);
    void* p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, off);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    printf("file %s off=%lld bytes=%zu resident_before=%zu\n", argv[2], (long long)off, bytes, resident(p, bytes));
    if (!strcmp(mode, "filereg")) {
      cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterReadOnly);
      printf("cudaHostRegister(ReadOnly) rc=%d %s resident_after_register=%zu\n", (int)e, cudaGetErrorString(e), resident(p, bytes));
      if (e != cudaSuccess) {
        cudaGetLastError();
        e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
        printf("cudaHostRegister(Default) rc=%d %s resident_after_register=%zu\n", (int)e, cudaGetErrorString(e), resident(p, bytes));
        if (e != cudaSuccess) { cudaGetLastError(); munmap(p, bytes); close(fd); return 1; }
      }
    }
    bench(mode, p, bytes, reps, dout);
    printf("resident_after=%zu\n", resident(p, bytes));
    if (!strcmp(mode, "filereg")) cudaHostUnregister(p);
    munmap(p, bytes); close(fd);
  } else { printf("modo desconocido\n"); return 2; }
  cudaFree(dout);
  return 0;
}
