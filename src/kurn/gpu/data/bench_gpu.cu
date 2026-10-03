// bench_gpu.cu: KURN's GPU harness (CUDA, Linux). Built by `kurn gpu harness` or the hand-run kit.
//
//   bench_gpu info                         device, clocks, NVML, idle power (JSON)
//   bench_gpu roofline [--secs S]          measured HBM read bandwidth, int8 mma.sync / dp4a peaks, launch overhead (JSON)
//   bench_gpu run --lib K.so --fmt F --N n --K k --M m [--reps R --secs S --quant 0|1 --cold 0|1 --seed s]
//                                          one KURN kernel: exactness + timing + energy (JSON lines)
//   bench_gpu matrix --plan P --out O.jsonl  the benchmark matrix: every implementation in the plan, interleaved
//
// Implementations measured on identical weight bytes in one process:
//   kurn:<name>   a generated kernel (dlopen'd .so implementing kurn_gpu.h), activation quantization included
//   cublas-fp16   cublasGemmEx on fp16-dequantized weights (activations converted f32 -> f16 inside the timing)
//   cublas-int8   cublasGemmEx int8 x int8 -> int32 on the q8 bytes: a tensor-core speed reference only (no block scales)
//   ggml          llama.cpp's ggml-cuda MUL_MAT (MMVQ / MMQ / cuBLAS, whatever ggml dispatches), when built with -DKURN_GGML
//
// Timing: weights rotate over enough device copies to exceed 4x L2 (cold, like decode streaming layer weights);
// KURN and cuBLAS calls are replayed from a CUDA graph; ggml runs its own graph compute. A round measures for
// --secs and records time per call and NVML board energy per call; rounds rotate the implementation order.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kurn_gpu.h"
#include "kurn_gpu_ref.h"

#ifdef KURN_GGML
#include "ggml-cuda.h"
#include "kurn_ggml_mm.h"
#endif

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) {                                                                               \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x);     \
      exit(3);                                                                                             \
    }                                                                                                      \
  } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// JSON number, or null when not finite (no NVML energy, no reference, ...)
static std::string jnum(double v, const char *fmt = "%.6g") {
  if (!std::isfinite(v)) return "null";
  char b[64];
  snprintf(b, sizeof b, fmt, v);
  return b;
}

static std::string jstr(const std::string &s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if ((unsigned char)c >= 0x20) o += c;
  }
  return o + "\"";
}

// ------------------------------------------------------------------ NVML (dlopen'd; no header or link dependency)
struct Nvml {
  void *h = nullptr, *dev = nullptr;
  int (*init)() = nullptr;
  int (*by_pci)(const char *, void **) = nullptr;
  int (*energy)(void *, unsigned long long *) = nullptr;
  int (*power)(void *, unsigned *) = nullptr;
  int (*clock)(void *, int, unsigned *) = nullptr;
  int (*temp)(void *, int, unsigned *) = nullptr;
  int (*reasons)(void *, unsigned long long *) = nullptr;
  int (*limit)(void *, unsigned *) = nullptr;
  int (*driver)(char *, unsigned) = nullptr;
  bool ok = false, has_energy = false;
  void open(int cuda_dev) {
    h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    init = (int (*)())dlsym(h, "nvmlInit_v2");
    by_pci = (int (*)(const char *, void **))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    energy = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
    power = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
    clock = (int (*)(void *, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
    temp = (int (*)(void *, int, unsigned *))dlsym(h, "nvmlDeviceGetTemperature");
    reasons = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!reasons) reasons = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
    limit = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
    driver = (int (*)(char *, unsigned))dlsym(h, "nvmlSystemGetDriverVersion");
    if (!init || !by_pci || init() != 0) return;
    char pci[64];
    if (cudaDeviceGetPCIBusId(pci, sizeof pci, cuda_dev) != cudaSuccess || by_pci(pci, &dev) != 0) return;
    ok = true;
    unsigned long long e;
    has_energy = energy && energy(dev, &e) == 0;
  }
  double joules() {  // board energy counter (mJ since driver load); NAN if unavailable
    unsigned long long e;
    return has_energy && energy(dev, &e) == 0 ? e * 1e-3 : NAN;
  }
  double watts() {
    unsigned p;
    return ok && power && power(dev, &p) == 0 ? p * 1e-3 : NAN;
  }
  unsigned clk(int which) {  // 1 = SM, 2 = memory (NVML_CLOCK_SM / NVML_CLOCK_MEM)
    unsigned v = 0;
    if (ok && clock) clock(dev, which, &v);
    return v;
  }
  unsigned temperature() {
    unsigned v = 0;
    if (ok && temp) temp(dev, 0, &v);
    return v;
  }
  unsigned long long throttle() {
    unsigned long long v = 0;
    if (ok && reasons) reasons(dev, &v);
    return v;
  }
};
static Nvml g_nvml;

// ------------------------------------------------------------------ device info and roofline kernels
struct Dev {
  cudaDeviceProp p;
  int id = 0, cc = 0;
  double nominal_bw = 0;  // GB/s from memory clock x bus width
  size_t l2 = 0;
};
static Dev g_dev;

static void dev_init() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n == 0) {
    printf("{\"kind\": \"error\", \"error\": %s}\n", jstr(e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device").c_str());
    exit(4);
  }
  CK(cudaGetDevice(&g_dev.id));
  CK(cudaGetDeviceProperties(&g_dev.p, g_dev.id));
  g_dev.cc = g_dev.p.major * 10 + g_dev.p.minor;
  int clk_khz = 0, bus = 0, l2 = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, g_dev.id);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, g_dev.id);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, g_dev.id);
  g_dev.nominal_bw = 2.0 * clk_khz * 1e3 * bus / 8 / 1e9;
  g_dev.l2 = (size_t)l2;
  g_nvml.open(g_dev.id);
}

__global__ void k_read(const uint4 *__restrict__ p, size_t n, unsigned *out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

__global__ void k_mma_peak(int iters, int *out) {
#if __CUDA_ARCH__ >= 800
  unsigned a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  int c[4][4] = {};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  int s = 0;
  for (int j = 0; j < 4; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 0x12345678) out[0] = s;
#endif
}

__global__ void k_dp4a_peak(int iters, int *out) {
  int a = threadIdx.x, b = a * 7, c[8] = {};
  for (int i = 0; i < iters; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = __dp4a(a + j, b, c[j]);
  int s = 0;
  for (int j = 0; j < 8; j++) s += c[j];
  if (s == 0x12345678) out[0] = s;
}

__global__ void k_empty() {}

static double idle_watts(double secs) {
  if (!g_nvml.ok) return NAN;
  double s = 0;
  int n = 0;
  double t0 = now_s();
  while (now_s() - t0 < secs) {
    double w = g_nvml.watts();
    if (!std::isnan(w)) s += w, n++;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return n ? s / n : NAN;
}

static int cmd_info() {
  char drv[96] = "?";
  if (g_nvml.ok && g_nvml.driver) g_nvml.driver(drv, sizeof drv);
  unsigned lim = 0;
  if (g_nvml.ok && g_nvml.limit) g_nvml.limit(g_nvml.dev, &lim);
  int rt = 0;
  cudaRuntimeGetVersion(&rt);
  printf("{\"kind\": \"info\", \"name\": %s, \"cc\": %d, \"sms\": %d, \"l2_bytes\": %zu, \"mem_bytes\": %zu, "
         "\"nominal_bw_gbs\": %.1f, \"smem_per_sm\": %zu, \"cuda_runtime\": %d, \"driver\": %s, \"nvml\": %s, "
         "\"nvml_energy\": %s, \"power_limit_w\": %.1f, \"idle_w\": %s, \"sm_mhz\": %u, \"mem_mhz\": %u, \"temp_c\": %u}\n",
         jstr(g_dev.p.name).c_str(), g_dev.cc, g_dev.p.multiProcessorCount, g_dev.l2, g_dev.p.totalGlobalMem, g_dev.nominal_bw,
         g_dev.p.sharedMemPerMultiprocessor, rt, jstr(drv).c_str(), g_nvml.ok ? "true" : "false",
         g_nvml.has_energy ? "true" : "false", lim * 1e-3, jnum(idle_watts(2.0), "%.1f").c_str(), g_nvml.clk(1), g_nvml.clk(2), g_nvml.temperature());
  return 0;
}

static int cmd_roofline(double secs) {
  size_t free_b, tot;
  CK(cudaMemGetInfo(&free_b, &tot));
  size_t bytes = std::min(free_b / 3, std::max((size_t)2 << 30, 16 * g_dev.l2));
  bytes &= ~(size_t)4095;
  uint4 *buf;
  unsigned *sink;
  int *isink;
  CK(cudaMalloc(&buf, bytes));
  CK(cudaMemset(buf, 1, bytes));
  CK(cudaMalloc(&sink, 64));
  CK(cudaMalloc(&isink, 64));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  int sms = g_dev.p.multiProcessorCount;
  // HBM read: best of a few grid shapes
  double best_bw = 0;
  for (int bpsm : {4, 8, 16}) {
    k_read<<<sms * bpsm, 512>>>(buf, bytes / 16, sink);
    CK(cudaDeviceSynchronize());
    int it = 0;
    double t0 = now_s();
    CK(cudaEventRecord(e0));
    while (now_s() - t0 < secs / 3) {
      k_read<<<sms * bpsm, 512>>>(buf, bytes / 16, sink);
      it++;
      if (it % 8 == 0) CK(cudaEventSynchronize(e0));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    best_bw = std::max(best_bw, (double)bytes * it / (ms * 1e-3) / 1e9);
  }
  auto peak = [&](bool mma) {
    int iters = 1 << 14;
    for (int rep = 0; rep < 2; rep++) {
      CK(cudaEventRecord(e0));
      if (mma)
        k_mma_peak<<<sms * 8, 256>>>(iters, isink);
      else
        k_dp4a_peak<<<sms * 8, 256>>>(iters, isink);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
    }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    double threads = (double)sms * 8 * 256;
    double ops = mma ? threads / 32 * iters * 4 * (2.0 * 16 * 8 * 32)  // 4 m16n8k32 per warp per iteration
                     : threads * iters * 8 * 8.0;                     // 8 dp4a per thread per iteration, 8 ops each
    return ops / (ms * 1e-3) / 1e12;
  };
  double mma_tops = g_dev.cc >= 80 ? peak(true) : NAN, dp4a_tops = peak(false);
  // launch overhead: direct launches and graph-replayed launches of an empty kernel
  const int nl = 20000;
  CK(cudaDeviceSynchronize());
  double t0 = now_s();
  for (int i = 0; i < nl; i++) k_empty<<<1, 32>>>();
  CK(cudaDeviceSynchronize());
  double direct_us = (now_s() - t0) / nl * 1e6;
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < 1000; i++) k_empty<<<1, 32, 0, st>>>();
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  t0 = now_s();
  for (int i = 0; i < 20; i++) CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  double graph_us = (now_s() - t0) / (20 * 1000) * 1e6;
  printf("{\"kind\": \"roofline\", \"name\": %s, \"cc\": %d, \"hbm_read_gbs\": %.1f, \"nominal_bw_gbs\": %.1f, \"buffer_bytes\": %zu, "
         "\"int8_mma_sync_tops\": %s, \"dp4a_tops\": %.2f, \"launch_us\": %.2f, \"graph_launch_us\": %.3f, \"sm_mhz\": %u, "
         "\"mem_mhz\": %u}\n",
         jstr(g_dev.p.name).c_str(), g_dev.cc, best_bw, g_dev.nominal_bw, bytes, jnum(mma_tops, "%.1f").c_str(), dp4a_tops, direct_us, graph_us,
         g_nvml.clk(1), g_nvml.clk(2));
  cudaFree(buf);
  return 0;
}

// ------------------------------------------------------------------ problem data shared by every implementation
struct Problem {
  std::string fmt;
  const kref_fmt *f = nullptr;
  int N = 0, K = 0;
  size_t wbytes = 0;
  std::vector<uint8_t> W;  // host weights, ggml blocks
  int copies = 1;
  std::vector<uint8_t *> dW;  // device copies (native blocks)
  std::vector<int> rows;      // rows checked against the reference
  std::vector<uint8_t> h16, h8;  // dequantized weights for the cuBLAS baselines (filled on first use)
};

struct Acts {
  int M = 0;
  std::vector<float> X;
  float *dX = nullptr;
  std::vector<int> cols;
  std::vector<double> ref_model;  // reference with f32 activations, [cols][rows]
};

static void pick(std::vector<int> &v, int n, int want) {
  v.clear();
  if (n <= want) {
    for (int i = 0; i < n; i++) v.push_back(i);
    return;
  }
  for (int i = 0; i < want; i++) v.push_back((int)((long)i * (n - 1) / (want - 1)));
}

static Problem make_problem(const std::string &fmt, int N, int K, uint64_t seed, bool cold) {
  Problem P;
  P.fmt = fmt;
  P.f = kref_format(fmt.c_str());
  if (!P.f) {
    fprintf(stderr, "unknown format %s\n", fmt.c_str());
    exit(2);
  }
  P.N = N, P.K = K;
  P.wbytes = (size_t)N * (K / P.f->block) * P.f->nbytes;
  P.W.resize(P.wbytes);
  kref_gen_weights(fmt.c_str(), N, K, seed, P.W.data());
  P.copies = cold ? (int)std::min<size_t>(512, std::max<size_t>(2, (4 * g_dev.l2 + P.wbytes - 1) / P.wbytes)) : 1;
  size_t free_b, tot;
  CK(cudaMemGetInfo(&free_b, &tot));
  while (P.copies > 1 && (size_t)P.copies * P.wbytes > free_b / 4) P.copies--;
  for (int i = 0; i < P.copies; i++) {
    uint8_t *d;
    CK(cudaMalloc(&d, P.wbytes));
    CK(cudaMemcpy(d, P.W.data(), P.wbytes, cudaMemcpyHostToDevice));
    P.dW.push_back(d);
  }
  pick(P.rows, N, 48);
  return P;
}

static void free_problem(Problem &P) {
  for (auto d : P.dW) cudaFree(d);
  P.dW.clear();
}

static Acts make_acts(const Problem &P, int M, uint64_t seed) {
  Acts A;
  A.M = M;
  A.X.resize((size_t)M * P.K);
  kref_gen_acts(P.K, M, seed, A.X.data());
  CK(cudaMalloc(&A.dX, A.X.size() * 4));
  CK(cudaMemcpy(A.dX, A.X.data(), A.X.size() * 4, cudaMemcpyHostToDevice));
  pick(A.cols, M, 4);
  kref_gemm(P.fmt.c_str(), P.W.data(), nullptr, A.X.data(), P.N, P.K, M, P.rows, A.cols, A.ref_model);
  return A;
}

static double relerr_vs(const std::vector<double> &ref, const std::vector<float> &Y, const Problem &P, const Acts &A) {
  double mx = 1e-30, err = 0;
  for (size_t ci = 0; ci < A.cols.size(); ci++)
    for (size_t ri = 0; ri < P.rows.size(); ri++) {
      double r = ref[ci * P.rows.size() + ri], y = Y[(size_t)A.cols[ci] * P.N + P.rows[ri]];
      mx = std::max(mx, fabs(r));
      err = std::max(err, std::isfinite(y) ? fabs(y - r) : INFINITY);
    }
  return err / mx;
}

// ------------------------------------------------------------------ implementations
struct Impl {
  std::string name, config, status = "ok";
  double relerr_exact = NAN, relerr_model = NAN;
  bool graphable = true;
  bool all_copies = false;  // one call() covers every weight copy (ggml: one graph holds all copies)
  virtual ~Impl() {}
  // prepare for (P, A); false = skip (status says why)
  virtual bool setup(Problem &P, Acts &A) = 0;
  // enqueue one call on weight copy i
  virtual void call(int i, cudaStream_t s) = 0;
  // run once and fill relerr_*; called after setup
  virtual void check(Problem &P, Acts &A) = 0;
  virtual void teardown() {}
  virtual double bytes(const Problem &P, const Acts &A) { return (double)P.wbytes + 4.0 * A.M * P.K + 4.0 * A.M * P.N; }
};

struct KurnImpl : Impl {
  void *lib = nullptr;
  kg_config_fn config_fn;
  kg_check_shape_fn check_shape;
  kg_prep_bytes_fn prep_bytes;
  kg_prepare_fn prepare;
  kg_xbytes_fn xbytes;
  kg_quant_fn quant, xblocks;
  kg_xblock_bytes_fn xblock_bytes;
  kg_run_fn run;
  bool include_quant = true;
  Problem *P = nullptr;
  Acts *A = nullptr;
  std::vector<uint8_t *> dP;
  void *dXq = nullptr, *dXb = nullptr;
  float *dY = nullptr;
  KurnImpl(const std::string &nm, const std::string &path, bool iq) {
    name = nm;
    include_quant = iq;
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      status = std::string("dlopen failed: ") + dlerror();
      return;
    }
#define SYM(v, n) v = (decltype(v))dlsym(lib, n)
    SYM(config_fn, "kg_config");
    SYM(check_shape, "kg_check_shape");
    SYM(prep_bytes, "kg_prep_bytes");
    SYM(prepare, "kg_prepare");
    SYM(xbytes, "kg_xbytes");
    SYM(quant, "kg_quant");
    SYM(xblocks, "kg_xblocks");
    SYM(xblock_bytes, "kg_xblock_bytes");
    SYM(run, "kg_run");
#undef SYM
    if (!config_fn || !run || !quant || !xblocks || !xblock_bytes) {
      status = "library does not implement kurn_gpu.h";
      return;
    }
    config = config_fn();
  }
  bool setup(Problem &p, Acts &a) override {
    if (status != "ok" && status.rfind("ok", 0) != 0) return false;
    P = &p, A = &a;
    if (config.find("weights=" + p.fmt + " ") == std::string::npos) {
      status = "kernel is for another format";
      return false;
    }
    if (check_shape(p.N, p.K, a.M)) {
      status = "shape not supported by this kernel";
      return false;
    }
    size_t pb = prep_bytes(p.N, p.K);
    for (int i = 0; i < p.copies && pb; i++) {
      uint8_t *d;
      CK(cudaMalloc(&d, pb));
      if (prepare(p.dW[i], d, p.N, p.K, 0)) {
        status = "kg_prepare failed";
        return false;
      }
      dP.push_back(d);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMalloc(&dXq, xbytes(p.K, a.M)));
    CK(cudaMalloc(&dXb, xblock_bytes(p.K, a.M)));
    CK(cudaMalloc(&dY, (size_t)a.M * p.N * 4));
    if (!include_quant) CK((cudaError_t)quant(a.dX, dXq, p.K, a.M, 0));
    return true;
  }
  void call(int i, cudaStream_t s) override {
    if (include_quant) quant(A->dX, dXq, P->K, A->M, s);
    run(dP.empty() ? (const void *)P->dW[i] : (const void *)dP[i], dXq, dY, P->N, P->K, A->M, s);
  }
  void check(Problem &p, Acts &a) override {
    CK(cudaMemset(dY, 0xFF, (size_t)a.M * p.N * 4));
    if (quant(a.dX, dXq, p.K, a.M, 0) || run(dP.empty() ? (const void *)p.dW[0] : (const void *)dP[0], dXq, dY, p.N, p.K, a.M, 0)) {
      status = "launch failed";
      return;
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)a.M * p.N);
    CK(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> xb(xblock_bytes(p.K, a.M));
    CK((cudaError_t)xblocks(a.dX, dXb, p.K, a.M, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(xb.data(), dXb, xb.size(), cudaMemcpyDeviceToHost));
    std::vector<double> ref;
    kref_gemm(p.fmt.c_str(), p.W.data(), xb.data(), nullptr, p.N, p.K, a.M, p.rows, a.cols, ref);
    relerr_exact = relerr_vs(ref, Y, p, a);
    relerr_model = relerr_vs(a.ref_model, Y, p, a);
    if (!(relerr_exact <= 1e-5)) status = "WRONG (exactness check failed)";
  }
  void teardown() override {
    for (auto d : dP) cudaFree(d);
    dP.clear();
    cudaFree(dXq);
    cudaFree(dXb);
    cudaFree(dY);
    dXq = dXb = nullptr, dY = nullptr;
  }
};

__global__ void k_f32_to_f16(const float *x, __half *y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half_rn(x[i]);
}
__global__ void k_f32_to_i8(const float *x, int8_t *y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = (int8_t)max(-127, min(127, __float2int_rn(x[i] * 32.f)));
}

struct CublasImpl : Impl {
  bool int8;
  cublasHandle_t h = nullptr;
  std::vector<void *> dWc;
  void *dXc = nullptr, *dY = nullptr, *ws = nullptr;
  Problem *P = nullptr;
  Acts *A = nullptr;
  explicit CublasImpl(bool i8) : int8(i8) {
    name = i8 ? "cublas-int8" : "cublas-fp16";
    config = i8 ? "cublasGemmEx int8 x int8 -> int32 (speed reference: no block scales, not exact)"
                : "cublasGemmEx fp16 x fp16 -> fp32, fp32 accumulate (dequantized weights)";
  }
  bool setup(Problem &p, Acts &a) override {
    P = &p, A = &a;
    if (int8 && (a.M % 4 || p.K % 4 || p.N % 4)) {
      status = "int8 GEMM needs M, N, K multiples of 4";
      return false;
    }
    if (!h && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
      status = "cublasCreate failed";
      return false;
    }
    CK(cudaMalloc(&ws, 32 << 20));
    cublasSetWorkspace(h, ws, 32 << 20);
    size_t el = int8 ? 1 : 2, wb = (size_t)p.N * p.K * el;
    int copies = (int)std::min<size_t>(64, std::max<size_t>(2, (4 * g_dev.l2 + wb - 1) / wb));
    if (p.copies == 1) copies = 1;
    std::vector<uint8_t> &host = int8 ? p.h8 : p.h16;
    std::vector<double> row(p.K);
    size_t rb = (size_t)(p.K / p.f->block) * p.f->nbytes;
    if (host.size() != wb) host.resize(wb);
    else rb = 0;  // already converted
    for (int n = 0; n < p.N && rb; n++) {
      kref_dequant_w(p.fmt.c_str(), p.W.data() + n * rb, p.K, row.data());
      for (int k = 0; k < p.K; k++) {
        if (int8) {
          host[(size_t)n * p.K + k] = (uint8_t)(int8_t)std::max(-127.0, std::min(127.0, std::round(row[k] * 64)));
        } else {
          __half hv = __float2half_rn((float)row[k]);
          memcpy(&host[((size_t)n * p.K + k) * 2], &hv, 2);
        }
      }
    }
    for (int i = 0; i < copies; i++) {
      void *d;
      CK(cudaMalloc(&d, wb));
      CK(cudaMemcpy(d, host.data(), wb, cudaMemcpyHostToDevice));
      dWc.push_back(d);
    }
    CK(cudaMalloc(&dXc, (size_t)a.M * p.K * el));
    CK(cudaMalloc(&dY, (size_t)a.M * p.N * 4));
    return true;
  }
  void call(int i, cudaStream_t s) override {
    cublasSetStream(h, s);
    size_t n = (size_t)A->M * P->K;
    if (int8)
      k_f32_to_i8<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(A->dX, (int8_t *)dXc, n);
    else
      k_f32_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(A->dX, (__half *)dXc, n);
    void *w = dWc[i % dWc.size()];
    if (int8) {
      const int alpha = 1, beta = 0;
      cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, P->N, A->M, P->K, &alpha, w, CUDA_R_8I, P->K, dXc, CUDA_R_8I, P->K, &beta, dY,
                   CUDA_R_32I, P->N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    } else {
      const float alpha = 1.f, beta = 0.f;
      cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, P->N, A->M, P->K, &alpha, w, CUDA_R_16F, P->K, dXc, CUDA_R_16F, P->K, &beta, dY,
                   CUDA_R_32F, P->N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
  }
  void check(Problem &p, Acts &a) override {
    call(0, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      status = "cuBLAS call failed";
      return;
    }
    if (int8) return;  // not the same math
    std::vector<float> Y((size_t)a.M * p.N);
    CK(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost));
    relerr_model = relerr_vs(a.ref_model, Y, p, a);
  }
  double bytes(const Problem &p, const Acts &a) override {
    return (double)p.N * p.K * (int8 ? 1 : 2) + 4.0 * a.M * p.K + 4.0 * a.M * p.N;
  }
  void teardown() override {
    for (auto d : dWc) cudaFree(d);
    dWc.clear();
    cudaFree(dXc);
    cudaFree(dY);
    cudaFree(ws);
    dXc = dY = ws = nullptr;
  }
  ~CublasImpl() {
    if (h) cublasDestroy(h);
  }
};

#ifdef KURN_GGML
struct GgmlImpl : Impl {
  ggml_backend_t be = nullptr;
  KgGgml g;
  GgmlImpl() {
    name = "ggml";
    config = "llama.cpp ggml-cuda MUL_MAT (its own dispatch: MMVQ / MMQ / cuBLAS)";
    graphable = false;
    all_copies = true;
  }
  bool setup(Problem &p, Acts &a) override {
    if (!be) be = ggml_backend_cuda_init(g_dev.id);
    if (!be) {
      status = "ggml_backend_cuda_init failed";
      return false;
    }
    bool ok = kg_ggml_setup(g, be, p.fmt.c_str(), p.f->block, p.f->nbytes, p.W.data(), p.wbytes, p.N, p.K, a.X.data(), a.M, p.copies);
    status = g.status;
    return ok;
  }
  void call(int, cudaStream_t) override { kg_ggml_compute(g, be); }
  void check(Problem &p, Acts &a) override {
    if (!kg_ggml_compute(g, be)) {
      status = "graph compute failed";
      return;
    }
    std::vector<float> Y((size_t)a.M * p.N);
    kg_ggml_output(g, Y.data(), Y.size());
    relerr_model = relerr_vs(a.ref_model, Y, p, a);
  }
  void teardown() override { kg_ggml_free(g); }
  ~GgmlImpl() {
    if (be) ggml_backend_free(be);
  }
};
#endif

// ------------------------------------------------------------------ measurement
struct Sample {
  double us = NAN, joules = NAN, watts = NAN;
  long calls = 0;
  unsigned sm = 0, mem = 0, temp = 0;
  unsigned long long throttle = 0;
};

static Sample measure(Impl &im, Problem &P, double secs, cudaStream_t st) {
  Sample s;
  int copies = P.copies;
  cudaGraphExec_t ge = nullptr;
  if (im.graphable) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < copies; i++) im.call(i, st);
    if (cudaStreamEndCapture(st, &g) != cudaSuccess || cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) {
      cudaGetLastError();
      im.graphable = false;
      ge = nullptr;
    }
  }
  auto once = [&]() {
    if (ge)
      CK(cudaGraphLaunch(ge, st));
    else if (im.all_copies)
      im.call(-1, st);
    else
      for (int i = 0; i < copies; i++) im.call(i, st);
  };
  for (int w = 0; w < 3; w++) once();  // warm up
  CK(cudaStreamSynchronize(st));
  double e0 = g_nvml.joules(), t0 = now_s();
  long reps = 0;
  double wsum = 0;
  int wn = 0;
  while (true) {
    once();
    reps++;
    if (reps % 4 == 0) {
      CK(cudaStreamSynchronize(st));
      double w = g_nvml.watts();
      if (!std::isnan(w)) wsum += w, wn++;
      if (now_s() - t0 >= secs) break;
    }
  }
  CK(cudaStreamSynchronize(st));
  double t1 = now_s(), e1 = g_nvml.joules();
  s.calls = reps * copies;
  s.us = (t1 - t0) / s.calls * 1e6;
  s.joules = (e1 - e0) / s.calls;
  s.watts = wn ? wsum / wn : NAN;
  s.sm = g_nvml.clk(1), s.mem = g_nvml.clk(2), s.temp = g_nvml.temperature(), s.throttle = g_nvml.throttle();
  if (ge) cudaGraphExecDestroy(ge);
  return s;
}

static void emit_check(FILE *out, Impl &im, Problem &P, Acts &A) {
  fprintf(out,
          "{\"kind\": \"check\", \"fmt\": %s, \"N\": %d, \"K\": %d, \"M\": %d, \"impl\": %s, \"config\": %s, \"status\": %s, "
          "\"relerr_exact\": %s, \"relerr_model\": %s, \"copies\": %d}\n",
          jstr(P.fmt).c_str(), P.N, P.K, A.M, jstr(im.name).c_str(), jstr(im.config).c_str(), jstr(im.status).c_str(),
          jnum(im.relerr_exact, "%.3e").c_str(), jnum(im.relerr_model, "%.3e").c_str(), P.copies);
  fflush(out);
}

static void emit_sample(FILE *out, Impl &im, Problem &P, Acts &A, int round, const Sample &s) {
  fprintf(out,
          "{\"kind\": \"sample\", \"fmt\": %s, \"N\": %d, \"K\": %d, \"M\": %d, \"impl\": %s, \"round\": %d, \"us\": %.4f, "
          "\"joules\": %s, \"watts\": %s, \"calls\": %ld, \"bytes\": %.0f, \"ops\": %.0f, \"sm_mhz\": %u, \"mem_mhz\": %u, "
          "\"temp_c\": %u, \"throttle\": %llu}\n",
          jstr(P.fmt).c_str(), P.N, P.K, A.M, jstr(im.name).c_str(), round, s.us, jnum(s.joules, "%.6e").c_str(), jnum(s.watts, "%.1f").c_str(), s.calls, im.bytes(P, A),
          2.0 * P.N * P.K * A.M, s.sm, s.mem, s.temp, s.throttle);
  fflush(out);
}

// ------------------------------------------------------------------ plan file
// fmt FORMAT
// kurn NAME LIB MMIN MMAX       (a KURN kernel used for batch sizes MMIN..MMAX; repeatable)
// shapes NxK ...
// batches M ...
// competitors ggml cublas-fp16 cublas-int8
// end
// Global lines: reps R, secs S, cold 0|1, quant 0|1, seed s.
struct KurnEntry {
  std::string name, lib;
  int mmin, mmax;
};
struct FmtPlan {
  std::string fmt;
  std::vector<KurnEntry> kurn;
  std::vector<std::pair<int, int>> shapes;
  std::vector<int> batches;
  std::vector<std::string> competitors;
};

static int cmd_matrix(const char *plan_path, const char *out_path) {
  FILE *pf = fopen(plan_path, "r");
  if (!pf) {
    fprintf(stderr, "cannot open plan %s\n", plan_path);
    return 2;
  }
  FILE *out = out_path ? fopen(out_path, "a") : stdout;
  int reps = 5, cold = 1, quant = 1;
  double secs = 0.5;
  uint64_t seed = 1;
  std::vector<FmtPlan> plans;
  char line[4096];
  FmtPlan cur;
  while (fgets(line, sizeof line, pf)) {
    std::vector<std::string> tok;
    for (char *t = strtok(line, " \t\r\n"); t; t = strtok(nullptr, " \t\r\n")) {
      if (t[0] == '#') break;
      tok.push_back(t);
    }
    if (tok.empty()) continue;
    const std::string &k = tok[0];
    if (k == "reps") reps = atoi(tok[1].c_str());
    else if (k == "secs") secs = atof(tok[1].c_str());
    else if (k == "cold") cold = atoi(tok[1].c_str());
    else if (k == "quant") quant = atoi(tok[1].c_str());
    else if (k == "seed") seed = strtoull(tok[1].c_str(), nullptr, 10);
    else if (k == "fmt") cur = FmtPlan(), cur.fmt = tok[1];
    else if (k == "kurn" && tok.size() == 5) cur.kurn.push_back({tok[1], tok[2], atoi(tok[3].c_str()), atoi(tok[4].c_str())});
    else if (k == "shapes")
      for (size_t i = 1; i < tok.size(); i++) {
        int n, kk;
        if (sscanf(tok[i].c_str(), "%dx%d", &n, &kk) == 2) cur.shapes.push_back({n, kk});
      }
    else if (k == "batches")
      for (size_t i = 1; i < tok.size(); i++) cur.batches.push_back(atoi(tok[i].c_str()));
    else if (k == "competitors")
      for (size_t i = 1; i < tok.size(); i++) cur.competitors.push_back(tok[i]);
    else if (k == "end") plans.push_back(cur);
  }
  fclose(pf);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  for (auto &fp : plans) {
    for (auto &sh : fp.shapes) {
      Problem P = make_problem(fp.fmt, sh.first, sh.second, seed + sh.first * 31 + sh.second, cold != 0);
      for (int M : fp.batches) {
        Acts A = make_acts(P, M, seed * 7 + M);
        std::vector<std::unique_ptr<Impl>> impls;
        for (auto &ke : fp.kurn)
          if (M >= ke.mmin && M <= ke.mmax) impls.emplace_back(new KurnImpl("kurn:" + ke.name, ke.lib, quant != 0));
        for (auto &c : fp.competitors) {
          if (c == "cublas-fp16") impls.emplace_back(new CublasImpl(false));
          else if (c == "cublas-int8") impls.emplace_back(new CublasImpl(true));
#ifdef KURN_GGML
          else if (c == "ggml") impls.emplace_back(new GgmlImpl());
#else
          else if (c == "ggml") {
            fprintf(out, "{\"kind\": \"skip\", \"fmt\": %s, \"N\": %d, \"K\": %d, \"M\": %d, \"impl\": \"ggml\", "
                         "\"reason\": \"harness built without llama.cpp (KURN_GGML)\"}\n", jstr(P.fmt).c_str(), P.N, P.K, M);
          }
#endif
        }
        std::vector<Impl *> live;
        for (auto &im : impls) {
          if (im->setup(P, A)) {
            im->check(P, A);
          }
          emit_check(out, *im, P, A);
          if (im->status.rfind("ok", 0) == 0) live.push_back(im.get());
          else im->teardown();
        }
        for (int r = 0; r < reps; r++)
          for (size_t j = 0; j < live.size(); j++) {
            Impl *im = live[(j + r) % live.size()];
            Sample s = measure(*im, P, secs, st);
            emit_sample(out, *im, P, A, r, s);
          }
        for (auto im : live) im->teardown();
        cudaFree(A.dX);
        fprintf(stderr, "[matrix] %s %dx%d M=%d: %zu implementations\n", fp.fmt.c_str(), P.N, P.K, M, live.size());
      }
      free_problem(P);
    }
  }
  if (out_path) fclose(out);
  return 0;
}

static int cmd_run(int argc, char **argv) {
  std::string lib, fmt;
  int N = 4096, K = 4096, M = 1, reps = 3, quant = 0, cold = 1;
  double secs = 0.3;
  uint64_t seed = 1;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "--lib") lib = v;
    else if (a == "--fmt") fmt = v;
    else if (a == "--N") N = atoi(v.c_str());
    else if (a == "--K") K = atoi(v.c_str());
    else if (a == "--M") M = atoi(v.c_str());
    else if (a == "--reps") reps = atoi(v.c_str());
    else if (a == "--secs") secs = atof(v.c_str());
    else if (a == "--quant") quant = atoi(v.c_str());
    else if (a == "--cold") cold = atoi(v.c_str());
    else if (a == "--seed") seed = strtoull(v.c_str(), nullptr, 10);
  }
  if (lib.empty() || fmt.empty()) {
    fprintf(stderr, "run needs --lib and --fmt\n");
    return 2;
  }
  Problem P = make_problem(fmt, N, K, seed, cold != 0);
  Acts A = make_acts(P, M, seed + 1);
  KurnImpl im("kurn", lib, quant != 0);
  if (im.setup(P, A)) im.check(P, A);
  emit_check(stdout, im, P, A);
  if (im.status != "ok") return 1;
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  for (int r = 0; r < reps; r++) emit_sample(stdout, im, P, A, r, measure(im, P, secs, st));
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: bench_gpu info | roofline [--secs S] | run --lib K.so --fmt F ... | matrix --plan P [--out O]\n");
    return 2;
  }
  dev_init();
  std::string cmd = argv[1];
  if (cmd == "info") return cmd_info();
  if (cmd == "roofline") return cmd_roofline(argc > 3 ? atof(argv[3]) : 3.0);
  if (cmd == "run") return cmd_run(argc, argv);
  if (cmd == "matrix") {
    const char *plan = nullptr, *out = nullptr;
    for (int i = 2; i + 1 < argc; i += 2) {
      if (!strcmp(argv[i], "--plan")) plan = argv[i + 1];
      if (!strcmp(argv[i], "--out")) out = argv[i + 1];
    }
    if (!plan) return 2;
    return cmd_matrix(plan, out);
  }
  fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
