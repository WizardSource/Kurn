// kurn GPU attention harness: correctness of a generated kga_* library against the float64 reference, and timing.
//
//   bench_gpu_attn run --lib attn.so --nq 1 --nkv 8192 --heads 32 --kv-heads 8 [--causal 1] [--pos0 -1] [--mask 0]
//                      [--layout 0] [--seed 1] [--tol 4e-3] [--tol-q8 4e-3] [--check-toks 16] [--secs 0.5] [--reps 3] [--cold-bytes 3e8]
//
// Data and reference as kurn_gpu_attn_ref.h (the CPU attention harness's distributions). Timing:
// - cold regime: the KV cache is replicated into enough "layers" to exceed --cold-bytes (default 300 MB, several
//   times the A100's 40 MB and the RTX 5090's 96 MB L2), and successive calls rotate through them, as a decode step
//   walks the layers. --cold-bytes 0 times one layer (hot, L2-resident when it fits).
// - one CUDA graph holds one call per layer (attention plus merge launches), so launch overhead is excluded like
//   in a graph-captured decode step; per-call time = graph time / layers.
// - NVML board energy per call when the driver exposes the energy counter.
// Output: JSON lines {"kind": "check" | "sample" | "error", ...}.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kurn_gpu_attn.h"
#include "kurn_gpu_attn_ref.h"

static void fail(const char *msg) {
  printf("{\"kind\": \"error\", \"error\": \"%s\"}\n", msg);
  exit(1);
}
#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      char b_[256];                                                              \
      snprintf(b_, sizeof b_, "%s: %s (line %d)", #x, cudaGetErrorString(e_), __LINE__); \
      fail(b_);                                                                  \
    }                                                                            \
  } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Nvml {  // dlopen'd; no header or link dependency
  void *h = nullptr, *dev = nullptr;
  int (*energy)(void *, unsigned long long *) = nullptr;
  int (*clock)(void *, int, unsigned *) = nullptr;
  bool ok = false;
  void open(int cuda_dev) {
    if (!(h = dlopen("libnvidia-ml.so.1", RTLD_NOW))) return;
    auto init = (int (*)())dlsym(h, "nvmlInit_v2");
    auto by_pci = (int (*)(const char *, void **))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    energy = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
    clock = (int (*)(void *, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
    char pci[64];
    unsigned long long e;
    ok = init && by_pci && energy && init() == 0 && cudaDeviceGetPCIBusId(pci, sizeof pci, cuda_dev) == cudaSuccess &&
         by_pci(pci, &dev) == 0 && energy(dev, &e) == 0;
  }
  double joules() {
    unsigned long long e;
    return ok && energy(dev, &e) == 0 ? e * 1e-3 : NAN;
  }
  unsigned clk(int which) {
    unsigned v = 0;
    if (ok && clock) clock(dev, which, &v);
    return v;
  }
};

static const char *arg_s(int argc, char **argv, const char *k, const char *def) {
  for (int i = 2; i < argc - 1; i++)
    if (!strcmp(argv[i], k)) return argv[i + 1];
  return def;
}
static double arg_f(int argc, char **argv, const char *k, double def) {
  const char *s = arg_s(argc, argv, k, nullptr);
  return s ? atof(s) : def;
}

template <class T>
static T *to_device(const std::vector<T> &v) {
  T *d = nullptr;
  CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
  if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static int cmd_run(int argc, char **argv) {
  const char *lib = arg_s(argc, argv, "--lib", nullptr);
  if (!lib) fail("--lib is required");
  void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  if (!h) fail(dlerror());
  auto cfg = (kga_config_fn)dlsym(h, "kga_config");
  auto check = (kga_check_fn)dlsym(h, "kga_check");
  auto splits = (kga_splits_fn)dlsym(h, "kga_splits");
  auto wsf = (kga_workspace_fn)dlsym(h, "kga_workspace");
  auto run = (kga_run_fn)dlsym(h, "kga_run");
  if (!cfg || !check || !splits || !wsf || !run) fail("library lacks kga_* symbols");

  kgar_problem p;
  const std::string config = cfg(&p.dk, &p.dv, &p.kv);
  p.mla = config.find("mla=1") != std::string::npos;
  p.nq = (int64_t)arg_f(argc, argv, "--nq", 1);
  p.nkv = (int64_t)arg_f(argc, argv, "--nkv", 4096);
  p.nh = (int)arg_f(argc, argv, "--heads", 32);
  p.nhkv = (int)arg_f(argc, argv, "--kv-heads", 8);
  p.causal = (int)arg_f(argc, argv, "--causal", 1);
  p.pos0 = (int64_t)arg_f(argc, argv, "--pos0", -1);
  p.use_mask = (int)arg_f(argc, argv, "--mask", 0);
  p.layout = (int)arg_f(argc, argv, "--layout", 0);
  const uint64_t seed = (uint64_t)arg_f(argc, argv, "--seed", 1);
  const double tol = arg_f(argc, argv, "--tol", 4e-3), tol_q8 = arg_f(argc, argv, "--tol-q8", 4e-3), secs = arg_f(argc, argv, "--secs", 0.5);
  const int reps = (int)arg_f(argc, argv, "--reps", 3);
  const double cold_bytes = arg_f(argc, argv, "--cold-bytes", 3e8);
  const int64_t check_toks = (int64_t)arg_f(argc, argv, "--check-toks", 16);
  if (p.mla) p.nhkv = 1;
  kgar_make(p, seed);

  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  Nvml nvml;
  nvml.open(dev);

  const size_t kv_bytes = p.k.size() + p.v.size();
  int nl = 1;
  if (cold_bytes > 0) nl = (int)std::min(64.0, std::max(2.0, std::ceil(cold_bytes / (double)kv_bytes)));
  float *dq = to_device(p.q);
  uint16_t *dmask = p.use_mask ? to_device(p.mask) : nullptr;
  float *dout = nullptr;
  const size_t on = (size_t)p.nq * p.nh * p.dv;
  CK(cudaMalloc(&dout, on * 4));
  std::vector<uint8_t *> dk(nl), dv(nl);
  for (int l = 0; l < nl; l++) {
    dk[l] = to_device(p.k);
    dv[l] = p.mla ? dk[l] : to_device(p.v);
  }
  std::vector<kga_args> args(nl);
  for (int l = 0; l < nl; l++) args[l] = kgar_args(p, dq, dk[l], dv[l], dmask, dout);
  if (int e = check(&args[0])) {
    char b[96];
    snprintf(b, sizeof b, "kga_check rejected the problem (%d)", e);
    fail(b);
  }
  void *ws = nullptr;
  const size_t wsb = wsf(&args[0]);
  if (wsb) CK(cudaMalloc(&ws, wsb));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  // correctness on layer 0 (output starts as NaN: every element must be written)
  CK(cudaMemset(dout, 0xFF, on * 4));
  if (int e = run(&args[0], ws, st)) {
    char b[160];
    snprintf(b, sizeof b, "kga_run failed (%d)%s", e, e == -4 ? ": tile exceeds this GPU's shared memory per block" : "");
    fail(b);
  }
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  std::vector<float> out(on);
  CK(cudaMemcpy(out.data(), dout, on * 4, cudaMemcpyDeviceToHost));
  const std::vector<int64_t> toks = kgar_check_tokens(p.nq, check_toks, seed);
  std::vector<double> ref;
  kgar_reference(p, toks, ref);
  double err = kgar_relerr(p, toks, ref, out.data()), err_q8 = 0.0;
  if (p.kv == KGA_KV_FP8) {  // FP8: also against the reference with q rounded to e4m3 as the kernel does
    std::vector<double> rq;
    kgar_reference(p, toks, rq, config.find("qsplit=1") != std::string::npos ? 2 : 1);
    err_q8 = kgar_relerr(p, toks, rq, out.data());
  }
  const bool ok = err <= tol && err_q8 <= tol_q8;
  printf("{\"kind\": \"check\", \"config\": \"%s\", \"device\": \"%s\", \"sm\": %d, \"sms\": %d, \"relerr\": %.4e, \"tol\": %.1e, "
         "\"relerr_q8\": %.4e, \"status\": \"%s\", \"splits\": %d, \"workspace\": %zu}\n",
         config.c_str(), prop.name, prop.major * 10 + prop.minor, prop.multiProcessorCount, err, tol, err_q8, ok ? "ok" : "FAIL",
         splits(&args[0]), wsb);
  fflush(stdout);
  if (secs <= 0) return ok ? 0 : 1;

  // timing: one graph = one call per layer
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int l = 0; l < nl; l++)
    if (run(&args[l], ws, st)) fail("kga_run failed during capture");
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));

  double pairs = 0;  // (query, key) pairs per head
  for (int64_t t = 0; t < p.nq; t++) {
    int64_t n = p.nkv;
    if (p.causal || p.use_mask) n = std::max<int64_t>(0, std::min<int64_t>(p.nkv, p.pos0 + t + 1));
    pairs += (double)n;
  }
  const double kv_seen = (p.causal || p.use_mask) ? (double)std::max<int64_t>(0, std::min<int64_t>(p.nkv, p.pos0 + p.nq)) : (double)p.nkv;
  const double flop = 2.0 * pairs * (p.dk + p.dv) * p.nh;
  const double bytes = kv_seen * p.nhkv * (p.rk + (p.mla ? 0 : p.rv)) + 4.0 * (p.q.size() + on);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; r++) {
    long launches = 0;
    const double j0 = nvml.joules(), t0 = now_s();
    CK(cudaEventRecord(e0, st));
    do {
      for (int i = 0; i < 8; i++) CK(cudaGraphLaunch(ge, st));
      launches += 8;
      CK(cudaStreamSynchronize(st));
    } while (now_s() - t0 < secs);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double j1 = nvml.joules(), calls = (double)launches * nl, us = ms * 1e3 / calls;
    printf("{\"kind\": \"sample\", \"round\": %d, \"us\": %.4f, \"GBps\": %.2f, \"TFLOPs\": %.4f, \"uJ\": %.3f, \"calls\": %.0f, "
           "\"layers\": %d, \"kv_bytes_per_layer\": %zu, \"sm_mhz\": %u, \"mem_mhz\": %u}\n",
           r, us, bytes / (us * 1e3), flop / (us * 1e6), (j1 - j0) / calls * 1e6, calls, nl, kv_bytes, nvml.clk(1), nvml.clk(2));
    fflush(stdout);
  }
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 2 || strcmp(argv[1], "run")) {
    fprintf(stderr, "usage: bench_gpu_attn run --lib LIB [--nq --nkv --heads --kv-heads --causal --pos0 --mask --layout ...]\n");
    return 2;
  }
  return cmd_run(argc, argv);
}
