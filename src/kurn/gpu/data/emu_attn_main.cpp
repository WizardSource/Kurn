// Runs one generated attention kernel on the CPU emulator against the exact reference:
//   emu_attn NQ NKV HEADS KV_HEADS CAUSAL POS0 MASK LAYOUT SEED
// (POS0 == -1: nkv - nq; below -1: literal, so the first rows see no key). Prints one JSON line: relerr, splits, shape. Every input buffer ends at a PROT_NONE guard
// page, and the output starts as NaN, so out-of-bounds reads crash and unwritten outputs fail.
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kurn_gpu_attn.h"
#include "kurn_gpu_attn_ref.h"

static void *guarded(size_t n) {
  size_t pg = (size_t)sysconf(_SC_PAGESIZE), r = (n + 15) / 16 * 16, tot = (r + pg - 1) / pg * pg + pg;
  char *m = (char *)mmap(nullptr, tot, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) { perror("mmap"); exit(1); }
  mprotect(m + tot - pg, pg, PROT_NONE);
  char *p = m + tot - pg - r;
  memset(p, 0xA5, r);
  return p;
}

template <class T>
static T *copy_guarded(const std::vector<T> &v) {
  T *p = (T *)guarded(v.size() * sizeof(T) + 16);
  if (!v.empty()) memcpy(p, v.data(), v.size() * sizeof(T));
  return p;
}

int main(int argc, char **argv) {
  if (argc != 10) { fprintf(stderr, "usage: emu_attn NQ NKV HEADS KV_HEADS CAUSAL POS0 MASK LAYOUT SEED\n"); return 2; }
  kgar_problem p;
  kga_config(&p.dk, &p.dv, &p.kv);
  p.nq = atoll(argv[1]);
  p.nkv = atoll(argv[2]);
  p.nh = atoi(argv[3]);
  p.nhkv = atoi(argv[4]);
  p.causal = atoi(argv[5]);
  p.pos0 = atoll(argv[6]);
  p.use_mask = atoi(argv[7]);
  p.layout = atoi(argv[8]);
  p.mla = KGA_MLA_BUILD;
  kgar_make(p, strtoull(argv[9], nullptr, 10));
  float *q = copy_guarded(p.q);
  uint8_t *k = copy_guarded(p.k);
  uint8_t *v = p.mla ? k : copy_guarded(p.v);
  uint16_t *mask = p.use_mask ? copy_guarded(p.mask) : nullptr;
  const size_t on = (size_t)p.nq * p.nh * p.dv;
  float *out = (float *)guarded(on * 4);
  memset(out, 0xFF, on * 4);
  kga_args a = kgar_args(p, q, k, v, mask, out);
  if (int e = kga_check(&a)) { printf("{\"error\": \"kga_check %d\"}\n", e); return 3; }
  const size_t wsb = kga_workspace(&a);
  void *ws = wsb ? guarded(wsb) : nullptr;
  // KGA_RUNS=n: n calls on the same workspace, the last one checked (state a call leaves behind breaks the next)
  const int runs = getenv("KGA_RUNS") ? atoi(getenv("KGA_RUNS")) : 1;
  for (int r = 0; r < runs; r++) {
    memset(out, 0xFF, on * 4);
    if (kga_run(&a, ws, nullptr)) { printf("{\"error\": \"kga_run failed\"}\n"); return 4; }
  }
  std::vector<int64_t> toks = kgar_check_tokens(p.nq, p.nq, 1);
  std::vector<double> ref;
  kgar_reference(p, toks, ref);
  const double err = kgar_relerr(p, toks, ref, out);
  char q8[64] = "";
  if (p.kv == KGA_KV_FP8) {  // also against the reference with q rounded to e4m3 as the kernel does
    std::vector<double> rq;
    kgar_reference(p, toks, rq, strstr(kga_config(nullptr, nullptr, nullptr), "qsplit=1") ? 2 : 1);
    snprintf(q8, sizeof q8, ", \"relerr_q8\": %.6e", kgar_relerr(p, toks, rq, out));
  }
  // KGA_CPU_LIB=path/to/kattn.so: run the CPU attention op (kurn_attn.h; kattn_args has kga_args's layout) on the same
  // arguments and report max |gpu - cpu| / max |cpu|
  char cpu[96] = "";
  if (const char *lib = getenv("KGA_CPU_LIB")) {
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    auto wsf = h ? (size_t(*)(const kga_args *, int))dlsym(h, "kattn_workspace") : nullptr;
    auto run = h ? (void (*)(const kga_args *, void *, int, int))dlsym(h, "kattn") : nullptr;
    if (!wsf || !run) { printf("{\"error\": \"KGA_CPU_LIB: %s\"}\n", h ? "missing kattn symbols" : dlerror()); return 5; }
    std::vector<float> co(on, NAN);
    kga_args ca = a;
    ca.out = co.data();
    std::vector<uint8_t> cws(wsf(&ca, 1) + 64, 0);
    run(&ca, cws.data(), 0, 1);
    double me = 0, mr = 0;
    for (size_t i = 0; i < on; i++) {
      me = fmax(me, fabs((double)out[i] - co[i]));
      mr = fmax(mr, fabs((double)co[i]));
      if (co[i] != co[i]) me = INFINITY;
    }
    snprintf(cpu, sizeof cpu, ", \"vs_cpu\": %.6e, \"cpu_relerr\": %.6e", me / (mr > 0 ? mr : 1), kgar_relerr(p, toks, ref, co.data()));
  }
  printf("{\"relerr\": %.6e, \"splits\": %d, \"nq\": %lld, \"nkv\": %lld, \"heads\": %d, \"kv_heads\": %d%s%s}\n", err, kga_splits(&a),
         (long long)p.nq, (long long)p.nkv, p.nh, p.nhkv, cpu, q8);
  return 0;
}
