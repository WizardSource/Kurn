// Runs one generated kernel on the CPU emulator: emu_run DIR
//   DIR/params.txt  "N K M FORMAT"
//   DIR/W.bin       weights, ggml blocks (N rows)
//   DIR/X.bin       activations, f32 [M][K]
// writes DIR/Xb.bin (ggml activation blocks from kg_xblocks), DIR/Y.bin (f32 [M][N]) and DIR/R.bin
// (f64 [M][N], the exact reference from kurn_gpu_ref.h on W and Xb). epi=swiglu: Y and R are [M][N/2].
// epi=fv: the fingerprint is set up before the first run; two more clean runs and one after corrupting the first
// prepared 16 x KT weight tile write DIR/FV.txt "blocks slots checks fails worst checks' fails'" (cumulative after the
// clean runs, with the worst passing |C - X| / tau, then after the corrupted one).
// Buffers end at a PROT_NONE guard page, so reads past the end crash instead of passing.
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kurn_gpu.h"
#include "kurn_gpu_ref.h"

static void *guarded(size_t n) {
  size_t pg = (size_t)sysconf(_SC_PAGESIZE), r = (n + 15) / 16 * 16, tot = (r + pg - 1) / pg * pg + pg;
  char *m = (char *)mmap(nullptr, tot, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) { perror("mmap"); exit(1); }
  mprotect(m + tot - pg, pg, PROT_NONE);
  char *p = m + tot - pg - r;
  memset(p, 0xA5, r);
  return p;
}

static std::vector<char> slurp(const std::string &p) {
  FILE *f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  std::vector<char> b;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  fclose(f);
  return b;
}

static void dump(const std::string &p, const void *d, size_t n) {
  FILE *f = fopen(p.c_str(), "wb");
  fwrite(d, 1, n, f);
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: emu_run DIR\n"); return 2; }
  std::string dir = argv[1];
  int N, K, M;
  char fmt[32];
  FILE *pf = fopen((dir + "/params.txt").c_str(), "r");
  if (!pf || fscanf(pf, "%d %d %d %31s", &N, &K, &M, fmt) != 4) { fprintf(stderr, "bad params.txt\n"); return 2; }
  fclose(pf);
  if (kg_check_shape(N, K, M)) { fprintf(stderr, "shape N=%d K=%d M=%d not supported by %s\n", N, K, M, kg_config()); return 3; }
  std::vector<char> w = slurp(dir + "/W.bin"), x = slurp(dir + "/X.bin");
  void *W = guarded(w.size());
  memcpy(W, w.data(), w.size());
  float *X = (float *)guarded(x.size());
  memcpy(X, x.data(), x.size());
  const void *Wr = W;
  void *P = nullptr;
  size_t pb = kg_prep_bytes(N, K);
  if (pb) {
    P = guarded(pb);
    if (kg_prepare(W, P, N, K, nullptr)) { fprintf(stderr, "kg_prepare failed\n"); return 4; }
    Wr = P;
  }
  const int epi = kg_epi(), nout = epi == 1 ? N / 2 : N;
  const int fvb = kg_fv_block();
  if (epi == 2) {
    std::vector<float> F;
    std::vector<uint16_t> G2;
    std::vector<uint32_t> sg;
    kref_fv_fingerprint(fmt, (const uint8_t *)W, N, K, fvb, 0x5EC12E7ULL, F, G2, sg);
    if (!P || kg_fv_setup(P, N, K, F.data(), G2.data(), sg.data(), 0x9E3779B9u, 8.f, nullptr)) {
      fprintf(stderr, "kg_fv_setup failed\n");
      return 4;
    }
  }
  const int act = kg_act();
  size_t xb = kg_xbytes(K, M);
  void *Xq = act == 1 ? (void *)X : guarded(xb ? xb : 16);
  if (act != 1 && kg_quant(X, Xq, K, M, nullptr)) { fprintf(stderr, "kg_quant failed\n"); return 4; }
  float *Y = (float *)guarded((size_t)M * nout * 4);
  memset(Y, 0xFF, (size_t)M * nout * 4);  // NaN: every output must be written
  if (kg_run(Wr, Xq, Y, N, K, M, nullptr)) { fprintf(stderr, "kg_run failed\n"); return 4; }
  dump(dir + "/Y.bin", Y, (size_t)M * nout * 4);
  if (epi == 2) {
    unsigned st[3], st2[3];
    std::vector<float> y2((size_t)M * N);
    if (kg_run(Wr, Xq, y2.data(), N, K, M, nullptr) || kg_fv_status(P, N, K, st, nullptr)) return 4;
    for (int i = 0; i < 512; i++) ((uint8_t *)P)[i] ^= 0x88;
    if (kg_run(Wr, Xq, y2.data(), N, K, M, nullptr) || kg_fv_status(P, N, K, st2, nullptr)) return 4;
    FILE *fv = fopen((dir + "/FV.txt").c_str(), "w");
    float worst;
    memcpy(&worst, &st[2], 4);
    fprintf(fv, "%d %d %u %u %.6g %u %u\n", (N + fvb - 1) / fvb, M < 16 ? M : 16, st[0], st[1], worst, st2[0], st2[1]);
    fclose(fv);
  }
  std::vector<int> rows(N), cols(M);
  for (int i = 0; i < N; i++) rows[i] = i;
  for (int i = 0; i < M; i++) cols[i] = i;
  std::vector<double> ref;
  if (act == 0) {
    size_t blk = kg_xblock_bytes(K, M);
    void *Xb = guarded(blk);
    if (kg_xblocks(X, Xb, K, M, nullptr)) { fprintf(stderr, "kg_xblocks failed\n"); return 4; }
    dump(dir + "/Xb.bin", Xb, blk);
    kref_gemm(fmt, (const uint8_t *)W, (const uint8_t *)Xb, nullptr, N, K, M, rows, cols, ref);
  } else {  // tensor-core engine: exact weights x f16-rounded activations
    std::vector<float> xh((size_t)K * M);
    for (size_t i = 0; i < xh.size(); i++) xh[i] = kref_round_f16(X[i]);
    dump(dir + "/Xb.bin", xh.data(), 0);
    kref_gemm(fmt, (const uint8_t *)W, nullptr, xh.data(), N, K, M, rows, cols, ref);
  }
  if (epi == 1) {
    std::vector<double> sw;
    kref_swiglu(ref, N, M, sw);
    ref.swap(sw);
  }
  dump(dir + "/R.bin", ref.data(), ref.size() * sizeof(double));
  return 0;
}
