// ggml_mm_check: run kurn_ggml_mm.h on ggml's CPU backend and compare with kurn_gpu_ref.h.
// Tests the ggml comparison path of the GPU harness without a GPU.
//   ggml_mm_check FMT N K M   -> one JSON line {fmt, status, relerr_model}
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ggml-cpu.h"
#include "kurn_ggml_mm.h"
#include "kurn_gpu_ref.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: ggml_mm_check FMT N K M\n");
    return 2;
  }
  const char *fmt = argv[1];
  int N = atoi(argv[2]), K = atoi(argv[3]), M = atoi(argv[4]);
  const kref_fmt *f = kref_format(fmt);
  if (!f) return 2;
  size_t wbytes = (size_t)N * (K / f->block) * f->nbytes;
  std::vector<uint8_t> W(wbytes);
  std::vector<float> X((size_t)M * K), Y((size_t)M * N);
  kref_gen_weights(fmt, N, K, 3, W.data());
  kref_gen_acts(K, M, 4, X.data());
  ggml_backend_t be = ggml_backend_cpu_init();
  KgGgml g;
  double err = NAN;
  if (kg_ggml_setup(g, be, fmt, f->block, f->nbytes, W.data(), wbytes, N, K, X.data(), M, 2) && kg_ggml_compute(g, be)) {
    kg_ggml_output(g, Y.data(), Y.size());
    std::vector<int> rows(N), cols(M);
    for (int i = 0; i < N; i++) rows[i] = i;
    for (int i = 0; i < M; i++) cols[i] = i;
    std::vector<double> ref;
    kref_gemm(fmt, W.data(), nullptr, X.data(), N, K, M, rows, cols, ref);
    double mx = 1e-30, e = 0;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        mx = std::max(mx, fabs(ref[(size_t)m * N + n]));
        e = std::max(e, fabs(ref[(size_t)m * N + n] - Y[(size_t)m * N + n]));
      }
    err = e / mx;
  }
  char e[32] = "null";
  if (std::isfinite(err)) snprintf(e, sizeof e, "%.3e", err);
  printf("{\"fmt\": \"%s\", \"status\": \"%s\", \"relerr_model\": %s}\n", fmt, g.status.c_str(), e);
  kg_ggml_free(g);
  ggml_backend_free(be);
  return 0;
}
