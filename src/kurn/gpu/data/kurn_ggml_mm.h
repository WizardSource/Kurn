// kurn_ggml_mm.h: one ggml MUL_MAT per weight copy on a ggml backend, for comparing KURN against
// llama.cpp's own kernels on identical weight bytes. Used by bench_gpu.cu (CUDA backend) and by
// ggml_mm_check.cpp (CPU backend, to test this code without a GPU).
#pragma once
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

struct KgGgml {
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  ggml_cgraph *gf = nullptr;
  std::vector<ggml_tensor *> outs;
  std::string status = "ok";
};

static void kg_ggml_quiet(enum ggml_log_level, const char *, void *) {}

// ggml type id for a KURN format name, checking that the block layout is the same
static int kg_ggml_type(const char *fmt, int block, int nbytes, std::string &why) {
  for (int t = 0; t < GGML_TYPE_COUNT; t++) {
    const char *n = ggml_type_name((ggml_type)t);
    if (!n || std::string(n) != fmt) continue;
    if ((int)ggml_blck_size((ggml_type)t) != block || (int)ggml_type_size((ggml_type)t) != nbytes) {
      why = "ggml's block layout for this name differs from KURN's";
      return -1;
    }
    return t;
  }
  why = "format not in this ggml build";
  return -1;
}

// Y_i = W_i x X for `copies` copies of W (N rows of K values, ggml blocks) and f32 X [M][K]
static bool kg_ggml_setup(KgGgml &g, ggml_backend_t be, const char *fmt, int block, int nbytes, const void *W, size_t wbytes, int N,
                          int K, const float *X, int M, int copies) {
  ggml_log_set(kg_ggml_quiet, nullptr);
  int t = kg_ggml_type(fmt, block, nbytes, g.status);
  if (t < 0) return false;
  ggml_init_params ip = {ggml_tensor_overhead() * (2 * copies + 8) + ggml_graph_overhead_custom(2 * copies + 8, false), nullptr, true};
  g.ctx = ggml_init(ip);
  std::vector<ggml_tensor *> ws;
  ggml_tensor *b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
  for (int i = 0; i < copies; i++) {
    ggml_tensor *w = ggml_new_tensor_2d(g.ctx, (ggml_type)t, K, N);
    ws.push_back(w);
    g.outs.push_back(ggml_mul_mat(g.ctx, w, b));
  }
  if (!ggml_backend_supports_op(be, g.outs[0])) {
    g.status = "not supported by this ggml backend (llama.cpp would run this MUL_MAT on the CPU)";
    return false;
  }
  g.gf = ggml_new_graph_custom(g.ctx, 2 * copies + 8, false);
  for (auto o : g.outs) ggml_build_forward_expand(g.gf, o);
  g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
  if (!g.buf) {
    g.status = "ggml buffer allocation failed";
    return false;
  }
  for (auto w : ws) ggml_backend_tensor_set(w, W, 0, wbytes);
  ggml_backend_tensor_set(b, X, 0, (size_t)M * K * sizeof(float));
  return true;
}

static bool kg_ggml_compute(KgGgml &g, ggml_backend_t be) { return ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS; }

// output of copy 0: Y[m * N + n] (ggml's mul_mat result is [M][N] row-major, same convention as KURN)
static void kg_ggml_output(KgGgml &g, float *Y, size_t n) { ggml_backend_tensor_get(g.outs[0], Y, 0, n * sizeof(float)); }

static void kg_ggml_free(KgGgml &g) {
  if (g.buf) ggml_backend_buffer_free(g.buf);
  if (g.ctx) ggml_free(g.ctx);
  g = KgGgml();
}
