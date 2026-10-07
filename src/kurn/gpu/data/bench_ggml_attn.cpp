// llama.cpp's CUDA flash attention (ggml_flash_attn_ext on the ggml-cuda backend) on exactly the problems the kurn
// harness runs (bench_gpu_attn.cu): same data and float64 reference (kurn_gpu_attn_ref.h), same cold-KV rotation and
// byte count, so the two lines compare like for like.
//
//   bench_ggml_attn run --kv f16|bf16|q8_0 --dk 128 [--dv D] [--mla 0] --nq 1 --nkv 8192 --heads 32 --kv-heads 8
//                       [--seed 1] [--tol 4e-3] [--check-toks 16] [--secs 0.5] [--reps 3] [--cold-bytes 3e8]
//
// The graph is what llama.cpp builds for decode: K/V as strided views of a token-major cache (row = all kv heads of a
// token), q permuted from [dk, heads, nq], an F16 causal mask, F32 accumulation (ggml_prec_set_acc), scale 1/sqrt(dk).
// MLA: V is a view of the first dv values of K, as llama.cpp's DeepSeek path. One graph holds one call per layer and
// is replayed (ggml-cuda captures it as a CUDA graph); per-call time = wall time of a batch of replays / calls.
// Output: JSON lines {"kind": "check" | "sample" | "error", "impl": "ggml-cuda", ...}.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "kurn_gpu_attn.h"
#include "kurn_gpu_attn_ref.h"

static void fail(const char *msg) {
  printf("{\"kind\": \"error\", \"impl\": \"ggml-cuda\", \"error\": \"%s\"}\n", msg);
  exit(1);
}
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const char *arg_s(int argc, char **argv, const char *k, const char *def) {
  for (int i = 2; i < argc - 1; i++)
    if (!strcmp(argv[i], k)) return argv[i + 1];
  return def;
}
static double arg_f(int argc, char **argv, const char *k, double def) {
  const char *s = arg_s(argc, argv, k, nullptr);
  return s ? atof(s) : def;
}

int main(int argc, char **argv) {
  if (argc < 2 || strcmp(argv[1], "run")) {
    fprintf(stderr, "usage: bench_ggml_attn run --kv f16|bf16|q8_0 --dk D [--dv D] [--mla 0|1] --nq --nkv --heads --kv-heads ...\n");
    return 2;
  }
  const std::string kv = arg_s(argc, argv, "--kv", "f16");
  kgar_problem p;
  p.kv = kv == "f16" ? KGA_KV_F16 : kv == "bf16" ? KGA_KV_BF16 : kv == "q8_0" ? KGA_KV_Q8_0 : -1;
  if (p.kv < 0) fail("--kv must be f16, bf16 or q8_0");
  const ggml_type T = kv == "f16" ? GGML_TYPE_F16 : kv == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_Q8_0;
  p.dk = (int)arg_f(argc, argv, "--dk", 128);
  p.mla = (int)arg_f(argc, argv, "--mla", p.dk == 576);
  p.dv = (int)arg_f(argc, argv, "--dv", p.dk == 576 ? 512 : p.dk);
  p.nq = (int64_t)arg_f(argc, argv, "--nq", 1);
  p.nkv = (int64_t)arg_f(argc, argv, "--nkv", 4096);
  p.nh = (int)arg_f(argc, argv, "--heads", 32);
  p.nhkv = p.mla ? 1 : (int)arg_f(argc, argv, "--kv-heads", 8);
  p.causal = 1;
  p.pos0 = -1;
  p.use_mask = 0;
  p.layout = 0;
  const uint64_t seed = (uint64_t)arg_f(argc, argv, "--seed", 1);
  const double tol = arg_f(argc, argv, "--tol", 4e-3), secs = arg_f(argc, argv, "--secs", 0.5), cold_bytes = arg_f(argc, argv, "--cold-bytes", 3e8);
  const int reps = (int)arg_f(argc, argv, "--reps", 3);
  const int64_t check_toks = (int64_t)arg_f(argc, argv, "--check-toks", 16);
  kgar_make(p, seed);

  ggml_backend_t be = ggml_backend_cuda_init(0);
  if (!be) fail("ggml_backend_cuda_init failed");
  char desc[256];
  ggml_backend_cuda_get_device_description(0, desc, sizeof desc);
  for (char *c = desc; *c; c++)
    if (*c == '"') *c = '\'';

  const size_t kv_bytes = p.k.size() + p.v.size();
  int nl = 1;
  if (cold_bytes > 0) nl = (int)std::min(64.0, std::max(2.0, std::ceil(cold_bytes / (double)kv_bytes)));

  // tensors: q, mask, and per layer a token-major K (and V) cache
  ggml_init_params ip = {ggml_tensor_overhead() * (size_t)(8 + 4 * nl), nullptr, true};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, p.dk, p.nh, p.nq);
  ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, p.nkv, p.nq);
  std::vector<ggml_tensor *> kc(nl), vc(nl);
  for (int l = 0; l < nl; l++) {
    kc[l] = ggml_new_tensor_2d(ctx, T, (int64_t)p.dk * p.nhkv, p.nkv);
    vc[l] = p.mla ? kc[l] : ggml_new_tensor_2d(ctx, T, (int64_t)p.dv * p.nhkv, p.nkv);
  }
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  if (!buf) fail("out of GPU memory for the KV layers");
  if (ggml_nbytes(kc[0]) != p.k.size() || (!p.mla && ggml_nbytes(vc[0]) != p.v.size())) fail("KV row size mismatch between kurn and ggml");
  ggml_backend_tensor_set(q, p.q.data(), 0, ggml_nbytes(q));
  std::vector<uint16_t> mk((size_t)p.nkv * p.nq);
  for (int64_t t = 0; t < p.nq; t++)
    for (int64_t j = 0; j < p.nkv; j++) mk[t * p.nkv + j] = j <= p.pos0 + t ? 0 : 0xFC00;
  ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2);
  for (int l = 0; l < nl; l++) {
    ggml_backend_tensor_set(kc[l], p.k.data(), 0, p.k.size());
    if (!p.mla) ggml_backend_tensor_set(vc[l], p.v.data(), 0, p.v.size());
  }

  // graph: one flash-attention call per layer
  ggml_init_params gp = {ggml_tensor_overhead() * (size_t)(16 + 8 * nl) + ggml_graph_overhead(), nullptr, true};
  ggml_context *gctx = ggml_init(gp);
  ggml_cgraph *gf = ggml_new_graph(gctx);
  ggml_tensor *qp = ggml_permute(gctx, q, 0, 2, 1, 3);
  std::vector<ggml_tensor *> outs(nl);
  for (int l = 0; l < nl; l++) {
    ggml_tensor *k = ggml_view_3d(gctx, kc[l], p.dk, p.nkv, p.nhkv, kc[l]->nb[1], ggml_row_size(T, p.dk), 0);
    ggml_tensor *v = ggml_view_3d(gctx, vc[l], p.dv, p.nkv, p.nhkv, vc[l]->nb[1], ggml_row_size(T, p.mla ? p.dk : p.dv), 0);
    ggml_tensor *o = ggml_flash_attn_ext(gctx, qp, k, v, mask, (float)(1.0 / sqrt((double)p.dk)), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    if (l == 0 && !ggml_backend_supports_op(be, o)) {
      char b[160];
      snprintf(b, sizeof b, "ggml-cuda does not support flash attention with %s KV, dk %d, dv %d in this build", kv.c_str(), p.dk, p.dv);
      fail(b);
    }
    outs[l] = o;
    ggml_build_forward_expand(gf, o);
  }
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  if (!ggml_gallocr_alloc_graph(ga, gf)) fail("graph allocation failed");
  if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) fail("graph compute failed");

  const size_t on = (size_t)p.nq * p.nh * p.dv;
  std::vector<float> out(on);
  ggml_backend_tensor_get(outs[0], out.data(), 0, on * 4);
  const std::vector<int64_t> toks = kgar_check_tokens(p.nq, check_toks, seed);
  std::vector<double> ref;
  kgar_reference(p, toks, ref);
  const double err = kgar_relerr(p, toks, ref, out.data());
  const bool ok = err <= tol;
  printf("{\"kind\": \"check\", \"impl\": \"ggml-cuda\", \"device\": \"%s\", \"kv\": \"%s\", \"dk\": %d, \"dv\": %d, \"mla\": %d, "
         "\"relerr\": %.4e, \"tol\": %.1e, \"status\": \"%s\", \"layers\": %d}\n",
         desc, kv.c_str(), p.dk, p.dv, p.mla, err, tol, ok ? "ok" : "FAIL", nl);
  fflush(stdout);
  if (secs <= 0) return ok ? 0 : 1;

  for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
  const double kv_seen = (double)std::max<int64_t>(0, std::min<int64_t>(p.nkv, p.pos0 + p.nq));
  double pairs = 0;
  for (int64_t t = 0; t < p.nq; t++) pairs += (double)std::max<int64_t>(0, std::min<int64_t>(p.nkv, p.pos0 + t + 1));
  const double flop = 2.0 * pairs * (p.dk + p.dv) * p.nh;
  const double bytes = kv_seen * p.nhkv * (p.rk + (p.mla ? 0 : p.rv)) + 4.0 * (p.q.size() + on);
  for (int r = 0; r < reps; r++) {
    long runs = 0;
    const double t0 = now_s();
    do {
      for (int i = 0; i < 8; i++) ggml_backend_graph_compute_async(be, gf);
      ggml_backend_synchronize(be);
      runs += 8;
    } while (now_s() - t0 < secs);
    const double calls = (double)runs * nl, us = (now_s() - t0) * 1e6 / calls;
    printf("{\"kind\": \"sample\", \"impl\": \"ggml-cuda\", \"round\": %d, \"us\": %.4f, \"GBps\": %.2f, \"TFLOPs\": %.4f, \"uJ\": NaN, "
           "\"calls\": %.0f, \"layers\": %d}\n",
           r, us, bytes / (us * 1e3), flop / (us * 1e6), calls, nl);
    fflush(stdout);
  }
  ggml_gallocr_free(ga);
  ggml_free(gctx);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(be);
  return ok ? 0 : 1;
}
