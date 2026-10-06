// kurn_gpu_attn_ref.h: host-side test problems and the exact reference for the GPU attention kernels (C++, no CUDA).
//
// Same data and reference semantics as the CPU attention harness (kurn/data/bench_attn.c): q ~ N(0, 3^2), k and v
// ~ N(0, 1) encoded in the KV format, an optional ggml-style fp16 causal mask with padded rows, and a double
// precision softmax(q k^T * scale + mask) v on exactly the K/V values the kernel reads (dequantized). Rows that see
// no key are 0. Layouts: 0 = token-major [token][kv head][d] (ggml views, the CPU harness), 1 = head-major
// [kv head][token][d] (kurn's engine.c cache).
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "kurn_gpu_attn.h"
#include "kurn_gpu_ref.h"

static inline uint16_t kgar_f2h(float f) {  // f32 -> f16 bits, round to nearest even (finite test data)
  const float r = kref_round_f16(f);
  uint32_t u;
  memcpy(&u, &r, 4);
  const uint16_t s = (uint16_t)((u >> 16) & 0x8000);
  const float a = fabsf(r);
  if (a == 0.f) return s;
  if (a < 6.103515625e-05f) return (uint16_t)(s | (uint16_t)lrintf(a * 16777216.f));  // subnormal: m * 2^-24
  int e;
  const float m = frexpf(a, &e);  // a = m 2^e, m in [0.5, 1)
  return (uint16_t)(s | ((e + 14) << 10) | ((uint16_t)lrintf(m * 2048.f) & 0x3FF));
}
static inline uint16_t kgar_f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static inline double kgar_bf2d(uint16_t h) {
  const uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

struct kgar_problem {
  int dk = 128, dv = 128, kv = KGA_KV_F16;
  int64_t nq = 1, nkv = 1024, pos0 = -1;
  int nh = 16, nhkv = 8, causal = 1, use_mask = 0, mla = 0, layout = 0;
  int64_t rk = 0, rv = 0, mask_s = 0;
  int64_t k_s_tok = 0, k_s_head = 0, v_s_tok = 0, v_s_head = 0;
  std::vector<float> q;
  std::vector<uint8_t> k, v;  // v empty when mla (v aliases k)
  std::vector<uint16_t> mask;
};

static inline int64_t kgar_row_bytes(int kv, int d) { return kv == KGA_KV_Q8_0 ? d / 32 * 34 : 2 * (int64_t)d; }

static inline void kgar_encode(int kv, const float *x, int d, uint8_t *dst) {
  if (kv == KGA_KV_F16) {
    for (int i = 0; i < d; i++) {
      const uint16_t h = kgar_f2h(x[i]);
      memcpy(dst + 2 * i, &h, 2);
    }
  } else if (kv == KGA_KV_BF16) {
    for (int i = 0; i < d; i++) {
      const uint16_t h = kgar_f2bf(x[i]);
      memcpy(dst + 2 * i, &h, 2);
    }
  } else {
    for (int b = 0; b < d / 32; b++) {
      float amax = 0;
      for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[32 * b + i]));
      const float dd = amax / 127.0f, id = dd ? 1.0f / dd : 0.0f;
      uint8_t *blk = dst + 34 * b;
      const uint16_t h = kgar_f2h(dd);
      memcpy(blk, &h, 2);
      for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)lrintf(x[32 * b + i] * id);
    }
  }
}

static inline void kgar_decode(int kv, const uint8_t *src, int d, double *out) {
  if (kv == KGA_KV_F16) {
    for (int i = 0; i < d; i++) out[i] = kref_h2d(kref_u16(src + 2 * i));
  } else if (kv == KGA_KV_BF16) {
    for (int i = 0; i < d; i++) out[i] = kgar_bf2d(kref_u16(src + 2 * i));
  } else {
    for (int b = 0; b < d / 32; b++) {
      const double dd = kref_h2d(kref_u16(src + 34 * b));
      for (int i = 0; i < 32; i++) out[32 * b + i] = dd * (int8_t)src[34 * b + 2 + i];
    }
  }
}

// Fill p.q / p.k / p.v / p.mask and the strides (p's shape fields must be set).
static inline void kgar_make(kgar_problem &p, uint64_t seed) {
  kref_rng r(seed);
  if (p.pos0 == -1) p.pos0 = p.nkv - p.nq;  // other negative values are literal: early rows see no key
  p.rk = kgar_row_bytes(p.kv, p.dk);
  p.rv = p.mla ? p.rk : kgar_row_bytes(p.kv, p.dv);
  p.q.resize((size_t)p.nq * p.nh * p.dk);
  for (auto &x : p.q) x = (float)(3.0 * r.gauss());
  const int64_t rows = p.nkv * p.nhkv;
  p.k.assign((size_t)(rows * p.rk), 0);
  p.v.assign(p.mla ? 0 : (size_t)(rows * p.rv), 0);
  std::vector<float> tmp(p.dk > p.dv ? p.dk : p.dv);
  p.k_s_tok = p.layout ? p.rk : p.nhkv * p.rk;
  p.k_s_head = p.layout ? p.nkv * p.rk : p.rk;
  p.v_s_tok = p.mla ? p.k_s_tok : p.layout ? p.rv : p.nhkv * p.rv;
  p.v_s_head = p.mla ? p.k_s_head : p.layout ? p.nkv * p.rv : p.rv;
  for (int64_t j = 0; j < p.nkv; j++)
    for (int g = 0; g < p.nhkv; g++) {
      for (int i = 0; i < p.dk; i++) tmp[i] = (float)r.gauss();
      kgar_encode(p.kv, tmp.data(), p.dk, p.k.data() + j * p.k_s_tok + g * p.k_s_head);
      if (!p.mla) {
        for (int i = 0; i < p.dv; i++) tmp[i] = (float)r.gauss();
        kgar_encode(p.kv, tmp.data(), p.dv, p.v.data() + j * p.v_s_tok + g * p.v_s_head);
      }
    }
  p.mask_s = (p.nkv + 63) / 64 * 64;
  p.mask.clear();
  if (p.use_mask) {
    p.mask.resize((size_t)(p.nq * p.mask_s));
    for (int64_t t = 0; t < p.nq; t++)
      for (int64_t j = 0; j < p.mask_s; j++) p.mask[t * p.mask_s + j] = (j <= p.pos0 + t && j < p.nkv) ? 0 : 0xFC00;
  }
}

// kga_args for the problem, given where its buffers live (device or host)
static inline kga_args kgar_args(const kgar_problem &p, const float *q, const void *k, const void *v, const uint16_t *mask, float *out) {
  kga_args a;
  memset(&a, 0, sizeof a);
  a.n_q = p.nq;
  a.n_kv = p.nkv;
  a.q_pos0 = p.pos0;
  a.n_head = p.nh;
  a.n_head_kv = p.nhkv;
  a.causal = p.use_mask ? 0 : p.causal;
  a.scale = (float)(1.0 / sqrt((double)p.dk));
  a.q = q;
  a.q_s_tok = (int64_t)p.nh * p.dk;
  a.q_s_head = p.dk;
  a.k = k;
  a.k_s_tok = p.k_s_tok;
  a.k_s_head = p.k_s_head;
  a.v = p.mla ? k : v;
  a.v_s_tok = p.v_s_tok;
  a.v_s_head = p.v_s_head;
  a.mask = p.use_mask ? mask : nullptr;
  a.mask_s_tok = p.mask_s;
  a.out = out;
  a.o_s_tok = (int64_t)p.nh * p.dv;
  a.o_s_head = p.dv;
  return a;
}

// Reference for query tokens `toks` (all heads): ref[(ti * nh + h) * dv + i]
static inline void kgar_reference(const kgar_problem &p, const std::vector<int64_t> &toks, std::vector<double> &ref) {
  ref.assign(toks.size() * p.nh * p.dv, 0.0);
  const double scale = (double)(float)(1.0 / sqrt((double)p.dk));
  std::vector<double> kd((size_t)p.nkv * p.dk), vd((size_t)p.nkv * p.dv), s(p.nkv);
  const int G = p.nh / p.nhkv;
  for (int g = 0; g < p.nhkv; g++) {
    for (int64_t j = 0; j < p.nkv; j++) {
      kgar_decode(p.kv, p.k.data() + j * p.k_s_tok + g * p.k_s_head, p.dk, &kd[j * p.dk]);
      if (p.mla)
        for (int i = 0; i < p.dv; i++) vd[j * p.dv + i] = kd[j * p.dk + i];
      else
        kgar_decode(p.kv, p.v.data() + j * p.v_s_tok + g * p.v_s_head, p.dv, &vd[j * p.dv]);
    }
    for (int hh = 0; hh < G; hh++) {
      const int h = g * G + hh;
      for (size_t ti = 0; ti < toks.size(); ti++) {
        const int64_t t = toks[ti];
        const float *qr = p.q.data() + (t * p.nh + h) * p.dk;
        int64_t lim = p.nkv - 1;
        if (p.causal || p.use_mask) lim = p.pos0 + t < lim ? p.pos0 + t : lim;
        double mx = -INFINITY;
        for (int64_t j = 0; j <= lim; j++) {
          double acc = 0;
          for (int i = 0; i < p.dk; i++) acc += (double)qr[i] * kd[j * p.dk + i];
          s[j] = acc * scale;
          if (s[j] > mx) mx = s[j];
        }
        if (lim < 0) continue;
        double sum = 0;
        for (int64_t j = 0; j <= lim; j++) sum += (s[j] = exp(s[j] - mx));
        double *o = &ref[(ti * p.nh + h) * p.dv];
        for (int64_t j = 0; j <= lim; j++) {
          const double w = s[j] / sum;
          for (int i = 0; i < p.dv; i++) o[i] += w * vd[j * p.dv + i];
        }
      }
    }
  }
}

// max |out - ref| / max |ref| over the checked tokens; NaN or Inf anywhere -> INFINITY
static inline double kgar_relerr(const kgar_problem &p, const std::vector<int64_t> &toks, const std::vector<double> &ref, const float *out) {
  double maxerr = 0, maxref = 0;
  for (size_t ti = 0; ti < toks.size(); ti++)
    for (int h = 0; h < p.nh; h++)
      for (int i = 0; i < p.dv; i++) {
        const double o = out[(toks[ti] * p.nh + h) * p.dv + i], r = ref[(ti * p.nh + h) * p.dv + i];
        if (!(fabs(o) < INFINITY)) return INFINITY;
        maxerr = fmax(maxerr, fabs(o - r));
        maxref = fmax(maxref, fabs(r));
      }
  return maxerr / (maxref > 0 ? maxref : 1);
}

// tokens to check: all if few, else first, last and a deterministic sample
static inline std::vector<int64_t> kgar_check_tokens(int64_t nq, int64_t want, uint64_t seed) {
  std::vector<int64_t> t;
  if (nq <= want) {
    for (int64_t i = 0; i < nq; i++) t.push_back(i);
    return t;
  }
  kref_rng r(seed ^ 0x5EED);
  t.push_back(0);
  t.push_back(nq - 1);
  while ((int64_t)t.size() < want) t.push_back((int64_t)(r.next() % (uint64_t)nq));
  return t;
}
