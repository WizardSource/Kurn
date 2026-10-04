// kurn_gpu_ref.h: host-side test data and exact references for the GPU kernels (C++, no CUDA).
//
// - kref_format(name): format descriptors for the GPU formats (ggml block layouts)
// - kref_gen_weights / kref_gen_acts: deterministic random weights (valid blocks) and f32 activations
// - kref_dequant_w / kref_dequant_x: exact dequantization to double (weights, and q8_0 / q8_K activation blocks)
// - kref_gemm: Y[m * N + n] = sum_k w(n, k) * x(m, k) in double, for chosen rows/columns
// Dequantize-then-dot in double equals KURN's Python references (kurn.formats) up to double rounding;
// tests/test_gpu_ref.py checks that.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <vector>

struct kref_fmt {
  const char *name;
  int block, nbytes, act_block, act_nbytes;  // weights: values / bytes per block; activation format likewise
};

static const kref_fmt KREF_FORMATS[] = {
    {"q8_0", 32, 34, 32, 34},     {"q4_0", 32, 18, 32, 34},    {"iq4_nl", 32, 18, 32, 34}, {"q4_K", 256, 144, 256, 292},
    {"q2_0", 64, 18, 32, 34},     {"tq2_0", 256, 66, 256, 292}, {"q1_0", 128, 18, 32, 34}, {"e8p", 256, 66, 256, 292},
};

static inline const kref_fmt *kref_format(const char *name) {
  for (const auto &f : KREF_FORMATS)
    if (!strcmp(f.name, name)) return &f;
  return nullptr;
}

// 256 rows x 8 of 2|a| (1, 3, 5) of the E8P codebook, row-major (kurn.ext.compress.E8P_ABS2)
static const char KREF_E8P_ABS2[] =
    "11111111111111331111131311111331111131131111313111113311111311131113113111131311111331111131111311311131113113111131311111331111"
    "13111113131111311311131113113111131311111331111131111113311111313111131131113111311311113131111133111111111111151111115111111511"
    "11115111111511111151111115111111511111111111333311131333111331331113331311133331113113331131313311313313113133311133113311331313"
    "11331331113331131133313111333311131113331311313313113313131133311313113313131313131313311313311313133131131333111331113313311313"
    "13311331133131131331313113313311133311131333113113331311133331113111133331113133311133133111333131131133311313133113133131133113"
    "31133131311333113131113331311313313113313131311331313131313133113133111331331131313313113133311133111133331113133311133133113113"
    "33113131331133113313111333131131331313113313311133311113333111313331131133313111333311111111133511111353111115331111313511113153"
    "11113315111133511111351311113531111151331111531311115331111311351113115311131315111313511113151311131531111331151113315111133511"
    "11111113111111311111131111113111111311111131111113111111311111111111133311113133111133131111333111131133111313131113133111133113"
    "11133131111333111131113311311313113113311131311311313131113133111133111311331131113313111133311113111133131113131311133113113113"
    "13113131131133111313111313131131131313111313311113311113133111311331131113313111133311113111113331111313311113313111311331113131"
    "31113311311311133113113131131311311331113131111331311131313113113131311131331111331111133311113133111311331131113313111133311111"
    "11111135111111531111131511111351111115131111153111113115111131511111351111115113111151311111531111131115111311511113151111135111"
    "11151113111511311115131111153111113111151131115111311511113151111135111111511113115111311151131111513111115311111311111513111151"
    "13111511131151111315111113511111151111131511113115111311151131111513111115311111311111153111115131111511311151113115111131511111"
    "35111111511111135111113151111311511131115113111151311111531111111113333311313333113313331133313311333313113333311311333313131333"
;

static inline double kref_h2d(uint16_t h) {
  int e = (h >> 10) & 0x1F, m = h & 0x3FF;
  double v = e == 0 ? ldexp((double)m, -24) : e == 31 ? INFINITY : ldexp((double)(m | 0x400), e - 25);
  return (h >> 15) ? -v : v;
}
// f32 -> nearest f16 (ties to even) -> f32: the activation rounding of the tensor-core engine (kg_act() 1 and 2)
static inline float kref_round_f16(float f) {
  if (!(fabsf(f) < 65520.f)) return f;  // inf / nan / overflow: not produced by the test data
  int e;
  frexpf(f, &e);  // f = m * 2^e, 0.5 <= |m| < 1
  int sh = e - 11;  // f16 keeps 11 significant bits
  if (e < -13) sh = -24;  // subnormal f16 spacing 2^-24
  return ldexpf(nearbyintf(ldexpf(f, -sh)), sh);
}
static inline uint16_t kref_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline float kref_f32(const uint8_t *p) {
  float f;
  memcpy(&f, p, 4);
  return f;
}

// weights of one row (K values) -> double
static inline void kref_dequant_w(const char *fmt, const uint8_t *w, int K, double *out) {
  static const int kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  const kref_fmt *f = kref_format(fmt);
  for (int b = 0; b < K / f->block; b++) {
    const uint8_t *p = w + (size_t)b * f->nbytes;
    double *o = out + (size_t)b * f->block;
    if (!strcmp(fmt, "q8_0")) {
      double d = kref_h2d(kref_u16(p));
      for (int i = 0; i < 32; i++) o[i] = d * (int8_t)p[2 + i];
    } else if (!strcmp(fmt, "q4_0") || !strcmp(fmt, "iq4_nl")) {
      double d = kref_h2d(kref_u16(p));
      bool nl = fmt[0] == 'i';
      for (int i = 0; i < 16; i++) {
        int lo = p[2 + i] & 15, hi = p[2 + i] >> 4;
        o[i] = d * (nl ? kv[lo] : lo - 8);
        o[i + 16] = d * (nl ? kv[hi] : hi - 8);
      }
    } else if (!strcmp(fmt, "q4_K")) {
      double d = kref_h2d(kref_u16(p)), dmin = kref_h2d(kref_u16(p + 2));
      const uint8_t *q = p + 4;
      for (int s = 0; s < 8; s++) {
        int sc = s < 4 ? q[s] & 63 : (q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4);
        int mn = s < 4 ? q[s + 4] & 63 : (q[s + 4] >> 4) | ((q[s] >> 6) << 4);
        for (int v = 0; v < 32; v++) {
          int nib = (p[16 + 32 * (s / 2) + v] >> (4 * (s & 1))) & 0xF;
          o[32 * s + v] = d * sc * nib - dmin * mn;
        }
      }
    } else if (!strcmp(fmt, "q2_0")) {
      double d = kref_h2d(kref_u16(p));
      for (int v = 0; v < 64; v++) o[v] = d * (((p[2 + v / 4] >> (2 * (v % 4))) & 3) - 1);
    } else if (!strcmp(fmt, "q1_0")) {
      double d = kref_h2d(kref_u16(p));
      for (int v = 0; v < 128; v++) o[v] = (p[2 + v / 8] >> (v % 8)) & 1 ? d : -d;
    } else if (!strcmp(fmt, "tq2_0")) {
      double d = kref_h2d(kref_u16(p + 64));
      for (int v = 0; v < 256; v++) o[v] = d * (((p[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3) - 1);
    } else if (!strcmp(fmt, "e8p")) {
      double d = kref_h2d(kref_u16(p));
      for (int g = 0; g < 32; g++) {
        int lo = p[2 + g], sg = p[34 + g];
        int par = __builtin_popcount(sg) & 1;
        const char *a = KREF_E8P_ABS2 + 8 * ((lo & 0x7F) | (par << 7));
        int t = lo & 0x80 ? 1 : -1;
        for (int i = 0; i < 8; i++) {
          int a2 = a[i] - '0';
          o[8 * g + i] = d * (((sg >> i) & 1 ? -2 * a2 : 2 * a2) + t);
        }
      }
    }
  }
}

// one row of q8_0 (act_block 32) or q8_K (256) activation blocks -> double
static inline void kref_dequant_x(const kref_fmt *f, const uint8_t *x, int K, double *out) {
  for (int b = 0; b < K / f->act_block; b++) {
    const uint8_t *p = x + (size_t)b * f->act_nbytes;
    if (f->act_block == 32) {
      double d = kref_h2d(kref_u16(p));
      for (int i = 0; i < 32; i++) out[32 * b + i] = d * (int8_t)p[2 + i];
    } else {
      double d = kref_f32(p);
      for (int i = 0; i < 256; i++) out[256 * b + i] = d * (int8_t)p[4 + i];
    }
  }
}

struct kref_rng {
  uint64_t s;
  explicit kref_rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  int range(int n) { return (int)(next() % (uint64_t)n); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double gauss() {
    double u = uniform() + 1e-300, v = uniform();
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
  }
};

// positive fp16 in [2^-6, 2^-1) (q4_K: d and dmin), or model-like small scales when `small`
static inline uint16_t kref_f16_scale(kref_rng &r, bool small) {
  return small ? (uint16_t)(((4 + r.range(3)) << 10) | r.range(1024)) : (uint16_t)(((9 + r.range(5)) << 10) | r.range(1024));
}

// random valid weight blocks for N rows of K values (small = model-like scale magnitudes)
static inline void kref_gen_weights(const char *fmt, int N, int K, uint64_t seed, uint8_t *out, bool small = true) {
  const kref_fmt *f = kref_format(fmt);
  kref_rng r(seed);
  size_t nb = (size_t)N * (K / f->block);
  for (size_t b = 0; b < nb; b++) {
    uint8_t *p = out + b * f->nbytes;
    for (int i = 0; i < f->nbytes; i++) p[i] = (uint8_t)r.next();
    uint16_t d = kref_f16_scale(r, small);
    if (!strcmp(fmt, "q8_0")) {
      for (int i = 0; i < 32; i++) p[2 + i] = (uint8_t)(int8_t)(r.range(255) - 127);
    }
    if (!strcmp(fmt, "tq2_0")) {
      p[64] = d & 0xFF, p[65] = d >> 8;
    } else {
      p[0] = d & 0xFF, p[1] = d >> 8;
    }
    if (!strcmp(fmt, "q4_K")) {
      uint16_t dm = kref_f16_scale(r, small);
      p[2] = dm & 0xFF, p[3] = dm >> 8;
    }
  }
}

static inline void kref_gen_acts(int K, int M, uint64_t seed, float *out) {
  kref_rng r(seed);
  for (size_t i = 0; i < (size_t)K * M; i++) out[i] = (float)r.gauss();
}

// Y[m * N + n] in double for the listed rows and columns; xs: either q8 blocks (xblocks != nullptr) or f32 X
static inline void kref_gemm(const char *fmt, const uint8_t *W, const uint8_t *xblocks, const float *X, int N, int K, int M,
                             const std::vector<int> &rows, const std::vector<int> &cols, std::vector<double> &out) {
  const kref_fmt *f = kref_format(fmt);
  size_t wrow = (size_t)(K / f->block) * f->nbytes, xrow = (size_t)(K / f->act_block) * f->act_nbytes;
  std::vector<double> w(K), x((size_t)K * cols.size());
  out.assign(rows.size() * cols.size(), 0.0);
  for (size_t ci = 0; ci < cols.size(); ci++) {
    int m = cols[ci];
    if (xblocks)
      kref_dequant_x(f, xblocks + (size_t)m * xrow, K, x.data() + (size_t)ci * K);
    else
      for (int k = 0; k < K; k++) x[(size_t)ci * K + k] = X[(size_t)m * K + k];
  }
  for (size_t ri = 0; ri < rows.size(); ri++) {
    kref_dequant_w(fmt, W + (size_t)rows[ri] * wrow, K, w.data());
    for (size_t ci = 0; ci < cols.size(); ci++) {
      const double *xc = x.data() + (size_t)ci * K;
      double s = 0;
      for (int k = 0; k < K; k++) s += w[k] * xc[k];
      out[ci * rows.size() + ri] = s;
    }
  }
  (void)N;
  (void)M;
}
