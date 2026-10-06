// kurn GPU attention kernel template (sm_80). kurn.gpu.attn emits the configuration #defines and the shared
// CUDA helpers (kurn.gpu.codegen.PRELUDE / CPASYNC, kurn.gpu.mma.HELPERS) in front of this file.
//
//   KGA_DK, KGA_DV    head dims
//   KGA_KV            KGA_KV_F16 | KGA_KV_BF16 | KGA_KV_Q8_0 | KGA_KV_FP8 (KV cache format; FP8 needs sm_89+)
//   KGA_TK            KV tokens per tile
//   KGA_WM, KGA_WN    warps along rows (16 rows each) and along the tile: for QK^T the WN warps of a row group
//                     split the tile's tokens, for PV they split the output columns (P goes through shared memory)
//   KGA_STAGES        cp.async pipeline depth for KV tiles
//   KGA_SPLIT         KV splits per (row tile, kv head); 0 = from the SM count (flash-decoding)
//   KGA_MLA           1: v aliases k (v = first DV values of each k row); the K tile is reused for PV
//   KGA_QSPLIT        FP8 only: 1 = q as hi + lo e4m3 tiles (two FP8 MMAs, about 8 bits of q), 0 = one e4m3 tile
//
// Rows are GQA-packed as in the CPU op: row r of kv head g is query token r / G, head g * G + r % G. A CTA owns
// 16 * WM rows x one kv head x one KV split. Scores are in the log2 domain (q pre-scaled by scale * log2 e) with
// lazy rescaling (the running max only moves when a tile's max exceeds it by more than 2^8), f16 (F16 / Q8_0 KV)
// or bf16 (BF16 KV) mma.sync m16n8k16 with f32 accumulation. Q8_0 tiles land raw (4-byte cp.async: 34-byte
// blocks) and are dequantized to f16 in shared memory. FP8 (e4m3) KV runs QK^T on FP8 mma.sync m16n8k32 (q quantized
// to e4m3 per row with a scale that maps the row's max to 448, undone on the scores) and PV on f16 mma.sync with V
// converted from e4m3 to f16 in shared memory (exact). Split results are merged by a second kernel from (max, sum,
// unnormalized O) partials in the workspace.
#include "kurn_gpu_attn.h"

#define KGA_LOG2E 1.4426950408889634f
#define KGA_RESCALE 8.0f
#define KGA_MAX_SPLIT 64
#define KGA_BM (16 * KGA_WM)
#define KGA_NT (32 * KGA_WM * KGA_WN)
#define KGA_TKW (KGA_TK / KGA_WN)  // tile tokens per warp in QK^T
#define KGA_DVW (KGA_DV / KGA_WN)  // output columns per warp in PV
#define KGA_FP8 (KGA_KV == KGA_KV_FP8)
#define KGA_QST (KGA_DK + 8)       // shared-memory row strides in halves (+16 bytes: conflict-free ldmatrix)
#define KGA_KST (KGA_DK + 8)
#define KGA_VST (KGA_MLA && !KGA_FP8 ? KGA_KST : KGA_DV + 8)
#define KGA_PST (KGA_TK + 8)
#define KGA_QST8 (KGA_DK + 16)     // FP8: row strides in bytes
#define KGA_KST8 (KGA_DK + 16)
#define KGA_VST8 (KGA_DV + 16)
#if KGA_FP8
#define KGA_RB(d) (d)
#define KGA_KBUF 0  // K stays e4m3 in the raw stages; V is converted into one f16 buffer
#elif KGA_KV == KGA_KV_Q8_0
#define KGA_RB(d) ((d) / 32 * 34)
#define KGA_KBUF 1  // f16 tiles are converted from raw staging; one buffer suffices
#else
#define KGA_RB(d) ((d) * 2)
#define KGA_KBUF KGA_STAGES
#endif
#define KGA_AL(x) (((x) + 15) / 16 * 16)
#if KGA_FP8
#define KGA_QT_BYTES KGA_AL(KGA_BM * KGA_QST8)
#define KGA_Q_BYTES ((1 + KGA_QSPLIT) * KGA_QT_BYTES + KGA_AL(KGA_BM * 4))  // e4m3 q tile(s) + per-row scale
#define KGA_K_BYTES 0
#define KGA_V_BYTES 0
#define KGA_RAW_BYTES (KGA_AL(KGA_TK * KGA_KST8) + (KGA_MLA ? 0 : KGA_AL(KGA_TK * KGA_VST8)))
#define KGA_VF_BYTES KGA_AL(KGA_TK * KGA_VST * 2)  // V converted to f16
#else
#define KGA_Q_BYTES KGA_AL(KGA_BM * KGA_QST * 2)
#define KGA_K_BYTES KGA_AL(KGA_TK * KGA_KST * 2)
#define KGA_V_BYTES (KGA_MLA ? 0 : KGA_AL(KGA_TK * KGA_VST * 2))
#define KGA_RAW_BYTES (KGA_KV == KGA_KV_Q8_0 ? KGA_AL(KGA_TK * (KGA_RB(KGA_DK) + (KGA_MLA ? 0 : KGA_RB(KGA_DV)))) : 0)
#define KGA_VF_BYTES 0
#endif
#define KGA_P_BYTES KGA_AL(KGA_BM * KGA_PST * 2)
#define KGA_RED_BYTES KGA_AL(KGA_WN * KGA_BM * 4)
#define KGA_SMEM (KGA_Q_BYTES + KGA_KBUF * (KGA_K_BYTES + KGA_V_BYTES) + KGA_STAGES * KGA_RAW_BYTES + KGA_VF_BYTES + KGA_P_BYTES + KGA_RED_BYTES)

#if KGA_TKW % 16 || KGA_DVW % 16 || KGA_DK % 64 || KGA_DV % 64
#error "tile and head dims must split into 16-wide warp slices (dims multiples of 64)"
#endif

KURN_FN void kldsm4t(uint32_t r[4], const void *p) {  // ldmatrix .trans: B fragments of a row-major (k x n) tile
#ifdef KURN_EMU
  kemu_ldmatrix(r, p, 4, 1);
#else
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}
#if KGA_FP8
KURN_FN void kga_mma8(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {  // D += A (16x32 e4m3) B (32x8 e4m3)
#ifdef KURN_EMU
  const unsigned bb[2] = {b0, b1};
  kemu_mma_e4m3_16832(d, a, bb, d);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
KURN_FN uint32_t kga_e4m3x2(float hi, float lo) {  // {lo, hi} -> two e4m3 bytes (RNE, saturating)
#ifdef KURN_EMU
  return kemu_cvt_e4m3x2(hi, lo);
#else
  unsigned short d; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo)); return d;
#endif
}
KURN_FN uint32_t kga_f16x2_e4m3x2(uint32_t v) {  // two e4m3 bytes -> f16x2 (exact)
#ifdef KURN_EMU
  return kemu_cvt_f16x2_e4m3x2(v);
#else
  uint32_t d; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((unsigned short)v)); return d;
#endif
}
#endif
KURN_FN void kga_mma(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if KGA_KV == KGA_KV_BF16
#ifdef KURN_EMU
  const unsigned bb[2] = {b0, b1};
  kemu_mma_bf16_16816(d, a, bb, d);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
#else
  kmma16816(d, a, b0, b1, d);
#endif
}
KURN_FN uint32_t kga_pack(float hi, float lo) {  // two f32 -> the MMA input type, {lo, hi}, round to nearest even
#if KGA_KV == KGA_KV_BF16
#ifdef KURN_EMU
  return kemu_cvt_bf16x2(hi, lo);
#else
  uint32_t d; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
#endif
#else
  return kcvt_h2(hi, lo);
#endif
}
KURN_FN float kga_exp2(float x) {  // 2^x; -inf -> 0
#ifdef KURN_EMU
  return exp2f(x);
#else
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
#endif
}
KURN_FN void kcp4(void *dst, const void *src, int bytes) {  // 4-byte async copy, zero-filled when bytes == 0
#ifdef KURN_EMU
  kemu_cp_async4(dst, src, bytes);
#else
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
#endif
}
KURN_FN float kga_ninf() { return __uint_as_float(0xff800000u); }

struct kga_plan {
  int G, R, nrt, nsplit;
  int64_t chunk;
};

static kga_plan kga_make_plan(const kga_args *a) {
  kga_plan p;
  p.G = a->n_head / a->n_head_kv;
  p.R = (int)(a->n_q * p.G);
  p.nrt = (p.R + KGA_BM - 1) / KGA_BM;
  int ns = KGA_SPLIT;
  if (ns <= 0) {  // about two waves of CTAs, at least 256 KV per split
    const long ctas = (long)p.nrt * a->n_head_kv, want = 2L * kurn_sms();
    ns = ctas < want ? (int)((want + ctas - 1) / ctas) : 1;
    const int64_t cap = a->n_kv / 256 > 1 ? a->n_kv / 256 : 1;
    if (ns > cap) ns = (int)cap;
  }
  if (ns < 1) ns = 1;
  if (ns > KGA_MAX_SPLIT) ns = KGA_MAX_SPLIT;
  p.chunk = ((a->n_kv + ns - 1) / ns + KGA_TK - 1) / KGA_TK * KGA_TK;
  p.nsplit = (int)((a->n_kv + p.chunk - 1) / p.chunk);  // no empty trailing splits
  if (p.nsplit < 1) p.nsplit = 1;
  return p;
}

// Issue the cp.async copies of KV tile [kv0, kv0 + TK) (tokens >= k1 zero-filled) into one pipeline stage.
KURN_FN void kga_load_tile(const kga_args &a, int g, int64_t kv0, int64_t k1, uint8_t *kdst, uint8_t *vdst, int tid) {
#if KGA_KV == KGA_KV_Q8_0
  constexpr int WK = KGA_RB(KGA_DK) / 4, WV = KGA_MLA ? 0 : KGA_RB(KGA_DV) / 4, W = WK + WV;
  for (int i = tid; i < KGA_TK * W; i += KGA_NT) {
    const int row = i / W, w = i % W;
    const int64_t j = kv0 + row;
    const bool ok = j < k1;
    if (w < WK) {
      const uint8_t *src = (const uint8_t *)a.k + (ok ? j * a.k_s_tok + g * a.k_s_head + 4 * w : 0);
      kcp4(kdst + row * KGA_RB(KGA_DK) + 4 * w, src, ok ? 4 : 0);
    } else {
      const uint8_t *src = (const uint8_t *)a.v + (ok ? j * a.v_s_tok + g * a.v_s_head + 4 * (w - WK) : 0);
      kcp4(vdst + row * KGA_RB(KGA_DV) + 4 * (w - WK), src, ok ? 4 : 0);
    }
  }
#elif KGA_FP8
  constexpr int CK = KGA_DK / 16, CV = KGA_MLA ? 0 : KGA_DV / 16, C = CK + CV;  // 16-byte chunks of e4m3 rows
  for (int i = tid; i < KGA_TK * C; i += KGA_NT) {
    const int row = i / C, c = i % C;
    const int64_t j = kv0 + row;
    const bool ok = j < k1;
    if (c < CK) {
      const uint8_t *src = (const uint8_t *)a.k + (ok ? j * a.k_s_tok + g * a.k_s_head + 16 * c : 0);
      kcp16(kdst + row * KGA_KST8 + 16 * c, src, ok ? 16 : 0);
    } else {
      const uint8_t *src = (const uint8_t *)a.v + (ok ? j * a.v_s_tok + g * a.v_s_head + 16 * (c - CK) : 0);
      kcp16(vdst + row * KGA_VST8 + 16 * (c - CK), src, ok ? 16 : 0);
    }
  }
#else
  constexpr int CK = KGA_DK / 8, CV = KGA_MLA ? 0 : KGA_DV / 8, C = CK + CV;
  for (int i = tid; i < KGA_TK * C; i += KGA_NT) {
    const int row = i / C, c = i % C;
    const int64_t j = kv0 + row;
    const bool ok = j < k1;
    if (c < CK) {
      const uint8_t *src = (const uint8_t *)a.k + (ok ? j * a.k_s_tok + g * a.k_s_head + 16 * c : 0);
      kcp16(kdst + (row * KGA_KST + 8 * c) * 2, src, ok ? 16 : 0);
    } else {
      const uint8_t *src = (const uint8_t *)a.v + (ok ? j * a.v_s_tok + g * a.v_s_head + 16 * (c - CK) : 0);
      kcp16(vdst + (row * KGA_VST + 8 * (c - CK)) * 2, src, ok ? 16 : 0);
    }
  }
#endif
}

#if KGA_FP8
// e4m3 rows (row stride `rst` bytes; the first DV values) -> f16 tile with row stride KGA_VST halves (exact)
KURN_FN void kga_cvt_v8(const uint8_t *raw, int rst, uint8_t *dst, int tid) {
  for (int i = tid; i < KGA_TK * (KGA_DV / 2); i += KGA_NT) {
    const int row = i / (KGA_DV / 2), e = 2 * (i % (KGA_DV / 2));
    kst_u32(dst + (row * KGA_VST + e) * 2, kga_f16x2_e4m3x2(kld_u16(raw + row * rst + e)));
  }
}
#endif

#if KGA_KV == KGA_KV_Q8_0
// raw Q8_0 rows (TK x d/32 blocks of 34 bytes) -> f16 tile with row stride `st` halves; d * q rounded to f16
KURN_FN void kga_deq_q8(const uint8_t *raw, uint8_t *dst, int d, int st, int tid) {
  const int rb = d / 32 * 34, npair = d / 2;
  for (int i = tid; i < KGA_TK * npair; i += KGA_NT) {
    const int row = i / npair, e = 2 * (i % npair);
    const uint8_t *blk = raw + row * rb + (e / 32) * 34;
    const float sc = KURN_H2F(kld_u16(blk));
    const int8_t q0 = (int8_t)blk[2 + e % 32], q1 = (int8_t)blk[3 + e % 32];
    kst_u32(dst + (row * st + e) * 2, kcvt_h2(sc * (float)q1, sc * (float)q0));
  }
}
#endif

static __global__ void __launch_bounds__(KGA_NT, 1) kga_main(kga_args a, kga_plan p, float *__restrict__ part) {
#ifdef KURN_EMU
  static uint8_t kga_smem_emu[KGA_SMEM + 16] __attribute__((aligned(16)));
  uint8_t *smem = kga_smem_emu;
#else
  extern __shared__ __align__(16) uint8_t kga_smem[];
  uint8_t *smem = kga_smem;
#endif
  uint8_t *Qs = smem;
  uint8_t *Ks = Qs + KGA_Q_BYTES;
  uint8_t *Vs = Ks + KGA_KBUF * KGA_K_BYTES;
  uint8_t *Raw = Vs + KGA_KBUF * KGA_V_BYTES;
  uint8_t *Vf = Raw + KGA_STAGES * KGA_RAW_BYTES;  // FP8: V as f16
  uint8_t *Ps = Vf + KGA_VF_BYTES;
  float *red = (float *)(Ps + KGA_P_BYTES);
  (void)Raw;
  (void)Vf;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / KGA_WN, wn = warp % KGA_WN;
  const int gq = lane >> 2, tq = lane & 3;
  const int rt = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
  const int G = p.G, R = p.R;
  const int r0 = rt * KGA_BM;

  // KV range of this CTA: its split, cut at the last position its rows can see
  int64_t hi = a.n_kv;
  if (a.causal) {
    const int rl = (r0 + KGA_BM < R ? r0 + KGA_BM : R) - 1;
    const int64_t lim = a.q_pos0 + rl / G + 1;
    hi = lim < hi ? (lim > 0 ? lim : 0) : hi;
  }
  const int64_t k0 = (int64_t)s * p.chunk < hi ? (int64_t)s * p.chunk : hi;
  const int64_t k1 = (int64_t)(s + 1) * p.chunk < hi ? (int64_t)(s + 1) * p.chunk : hi;

  // Q tile, pre-scaled into the log2 domain, rounded to the MMA type; padding rows are 0
  const float qs = a.scale * KGA_LOG2E;
#if KGA_FP8
  // e4m3 q with a per-row scale mapping the row's max |q * qs| to 448; the scores are multiplied back by its inverse
  float *qsc = (float *)(Qs + (1 + KGA_QSPLIT) * KGA_QT_BYTES);
  for (int row = tid; row < KGA_BM; row += KGA_NT) {
    const int r = r0 + row;
    float amax = 0.f;
    if (r < R) {
      const float *q = a.q + (int64_t)(r / G) * a.q_s_tok + (int64_t)(g * G + r % G) * a.q_s_head;
      for (int e = 0; e < KGA_DK; e++) amax = fmaxf(amax, fabsf(q[e] * qs));
    }
    qsc[row] = amax > 0.f ? 448.f / amax : 1.f;
  }
  __syncthreads();
  for (int i = tid; i < KGA_BM * (KGA_DK / 2); i += KGA_NT) {
    const int row = i / (KGA_DK / 2), e = 2 * (i % (KGA_DK / 2)), r = r0 + row;
    float x0 = 0.f, x1 = 0.f;
    if (r < R) {
      const float *q = a.q + (int64_t)(r / G) * a.q_s_tok + (int64_t)(g * G + r % G) * a.q_s_head + e;
      x0 = q[0] * qs * qsc[row];
      x1 = q[1] * qs * qsc[row];
    }
    const uint32_t hi = kga_e4m3x2(x1, x0);
    kst_u16(Qs + row * KGA_QST8 + e, hi);
#if KGA_QSPLIT
    const uint32_t h2 = kga_f16x2_e4m3x2(hi);  // the residual q - hi, itself rounded to e4m3
    kst_u16(Qs + KGA_QT_BYTES + row * KGA_QST8 + e, kga_e4m3x2(x1 - kh2f_hi(h2), x0 - kh2f_lo(h2)));
#endif
  }
#else
  for (int i = tid; i < KGA_BM * (KGA_DK / 2); i += KGA_NT) {
    const int row = i / (KGA_DK / 2), e = 2 * (i % (KGA_DK / 2)), r = r0 + row;
    float x0 = 0.f, x1 = 0.f;
    if (r < R) {
      const float *q = a.q + (int64_t)(r / G) * a.q_s_tok + (int64_t)(g * G + r % G) * a.q_s_head + e;
      x0 = q[0] * qs;
      x1 = q[1] * qs;
    }
    kst_u32(Qs + (row * KGA_QST + e) * 2, kga_pack(x1, x0));
  }
#endif

  // per thread: rows ra = 16 wm + gq and ra + 8 of the CTA tile
  const int ra = 16 * wm + gq;
  const int rga = r0 + ra, rgb = rga + 8;
  const bool va = rga < R, vb = rgb < R;
  const int64_t lima = va ? a.q_pos0 + rga / G : -1, limb = vb ? a.q_pos0 + rgb / G : -1;  // last visible kv (causal)
  const uint16_t *mra = a.mask && va ? a.mask + (int64_t)(rga / G) * a.mask_s_tok : nullptr;
  const uint16_t *mrb = a.mask && vb ? a.mask + (int64_t)(rgb / G) * a.mask_s_tok : nullptr;
#if KGA_FP8
  __syncthreads();
  const float iqa = 1.f / qsc[ra], iqb = 1.f / qsc[ra + 8];
#endif
  float m[2] = {kga_ninf(), kga_ninf()}, l[2] = {0.f, 0.f};
  float o[KGA_DVW / 8][4];
#pragma unroll
  for (int n = 0; n < KGA_DVW / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

  const int ntiles = (int)((k1 - k0 + KGA_TK - 1) / KGA_TK);
  auto kbuf = [&](int st) { return Ks + (KGA_KBUF == 1 ? 0 : st) * KGA_K_BYTES; };
  auto vbuf = [&](int st) { return KGA_MLA ? kbuf(st) : Vs + (KGA_KBUF == 1 ? 0 : st) * KGA_V_BYTES; };
  (void)vbuf;
#if KGA_KV == KGA_KV_Q8_0
  auto ldst_k = [&](int st) { return Raw + st * KGA_RAW_BYTES; };
  auto ldst_v = [&](int st) { return Raw + st * KGA_RAW_BYTES + KGA_TK * KGA_RB(KGA_DK); };
#elif KGA_FP8
  auto ldst_k = [&](int st) { return Raw + st * KGA_RAW_BYTES; };
  auto ldst_v = [&](int st) { return Raw + st * KGA_RAW_BYTES + KGA_AL(KGA_TK * KGA_KST8); };
#else
  auto ldst_k = [&](int st) { return kbuf(st); };
  auto ldst_v = [&](int st) { return vbuf(st); };
#endif
  for (int st = 0; st < KGA_STAGES - 1; st++) {
    if (st < ntiles) kga_load_tile(a, g, k0 + (int64_t)st * KGA_TK, k1, ldst_k(st), ldst_v(st), tid);
    kcp_commit();
  }

  for (int it = 0; it < ntiles; it++) {
    const int st = it % KGA_STAGES;
    {
      const int nx = it + KGA_STAGES - 1;
      if (nx < ntiles) kga_load_tile(a, g, k0 + (int64_t)nx * KGA_TK, k1, ldst_k(nx % KGA_STAGES), ldst_v(nx % KGA_STAGES), tid);
      kcp_commit();
    }
    KCP_WAIT(KGA_STAGES - 1);
    __syncthreads();
#if KGA_KV == KGA_KV_Q8_0
    kga_deq_q8(ldst_k(st), kbuf(st), KGA_DK, KGA_KST, tid);
#if !KGA_MLA
    kga_deq_q8(ldst_v(st), vbuf(st), KGA_DV, KGA_VST, tid);
#endif
    __syncthreads();
#endif
#if KGA_FP8
    kga_cvt_v8(KGA_MLA ? ldst_k(st) : ldst_v(st), KGA_MLA ? KGA_KST8 : KGA_VST8, Vf, tid);
    __syncthreads();
    const uint8_t *Kt = ldst_k(st), *Vt = Vf;
#else
    const uint8_t *Kt = kbuf(st), *Vt = vbuf(st);
#endif
    const int64_t kv0 = k0 + (int64_t)it * KGA_TK;

    // S = Q K^T for rows 16 wm.., tile tokens wn * TKW..
    float sc[KGA_TKW / 8][4];
#pragma unroll
    for (int n = 0; n < KGA_TKW / 8; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#if KGA_FP8
#pragma unroll 2
    for (int kk = 0; kk < KGA_DK / 32; kk++) {  // m16n8k32: 32 e4m3 values = the same 16-byte ldmatrix rows as f16
      uint32_t af[4];
      const uint8_t *qa = Qs + (16 * wm + (lane & 7) + 8 * ((lane >> 3) & 1)) * KGA_QST8 + 32 * kk + 16 * (lane >> 4);
      kldsm4(af, qa);
#if KGA_QSPLIT
      uint32_t al[4];
      kldsm4(al, qa + KGA_QT_BYTES);
#endif
#pragma unroll
      for (int np = 0; np < KGA_TKW / 16; np++) {
        uint32_t bf[4];
        const int j = lane >> 3;
        kldsm4(bf, Kt + (wn * KGA_TKW + 16 * np + (lane & 7) + 8 * (j >> 1)) * KGA_KST8 + 32 * kk + 16 * (j & 1));
        kga_mma8(sc[2 * np], af, bf[0], bf[1]);
        kga_mma8(sc[2 * np + 1], af, bf[2], bf[3]);
#if KGA_QSPLIT
        kga_mma8(sc[2 * np], al, bf[0], bf[1]);
        kga_mma8(sc[2 * np + 1], al, bf[2], bf[3]);
#endif
      }
    }
#pragma unroll
    for (int n = 0; n < KGA_TKW / 8; n++) {
      sc[n][0] *= iqa;
      sc[n][1] *= iqa;
      sc[n][2] *= iqb;
      sc[n][3] *= iqb;
    }
#else
#pragma unroll 2
    for (int kk = 0; kk < KGA_DK / 16; kk++) {
      uint32_t af[4];
      kldsm4(af, Qs + ((16 * wm + (lane & 7) + 8 * ((lane >> 3) & 1)) * KGA_QST + 16 * kk + 8 * (lane >> 4)) * 2);
#pragma unroll
      for (int np = 0; np < KGA_TKW / 16; np++) {
        uint32_t bf[4];
        const int j = lane >> 3;
        kldsm4(bf, Kt + ((wn * KGA_TKW + 16 * np + (lane & 7) + 8 * (j >> 1)) * KGA_KST + 16 * kk + 8 * (j & 1)) * 2);
        kga_mma(sc[2 * np], af, bf[0], bf[1]);
        kga_mma(sc[2 * np + 1], af, bf[2], bf[3]);
      }
    }
#endif

    // masks: tile tail, causal, explicit fp16 mask, padding rows; per-row tile max
    float tmax[2] = {kga_ninf(), kga_ninf()};
#pragma unroll
    for (int n = 0; n < KGA_TKW / 8; n++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int64_t j = kv0 + wn * KGA_TKW + 8 * n + 2 * tq + (i & 1);
        const int hb = i >> 1;
        const bool vis = j < k1 && (hb ? vb : va) && (!a.causal || j <= (hb ? limb : lima));
        float x = vis ? sc[n][i] : kga_ninf();
        const uint16_t *mr = hb ? mrb : mra;
        if (vis && mr) x += KURN_H2F(kld_u16((const uint8_t *)(mr + j))) * KGA_LOG2E;
        sc[n][i] = x;
        tmax[hb] = fmaxf(tmax[hb], x);
      }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(KURN_FULL, tmax[h], 1));
      tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(KURN_FULL, tmax[h], 2));
    }
#if KGA_WN > 1
    if (tq == 0) {
      red[wn * KGA_BM + ra] = tmax[0];
      red[wn * KGA_BM + ra + 8] = tmax[1];
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < KGA_WN; w++) {
      tmax[0] = fmaxf(tmax[0], red[w * KGA_BM + ra]);
      tmax[1] = fmaxf(tmax[1], red[w * KGA_BM + ra + 8]);
    }
#endif
    // lazy rescale: every warp of a row group takes the same decision from the same values
    float alpha[2] = {1.f, 1.f};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      if (tmax[h] != kga_ninf() && (m[h] == kga_ninf() || tmax[h] > m[h] + KGA_RESCALE)) {
        alpha[h] = m[h] == kga_ninf() ? 0.f : kga_exp2(m[h] - tmax[h]);
        l[h] *= alpha[h];
        m[h] = tmax[h];
      }
    }
    const float mu[2] = {m[0] == kga_ninf() ? 0.f : m[0], m[1] == kga_ninf() ? 0.f : m[1]};
#pragma unroll
    for (int n = 0; n < KGA_TKW / 8; n++) {
      const float p0 = kga_exp2(sc[n][0] - mu[0]), p1 = kga_exp2(sc[n][1] - mu[0]);
      const float p2 = kga_exp2(sc[n][2] - mu[1]), p3 = kga_exp2(sc[n][3] - mu[1]);
      l[0] += p0 + p1;
      l[1] += p2 + p3;
      const int col = wn * KGA_TKW + 8 * n + 2 * tq;
      kst_u32(Ps + (ra * KGA_PST + col) * 2, kga_pack(p1, p0));
      kst_u32(Ps + ((ra + 8) * KGA_PST + col) * 2, kga_pack(p3, p2));
    }
#pragma unroll
    for (int n = 0; n < KGA_DVW / 8; n++) {
      o[n][0] *= alpha[0];
      o[n][1] *= alpha[0];
      o[n][2] *= alpha[1];
      o[n][3] *= alpha[1];
    }
#if KGA_WN > 1
    __syncthreads();
#else
    __syncwarp();
#endif

    // O += P V for rows 16 wm.., output columns wn * DVW..
#pragma unroll 2
    for (int kk = 0; kk < KGA_TK / 16; kk++) {
      uint32_t af[4];
      kldsm4(af, Ps + ((16 * wm + (lane & 7) + 8 * ((lane >> 3) & 1)) * KGA_PST + 16 * kk + 8 * (lane >> 4)) * 2);
#pragma unroll
      for (int np = 0; np < KGA_DVW / 16; np++) {
        uint32_t bf[4];
        const int j = lane >> 3;
        kldsm4t(bf, Vt + ((16 * kk + (lane & 7) + 8 * (j & 1)) * KGA_VST + wn * KGA_DVW + 16 * np + 8 * (j >> 1)) * 2);
        kga_mma(o[2 * np], af, bf[0], bf[1]);
        kga_mma(o[2 * np + 1], af, bf[2], bf[3]);
      }
    }
    __syncthreads();  // the next iteration refills this stage and P
  }
  KCP_WAIT(0);

  // row sums: quad lanes, then the WN warps of the row group
#pragma unroll
  for (int h = 0; h < 2; h++) {
    l[h] += __shfl_xor_sync(KURN_FULL, l[h], 1);
    l[h] += __shfl_xor_sync(KURN_FULL, l[h], 2);
  }
#if KGA_WN > 1
  if (tq == 0) {
    red[wn * KGA_BM + ra] = l[0];
    red[wn * KGA_BM + ra + 8] = l[1];
  }
  __syncthreads();
  l[0] = l[1] = 0.f;
#pragma unroll
  for (int w = 0; w < KGA_WN; w++) {
    l[0] += red[w * KGA_BM + ra];
    l[1] += red[w * KGA_BM + ra + 8];
  }
#endif

#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int r = h ? rgb : rga;
    if (r >= R) continue;
    if (p.nsplit == 1) {
      const float inv = l[h] > 0.f ? 1.f / l[h] : 0.f;
      float *dst = a.out + (int64_t)(r / G) * a.o_s_tok + (int64_t)(g * G + r % G) * a.o_s_head + wn * KGA_DVW + 2 * tq;
#pragma unroll
      for (int n = 0; n < KGA_DVW / 8; n++) {
        dst[8 * n] = o[n][2 * h] * inv;
        dst[8 * n + 1] = o[n][2 * h + 1] * inv;
      }
    } else {
      float *dst = part + (((int64_t)g * p.nsplit + s) * R + r) * (KGA_DV + 2);
      if (wn == 0 && tq == 0) {
        dst[0] = m[h];
        dst[1] = l[h];
      }
      dst += 2 + wn * KGA_DVW + 2 * tq;
#pragma unroll
      for (int n = 0; n < KGA_DVW / 8; n++) {
        dst[8 * n] = o[n][2 * h];
        dst[8 * n + 1] = o[n][2 * h + 1];
      }
    }
  }
}

// LSE merge of the split partials: one block per (row, kv head)
static __global__ void __launch_bounds__(128) kga_merge(kga_args a, kga_plan p, const float *__restrict__ part) {
  const int r = blockIdx.x, g = blockIdx.y, G = p.G, R = p.R;
  const int64_t rs = (int64_t)R * (KGA_DV + 2);
  const float *base = part + ((int64_t)g * p.nsplit * R + r) * (KGA_DV + 2);
  float M = kga_ninf();
  for (int s = 0; s < p.nsplit; s++) M = fmaxf(M, base[s * rs]);
  float L = 0.f;
  if (M != kga_ninf())
    for (int s = 0; s < p.nsplit; s++)
      if (base[s * rs] != kga_ninf()) L += kga_exp2(base[s * rs] - M) * base[s * rs + 1];
  const float inv = L > 0.f ? 1.f / L : 0.f;
  float *dst = a.out + (int64_t)(r / G) * a.o_s_tok + (int64_t)(g * G + r % G) * a.o_s_head;
  for (int d = threadIdx.x; d < KGA_DV; d += 128) {
    float acc = 0.f;
    if (M != kga_ninf())
      for (int s = 0; s < p.nsplit; s++)
        if (base[s * rs] != kga_ninf()) acc += kga_exp2(base[s * rs] - M) * base[s * rs + 2 + d];
    dst[d] = acc * inv;
  }
}

extern "C" {
const char *kga_config(int *dk, int *dv, int *kv_format) {
  if (dk) *dk = KGA_DK;
  if (dv) *dv = KGA_DV;
  if (kv_format) *kv_format = KGA_KV;
  return KGA_CONFIG;
}

int kga_check(const kga_args *a) {
  if (a->n_q < 1 || a->n_kv < 1 || a->n_head < 1 || a->n_head_kv < 1 || a->n_head % a->n_head_kv) return -1;
  if ((int64_t)a->n_q * (a->n_head / a->n_head_kv) > (1LL << 30)) return -2;
  const uintptr_t al = KGA_KV == KGA_KV_Q8_0 ? 4 : 16;
  if ((uintptr_t)a->k % al || a->k_s_tok % (int64_t)al || a->k_s_head % (int64_t)al) return -3;
  if ((uintptr_t)a->v % al || a->v_s_tok % (int64_t)al || a->v_s_head % (int64_t)al) return -3;
  if (KGA_MLA && (a->v != a->k || a->v_s_tok != a->k_s_tok || a->v_s_head != a->k_s_head)) return -4;
  if (a->mask && a->mask_s_tok < a->n_kv) return -5;
  return 0;
}

int kga_splits(const kga_args *a) { return kga_make_plan(a).nsplit; }

size_t kga_workspace(const kga_args *a) {
  const kga_plan p = kga_make_plan(a);
  return p.nsplit > 1 ? sizeof(float) * (size_t)a->n_head_kv * p.nsplit * p.R * (KGA_DV + 2) : 0;
}

int kga_run(const kga_args *a, void *ws, cudaStream_t s) {
  if (kga_check(a)) return -1;
  const kga_plan p = kga_make_plan(a);
  if (p.nsplit > 1 && !ws) return -2;
#ifndef KURN_EMU
  static int attr = 0;  // 1: ready; -1: this device's per-block shared memory is too small for the tile
  if (!attr) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    attr = optin >= KGA_SMEM && cudaFuncSetAttribute(kga_main, cudaFuncAttributeMaxDynamicSharedMemorySize, KGA_SMEM) == cudaSuccess ? 1 : -1;
  }
  if (attr < 0) return -4;  // e.g. an A100-sized tile (> 99 KB) on an sm_86/89/120 GPU: pick a smaller tk
#endif
  const kga_args args = *a;
  float *part = (float *)ws;
  KURN_LAUNCH_SMEM(kga_main, dim3((unsigned)p.nrt, (unsigned)a->n_head_kv, (unsigned)p.nsplit), dim3(KGA_NT), KGA_SMEM, s, args, p, part);
  if (p.nsplit > 1) KURN_LAUNCH(kga_merge, dim3((unsigned)p.R, (unsigned)a->n_head_kv), dim3(128), s, args, p, (const float *)part);
#ifdef KURN_EMU
  return 0;
#else
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
#endif
}
}
