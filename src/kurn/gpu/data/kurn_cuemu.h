// kurn_cuemu.h: run KURN's generated CUDA kernels on the CPU, with per-thread semantics.
//
// Every CUDA thread is a fiber (ucontext). A fiber runs until it reaches a block barrier
// (__syncthreads) or a warp collective (__shfl*_sync, __syncwarp, mma.sync), then yields.
// The scheduler releases a barrier only when every live thread of the block (or all 32 lanes
// of the warp) has arrived, computes warp collectives from all lanes' operands at once, and
// reports a deadlock (divergent barrier, partial-warp collective) instead of hanging.
// Run-to-barrier scheduling makes missing __syncthreads show up as wrong results: a thread
// that reads shared memory before its producers ran sees stale data.
//
// cp.async copies are deferred until the issuing thread's cp.async.wait_group releases their
// group, so reading shared memory without waiting also reads stale data. Vector and word loads
// through the kurn_ld* helpers check alignment (a misaligned 16-byte load faults on a GPU).
//
// Blocks run one after another on the calling OS thread, so `__shared__` maps to `static`.
// Only the subset of CUDA that KURN's code generator emits is provided.
#pragma once
#include <ucontext.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <functional>
#include <vector>

#define KURN_EMU 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __constant__ static const
#define __shared__ static
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return uint4{x, y, z, w}; }
typedef void *cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0

namespace kemu {

enum State { RUN, BAR, WARP, DONE };
enum Op { OP_SHFL_XOR, OP_SHFL_IDX, OP_SYNCWARP, OP_MMA_S8, OP_MMA_F16, OP_MMA_BF16, OP_MMA_E4M3, OP_LDSM };

struct CpAsync {
  void *dst;
  unsigned char src[16];
  int n;
};

struct Thread {
  ucontext_t ctx;
  dim3 tid;
  int linear = 0;
  State state = RUN;
  Op op = OP_SYNCWARP;
  uint32_t in[12];
  uint32_t out[4];
  std::vector<std::vector<CpAsync>> groups;  // committed cp.async groups, oldest first
  std::vector<CpAsync> open;                 // issued, not yet committed
};

struct Ctx {
  std::vector<Thread> threads;
  std::vector<char *> stacks;
  ucontext_t sched;
  dim3 bid, bdim, gdim;
  Thread *cur = nullptr;
  std::function<void()> body;
  uint64_t seed = 0;
};

inline Ctx &ctx() {
  static Ctx c;
  return c;
}

[[noreturn]] inline void fail(const char *msg) {
  Ctx &c = ctx();
  fprintf(stderr, "kurn_cuemu: %s (block %u,%u,%u", msg, c.bid.x, c.bid.y, c.bid.z);
  if (c.cur) fprintf(stderr, ", thread %d", c.cur->linear);
  fprintf(stderr, ")\n");
  exit(70);
}

inline void yield_to_sched() {
  Ctx &c = ctx();
  swapcontext(&c.cur->ctx, &c.sched);
}

inline void entry() {
  Ctx &c = ctx();
  c.body();
  if (!c.cur->open.empty() || !c.cur->groups.empty()) {
    // a kernel may legally exit with copies in flight only if it never reads them; apply them anyway
    for (auto &g : c.cur->groups)
      for (auto &e : g) memcpy(e.dst, e.src, e.n);
    for (auto &e : c.cur->open) memcpy(e.dst, e.src, e.n);
  }
  c.cur->state = DONE;
  swapcontext(&c.cur->ctx, &c.sched);
}

float h2f_bits(uint16_t h);
// FP8 E4M3 (e4m3fn: bias 7, no infinity, 0x7F / 0xFF NaN, max 448)
inline float e4m3_to_f(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  if (e == 15 && m == 7) return NAN;
  const float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf((float)m, -9);
  return s ? -v : v;
}
inline float bf2f_bits(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

inline void resolve_warp(std::vector<Thread> &t, int w0) {
  Thread *L = &t[w0];
  Op op = L[0].op;
  for (int l = 1; l < 32; l++)
    if (L[l].op != op) fail("lanes of one warp reached different collectives");
  if (op == OP_MMA_E4M3) {
    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32: the byte fragment layouts of the int8 m16n8k32 MMA
    float A[16][32], B[32][8], C[16][8];
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int i = 0; i < 16; i++)
        A[g + 8 * ((i / 4) & 1)][4 * tq + (i & 3) + (i >= 8 ? 16 : 0)] = e4m3_to_f((uint8_t)(L[l].in[i / 4] >> (8 * (i & 3))));
      for (int i = 0; i < 8; i++) B[4 * tq + (i & 3) + (i >= 4 ? 16 : 0)][g] = e4m3_to_f((uint8_t)(L[l].in[4 + i / 4] >> (8 * (i & 3))));
      for (int i = 0; i < 4; i++) memcpy(&C[g + 8 * (i >= 2)][2 * tq + (i & 1)], &L[l].in[6 + i], 4);
    }
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int i = 0; i < 4; i++) {
        int r = g + 8 * (i >= 2), col = 2 * tq + (i & 1);
        float acc = C[r][col];
        for (int k = 0; k < 32; k++) acc += A[r][k] * B[k][col];  // e4m3 x e4m3 products are exact in f32
        memcpy(&L[l].out[i], &acc, 4);
      }
    }
    for (int l = 0; l < 32; l++) L[l].state = RUN;
    return;
  }
  if (op == OP_SHFL_XOR || op == OP_SHFL_IDX) {
    for (int l = 0; l < 32; l++) {
      int src = op == OP_SHFL_XOR ? (l ^ (int)L[l].in[1]) : ((int)L[l].in[1] & 31);
      int width = (int)L[l].in[2];
      if (op == OP_SHFL_IDX) src = (l & ~(width - 1)) | (src & (width - 1));
      if (src < 0 || src > 31) fail("shuffle source lane out of range");
      L[l].out[0] = L[src].in[0];
    }
  } else if (op == OP_MMA_S8) {
    // mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, PTX ISA fragment layouts:
    //   A (16x32, row): a_i (byte i of regs 0..3): row = g + 8*((i/4)&1), col = 4t + (i&3) + 16*(i>=8)
    //   B (32x8, col):  b_i (byte i of regs 0..1): row = 4t + (i&3) + 16*(i>=4), col = g
    //   C/D (16x8):     c_i: row = g + 8*(i>=2), col = 2t + (i&1)
    int8_t A[16][32], B[32][8];
    int32_t C[16][8];
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int i = 0; i < 16; i++) {
        int8_t v = (int8_t)(L[l].in[i / 4] >> (8 * (i & 3)));
        A[g + 8 * ((i / 4) & 1)][4 * tq + (i & 3) + (i >= 8 ? 16 : 0)] = v;
      }
      for (int i = 0; i < 8; i++) {
        int8_t v = (int8_t)(L[l].in[4 + i / 4] >> (8 * (i & 3)));
        B[4 * tq + (i & 3) + (i >= 4 ? 16 : 0)][g] = v;
      }
      for (int i = 0; i < 4; i++) C[g + 8 * (i >= 2)][2 * tq + (i & 1)] = (int32_t)L[l].in[6 + i];
    }
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int i = 0; i < 4; i++) {
        int r = g + 8 * (i >= 2), col = 2 * tq + (i & 1);
        int32_t acc = C[r][col];
        for (int k = 0; k < 32; k++) acc += (int32_t)A[r][k] * (int32_t)B[k][col];
        L[l].out[i] = (uint32_t)acc;
      }
    }
  }
  else if (op == OP_MMA_F16 || op == OP_MMA_BF16) {
    // mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32 fragment layouts (PTX ISA, same for both types):
    //   A (16x16): reg r holds 2 halves (lo, hi) at row g + 8*(r&1), col 2t + (lo/hi) + 8*(r>>1)
    //   B (16x8):  reg r holds 2 halves at row(k) 2t + (lo/hi) + 8*r, col g
    //   C/D (16x8): c_i at row g + 8*(i>=2), col 2t + (i&1)
    float (*cvt)(uint16_t) = op == OP_MMA_F16 ? h2f_bits : bf2f_bits;
    float A[16][16], B[16][8], C[16][8];
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int r = 0; r < 4; r++)
        for (int h = 0; h < 2; h++) A[g + 8 * (r & 1)][2 * tq + h + 8 * (r >> 1)] = cvt((uint16_t)(L[l].in[r] >> (16 * h)));
      for (int r = 0; r < 2; r++)
        for (int h = 0; h < 2; h++) B[2 * tq + h + 8 * r][g] = cvt((uint16_t)(L[l].in[4 + r] >> (16 * h)));
      for (int i = 0; i < 4; i++) {
        float f;
        memcpy(&f, &L[l].in[6 + i], 4);
        C[g + 8 * (i >= 2)][2 * tq + (i & 1)] = f;
      }
    }
    for (int l = 0; l < 32; l++) {
      int g = l >> 2, tq = l & 3;
      for (int i = 0; i < 4; i++) {
        int r = g + 8 * (i >= 2), col = 2 * tq + (i & 1);
        float acc = C[r][col];
        for (int k = 0; k < 16; k++) acc += A[r][k] * B[k][col];  // f16 x f16 and bf16 x bf16 products are exact in f32
        memcpy(&L[l].out[i], &acc, 4);
      }
    }
  } else if (op == OP_LDSM) {
    // ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16: lanes 8j..8j+7 give the row addresses of matrix j;
    // lane l receives, for matrix j, the halves (row l/4, cols 2(l%4), 2(l%4)+1) (or transposed with .trans)
    int n = (int)L[0].in[2], trans = (int)L[0].in[3];
    for (int l = 1; l < 32; l++)
      if ((int)L[l].in[2] != n || (int)L[l].in[3] != trans) fail("ldmatrix: lanes disagree on .num/.trans");
    for (int j = 0; j < n; j++) {
      const uint16_t *row[8];
      for (int r = 0; r < 8; r++) {
        uint64_t a = (uint64_t)L[8 * j + r].in[0] | ((uint64_t)L[8 * j + r].in[1] << 32);
        if (a % 16) fail("ldmatrix: row address not 16-byte aligned");
        row[r] = (const uint16_t *)(uintptr_t)a;
      }
      for (int l = 0; l < 32; l++) {
        int rr = l >> 2, cc = 2 * (l & 3);
        uint16_t lo = trans ? row[cc][rr] : row[rr][cc], hi = trans ? row[cc + 1][rr] : row[rr][cc + 1];
        L[l].out[j] = (uint32_t)lo | ((uint32_t)hi << 16);
      }
    }
  }
  for (int l = 0; l < 32; l++) L[l].state = RUN;
}

inline void launch(dim3 grid, dim3 block, std::function<void()> body) {
  Ctx &c = ctx();
  int nt = (int)(block.x * block.y * block.z);
  if (nt <= 0 || nt > 1024) fail("block size must be 1..1024");
  if (nt % 32) fail("emulator needs whole warps (block size a multiple of 32)");
  const size_t STACK = 256 * 1024;
  while ((int)c.stacks.size() < nt) c.stacks.push_back((char *)malloc(STACK));
  c.bdim = block;
  c.gdim = grid;
  if (const char *s = getenv("KEMU_SEED")) c.seed = strtoull(s, nullptr, 10) * 2654435761ULL + 1;
  c.body = body;
  c.threads.assign(nt, Thread());
  for (unsigned bz = 0; bz < grid.z; bz++)
    for (unsigned by = 0; by < grid.y; by++)
      for (unsigned bx = 0; bx < grid.x; bx++) {
        c.bid = dim3(bx, by, bz);
        for (int i = 0; i < nt; i++) {
          Thread &t = c.threads[i];
          t = Thread();
          t.linear = i;
          t.tid = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
          getcontext(&t.ctx);
          t.ctx.uc_stack.ss_sp = c.stacks[i];
          t.ctx.uc_stack.ss_size = STACK;
          t.ctx.uc_link = nullptr;
          makecontext(&t.ctx, (void (*)())entry, 0);
        }
        int live = nt;
        while (live) {
          bool ran = false;
          // KEMU_SEED: random thread order and random delays, so warps drift apart and races
          // (write-after-read on shared memory, missing barriers) are exposed
          std::vector<int> order(nt);
          for (int i = 0; i < nt; i++) order[i] = i;
          if (c.seed) {
            for (int i = nt - 1; i > 0; i--) {
              c.seed = c.seed * 6364136223846793005ULL + 1442695040888963407ULL;
              int j = (int)((c.seed >> 33) % (uint64_t)(i + 1));
              int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
          }
          // KEMU_SEED also holds back whole warps for a pass, so warps drift apart and cross-warp races on shared
          // memory (a missing __syncthreads between a producer warp and a consumer warp) are exposed
          uint64_t warp_hold = 0;
          if (c.seed && nt > 32) {
            for (int w = 0; w < nt / 32 && w < 64; w++) {
              c.seed = c.seed * 6364136223846793005ULL + 1442695040888963407ULL;
              if ((c.seed >> 41) % 3 == 0) warp_hold |= 1ULL << w;
            }
            if (warp_hold == ((nt / 32 >= 64) ? ~0ULL : (1ULL << (nt / 32)) - 1)) warp_hold &= warp_hold - 1;
          }
          for (int ii = 0; ii < nt; ii++) {
            Thread &t = c.threads[order[ii]];
            if (t.state != RUN) continue;
            if (warp_hold >> ((order[ii] / 32) & 63) & 1) continue;
            if (c.seed) {
              c.seed = c.seed * 6364136223846793005ULL + 1442695040888963407ULL;
              if ((c.seed >> 40) % 3 == 0 && ii + 1 < nt) continue;  // delay this thread one pass
            }
            c.cur = &t;
            swapcontext(&c.sched, &t.ctx);
            ran = true;
          }
          c.cur = nullptr;
          live = 0;
          int at_bar = 0;
          for (auto &t : c.threads) live += t.state != DONE, at_bar += t.state == BAR;
          bool released = false;
          for (int w = 0; w < nt; w += 32) {
            int n_warp = 0, n_done = 0;
            for (int l = 0; l < 32; l++) n_warp += c.threads[w + l].state == WARP, n_done += c.threads[w + l].state == DONE;
            if (n_warp == 32) {
              resolve_warp(c.threads, w);
              released = true;
            } else if (n_warp && n_done) {
              fail("warp collective with exited lanes (full-mask collectives need all 32 lanes)");
            }
          }
          if (!released && live && at_bar == live) {
            for (auto &t : c.threads)
              if (t.state == BAR) t.state = RUN;
            released = true;
          }
          bool runnable = false;
          for (auto &t : c.threads) runnable |= t.state == RUN;
          if (live && !ran && !released && !runnable)
            fail("deadlock: threads wait at different barriers (divergent __syncthreads or warp op)");
        }
      }
}

inline uint32_t warp_op(Op op, const uint32_t *in, int n) {
  Ctx &c = ctx();
  Thread *t = c.cur;
  t->op = op;
  for (int i = 0; i < n; i++) t->in[i] = in[i];
  t->state = WARP;
  yield_to_sched();
  return t->out[0];
}

inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

}  // namespace kemu

#define threadIdx (kemu::ctx().cur->tid)
#define blockIdx (kemu::ctx().bid)
#define blockDim (kemu::ctx().bdim)
#define gridDim (kemu::ctx().gdim)

inline void __syncthreads() {
  kemu::ctx().cur->state = kemu::BAR;
  kemu::yield_to_sched();
}
inline void __syncwarp(unsigned = 0xffffffffu) { kemu::warp_op(kemu::OP_SYNCWARP, nullptr, 0); }

inline unsigned __shfl_xor_sync(unsigned, unsigned v, int m, int w = 32) {
  uint32_t in[3] = {v, (uint32_t)m, (uint32_t)w};
  return kemu::warp_op(kemu::OP_SHFL_XOR, in, 3);
}
inline int __shfl_xor_sync(unsigned mask, int v, int m, int w = 32) { return (int)__shfl_xor_sync(mask, (unsigned)v, m, w); }
inline float __shfl_xor_sync(unsigned mask, float v, int m, int w = 32) {
  return kemu::u2f(__shfl_xor_sync(mask, kemu::f2u(v), m, w));
}
inline unsigned __shfl_sync(unsigned, unsigned v, int src, int w = 32) {
  uint32_t in[3] = {v, (uint32_t)src, (uint32_t)w};
  return kemu::warp_op(kemu::OP_SHFL_IDX, in, 3);
}
inline int __shfl_sync(unsigned mask, int v, int src, int w = 32) { return (int)__shfl_sync(mask, (unsigned)v, src, w); }
inline float __shfl_sync(unsigned mask, float v, int src, int w = 32) { return kemu::u2f(__shfl_sync(mask, kemu::f2u(v), src, w)); }

inline int __dp4a(int a, int b, int c) {
  for (int i = 0; i < 4; i++) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
  return c;
}
inline unsigned __vsub4(unsigned a, unsigned b) {
  unsigned r = 0;
  for (int i = 0; i < 4; i++) r |= ((((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF)) & 0xFF) << (8 * i);
  return r;
}
inline unsigned __vadd4(unsigned a, unsigned b) {
  unsigned r = 0;
  for (int i = 0; i < 4; i++) r |= ((((a >> (8 * i)) & 0xFF) + ((b >> (8 * i)) & 0xFF)) & 0xFF) << (8 * i);
  return r;
}
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
  uint64_t v = ((uint64_t)y << 32) | x;
  unsigned r = 0;
  for (int i = 0; i < 4; i++) {
    unsigned sel = (s >> (4 * i)) & 0xF;
    if (sel & 8) kemu::fail("__byte_perm sign-replicate mode is not emulated");
    r |= (unsigned)((v >> (8 * sel)) & 0xFF) << (8 * i);
  }
  return r;
}
inline int __float2int_rn(float f) { return (int)std::nearbyintf(f); }
inline unsigned __float_as_uint(float f) { return kemu::f2u(f); }
inline float __uint_as_float(unsigned u) { return kemu::u2f(u); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

// IEEE half <-> float, round to nearest even (same as __float2half_rn and F16C).
inline float kemu_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF, u;
  if (e == 0) {
    if (m == 0) {
      u = s;
    } else {
      e = 127 - 15 + 1;
      while (!(m & 0x400)) m <<= 1, e--;
      u = s | (e << 23) | ((m & 0x3FF) << 13);
    }
  } else if (e == 31) {
    u = s | 0x7F800000 | (m << 13);
  } else {
    u = s | ((e + 112) << 23) | (m << 13);
  }
  return kemu::u2f(u);
}
inline uint16_t kemu_f2h(float f) {
  uint32_t x = kemu::f2u(f), s = (x >> 16) & 0x8000;
  int e = (int)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(s | 0x7C00 | (m ? 0x200 : 0));
  if (e >= 31) return (uint16_t)(s | 0x7C00);
  if (e <= 0) {
    if (e < -10) return (uint16_t)s;
    m |= 0x800000;
    int shift = 14 - e;
    uint32_t hm = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1))) hm++;
    return (uint16_t)(s | hm);
  }
  uint32_t hm = m >> 13, rem = m & 0x1FFF;
  uint32_t h = s | ((uint32_t)e << 10) | hm;
  if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) h++;
  return (uint16_t)h;
}

inline float kemu::h2f_bits(uint16_t h) { return kemu_h2f(h); }

// f16 x2 arithmetic, rounded to nearest even per lane (exact for the small integers KURN's dequantizers produce)
inline uint32_t kemu_h2op(uint32_t a, uint32_t b, uint32_t c, int op) {
  uint32_t r = 0;
  for (int h = 0; h < 2; h++) {
    float x = kemu_h2f((uint16_t)(a >> 16 * h)), y = kemu_h2f((uint16_t)(b >> 16 * h)), z = kemu_h2f((uint16_t)(c >> 16 * h));
    double v = op == 0 ? (double)x * y + z : op == 1 ? (double)x - y : op == 2 ? (double)x + y : (double)x * y;
    r |= (uint32_t)kemu_f2h((float)v) << (16 * h);
  }
  return r;
}
inline uint32_t kemu_hfma2(uint32_t a, uint32_t b, uint32_t c) { return kemu_h2op(a, b, c, 0); }
inline uint32_t kemu_hsub2(uint32_t a, uint32_t b) { return kemu_h2op(a, b, 0, 1); }
inline uint32_t kemu_hadd2(uint32_t a, uint32_t b) { return kemu_h2op(a, b, 0, 2); }
inline uint32_t kemu_cvt_f16x2(float hi, float lo) { return (uint32_t)kemu_f2h(lo) | ((uint32_t)kemu_f2h(hi) << 16); }

inline void kemu_mma_f16_16816(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
  uint32_t in[10] = {a[0], a[1], a[2], a[3], b[0], b[1], 0, 0, 0, 0};
  memcpy(&in[6], c, 16);
  kemu::warp_op(kemu::OP_MMA_F16, in, 10);
  memcpy(d, kemu::ctx().cur->out, 16);
}
inline void kemu_mma_bf16_16816(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
  uint32_t in[10] = {a[0], a[1], a[2], a[3], b[0], b[1], 0, 0, 0, 0};
  memcpy(&in[6], c, 16);
  kemu::warp_op(kemu::OP_MMA_BF16, in, 10);
  memcpy(d, kemu::ctx().cur->out, 16);
}
// f32 -> bf16, round to nearest even (same as cvt.rn.bf16x2.f32; NaN stays NaN)
inline uint16_t kemu_f2bf(float f) {
  uint32_t u = kemu::f2u(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
inline uint32_t kemu_cvt_bf16x2(float hi, float lo) { return (uint32_t)kemu_f2bf(lo) | ((uint32_t)kemu_f2bf(hi) << 16); }
inline void kemu_mma_e4m3_16832(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
  uint32_t in[10] = {a[0], a[1], a[2], a[3], b[0], b[1], 0, 0, 0, 0};
  memcpy(&in[6], c, 16);
  kemu::warp_op(kemu::OP_MMA_E4M3, in, 10);
  memcpy(d, kemu::ctx().cur->out, 16);
}
// f32 -> e4m3, round to nearest even, saturating to +-448 (cvt.rn.satfinite.e4m3x2.f32); NaN -> 0x7F
inline uint8_t kemu_f2e4m3(float x) {
  if (x != x) return 0x7F;
  const uint8_t s = std::signbit(x) ? 0x80 : 0;
  const float a = fabsf(x);
  if (a >= 448.f) return s | 0x7E;
  if (a < 0.015625f) return s | (uint8_t)std::nearbyintf(a * 512.f);  // subnormal (q = 8 is the smallest normal, code 8)
  int e;
  frexpf(a, &e);  // a = f * 2^e, f in [0.5, 1)
  e -= 1;         // a = 1.m * 2^e
  int q = (int)std::nearbyintf(ldexpf(a, 3 - e));  // 8 .. 16
  if (q == 16) q = 8, e++;
  if (e + 7 > 15 || (e + 7 == 15 && q - 8 == 7)) return s | 0x7E;
  return s | (uint8_t)(((e + 7) << 3) | (q - 8));
}
inline uint32_t kemu_cvt_e4m3x2(float hi, float lo) { return (uint32_t)kemu_f2e4m3(lo) | ((uint32_t)kemu_f2e4m3(hi) << 8); }
// e4m3x2 -> f16x2 (cvt.rn.f16x2.e4m3x2; exact): low byte -> low half
inline uint32_t kemu_cvt_f16x2_e4m3x2(uint32_t v) {
  return (uint32_t)kemu_f2h(kemu::e4m3_to_f((uint8_t)v)) | ((uint32_t)kemu_f2h(kemu::e4m3_to_f((uint8_t)(v >> 8))) << 16);
}
inline void kemu_ldmatrix(unsigned *r, const void *addr, int n, int trans) {
  uint64_t a = (uint64_t)(uintptr_t)addr;
  uint32_t in[4] = {(uint32_t)a, (uint32_t)(a >> 32), (uint32_t)n, (uint32_t)trans};
  kemu::warp_op(kemu::OP_LDSM, in, 4);
  for (int i = 0; i < n; i++) r[i] = kemu::ctx().cur->out[i];
}

// atomics and fences: fibers only switch at barriers and warp collectives, so plain operations are atomic here
inline int atomicAdd(int *p, int v) { int o = *p; *p = o + v; return o; }
inline int atomicExch(int *p, int v) { int o = *p; *p = v; return o; }
inline void __threadfence() {}
inline int kemu_sm_count() { const char *s = getenv("KEMU_SMS"); return s ? atoi(s) : 8; }

// int8 tensor-core MMA as a warp collective
inline void kemu_mma_s8_16832(int d[4], const unsigned a[4], const unsigned b[2], const int c[4]) {
  uint32_t in[10] = {a[0], a[1], a[2], a[3], b[0], b[1], (uint32_t)c[0], (uint32_t)c[1], (uint32_t)c[2], (uint32_t)c[3]};
  kemu::warp_op(kemu::OP_MMA_S8, in, 10);
  kemu::Thread *t = kemu::ctx().cur;
  for (int i = 0; i < 4; i++) d[i] = (int)t->out[i];
}

// A cp.async copy lands at its wait_group, so reading shared memory without waiting reads stale data. Under KEMU_SEED
// half of the copies land at issue instead (also legal on hardware), so refilling a buffer that other warps still
// read (a missing barrier before the next stage's copies) overwrites their data.
inline void kemu_cp_issue(const kemu::CpAsync &e) {
  kemu::Ctx &c = kemu::ctx();
  if (c.seed) {
    c.seed = c.seed * 6364136223846793005ULL + 1442695040888963407ULL;
    if ((c.seed >> 45) & 1) {
      memcpy(e.dst, e.src, e.n);
      return;
    }
  }
  c.cur->open.push_back(e);
}
// cp.async (16 bytes, zero-filled past src_size), deferred until wait_group
inline void kemu_cp_async16(void *dst, const void *src, int src_size) {
  if ((uintptr_t)dst % 16 || (src_size && (uintptr_t)src % 16)) kemu::fail("misaligned cp.async (16-byte copies need 16-byte alignment)");
  kemu::CpAsync e;
  e.dst = dst;
  e.n = 16;
  memset(e.src, 0, 16);
  if (src_size) memcpy(e.src, src, src_size);
  kemu_cp_issue(e);
}
// cp.async.ca 4-byte copy (zero-filled when src_size == 0), deferred until wait_group
inline void kemu_cp_async4(void *dst, const void *src, int src_size) {
  if ((uintptr_t)dst % 4 || (src_size && (uintptr_t)src % 4)) kemu::fail("misaligned cp.async (4-byte copies need 4-byte alignment)");
  kemu::CpAsync e;
  e.dst = dst;
  e.n = 4;
  memset(e.src, 0, 16);
  if (src_size) memcpy(e.src, src, src_size);
  kemu_cp_issue(e);
}
inline void kemu_cp_commit() {
  kemu::Thread *t = kemu::ctx().cur;
  t->groups.push_back(t->open);
  t->open.clear();
}
inline void kemu_cp_wait(int pending) {
  kemu::Thread *t = kemu::ctx().cur;
  while ((int)t->groups.size() > pending) {
    for (auto &e : t->groups.front()) memcpy(e.dst, e.src, e.n);
    t->groups.erase(t->groups.begin());
  }
}

inline void kemu_check_align(const void *p, int n, const char *what) {
  if ((uintptr_t)p % n) {
    char msg[128];
    snprintf(msg, sizeof msg, "misaligned %s (address %% %d = %d)", what, n, (int)((uintptr_t)p % n));
    kemu::fail(msg);
  }
}

#define KURN_LAUNCH(kernel, grid, block, stream, ...) kemu::launch((grid), (block), [=]() { kernel(__VA_ARGS__); })
#define KURN_LAUNCH_SMEM(kernel, grid, block, smem, stream, ...) kemu::launch((grid), (block), [=]() { kernel(__VA_ARGS__); })
