"""CUDA C++ generation for `target cuda`.

One resolved config -> one self-contained `.cu` file implementing the kurn_gpu.h ABI:

    kg_config()                         the config string
    kg_check_shape(N, K, M)             0 if the kernel supports the shape
    kg_prep_bytes(N, K)                 bytes of the prepared (repacked) weights; 0 = use W as is
    kg_prepare(W, Wp, N, K, s)          native ggml blocks -> prepared layout (layout split)
    kg_xbytes(K, M)                     bytes of the quantized activations in the kernel's layout
    kg_quant(X, Xq, K, M, s)            f32 activations -> the kernel's activation layout
    kg_xblock_bytes(K, M)               bytes of M rows of ggml activation blocks
    kg_xblocks(X, Xb, K, M, s)          f32 activations -> ggml activation blocks (q8_0 / q8_K), for references
    kg_run(W, Xq, Y, N, K, M, s)        Y[m * N + n] = dot(W row n, activation m)

All pointers are device pointers. The same file compiles with nvcc, and with a host C++
compiler against data/kurn_cuemu.h (-DKURN_EMU) to run the kernels on the CPU emulator.

Only sm_80 features are emitted (dp4a, int8 mma.sync m16n8k32, cp.async), so the code runs on
Ampere, Ada, Hopper and Blackwell.
"""

from .spec import ACT, FORMATS, codegen_keys, gemm_smem, threads

KV_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

PRELUDE = r"""#ifdef KURN_EMU
#include "kurn_cuemu.h"
#define KURN_H2F(h) kemu_h2f((uint16_t)(h))
#define KURN_F2H(f) kemu_f2h(f)
#define KURN_POPC(x) __builtin_popcount(x)
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>
#define KURN_H2F(h) __half2float(__ushort_as_half((unsigned short)(h)))
#define KURN_F2H(f) __half_as_ushort(__float2half_rn(f))
#define KURN_POPC(x) __popc(x)
#define KURN_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#define KURN_LAUNCH_SMEM(kernel, grid, block, smem, stream, ...) kernel<<<(grid), (block), (smem), (stream)>>>(__VA_ARGS__)
#endif
#define KURN_FULL 0xffffffffu
#define KURN_FN static __device__ __forceinline__ __attribute__((unused))

KURN_FN uint32_t kld_u8(const uint8_t *p) { return *p; }
KURN_FN uint32_t kld_u16(const uint8_t *p) {
#ifdef KURN_EMU
  kemu_check_align(p, 2, "16-bit load");
  uint16_t v; memcpy(&v, p, 2); return v;
#else
  return *(const uint16_t *)p;
#endif
}
KURN_FN uint32_t kld_u32(const void *p) {
#ifdef KURN_EMU
  kemu_check_align(p, 4, "32-bit load");
  uint32_t v; memcpy(&v, p, 4); return v;
#else
  return *(const uint32_t *)p;
#endif
}
KURN_FN uint32_t kld_u32_b2(const uint8_t *p) { return kld_u16(p) | (kld_u16(p + 2) << 16); }
KURN_FN uint2 kld_v2(const void *p) {
#ifdef KURN_EMU
  kemu_check_align(p, 8, "8-byte vector load");
  uint2 v; memcpy(&v, p, 8); return v;
#else
  return *(const uint2 *)p;
#endif
}
KURN_FN uint4 kld_v4(const void *p) {
#ifdef KURN_EMU
  kemu_check_align(p, 16, "16-byte vector load");
  uint4 v; memcpy(&v, p, 16); return v;
#else
  return *(const uint4 *)p;
#endif
}
KURN_FN float kld_f32(const uint8_t *p) { return __uint_as_float(kld_u32(p)); }
KURN_FN void kst_u16(uint8_t *p, uint32_t v) {
#ifdef KURN_EMU
  kemu_check_align(p, 2, "16-bit store");
  uint16_t x = (uint16_t)v; memcpy(p, &x, 2);
#else
  *(uint16_t *)p = (uint16_t)v;
#endif
}
KURN_FN void kst_u32(void *p, uint32_t v) {
#ifdef KURN_EMU
  kemu_check_align(p, 4, "32-bit store");
  memcpy(p, &v, 4);
#else
  *(uint32_t *)p = v;
#endif
}
"""

MMA = r"""
// int8 tensor cores: D = A (16x32, row) * B (32x8, col), s32 accumulate (sm_80+)
KURN_FN void kmma(int d[4], const unsigned a[4], const unsigned b[2]) {
#ifdef KURN_EMU
  const int z[4] = {0, 0, 0, 0};
  kemu_mma_s8_16832(d, a, b, z);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0), "r"(0), "r"(0));
#endif
}
"""

CPASYNC = r"""
// 16-byte async global->shared copy, zero-filled when bytes == 0 (sm_80+)
KURN_FN void kcp16(void *dst, const void *src, int bytes) {
#ifdef KURN_EMU
  kemu_cp_async16(dst, src, bytes);
#else
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
#endif
}
KURN_FN void kcp_commit() {
#ifdef KURN_EMU
  kemu_cp_commit();
#else
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
#ifdef KURN_EMU
#define KCP_WAIT(n) kemu_cp_wait(n)
#else
#define KCP_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#endif
"""


def _u32(bs):
    return [int.from_bytes(bytes(b & 0xFF for b in bs[i : i + 4]), "little") for i in range(0, len(bs), 4)]


def _iq4_table():
    t = _u32(KV_IQ4NL)
    return (
        "// IQ4_NL: 4 nibble indices (one per byte) -> 4 int8 codebook values\n"
        "KURN_FN uint32_t kiq4(uint32_t q) {\n"
        "  const uint32_t sel = (q & 0x7) | ((q >> 4) & 0x70) | ((q >> 8) & 0x700) | ((q >> 12) & 0x7000);\n"
        f"  const uint32_t lo = __byte_perm(0x{t[0]:08x}u, 0x{t[1]:08x}u, sel), hi = __byte_perm(0x{t[2]:08x}u, 0x{t[3]:08x}u, sel);\n"
        "  const uint32_t m = ((q >> 3) & 0x01010101u) * 0xFFu;\n"
        "  return (lo & ~m) | (hi & m);\n"
        "}\n"
    )


def _e8p_table():
    from ..ext.compress import e8p_abs_u64

    words = []
    for v in e8p_abs_u64():
        words += [v & 0xFFFFFFFF, v >> 32]
    lines = ",\n".join("  " + ", ".join(f"0x{w:08x}u" for w in words[i : i + 8]) for i in range(0, len(words), 8))
    return (
        "// E8P: 256 rows x 8 bytes of 4|a_i| (2, 6, 10), rows 0-127 even coordinate parity\n"
        f"__constant__ uint32_t kE8P[512] = {{\n{lines}}};\n"
        "KURN_FN uint32_t kspread4(uint32_t n) { return (n | (n << 7) | (n << 14) | (n << 21)) & 0x01010101u; }\n"
        "// one 8-weight vector (code byte lo, sign byte hi) -> 2 words of q = 4c (odd ints in [-11, 11])\n"
        "KURN_FN void ke8p(const uint32_t *tab, uint32_t lo, uint32_t hi, uint32_t &w0, uint32_t &w1) {\n"
        "  const uint32_t row = (lo & 0x7F) | ((KURN_POPC(hi) & 1) << 7);\n"
        "  const uint32_t a0 = tab[2 * row], a1 = tab[2 * row + 1];\n"
        "  const uint32_t m0 = kspread4(hi & 0xF) * 0xFFu, m1 = kspread4(hi >> 4) * 0xFFu;\n"
        "  const uint32_t t = (lo & 0x80) ? 0x01010101u : 0xFFFFFFFFu;\n"
        "  w0 = __vadd4((a0 & ~m0) | (__vsub4(0u, a0) & m0), t);\n"
        "  w1 = __vadd4((a1 & ~m1) | (__vsub4(0u, a1) & m1), t);\n"
        "}\n"
    )


def _act_macros(act, xs=False):
    if act == "q8_0" and xs:  # split: int8 plane [M][K] + float scale plane [M][K/32]
        return "#define XD(x, c) ((x)[c])\n#define XP(x, c) ((x) + (size_t)(c) * 32)\n"
    if act == "q8_0":
        return (
            "#define XD(x, c) KURN_H2F(kld_u16((x) + (size_t)(c) * 34))\n"
            "#define XQ(x, c, off) (int)kld_u32_b2((x) + (size_t)(c) * 34 + 2 + (off))\n"
        )
    return (
        "#define XD8K(x, b) kld_f32((x) + (size_t)(b) * 292)\n"
        "#define XQ(x, c, off) (int)kld_u32((x) + (size_t)((c) >> 3) * 292 + 4 + ((c) & 7) * 32 + (off))\n"
        "#define XBS(x, c, h) (int)(int16_t)kld_u16((x) + (size_t)((c) >> 3) * 292 + 260 + 2 * (((c) & 7) * 2 + (h)))\n"
    )


def _loadq(dst, ptr, nb, aligned):
    """C statements loading nb quant bytes at ptr into uint32_t dst[nb/4] (or dst[0] for nb < 4)."""
    if nb < 4:
        return [f"uint32_t {dst}[1] = {{{'kld_u16' if nb == 2 else 'kld_u8'}({ptr})}};"]
    nw = nb // 4
    out = [f"uint32_t {dst}[{nw}];"]
    if not aligned:
        out.append(f"for (int i = 0; i < {nw}; i++) {dst}[i] = kld_u32_b2({ptr} + 4 * i);")
    elif nb >= 16:
        for v in range(nb // 16):
            out.append(f"{{ const uint4 t = kld_v4({ptr} + {16 * v}); {dst}[{4 * v}] = t.x; {dst}[{4 * v + 1}] = t.y; "
                       f"{dst}[{4 * v + 2}] = t.z; {dst}[{4 * v + 3}] = t.w; }}")  # fmt: skip
    elif nb == 8:
        out.append(f"{{ const uint2 t = kld_v2({ptr}); {dst}[0] = t.x; {dst}[1] = t.y; }}")
    else:
        out.append(f"{dst}[0] = kld_u32({ptr});")
    return out


def _xload(dst, ptr, nb):
    """Aligned vector loads of nb activation bytes (xlayout split) into uint32_t dst[], indented for a COLS loop."""
    return ["  " + ln for ln in _loadq(dst, ptr, nb, True)]


def row_bytes(fmt, k):
    f = FORMATS[fmt]
    return k // f["block"] * f["nbytes"]


# --------------------------------------------------------------------------- GEMV units
# Each emitter returns the body of
#   kunit(const uint8_t *wq, const uint8_t *ws, int u, int p, const uint8_t *const *xr, float *acc, tab)
# accumulating one lane's share (part p of SUB) of work unit u into acc[COLS].


def _unit_q8_0(c, split):
    nb = 32 // c["sub"]
    if split:
        s = _loadq("q", "wq + (size_t)u * 32 + p * %d" % nb, nb, True) + ["const float dw = KURN_H2F(kld_u16(ws + 2 * (size_t)u));"]
    else:
        s = ["const uint8_t *b = wq + (size_t)u * 34;", "const float dw = KURN_H2F(kld_u16(b));"]
        s += _loadq("q", "b + 2 + p * %d" % nb, nb, False)
    s += ["#pragma unroll", "for (int j = 0; j < COLS; j++) {"]
    if c["xlayout"] == "split":
        s += _xload("xa", f"XP(xr[j], u) + p * {nb}", nb)
        x, xd = "(int)xa[i]", "xd[j]"
    else:
        x, xd = f"XQ(xr[j], u, p * {nb} + 4 * i)", "xr[j]"
    s += [
        "  int s = 0;",
        "#pragma unroll",
        f"  for (int i = 0; i < {nb // 4}; i++) s = __dp4a((int)q[i], {x}, s);",
        f"  acc[j] += dw * XD({xd}, u) * (float)s;",
        "}",
    ]
    return s


def _unit_q4(c, split, iq4):
    nb = 16 // c["sub"]
    if split:
        s = _loadq("q", "wq + (size_t)u * 16 + p * %d" % nb, nb, True) + ["const float dw = KURN_H2F(kld_u16(ws + 2 * (size_t)u));"]
    else:
        s = ["const uint8_t *b = wq + (size_t)u * 18;", "const float dw = KURN_H2F(kld_u16(b));"]
        s += _loadq("q", "b + 2 + p * %d" % nb, nb, False)
    if iq4:
        dec = ("kiq4(q[i] & 0x0F0F0F0Fu)", "kiq4((q[i] >> 4) & 0x0F0F0F0Fu)")
    else:
        dec = ("__vsub4(q[i] & 0x0F0F0F0Fu, 0x08080808u)", "__vsub4((q[i] >> 4) & 0x0F0F0F0Fu, 0x08080808u)")
    s += [
        f"uint32_t lo[{nb // 4}], hi[{nb // 4}];",
        "#pragma unroll",
        f"for (int i = 0; i < {nb // 4}; i++) {{ lo[i] = {dec[0]}; hi[i] = {dec[1]}; }}",
        "#pragma unroll",
        "for (int j = 0; j < COLS; j++) {",
    ]
    if c["xlayout"] == "split":
        s += _xload("xl", f"XP(xr[j], u) + p * {nb}", nb) + _xload("xh", f"XP(xr[j], u) + 16 + p * {nb}", nb)
        xl, xh, xd = "(int)xl[i]", "(int)xh[i]", "xd[j]"
    else:
        xl, xh, xd = f"XQ(xr[j], u, p * {nb} + 4 * i)", f"XQ(xr[j], u, 16 + p * {nb} + 4 * i)", "xr[j]"
    s += [
        "  int s = 0;",
        "#pragma unroll",
        f"  for (int i = 0; i < {nb // 4}; i++) {{",
        f"    s = __dp4a((int)lo[i], {xl}, s);",
        f"    s = __dp4a((int)hi[i], {xh}, s);",
        "  }",
        f"  acc[j] += dw * XD({xd}, u) * (float)s;",
        "}",
    ]
    return s


def _unit_q4_K(c, split):
    nb = 32 // c["sub"]
    s = ["const int b = u >> 2, jj = u & 3;"]
    if split:
        s += ["const uint8_t *meta = ws + (size_t)b * 16;"]
        s += _loadq("q", f"wq + (size_t)u * 32 + p * {nb}", nb, True)
    else:
        s += ["const uint8_t *meta = wq + (size_t)b * 144;"]
        s += _loadq("q", f"meta + 16 + 32 * jj + p * {nb}", nb, True)
    s += [
        "const float d = KURN_H2F(kld_u16(meta)), dmin = KURN_H2F(kld_u16(meta + 2));",
        "const uint32_t sw0 = kld_u32(meta + 4), sw1 = kld_u32(meta + 8), sw2 = kld_u32(meta + 12);",
        "const uint32_t sw[3] = {sw0, sw1, sw2};",
        "#define SB(k) ((sw[(k) >> 2] >> (8 * ((k) & 3))) & 0xFF)",
        "int sc[2], mn[2];",
        "#pragma unroll",
        "for (int h = 0; h < 2; h++) {",
        "  const int sb = 2 * jj + h;",
        "  if (sb < 4) { sc[h] = SB(sb) & 63; mn[h] = SB(sb + 4) & 63; }",
        "  else { sc[h] = (SB(sb + 4) & 0xF) | ((SB(sb - 4) >> 6) << 4); mn[h] = (SB(sb + 4) >> 4) | ((SB(sb) >> 6) << 4); }",
        "}",
        "#undef SB",
        "const int c0 = b * 8 + 2 * jj;",
        "#pragma unroll",
        "for (int j = 0; j < COLS; j++) {",
        "  int s0 = 0, s1 = 0, m0 = 0, m1 = 0;",
        "#pragma unroll",
        f"  for (int i = 0; i < {nb // 4}; i++) {{",
        f"    const int x0 = XQ(xr[j], c0, p * {nb} + 4 * i), x1 = XQ(xr[j], c0 + 1, p * {nb} + 4 * i);",
        "    s0 = __dp4a((int)(q[i] & 0x0F0F0F0Fu), x0, s0);",
        "    s1 = __dp4a((int)((q[i] >> 4) & 0x0F0F0F0Fu), x1, s1);",
    ]
    if c["mins"] == "dp4a":
        s += ["    m0 = __dp4a(0x01010101, x0, m0);", "    m1 = __dp4a(0x01010101, x1, m1);"]
    s += ["  }"]
    if c["mins"] == "bsums":
        s += [
            "#pragma unroll",
            f"  for (int h = 0; h < {nb // 16}; h++) {{",
            f"    m0 += XBS(xr[j], c0, p * {nb // 16} + h);",
            f"    m1 += XBS(xr[j], c0 + 1, p * {nb // 16} + h);",
            "  }",
        ]
    s += [
        "  acc[j] += XD8K(xr[j], b) * (d * (float)(sc[0] * s0 + sc[1] * s1) - dmin * (float)(mn[0] * m0 + mn[1] * m1));",
        "}",
    ]
    return s


def _unit_crumbs(c, split, bits):
    """q2_0 (2-bit, 64 values / 16 bytes per unit) and q1_0 (1-bit, 128 values / 16 bytes)."""
    nb = 16 // c["sub"]
    vpb = 8 // bits  # values per byte
    vpl = nb * vpb  # values per lane
    nch = max(1, vpl // 32)  # activation chunks per lane
    cpu = 64 // 32 if bits == 2 else 128 // 32  # chunks per unit
    if split:
        s = _loadq("q", f"wq + (size_t)u * 16 + p * {nb}", nb, True) + ["const float dw = KURN_H2F(kld_u16(ws + 2 * (size_t)u));"]
    else:
        s = ["const uint8_t *b = wq + (size_t)u * 18;", "const float dw = KURN_H2F(kld_u16(b));"]
        s += _loadq("q", f"b + 2 + p * {nb}", nb, False)
    # expanded words: e[w] holds 4 consecutive values; value index of word w within the lane = 4 w
    nwords = vpl // 4
    s += [f"uint32_t e[{nwords}];"]
    if bits == 2:
        s += [
            "#pragma unroll",
            f"for (int k = 0; k < {nb}; k++) {{",
            "  const uint32_t x = (q[k >> 2] >> (8 * (k & 3))) & 0xFF;",
            "  e[k] = __vsub4((x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u, 0x01010101u);",
            "}",
        ]
    else:
        s += [
            "#pragma unroll",
            f"for (int k = 0; k < {nb}; k++) {{",
            "  const uint32_t x = (q[k >> 2] >> (8 * (k & 3))) & 0xFF;",
            "#pragma unroll",
            "  for (int h = 0; h < 2; h++) {",
            "    const uint32_t n = (x >> (4 * h)) & 0xF;",
        ]
        if c["unpack"] == "lut":
            s += ["    e[2 * k + h] = tab[n];"]
        else:
            s += ["    const uint32_t z = (n | (n << 7) | (n << 14) | (n << 21)) & 0x01010101u;",
                  "    e[2 * k + h] = __vsub4(z + z, 0x01010101u);"]  # fmt: skip
        s += ["  }", "}"]
    s += [
        f"const int v0 = p * {vpl};  // first value of this lane within the unit",
        "#pragma unroll",
        "for (int j = 0; j < COLS; j++) {",
        "  float a = 0.f;",
        "#pragma unroll",
        f"  for (int ch = 0; ch < {nch}; ch++) {{",
        f"    const int cc = u * {cpu} + ((v0 + ch * 32) >> 5);",
    ]
    if c["xlayout"] == "split":
        s += ["  " + ln for ln in _xload("xa", "XP(xr[j], cc) + ((v0 + ch * 32) & 31)", 4 * min(8, nwords))]
        x, xd = "(int)xa[w]", "xd[j]"
    else:
        x, xd = "XQ(xr[j], cc, ((v0 + ch * 32) & 31) + 4 * w)", "xr[j]"
    s += [
        "    int s = 0;",
        "#pragma unroll",
        f"    for (int w = 0; w < {min(8, nwords)}; w++) s = __dp4a((int)e[ch * 8 + w], {x}, s);",
        f"    a += XD({xd}, cc) * (float)s;",
        "  }",
        "  acc[j] += dw * a;",
        "}",
    ]
    return s


def _unit_tq2_0(c, split):
    nb = 32 // c["sub"]
    s = ["const int b = u >> 1, hh = u & 1;"]
    if split:
        s += _loadq("q", f"wq + (size_t)u * 32 + p * {nb}", nb, True) + ["const float dw = KURN_H2F(kld_u16(ws + 2 * (size_t)b));"]
    else:
        s += ["const uint8_t *blk = wq + (size_t)b * 66;", "const float dw = KURN_H2F(kld_u16(blk + 64));"]
        s += _loadq("q", f"blk + 32 * hh + p * {nb}", nb, False)
    s += [
        "#pragma unroll",
        "for (int j = 0; j < COLS; j++) {",
        "  int s = 0;",
        "#pragma unroll",
        f"  for (int i = 0; i < {nb // 4}; i++) {{",
        "#pragma unroll",
        "    for (int sh = 0; sh < 4; sh++)",
        f"      s = __dp4a((int)__vsub4((q[i] >> (2 * sh)) & 0x03030303u, 0x01010101u), XQ(xr[j], b * 8 + 4 * hh + sh, p * {nb} + 4 * i), s);",
        "  }",
        "  acc[j] += dw * XD8K(xr[j], b) * (float)s;",
        "}",
    ]
    return s


def _unit_e8p(c, split):
    nv = 4 // c["sub"]  # vectors of 8 per lane
    s = ["const int b = u >> 3;"]
    if split:
        s += ["const uint8_t *cb = wq + (size_t)u * 8;", "const float dw = KURN_H2F(kld_u16(ws + 2 * (size_t)b));"]
        if nv == 4:
            s += ["const uint32_t los = kld_u32(cb), his = kld_u32(cb + 4);"]
        elif nv == 2:
            s += ["const uint32_t los = kld_u16(cb + 2 * p), his = kld_u16(cb + 4 + 2 * p);"]
        else:
            s += ["const uint32_t los = kld_u8(cb + p), his = kld_u8(cb + 4 + p);"]
    else:
        s += ["const uint8_t *blk = wq + (size_t)b * 66;", "const float dw = KURN_H2F(kld_u16(blk));",
              f"const int g0 = (u & 7) * 4 + p * {nv};", "uint32_t los = 0, his = 0;",
              "#pragma unroll", f"for (int v = 0; v < {nv}; v++) {{ los |= kld_u8(blk + 2 + g0 + v) << (8 * v); "
              "his |= kld_u8(blk + 34 + g0 + v) << (8 * v); }"]  # fmt: skip
    s += [
        f"uint32_t w[{2 * nv}];",
        "#pragma unroll",
        f"for (int v = 0; v < {nv}; v++) ke8p(tab, (los >> (8 * v)) & 0xFF, (his >> (8 * v)) & 0xFF, w[2 * v], w[2 * v + 1]);",
        "#pragma unroll",
        "for (int j = 0; j < COLS; j++) {",
        "  int s = 0;",
        "#pragma unroll",
        f"  for (int i = 0; i < {2 * nv}; i++) s = __dp4a((int)w[i], XQ(xr[j], u, 32 * p / {c['sub']} + 4 * i), s);",
        "  acc[j] += dw * XD8K(xr[j], b) * (float)s;",
        "}",
    ]
    return s


UNITS = {
    "q8_0": _unit_q8_0,
    "q4_0": lambda c, sp: _unit_q4(c, sp, False),
    "iq4_nl": lambda c, sp: _unit_q4(c, sp, True),
    "q4_K": _unit_q4_K,
    "q2_0": lambda c, sp: _unit_crumbs(c, sp, 2),
    "q1_0": lambda c, sp: _unit_crumbs(c, sp, 1),
    "tq2_0": _unit_tq2_0,
    "e8p": _unit_e8p,
}


def _table_setup(c):
    """(device helpers, kernel-prologue statements, tab argument) for formats with lookup tables."""
    w = c["weights"]
    if w == "e8p":
        return (_e8p_table(), ["__shared__ uint32_t tab[512];",
                               "for (int i = threadIdx.x; i < 512; i += blockDim.x) tab[i] = kE8P[i];", "__syncthreads();"])  # fmt: skip
    if w == "q1_0" and c.get("unpack") == "lut":
        return ("", ["__shared__ uint32_t tab[16];",
                     "if (threadIdx.x < 16) { const uint32_t n = threadIdx.x, z = (n | (n << 7) | (n << 14) | (n << 21)) & 0x01010101u; "
                     "tab[n] = __vsub4(z + z, 0x01010101u); }", "__syncthreads();"])  # fmt: skip
    if w == "iq4_nl":
        return (_iq4_table(), ["const uint32_t *tab = nullptr; (void)tab;"])
    return ("", ["const uint32_t *tab = nullptr; (void)tab;"])


def _lb(nt, c):
    return f"{nt}, {max(1, c['minb'])}"  # an explicit minimum of 1 block stops ptxas from capping registers and spilling


def _indent(lines, n):
    return "\n".join((" " * n + ln) if ln and not ln.startswith("#") else ln for ln in lines)


# --------------------------------------------------------------------------- shape helpers (C)


def _shape_c(c):
    f = FORMATS[c["weights"]]
    act = ACT[f["act"]]
    blk = f["block"]
    kmul = max(blk, act["block"]) if c["op"] == "gemv" else 32
    rowq = f"((size_t)K * {f['ub']} / {f['unit']})"
    rows = f"((size_t)K / {blk} * {f['sb']})"
    xrow = f"((size_t)K / {act['block']} * {act['nbytes']})"
    return kmul, rowq, rows, xrow


# --------------------------------------------------------------------------- repack + quantize


def _repack_body(fmt):
    """Statements copying weight block (r, b) from src to the split planes q (quant) and s (scale)."""
    copy = lambda n, so, do: f"for (int i = 0; i < {n}; i++) q[{do} + i] = src[{so} + i];"  # noqa: E731
    if fmt == "q8_0":
        return [copy(32, 2, 0), "s[0] = src[0]; s[1] = src[1];"]
    if fmt in ("q4_0", "iq4_nl", "q2_0", "q1_0"):
        return [copy(16, 2, 0), "s[0] = src[0]; s[1] = src[1];"]
    if fmt == "q4_K":
        return [copy(128, 16, 0), "for (int i = 0; i < 16; i++) s[i] = src[i];"]
    if fmt == "tq2_0":
        return [copy(64, 0, 0), "s[0] = src[64]; s[1] = src[65];"]
    if fmt == "e8p":  # per 32-value chunk: 4 code bytes then 4 sign bytes
        return ["for (int ch = 0; ch < 8; ch++) for (int i = 0; i < 4; i++) { q[8 * ch + i] = src[2 + 4 * ch + i]; "
                "q[8 * ch + 4 + i] = src[34 + 4 * ch + i]; }", "s[0] = src[0]; s[1] = src[1];"]  # fmt: skip
    raise KeyError(fmt)


def _quant_kernels(act):
    q80 = r"""
// f32 -> q8_0 blocks (ggml quantize_row_q8_0_ref rounding), one warp per block of 32.
// split != 0: int8 plane Q[M][K] plus float scale plane D[M][K/32] (scale = the fp16-rounded d)
static __global__ void __attribute__((unused)) kg_quant_q8_0(const float *__restrict__ X, uint8_t *__restrict__ Q, float *__restrict__ D, int K, int M, int split) {
  const int lane = threadIdx.x & 31, nb = K / 32;
  const long blk = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const bool live = blk < (long)M * nb;
  const long bb = live ? blk : 0;
  const int m = (int)(bb / nb), c = (int)(bb % nb);
  const float v = X[(size_t)m * K + c * 32 + lane];
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(KURN_FULL, amax, o));
  const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
  const int q = (int)roundf(v * id);
  if (!live) return;
  if (!split) {
    uint8_t *b = Q + ((size_t)m * nb + c) * 34;
    if (lane == 0) kst_u16(b, KURN_F2H(d));
    b[2 + lane] = (uint8_t)(int8_t)q;
  } else {
    Q[(size_t)m * K + c * 32 + lane] = (uint8_t)(int8_t)q;
    if (lane == 0) D[(size_t)m * nb + c] = KURN_H2F(KURN_F2H(d));
  }
}
"""
    q8k = r"""
// f32 -> q8_K blocks (ggml quantize_row_q8_K_ref), one warp per block of 256 (8 values per lane)
static __global__ void kg_quant_q8_K(const float *__restrict__ X, uint8_t *__restrict__ Q, int K, int M) {
  const int lane = threadIdx.x & 31, nb = K / 256;
  const long blk = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const bool live = blk < (long)M * nb;
  const long bb = live ? blk : 0;
  const int m = (int)(bb / nb), c = (int)(bb % nb);
  const float *x = X + (size_t)m * K + (size_t)c * 256 + lane * 8;
  float v[8], amax = 0.f, mx = 0.f;
  int idx = 1 << 30;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[i] = x[i];
    const float ax = fabsf(v[i]);
    if (ax > amax) { amax = ax; mx = v[i]; idx = lane * 8 + i; }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {  // largest |x|, first index on ties (ggml scans in order)
    const float oa = __shfl_xor_sync(KURN_FULL, amax, o), om = __shfl_xor_sync(KURN_FULL, mx, o);
    const int oi = __shfl_xor_sync(KURN_FULL, idx, o);
    if (oa > amax || (oa == amax && oi < idx)) { amax = oa; mx = om; idx = oi; }
  }
  uint8_t *b = Q + ((size_t)m * nb + c) * 292;
  int q[8], s8 = 0;
  const float iscale = amax != 0.f ? -127.f / mx : 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    q[i] = amax != 0.f ? min(127, __float2int_rn(iscale * v[i])) : 0;
    s8 += q[i];
  }
  const int pair = __shfl_xor_sync(KURN_FULL, s8, 1);
  if (!live) return;
  if (lane == 0) kst_u32(b, __float_as_uint(amax != 0.f ? 1.f / iscale : 0.f));
#pragma unroll
  for (int i = 0; i < 8; i++) b[4 + lane * 8 + i] = (uint8_t)(int8_t)q[i];
  if (!(lane & 1)) kst_u16(b + 260 + 2 * (lane >> 1), (uint32_t)(uint16_t)(int16_t)(s8 + pair));
}
"""
    return q80 + (q8k if act == "q8_K" else "")


# --------------------------------------------------------------------------- GEMV


def _gemv_kernel(c):
    w = c["weights"]
    split = c["layout"] == "split"
    tables, prologue = _table_setup(c)
    nt = threads(c)
    xs = c["xlayout"] == "split"
    # dev2 keeps 32-bit column offsets instead of per-column pointers; with xlayout split the offset is cj * K, so the
    # column's scale row (K / 32 floats per column) starts at offset / 32
    body = [ln.replace("xr[j]", "(xb + xo[j])").replace("xd[j]", "(xdb + (xo[j] >> 5))") for ln in UNITS[w](c, split)]
    tpr = c["tpr"]
    red = []
    for m in (16, 8, 4, 2, 1):
        if m < min(tpr, 32):
            red.append(f"    v += __shfl_xor_sync(KURN_FULL, v, {m});")
    if tpr > 32:
        tail = [
            f"  __shared__ float red[{c['rpb'] * (tpr // 32) * c['cols']}];",
            "  #pragma unroll",
            "  for (int j = 0; j < COLS; j++) {",
            "    float v = acc[j];",
            *red,
            f"    if ((lid & 31) == 0) red[(lr * {tpr // 32} + (lid >> 5)) * COLS + j] = v;",
            "  }",
            "  __syncthreads();",
            "  if (lid == 0 && row < N) {",
            "    for (int j = 0; j < COLS; j++) {",
            "      float v = 0.f;",
            f"      for (int i = 0; i < {tpr // 32}; i++) v += red[(lr * {tpr // 32} + i) * COLS + j];",
            "      if (col0 + j < M) Y[(size_t)(col0 + j) * N + row] = v;",
            "    }",
            "  }",
        ]
    else:
        tail = [
            "  #pragma unroll",
            "  for (int j = 0; j < COLS; j++) {",
            "    float v = acc[j];",
            *red,
            "    if (lid == 0 && row < N && col0 + j < M) Y[(size_t)(col0 + j) * N + row] = v;",
            "  }",
        ]
    kmul, rowq, rows, xrow = _shape_c(c)
    f = FORMATS[w]
    ws = f"W + (size_t)N * {rowq} + (size_t)wrow * {rows}" if split else "wq"
    rowb = rowq if split else f"((size_t)K / {f['block']} * {f['nbytes']})"
    tab_arg = "const uint32_t *__restrict__ tab"
    xd_param, xd_arg = ("const float *__restrict__ xdb, ", "xdb, ") if xs else ("", "")
    xinit = "min(col0 + j, M - 1) * K" if xs else f"min(col0 + j, M - 1) * (int){xrow}"
    xdecl = "\n  const float *xdb = (const float *)(X + (size_t)M * K);" if xs else ""
    return f"""{tables}{_act_macros(f["act"], xs)}
#define COLS {c["cols"]}
#define SUB {c["sub"]}
KURN_FN void kunit(const uint8_t *__restrict__ wq, const uint8_t *__restrict__ ws, int u, int p,
                                             const uint8_t *__restrict__ xb, const int *xo, {xd_param}float *acc, {tab_arg}) {{
  (void)ws; (void)tab;
{_indent(body, 2)}
}}

static __global__ void __launch_bounds__({_lb(nt, c)})
kg_gemv(const uint8_t *__restrict__ W, const uint8_t *__restrict__ X, float *__restrict__ Y, int N, int K, int M) {{
{_indent(prologue, 2)}
  const int lr = threadIdx.x / {tpr}, lid = threadIdx.x % {tpr};
  const int row = blockIdx.x * {c["rpb"]} + lr, wrow = row < N ? row : N - 1;
  const int col0 = blockIdx.y * COLS;
  const uint8_t *wq = W + (size_t)wrow * {rowb};
  const uint8_t *ws = {ws};
  int xo[COLS];  // 32-bit column offsets (columns past M repeat the last one; their results are not stored)
  float acc[COLS];
  #pragma unroll
  for (int j = 0; j < COLS; j++) {{ xo[j] = {xinit}; acc[j] = 0.f; }}{xdecl}
  const int work = K / {f["unit"]} * SUB;
  int w0 = lid;
  // main loop without bounds checks, so the {c["unroll"]} units' loads can all be issued before their math
  for (; w0 + {(c["unroll"] - 1) * tpr} < work; w0 += {tpr * c["unroll"]}) {{
    #pragma unroll
    for (int k = 0; k < {c["unroll"]}; k++) {{
      const int w = w0 + k * {tpr};
      kunit(wq, ws, w / SUB, w % SUB, X, xo, {xd_arg}acc, tab);
    }}
  }}
  for (; w0 < work; w0 += {tpr}) kunit(wq, ws, w0 / SUB, w0 % SUB, X, xo, {xd_arg}acc, tab);
{chr(10).join(tail)}
}}
"""


# --------------------------------------------------------------------------- GEMM (int8 mma.sync)


# --------------------------------------------------------------------------- ABI


def _abi(c):
    w = c["weights"]
    f = FORMATS[w]
    act = f["act"]
    gemm = c["op"] == "gemm"
    split = c["layout"] == "split"
    kmul, rowq, rows, xrow = _shape_c(c)
    cfg = " ".join(f"{k}={c[k]}" for k in codegen_keys(c["op"]))
    if gemm:
        xbytes = "(size_t)M * K + (size_t)M * (K / 32) * 4"
        quant = ("  KURN_LAUNCH(kg_quant_q8_0, dim3((unsigned)(((long)M * (K / 32) + 3) / 4)), dim3(128), s, X, (uint8_t *)Xq, "
                 "(float *)((uint8_t *)Xq + (size_t)M * K), K, M, 1);")  # fmt: skip
        grid = f"dim3((unsigned)((N + {c['bm']} - 1) / {c['bm']}), (unsigned)((M + {c['bn']} - 1) / {c['bn']}))"
        run = f"  KURN_LAUNCH(kg_gemm, {grid}, dim3({threads(c)}), s, (const uint8_t *)W, (const uint8_t *)Xq, Y, N, K, M);"
    else:
        if c["xlayout"] == "split":  # aligned int8 plane [M][K] + float scale plane [M][K/32] (split q8_0 quantizer)
            xbytes = "(size_t)M * K + (size_t)M * (K / 32) * 4"
            quant = ("  KURN_LAUNCH(kg_quant_q8_0, dim3((unsigned)(((long)M * (K / 32) + 3) / 4)), dim3(128), s, X, (uint8_t *)Xq, "
                     "(float *)((uint8_t *)Xq + (size_t)M * K), K, M, 1);")  # fmt: skip
        else:
            xbytes = f"(size_t)M * {xrow}"
            quant = None
        grid = f"dim3((unsigned)((N + {c['rpb']} - 1) / {c['rpb']}), (unsigned)((M + {c['cols']} - 1) / {c['cols']}))"
        run = f"  KURN_LAUNCH(kg_gemv, {grid}, dim3({threads(c)}), s, (const uint8_t *)W, (const uint8_t *)Xq, Y, N, K, M);"
    xblocks = (
        "  KURN_LAUNCH(kg_quant_q8_0, dim3((unsigned)(((long)M * (K / 32) + 3) / 4)), dim3(128), s, X, (uint8_t *)Xb, (float *)nullptr, K, M, 0);"
        if act == "q8_0"
        else "  KURN_LAUNCH(kg_quant_q8_K, dim3((unsigned)(((long)M * (K / 256) + 3) / 4)), dim3(128), s, X, (uint8_t *)Xb, K, M);"
    )
    quant = quant or xblocks.replace("Xb", "Xq")
    err = "#ifdef KURN_EMU\n  return 0;\n#else\n  return (int)cudaGetLastError();\n#endif"
    rowb = f"((size_t)K / {f['block']} * {f['nbytes']})"
    if split:
        prep_bytes = f"(size_t)N * ({rowq} + {rows})"
        prepare = f"""  const long nblk = (long)N * (K / {f["block"]});
  KURN_LAUNCH(kg_repack, dim3((unsigned)((nblk + 127) / 128)), dim3(128), s, (const uint8_t *)W, (uint8_t *)Wp, N, K);
{err}"""
    else:
        prep_bytes = "0"
        prepare = "  (void)W; (void)Wp; (void)N; (void)K; (void)s;\n  return 0;"
    repack = ""
    if split:
        repack = f"""
// native ggml blocks -> split layout: quant plane [N][{rowq}] then scale plane [N][K/{f["block"]} * {f["sb"]}]
static __global__ void kg_repack(const uint8_t *__restrict__ W, uint8_t *__restrict__ P, int N, int K) {{
  const int nb = K / {f["block"]};
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)N * nb) return;
  const int r = (int)(i / nb), bl = (int)(i % nb);
  const uint8_t *src = W + (size_t)r * {rowb} + (size_t)bl * {f["nbytes"]};
  uint8_t *q = P + (size_t)r * {rowq} + (size_t)bl * {f["ub"] * f["block"] // f["unit"]};
  uint8_t *s = P + (size_t)N * {rowq} + (size_t)r * {rows} + (size_t)bl * {f["sb"]};
{_indent(_repack_body(w), 2)}
}}
"""
    return f"""{repack}
extern "C" {{
const char *kg_config(void) {{ return "{cfg}"; }}
int kg_act(void) {{ return 0; }}  // 0: kg_quant -> q8 activations; reference from kg_xblocks
int kg_check_shape(int N, int K, int M) {{ return (N > 0 && M > 0 && K > 0 && K % {kmul} == 0) ? 0 : -1; }}
size_t kg_prep_bytes(int N, int K) {{ (void)N; (void)K; return {prep_bytes}; }}
size_t kg_xbytes(int K, int M) {{ return {xbytes}; }}
int kg_prepare(const void *W, void *Wp, int N, int K, cudaStream_t s) {{
{prepare}
}}
int kg_quant(const float *X, void *Xq, int K, int M, cudaStream_t s) {{
{quant}
{err}
}}
size_t kg_xblock_bytes(int K, int M) {{ return (size_t)M * (K / {ACT[act]["block"]}) * {ACT[act]["nbytes"]}; }}
int kg_xblocks(const float *X, void *Xb, int K, int M, cudaStream_t s) {{
{xblocks}
{err}
}}
int kg_run(const void *W, const void *Xq, float *Y, int N, int K, int M, cudaStream_t s) {{
  if (kg_check_shape(N, K, M)) return -1;
{run}
{err}
}}
}}
"""


def generate(c):
    """Resolved cuda config -> CUDA C++ source implementing the kurn_gpu.h ABI."""
    cfg = " ".join(f"{k}={c[k]}" for k in codegen_keys(c["op"]))
    head = f"// generated by kurn: {c['weights']} {c['op']} for CUDA ({FORMATS[c['weights']]['doc']})\n// {cfg}\n"
    if c["op"] == "gemm":
        from . import mma

        tables = _iq4_table() if c["weights"] == "iq4_nl" else ""
        return (head + PRELUDE + CPASYNC + mma.HELPERS + tables + mma.kernel(c) + mma.repack_kernel(c) + mma.x16_kernel(c)
                + mma.abi(c, cfg))  # fmt: skip
    act = FORMATS[c["weights"]]["act"]
    return head + PRELUDE + _quant_kernels(act) + _gemv_kernel(c) + _abi(c)


def describe(c):
    """One-line summary of the kernel's static resources (for reports)."""
    if c["op"] == "gemm":
        return f"gemm {c['weights']} bm={c['bm']} bn={c['bn']} bk={c['bk']} threads={threads(c)} smem={gemm_smem(c)}B stages={c['stages']}"
    return f"gemv {c['weights']} threads={threads(c)} tpr={c['tpr']} cols={c['cols']} layout={c['layout']}"
