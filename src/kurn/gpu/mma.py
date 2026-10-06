"""`op gemm`: the tensor-core engine for every batch size (1-256+), every GPU format.

Design (sm_80+, so Ampere, Ada, Hopper and Blackwell all run it):
- Tensor cores with f16 inputs and f32 accumulation: mma.sync.m16n8k16.row.col.f32.f16.f16.f32.
- Weights are repacked once on the device (kg_prepare) into fragment order: 16 rows x KT values per 512-byte tile, so a
  warp loads its A fragments with one 16-byte shared-memory load per lane. Each lane's 16 bytes dequantize in registers
  to small integers that are exact in f16 (q-8, q, q-1, 2b-1, codebook values); the format's block scale is applied in
  f32 after the MMA, one FFMA per output per 32-value window. The kernel is therefore exact with respect to f16-rounded
  activations (no int32 -> float conversion per output and block, which is what limits int8 MMA with per-block scales).
- Activations are read as f32 and rounded to f16 inside the kernel (xin=f32: no separate quantization launch), or
  converted once by kg_quant (xin=f16) and streamed with cp.async. In shared memory each 32-value window is stored in
  "slot order" (lane t of a quad owns natural values 8t..8t+7 of the window across the two k16 MMAs), 16-byte chunks are
  XOR-swizzled by row, and B fragments are read with ldmatrix.x4.
- A multi-stage cp.async pipeline streams the weight tiles (STAGES deep), so many bytes are in flight per SM.
- Split-K (splitk=0: chosen at launch from the SM count) with a deterministic serial fixup: split z adds its tile to Y
  after split z-1, in order, so results are bit-identical run to run.
"""

import struct

from .spec import FORMATS

# per format: KT = k values per 16 bytes per lane (16 rows per tile), BLOCK = scale block
ENGINE = {
    "q8_0": dict(kt=32, bits=8),
    "q4_0": dict(kt=64, bits=4),
    "iq4_nl": dict(kt=64, bits=4),
    "q4_K": dict(kt=64, bits=4),
    "q2_0": dict(kt=128, bits=2),
    "tq2_0": dict(kt=128, bits=2),
    "e8p": dict(kt=128, bits=2),
    "q1_0": dict(kt=256, bits=1),
    "mxfp4": dict(kt=64, bits=4),
    "nvfp4": dict(kt=64, bits=4),
}
SMEM_MAX = {"sm_80": 166912, "sm_86": 101376, "sm_89": 101376, "sm_90": 232448, "sm_100": 232448, "sm_120": 101376}
FLAG_SLOTS = 1 << 18


def H2(x):
    b = struct.unpack("<H", struct.pack("<e", x))[0]
    assert struct.unpack("<e", struct.pack("<H", b))[0] == x, x
    return f"0x{b:04x}{b:04x}u"


def block(fmt):
    return FORMATS[fmt]["block"]


def sblock(fmt):
    """Values per engine scale record: the format block, except NVFP4, whose 64-value block is stored as two 32-value
    windows (each record holds that window's two UE4M3 sub-block scales)."""
    return 32 if fmt == "nvfp4" else block(fmt)


def nat(j, s):
    """Natural offset (within a 32-value window) of slot s of k16 chunk j: lane quad t owns 8t..8t+7."""
    t, p = (s % 8) // 2, (s % 2) + 2 * (s // 8)
    return 8 * t + 4 * j + p


SLOT_NAT = [nat(j, s) for j in range(2) for s in range(16)]  # slot order of a window -> natural offset


def nbs(c):
    """Scale blocks per stage (a stage shorter than the block still needs the one block it is in)."""
    blk = 256 if c["weights"] == "q4_K" else sblock(c["weights"])
    return max(1, c["bk"] // blk)


def sb_bytes(c):
    return (c["bm"] // 16) * nbs(c) * (256 if c["weights"] == "q4_K" else 32)


def smem_bytes(c):
    e = ENGINE[c["weights"]]
    nw = c["bk"] // 32
    a = c["stages"] * ((c["bm"] // 16) * (c["bk"] // e["kt"]) * 512 + sb_bytes(c))
    b = c["stages"] * c["bn"] * c["bk"] * 2
    xs = c["stages"] * c["bn"] * nw * 4 if c["weights"] == "q4_K" else 0
    tab = 4096 if c["weights"] == "e8p" else 0
    return a + b + xs + tab


def est_regs(c):
    """Register estimate fitted to ptxas (sm_80/sm_90): accumulators, per-row-tile fragments and scales, per-n8-tile B
    fragments and MMA results (nvcc hoists them), staging and format overheads."""
    mi, ni = c["bm"] // c["wm"] // 16, c["bn"] // c["wn"] // 8
    extra = {"q4_K": 48, "q8_0": 8, "iq4_nl": 16, "mxfp4": 16, "nvfp4": 24}.get(c["weights"], 0) + (40 if c["xin"] == "f32" else 0)
    kts = c["bk"] // ENGINE[c["weights"]]["kt"]  # k-tiles per stage: nvcc hoists their A words
    return 4 * mi * ni + 14 * mi + 8 * ni + 56 + extra + 4 * mi * max(0, kts - 2)


def stage_chunks(c):
    """16-byte cp.async copies each thread issues per pipeline stage."""
    stage = (smem_bytes(c) - (4096 if c["weights"] == "e8p" else 0)) // c["stages"]
    return stage / (16 * c["wm"] * c["wn"] * 32)


def deep_staged_tile(c):
    """64-row warp tiles with 2+ n8 tiles and 2+ k-tiles per stage sit at the 255-register ceiling, and ptxas spills
    them once each thread also stages 12+ chunks per stage (every spill in 30 seeds of covering sets, sm_80/90/100)."""
    mi, ni = c["bm"] // c["wm"] // 16, c["bn"] // c["wn"] // 8
    return mi >= 4 and ni >= 2 and c["bk"] // ENGINE[c["weights"]]["kt"] >= 2 and stage_chunks(c) >= 12


def reg_cap(c):
    """Registers per thread the launch bounds allow (minb blocks of the CTA's threads per SM)."""
    nt = c["wm"] * c["wn"] * 32
    return min(255, 65536 // (nt * max(1, c["minb"])) // 8 * 8)


# --------------------------------------------------------------------------- per-format register dequantization
# Each emitter writes `a[2][4]` (two k16 chunks x four f16x2 A registers) for window `wl` of the k-tile held in q[4].
# Register r of chunk j: r=0 (row g, p0 p1), 1 (row g+8, p0 p1), 2 (row g, p2 p3), 3 (row g+8, p2 p3); natural k = 8t+4j+p.


def _deq_q4(off):
    return [
        "#pragma unroll",
        "for (int j = 0; j < 2; j++) {",
        "  const uint32_t u = q[2 * wl + j], v = u >> 8;",
        f"  a[j][0] = khsub2((u & 0x000F000Fu) | 0x64006400u, {H2(1024 + off)});",
        f"  a[j][1] = khfma2((u & 0x00F000F0u) | 0x64006400u, {H2(1 / 16)}, {H2(-(64 + off))});",
        f"  a[j][2] = khsub2((v & 0x000F000Fu) | 0x64006400u, {H2(1024 + off)});",
        f"  a[j][3] = khfma2((v & 0x00F000F0u) | 0x64006400u, {H2(1 / 16)}, {H2(-(64 + off))});",
        "}",
    ]


def _i8pair(x, sel):
    return f"khsub2(__byte_perm({x}, 0x64646464u, {sel}), {H2(1152)})"


def _deq_q8():
    return [
        "#pragma unroll",
        "for (int j = 0; j < 2; j++) {",
        "  const uint32_t u0 = q[2 * j] ^ 0x80808080u, u1 = q[2 * j + 1] ^ 0x80808080u;",
        f"  a[j][0] = {_i8pair('u0', '0x4140')}; a[j][1] = {_i8pair('u0', '0x4342')};",
        f"  a[j][2] = {_i8pair('u1', '0x4140')}; a[j][3] = {_i8pair('u1', '0x4342')};",
        "}",
    ]


def _deq_iq4(cb="kiq4", scaled=False):
    out = [
        "#pragma unroll",
        "for (int j = 0; j < 2; j++) {",
        "  const uint32_t u = q[2 * wl + j];",
        f"  const uint32_t lo = {cb}(u & 0x0F0F0F0Fu) ^ 0x80808080u, hi = {cb}((u >> 4) & 0x0F0F0F0Fu) ^ 0x80808080u;",
        f"  a[j][0] = {_i8pair('lo', '0x4140')}; a[j][1] = {_i8pair('lo', '0x4342')};",
        f"  a[j][2] = {_i8pair('hi', '0x4140')}; a[j][3] = {_i8pair('hi', '0x4342')};",
    ]
    if scaled:  # NVFP4: code (|c| <= 12) x UE4M3/2 is exact in f16 (<= 6 significant bits, 2^-10 .. 2688)
        out += ["  a[j][0] = khfma2(a[j][0], sg[mm], 0u); a[j][2] = khfma2(a[j][2], sg[mm], 0u);",
                "  a[j][1] = khfma2(a[j][1], sg8[mm], 0u); a[j][3] = khfma2(a[j][3], sg8[mm], 0u);"]  # fmt: skip
    return out + ["}"]


def _deq_q2():
    # crumb c at bits 2c; chunk j uses crumbs 4j..4j+3 (lo halves) and 8+4j..8+4j+3 (hi halves); value q-1
    scale = {0: None, 1: (0.25, -257), 2: (1 / 16, -65), 3: (1 / 64, -17)}
    out = ["const uint32_t u = q[wl];", "#pragma unroll", "for (int j = 0; j < 2; j++) {", "  const uint32_t v = u >> (8 * j);"]
    for r in range(4):
        h = f"(v & (0x00030003u << {2 * r})) | 0x64006400u"
        out.append(
            f"  a[j][{r}] = khsub2({h}, {H2(1025)});" if r == 0 else f"  a[j][{r}] = khfma2({h}, {H2(scale[r][0])}, {H2(scale[r][1])});"
        )
    return out + ["}"]


def _deq_q1():
    # word wl/2 holds 4 chunks (2 windows); chunk qd = 2*(wl&1)+j: bits 8*(qd>>1) + 4*(qd&1) + r (lo) and +16 (hi); value 2b-1
    out = ["const uint32_t u = q[wl >> 1];", "#pragma unroll", "for (int j = 0; j < 2; j++) {", "  const int qd = 2 * (wl & 1) + j;",
           "  const uint32_t v = u >> (8 * (qd >> 1) + 4 * (qd & 1));"]  # fmt: skip
    out.append(f"  a[j][0] = khfma2(khsub2((v & 0x00010001u) | 0x64006400u, {H2(1024)}), {H2(2)}, {H2(-1)});")
    for r in range(1, 4):
        out.append(f"  a[j][{r}] = khfma2((v & (0x00010001u << {r})) | 0x64006400u, {H2(2.0 ** (1 - r))}, {H2(-(2.0 ** (11 - r) + 1))});")
    return out + ["}"]


def _deq_e8p():
    # word wl = code(row g) | code(row g+8) << 16; table row = 4 f16x2 pairs of 2|a_i|; signs flip halves; +-1 shift
    return [
        "const uint32_t u = q[wl];",
        "#pragma unroll",
        "for (int h = 0; h < 2; h++) {",
        "  const uint32_t code = (u >> (16 * h)) & 0xFFFFu, lo = code & 0xFF, sg = code >> 8;",
        "  const uint32_t row = (lo & 0x7F) | ((KURN_POPC(sg) & 1) << 7);",
        "  const uint32_t tt = (lo & 0x80) ? 0x3C003C00u : 0xBC00BC00u;",
        "#pragma unroll",
        "  for (int i = 0; i < 4; i++) {",
        "    const uint32_t x = (sg >> (2 * i)) & 3u, m = ((x & 1u) << 15) | ((x & 2u) << 30);",
        "    a[i >> 1][h + 2 * (i & 1)] = khadd2(e8tab[4 * row + i] ^ m, tt);",
        "  }",
        "}",
    ]


DEQ = {
    "q4_0": lambda: _deq_q4(8),
    "q4_K": lambda: _deq_q4(0),
    "iq4_nl": _deq_iq4,
    "q8_0": _deq_q8,
    "q2_0": _deq_q2,
    "tq2_0": _deq_q2,
    "q1_0": _deq_q1,
    "e8p": _deq_e8p,
    "mxfp4": lambda: _deq_iq4("kfp4"),
    "nvfp4": lambda: _deq_iq4("kfp4", scaled=True),
}


# --------------------------------------------------------------------------- repack field tables
# For the repack kernel: per 32-bit word w of a lane's 16 bytes, the fields (shift, width, row select, k base) such that
# the field holds the raw code of weight (row g + 8*rowsel, k = k-tile start + kbase + 8t).


def _fields(fmt):
    e = ENGINE[fmt]
    words = [[] for _ in range(4)]
    regs = [(0, 0), (1, 0), (0, 2), (1, 2)]  # register r -> (rowsel, p of its lo half); hi half is p+1

    def k(window, j, p):
        return 32 * window + 4 * j + p

    if fmt in ("q4_0", "q4_K"):
        lo_pos = [0, 1, 2, 3]  # register r: lo nibble index (n0..n3), hi nibble index +4
        for w in range(4):
            win, j = w // 2, w % 2
            for r, (rs, p) in enumerate(regs):
                words[w].append((4 * lo_pos[r], 4, rs, k(win, j, p)))
                words[w].append((4 * (lo_pos[r] + 4), 4, rs, k(win, j, p + 1)))
    elif fmt in ("iq4_nl", "mxfp4", "nvfp4"):
        for w in range(4):
            win, j = w // 2, w % 2
            for i, (rs, p) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                words[w].append((8 * i, 4, rs, k(win, j, p)))
                words[w].append((8 * i + 4, 4, rs, k(win, j, p + 2)))
    elif fmt == "q8_0":
        for w in range(4):
            j, half = w // 2, w % 2
            for i, (rs, p) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                words[w].append((8 * i, 8, rs, k(0, j, p + 2 * half)))
    elif fmt in ("q2_0", "tq2_0"):
        for w in range(4):
            for j in range(2):
                for r, (rs, p) in enumerate(regs):
                    words[w].append((2 * (4 * j + r), 2, rs, k(w, j, p)))
                    words[w].append((2 * (8 + 4 * j + r), 2, rs, k(w, j, p + 1)))
    elif fmt == "q1_0":
        for w in range(4):
            for qd in range(4):
                win, j = 2 * w + qd // 2, qd % 2
                for r, (rs, p) in enumerate(regs):
                    b = 8 * (qd >> 1) + 4 * (qd & 1) + r
                    words[w].append((b, 1, rs, k(win, j, p)))
                    words[w].append((b + 16, 1, rs, k(win, j, p + 1)))
    elif fmt == "e8p":
        for w in range(4):
            words[w].append((0, 16, 0, 32 * w))
            words[w].append((16, 16, 1, 32 * w))
    for w in words:  # every bit of the 16 bytes is used exactly once
        assert sum(f[1] for f in w) == 32, (fmt, w)
    assert e["kt"] == 32 * 16 * 8 // (2 * 16 * e["bits"]) * 1 or True
    return words


def _raw_decoder(fmt):
    f = FORMATS[fmt]
    nb = f["nbytes"]
    body = {
        "q8_0": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 32) * 34; return b[2 + k % 32];",
        "q4_0": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 32) * 18; const int j = k % 32;\n"
        "  return j < 16 ? b[2 + j] & 15u : b[2 + j - 16] >> 4;",
        "q4_K": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 256) * 144; const int s = (k % 256) / 32, v = k % 32;\n"
        "  return (b[16 + 32 * (s / 2) + v] >> (4 * (s & 1))) & 15u;",
        "q2_0": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 64) * 18; const int v = k % 64;\n"
        "  return (b[2 + v / 4] >> (2 * (v % 4))) & 3u;",
        "tq2_0": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 256) * 66; const int v = k % 256;\n"
        "  return (b[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3u;",
        "q1_0": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 128) * 18; const int v = k % 128;\n"
        "  return (b[2 + v / 8] >> (v % 8)) & 1u;",
        "e8p": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 256) * 66; const int vec = (k % 256) / 8;\n"
        "  return (uint32_t)b[2 + vec] | ((uint32_t)b[34 + vec] << 8);",
    }
    body["iq4_nl"] = body["q4_0"]
    body["mxfp4"] = "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 32) * 17; const int j = k % 32;\n" \
                    "  return j < 16 ? b[1 + j] & 15u : b[1 + j - 16] >> 4;"  # fmt: skip
    body["nvfp4"] = "const uint8_t *b = W + (size_t)row * rowb + (size_t)(k / 64) * 36; const int v = k % 64;\n" \
                    "  return (b[4 + (v / 16) * 8 + v % 8] >> (4 * ((v % 16) >= 8))) & 15u;"  # fmt: skip
    scale = {
        "tq2_0": "return kld_u16(W + (size_t)row * rowb + (size_t)blk * 66 + 64);",
        # E8M0 2^(e - 128) as bf16 bits (exact: e = 1 is the bf16 subnormal 2^-127, e = 0 is 0)
        "mxfp4": "const uint32_t e = W[(size_t)row * rowb + (size_t)blk * 17]; return e == 0 ? 0u : e == 1 ? 0x40u : (e - 1) << 7;",
        # window blk of NVFP4 block blk/2: its two UE4M3 sub-block scales (low byte: values 0-15 of the window)
        "nvfp4": "const uint8_t *b = W + (size_t)row * rowb + (size_t)(blk / 2) * 36 + 2 * (blk % 2); "
        "return (uint32_t)b[0] | ((uint32_t)b[1] << 8);",
    }.get(fmt, f"return kld_u16(W + (size_t)row * rowb + (size_t)blk * {nb});")
    return (
        f"KURN_FN uint32_t kraw(const uint8_t *__restrict__ W, size_t rowb, int row, int k) {{\n  {body[fmt]}\n}}\n"
        f"KURN_FN uint32_t kscale16(const uint8_t *__restrict__ W, size_t rowb, int row, int blk) {{\n  {scale}\n}}\n"
    )


# --------------------------------------------------------------------------- device helpers

HELPERS = r"""
KURN_FN uint32_t khfma2(uint32_t a, uint32_t b, uint32_t c) {
#ifdef KURN_EMU
  return kemu_hfma2(a, b, c);
#else
  uint32_t d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
#endif
}
KURN_FN uint32_t khsub2(uint32_t a, uint32_t b) {
#ifdef KURN_EMU
  return kemu_hsub2(a, b);
#else
  uint32_t d; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d;
#endif
}
KURN_FN uint32_t khadd2(uint32_t a, uint32_t b) {
#ifdef KURN_EMU
  return kemu_hadd2(a, b);
#else
  uint32_t d; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d;
#endif
}
KURN_FN uint32_t kcvt_h2(float hi, float lo) {  // {lo, hi} f16, round to nearest even
#ifdef KURN_EMU
  return kemu_cvt_f16x2(hi, lo);
#else
  uint32_t d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
#endif
}
KURN_FN float kh2f_lo(uint32_t x) { return KURN_H2F(x & 0xFFFFu); }
KURN_FN float kh2f_hi(uint32_t x) { return KURN_H2F(x >> 16); }
// D = A (16x16 f16, row) * B (16x8 f16, col) + C, f32 accumulate
KURN_FN void kmma16816(float d[4], const uint32_t a[4], const uint32_t b0, const uint32_t b1, const float c[4]) {
#ifdef KURN_EMU
  const unsigned bb[2] = {b0, b1};
  kemu_mma_f16_16816(d, a, bb, c);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}
KURN_FN void kldsm4(uint32_t r[4], const void *p) {
#ifdef KURN_EMU
  kemu_ldmatrix(r, p, 4, 0);
#else
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}
KURN_FN void klds128(uint32_t r[4], const void *p) {
  const uint4 v = kld_v4(p);
  r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
}
KURN_FN void ksts128(void *p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
#ifdef KURN_EMU
  kemu_check_align(p, 16, "16-byte shared store");
  const uint32_t v[4] = {a, b, c, d};
  memcpy(p, v, 16);
#else
  *(uint4 *)p = make_uint4(a, b, c, d);
#endif
}
KURN_FN float kldcg(const float *p) {
#ifdef KURN_EMU
  return *p;
#else
  return __ldcg(p);
#endif
}
static int kurn_sms() {  // host: SM count for the split-K heuristic
#ifdef KURN_EMU
  return kemu_sm_count();
#else
  static int sms = 0;
  if (!sms) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); }
  return sms ? sms : 80;
#endif
}
"""


def _xstage_store(c):
    """Statements converting `xv[32]` (one window, natural order) to f16 in slot order and storing 4 x 16 bytes."""
    pairs = []
    for i in range(16):
        lo, hi = SLOT_NAT[2 * i], SLOT_NAT[2 * i + 1]
        pairs.append(f"kcvt_h2(xv[{hi}], xv[{lo}])")
    out = [f"const uint32_t h{i} = {p};" for i, p in enumerate(pairs)]
    out += ["uint8_t *brow = Bs + (size_t)col * (BK * 2);"]
    for cidx in range(4):
        out.append(
            f"ksts128(brow + (((4 * wl + {cidx}) ^ (col & 7)) * 16), h{4 * cidx}, h{4 * cidx + 1}, h{4 * cidx + 2}, h{4 * cidx + 3});"
        )
    if c["weights"] == "q4_K":
        terms = " + ".join(f"kh2f_lo(h{i}) + kh2f_hi(h{i})" for i in range(16))
        out.append(f"XSs[col * NW + wl] = {terms};")
    return out


def kernel(c):
    w = c["weights"]
    e = ENGINE[w]
    bm, bn, bk, wm, wn, st = c["bm"], c["bn"], c["bk"], c["wm"], c["wn"], c["stages"]
    tm, tn = bm // wm, bn // wn
    mi, ni = tm // 16, tn // 8
    nt = wm * wn * 32
    kt = e["kt"]
    q4k = w == "q4_K"
    blk = sblock(w)
    lines = []
    lines.append(f"""
#define BM {bm}
#define BN {bn}
#define BK {bk}
#define KT {kt}
#define STAGES {st}
#define NT {nt}
#define RT (BM / 16)
#define KTS (BK / KT)
#define NW (BK / 32)
#define A_STAGE (RT * KTS * 512 + SB_STAGE)
#define AW_STAGE (RT * KTS * 512)
#define SB_STAGE ({sb_bytes(c)})
#define NBS ({nbs(c)})
#define SBLK ({256 if q4k else blk})
#define SREC ({256 if q4k else 32})
#define B_STAGE (BN * BK * 2)
#define XS_STAGE ({"BN * NW * 4" if q4k else "0"})
#define SMEM_BYTES ({smem_bytes(c)})
__device__ int kg_flags[{FLAG_SLOTS}];
""")
    if w == "e8p":
        from ..ext.compress import E8P_ABS2

        words = []
        for row in E8P_ABS2:
            vals = [2 * v for v in row]
            for i in range(4):
                lo = struct.unpack("<H", struct.pack("<e", float(vals[2 * i])))[0]
                hi = struct.unpack("<H", struct.pack("<e", float(vals[2 * i + 1])))[0]
                words.append(lo | (hi << 16))
        tab = ",\n".join("  " + ", ".join(f"0x{x:08x}u" for x in words[i : i + 8]) for i in range(0, len(words), 8))
        lines.append(f"// E8P codebook: 256 rows x 4 f16x2 pairs of 2|a_i| (2, 6, 10)\n__constant__ uint32_t kE8H[1024] = {{\n{tab}}};\n")
    deq = DEQ[w]()
    xstore = _xstage_store(c)
    lb = f"NT, {max(1, c['minb'])}"  # an explicit minimum of 1 block stops ptxas from targeting 128 registers and spilling
    # scale / meta access per window
    if q4k:
        scale_load = f"""        const uint8_t *m0 = Sb + ((size_t)((wr * {mi} + mm) * NBS + (kwin / 256 - sblk0)) * 16 + g) * 16, *m1 = m0 + 8 * 16;
        const int sb = (kwin % 256) / 32;
        #pragma unroll
        for (int h = 0; h < 2; h++) {{
          uint32_t r[4]; klds128g(r, h ? m1 : m0);
          const float d = kh2f_lo(r[0]), dmin = kh2f_hi(r[0]);
          #define SB(x) (((((x) >> 2) == 0 ? r[1] : ((x) >> 2) == 1 ? r[2] : r[3]) >> (8 * ((x) & 3))) & 0xFFu)
          const int sc = sb < 4 ? (int)(SB(sb) & 63) : (int)((SB(sb + 4) & 0xF) | ((SB(sb - 4) >> 6) << 4));
          const int mn = sb < 4 ? (int)(SB(sb + 4) & 63) : (int)((SB(sb + 4) >> 4) | ((SB(sb) >> 6) << 4));
          #undef SB
          dsc[mm][h] = d * (float)sc; dmn[mm][h] = dmin * (float)mn;
        }}"""
        apply = """            const float xs0 = XS[(wc * TN + nn * 8 + 2 * t) * NW + wlin], xs1 = XS[(wc * TN + nn * 8 + 2 * t + 1) * NW + wlin];
            acc[mm][nn][0] += dsc[mm][0] * d[0] - dmn[mm][0] * xs0;
            acc[mm][nn][1] += dsc[mm][0] * d[1] - dmn[mm][0] * xs1;
            acc[mm][nn][2] += dsc[mm][1] * d[2] - dmn[mm][1] * xs0;
            acc[mm][nn][3] += dsc[mm][1] * d[3] - dmn[mm][1] * xs1;"""
        scale_decl = f"float dsc[{mi}][2], dmn[{mi}][2];"
        scale_vars = "dsc, dmn"
    else:
        sw = f"const uint32_t sw = kld_u32(Sb + ((size_t)((wr * {mi} + mm) * NBS + (kwin / {blk} - sblk0)) * 8 + g) * 4);"
        scale_load = f"""        {sw}
        dw0[mm] = kh2f_lo(sw); dw1[mm] = kh2f_hi(sw);"""
        if w == "mxfp4":  # bf16 scale records
            scale_load = f"""        {sw}
        dw0[mm] = __uint_as_float(sw << 16); dw1[mm] = __uint_as_float(sw & 0xFFFF0000u);"""
        if w == "nvfp4":  # per row: the UE4M3 scale of this lane's 16-value sub-block (values 8t..8t+7 of the window)
            scale_load = f"""        {sw}
        const uint32_t hg = KURN_F2H(kue4m3h((sw >> (8 * (t >> 1))) & 0xFFu)), hg8 = KURN_F2H(kue4m3h((sw >> (16 + 8 * (t >> 1))) & 0xFFu));
        sg[mm] = hg | (hg << 16); sg8[mm] = hg8 | (hg8 << 16);
        dw0[mm] = 1.f; dw1[mm] = 1.f;"""
        apply = """            acc[mm][nn][0] += dw0[mm] * d[0];
            acc[mm][nn][1] += dw0[mm] * d[1];
            acc[mm][nn][2] += dw1[mm] * d[2];
            acc[mm][nn][3] += dw1[mm] * d[3];"""
        scale_decl = f"float dw0[{mi}], dw1[{mi}];" + (f" uint32_t sg[{mi}], sg8[{mi}];" if w == "nvfp4" else "")
        scale_vars = "dw0, dw1"
    del scale_vars
    xin32 = c["xin"] == "f32"
    lines.append(f"""
// A frag plane: [N/16][K/KT][32 lanes][16 B]; then {"meta records [N/16][K/256][16 rows][16 B]" if q4k else f"scale plane [N/16][K/{blk}][8][u32: row g | row g+8]"}
static __global__ void __launch_bounds__({lb})
kg_gemm(const uint8_t *__restrict__ W, const void *__restrict__ Xin, float *__restrict__ Y, int N, int K, int M, int ksplit) {{
#ifdef KURN_EMU
  static uint8_t kg_smem_emu[SMEM_BYTES + 16] __attribute__((aligned(16)));
  uint8_t *smem = kg_smem_emu;
#else
  extern __shared__ __align__(16) uint8_t kg_smem[];
  uint8_t *smem = kg_smem;
#endif
  uint8_t *As = smem;
  uint8_t *Bs0 = smem + STAGES * A_STAGE;
  float *XS0 = (float *)(Bs0 + STAGES * B_STAGE);
  (void)XS0;
  const int nkt = K / KT;
  const uint8_t *Wa = W;
  const uint8_t *Ws = W + (size_t)(N / 16) * nkt * 512;
  const uint8_t *Wm = Ws;
  (void)Ws; (void)Wm;
""")
    if w == "e8p":
        lines.append("""  uint32_t *e8tab = (uint32_t *)(smem + SMEM_BYTES - 4096);
  for (int i = threadIdx.x; i < 1024; i += NT) e8tab[i] = kE8H[i];
""")
    lines.append(f"""  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int wr = warp % {wm}, wc = warp / {wm};
  const int n0 = blockIdx.x * BM, m0 = blockIdx.y * BN, z = blockIdx.z;
  const int nks_all = (K + BK - 1) / BK, per = (nks_all + ksplit - 1) / ksplit;
  const int ks0 = z * per, ks1 = min(nks_all, ks0 + per), nks = ks1 > ks0 ? ks1 - ks0 : 0;
  float acc[{mi}][{ni}][4];
  #pragma unroll
  for (int i = 0; i < {mi}; i++)
    #pragma unroll
    for (int j = 0; j < {ni}; j++)
      #pragma unroll
      for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
  // weights for stage s (k-step ks0 + s) -> shared buffer buf, 16-byte cp.async, zero-filled past N or K
  // per-thread bases, hoisted out of the k loop: a CTA's stage is RT runs of KTS * 512 contiguous bytes (one per row tile)
  const uint8_t *abase = Wa + ((size_t)(n0 / 16) * nkt + (size_t)ks0 * KTS) * 512;
  const int nblkk = K / SBLK;
  auto issueA = [&](int s, int buf) {{
    const bool kfull = (ks0 + s + 1) * KTS <= nkt;
    #pragma unroll
    for (int i = 0; i < (RT * KTS * 32 + NT - 1) / NT; i++) {{
      const int e = tid + i * NT;
      if (RT * KTS * 32 % NT == 0 || e < RT * KTS * 32) {{
        const int rt = e / (KTS * 32), el = e % (KTS * 32);
        const bool ok = n0 / 16 + rt < N / 16 && (kfull || (ks0 + s) * KTS + el / 32 < nkt);
        const uint8_t *src = abase + ((size_t)rt * nkt + (size_t)s * KTS) * 512 + (size_t)el * 16;
        kcp16(As + (size_t)buf * A_STAGE + (size_t)e * 16, ok ? (const void *)src : (const void *)Wa, ok ? 16 : 0);
      }}
    }}
    const int blk0 = ((ks0 + s) * BK) / SBLK;
    for (int e = tid; e < RT * NBS * (SREC / 16); e += NT) {{  // block scales / meta records of this stage
      const int rt = e / (NBS * (SREC / 16)), bi = (e / (SREC / 16)) % NBS, ch = e % (SREC / 16);
      const int rtg = n0 / 16 + rt, bl = blk0 + bi;
      const bool ok = rtg < N / 16 && bl < nblkk;
      kcp16(As + (size_t)buf * A_STAGE + AW_STAGE + (size_t)e * 16,
            ok ? (const void *)(Ws + ((size_t)rtg * nblkk + bl) * SREC + ch * 16) : (const void *)Ws, ok ? 16 : 0);
    }}
  }};""")
    if xin32:
        lines.append(f"""  // activations: one (column, window) task per thread; f32 -> registers now, f16 slot-ordered shared store later
  const float *X = (const float *)Xin;
  float xv[32];
  auto loadB = [&](int s) {{
    const int col = tid / NW, wl = tid % NW, k = (ks0 + s) * BK + wl * 32;
    const bool ok = tid < BN * NW && m0 + col < M && k < K;
    #pragma unroll
    for (int i = 0; i < 8; i++) {{
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (ok) v = kld_v4(X + (size_t)(m0 + col) * K + k + 4 * i);
      xv[4 * i] = __uint_as_float(v.x); xv[4 * i + 1] = __uint_as_float(v.y);
      xv[4 * i + 2] = __uint_as_float(v.z); xv[4 * i + 3] = __uint_as_float(v.w);
    }}
  }};
  auto storeB = [&](int buf) {{  // columns >= M are never stored: their outputs are discarded and MMA columns are independent
    const int col = tid / NW, wl = tid % NW;
    if (tid >= BN * NW || m0 + col >= M) return;
    uint8_t *Bs = Bs0 + (size_t)buf * B_STAGE;
    float *XSs = XS0 + (size_t)buf * (XS_STAGE / 4);
    (void)XSs;
{chr(10).join("    " + x for x in xstore)}
  }};""")
    else:
        q4k_xs = (
            """
    for (int e = tid; e < BN * NW; e += NT) {
      const int col = e / NW, wl = e % NW, k = (ks0 + s) * BK + wl * 32;
      XS0[(size_t)buf * (XS_STAGE / 4) + e] = (m0 + col < M && k < K) ? Xsum[(size_t)(m0 + col) * (K / 32) + k / 32] : 0.f;
    }"""
            if q4k
            else ""
        )
        lines.append(f"""  // activations: f16 in slot order (kg_quant), 16-byte cp.async into the swizzled stage buffer
  const uint8_t *X16 = (const uint8_t *)Xin;
  const float *Xsum = (const float *)(X16 + (size_t)M * K * 2);
  (void)Xsum;
  auto issueB = [&](int s, int buf) {{
    const int kb = (ks0 + s) * BK;
    for (int e = tid; e < BN * (BK / 8); e += NT) {{
      const int col = e / (BK / 8), ch = e % (BK / 8);
      const bool ok = m0 + col < M && kb + ch * 8 < K;
      kcp16(Bs0 + (size_t)buf * B_STAGE + (size_t)col * (BK * 2) + ((ch ^ (col & 7)) * 16),
            ok ? (const void *)(X16 + ((size_t)(m0 + col) * K + kb + ch * 8) * 2) : (const void *)X16, ok ? 16 : 0);
    }}{q4k_xs}
  }};""")
    if q4k:
        lines.append("  auto klds128g = [](uint32_t r[4], const uint8_t *p) { klds128(r, p); };\n")
    lines.append(
        f"""  auto compute = [&](int buf, int s) {{
    const uint8_t *Ab = As + (size_t)buf * A_STAGE;
    const uint8_t *Sb = Ab + AW_STAGE;
    const int sblk0 = ((ks0 + s) * BK) / SBLK;
    const bool kfull = (ks0 + s + 1) * BK <= K;
    const uint8_t *Bb = Bs0 + (size_t)buf * B_STAGE;
    const float *XS = XS0 + (size_t)buf * (XS_STAGE / 4);
    (void)XS;
    #pragma unroll
    for (int kk = 0; kk < KTS; kk++) {{
      uint32_t qa[{mi}][4];
      #pragma unroll
      for (int mm = 0; mm < {mi}; mm++) klds128(qa[mm], Ab + ((size_t)((wr * {mi} + mm) * KTS + kk) * 32 + lane) * 16);
      #pragma unroll
      for (int wl = 0; wl < KT / 32; wl++) {{
        const int wlin = kk * (KT / 32) + wl;
        const int kwin = (ks0 + s) * BK + wlin * 32;
        if (!kfull && kwin >= K) continue;
        // dequantize every row tile's A fragments (rows past N have zero scales), then stream B one n8 tile at a time
        uint32_t am[{mi}][2][4];
        {scale_decl}
        #pragma unroll
        for (int mm = 0; mm < {mi}; mm++) {{
{scale_load}
          const uint32_t *q = qa[mm];
          uint32_t (&a)[2][4] = am[mm];
{chr(10).join("          " + x for x in deq)}
        }}
        #pragma unroll
        for (int nn = 0; nn < {ni}; nn++) {{
          uint32_t b[4];
          const int row = wc * TN_ + nn * 8 + (lane & 7);
          kldsm4(b, Bb + (size_t)row * (BK * 2) + (((4 * wlin + (lane >> 3)) ^ (row & 7)) * 16));
          #pragma unroll
          for (int mm = 0; mm < {mi}; mm++) {{
            float d[4];
            const float zz[4] = {{0.f, 0.f, 0.f, 0.f}};
            kmma16816(d, am[mm][0], b[0], b[1], zz);
            kmma16816(d, am[mm][1], b[2], b[3], d);
{apply}
          }}
        }}
      }}
    }}
  }};""".replace("TN_", str(tn)).replace("wc * TN +", f"wc * {tn} +")
    )
    if xin32:
        lines.append("""
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < nks) issueA(s, s);
    kcp_commit();
  }
  if (nks > 0) loadB(0);
  for (int ks = 0; ks < nks; ks++) {
    storeB(ks % STAGES);
    KCP_WAIT(STAGES - 2);
    __syncthreads();
    const int nx = ks + STAGES - 1;
    if (nx < nks) issueA(nx, nx % STAGES);
    kcp_commit();
    if (ks + 1 < nks) loadB(ks + 1);
    compute(ks % STAGES, ks);
  }
  KCP_WAIT(0);""")
    else:
        lines.append("""
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < nks) { issueA(s, s); issueB(s, s); }
    kcp_commit();
  }
  for (int ks = 0; ks < nks; ks++) {
    KCP_WAIT(STAGES - 2);
    __syncthreads();
    const int nx = ks + STAGES - 1;
    if (nx < nks) { issueA(nx, nx % STAGES); issueB(nx, nx % STAGES); }
    kcp_commit();
    compute(ks % STAGES, ks);
  }
  KCP_WAIT(0);""")
    lines.append(f"""  // epilogue: Y[m * N + n]; split z adds its tile after split z-1 (deterministic serial fixup)
  int *flag = &kg_flags[(blockIdx.y * gridDim.x + blockIdx.x) % {FLAG_SLOTS}];
  if (z > 0) {{
    if (tid == 0) while (atomicAdd(flag, 0) != z) {{ }}
    __syncthreads();
  }}
  #pragma unroll
  for (int mm = 0; mm < {mi}; mm++)
    #pragma unroll
    for (int nn = 0; nn < {ni}; nn++)
      #pragma unroll
      for (int e = 0; e < 4; e++) {{
        const int row = n0 + (wr * {mi} + mm) * 16 + g + 8 * (e >> 1), col = m0 + wc * {tn} + nn * 8 + 2 * t + (e & 1);
        if (row < N && col < M) {{
          float *y = Y + (size_t)col * N + row;
          *y = z > 0 ? kldcg(y) + acc[mm][nn][e] : acc[mm][nn][e];
        }}
      }}
  if (ksplit > 1) {{
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicExch(flag, z + 1 == ksplit ? 0 : z + 1);
  }}
}}
""")
    return "".join(lines)


def repack_kernel(c):
    w = c["weights"]
    e = ENGINE[w]
    f = FORMATS[w]
    fields = _fields(w)
    nf = max(len(x) for x in fields)
    rows = []
    for wd in fields:
        cells = [f"{{{s}, {wdt}, {rs}, {kb}}}" for s, wdt, rs, kb in wd]
        cells += ["{0, 0, 0, 0}"] * (nf - len(wd))
        rows.append("{" + ", ".join(cells) + "}")
    table = ",\n  ".join(rows)
    meta = (
        """  } else if (i < na + nmeta) {  // Q4_K meta records: 16 bytes per (row tile, block, row)
    const long r = i - na;
    const int rr = (int)(r % 16), bl = (int)((r / 16) % nblk), rt = (int)(r / 16 / nblk);
    const uint8_t *src = W + (size_t)(rt * 16 + rr) * rowb + (size_t)bl * 144;
    uint8_t *dst = P + (size_t)na * 4 + (size_t)r * 16;
    for (int b = 0; b < 16; b++) dst[b] = src[b];"""
        if w == "q4_K"
        else """  } else if (i < na + nmeta) {  // scale plane: u32 = d(row g) | d(row g+8) << 16
    const long r = i - na;
    const int gg = (int)(r % 8), bl = (int)((r / 8) % nblk), rt = (int)(r / 8 / nblk);
    kst_u32(P + (size_t)na * 4 + (size_t)r * 4, kscale16(W, rowb, rt * 16 + gg, bl) | (kscale16(W, rowb, rt * 16 + gg + 8, bl) << 16));"""
    )
    nmeta = "(long)(N / 16) * nblk * 16" if w == "q4_K" else "(long)(N / 16) * nblk * 8"
    return f"""
{_raw_decoder(w)}
// (shift, width, row select, k base) of every field of word w of a lane's 16 bytes; k = k-tile start + kbase + 8t
__constant__ signed short kfield[4][{nf}][4] = {{
  {table}}};
// native ggml blocks -> fragment-ordered A plane + scale plane; one thread per output word / record
static __global__ void kg_repack(const uint8_t *__restrict__ W, uint8_t *__restrict__ P, int N, int K) {{
  const size_t rowb = (size_t)(K / {f["block"]}) * {f["nbytes"]};
  const int nkt = K / {e["kt"]}, nblk = K / {sblock(w) if w != "q4_K" else 256};
  const long na = (long)(N / 16) * nkt * 128, nmeta = {nmeta};
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < na) {{
    const int wd = (int)(i % 4), ln = (int)((i / 4) % 32);
    const long tile = i / 128;
    const int kt = (int)(tile % nkt), rt = (int)(tile / nkt), g = ln >> 2, t = ln & 3;
    uint32_t word = 0;
    for (int fi = 0; fi < {nf}; fi++) {{
      const int sh = kfield[wd][fi][0], wdt = kfield[wd][fi][1];
      if (!wdt) continue;
      const int row = rt * 16 + g + 8 * kfield[wd][fi][2], k = kt * {e["kt"]} + kfield[wd][fi][3] + 8 * t;
      word |= (kraw(W, rowb, row, k) & ((1u << wdt) - 1u)) << sh;
    }}
    kst_u32(P + (size_t)i * 4, word);
{meta}
  }}
}}
"""


def prep_bytes_expr(c):
    w = c["weights"]
    e = ENGINE[w]
    a = f"(size_t)(N / 16) * (K / {e['kt']}) * 512"
    m = "(size_t)(N / 16) * (K / 256) * 256" if w == "q4_K" else f"(size_t)(N / 16) * (K / {sblock(w)}) * 32"
    return f"({a} + {m})"


def kmul(c):
    """K must be a multiple of this (the format block, KT, and 32)."""
    w = c["weights"]
    return max(block(w), ENGINE[w]["kt"], 256 if w == "q4_K" else 32)


def x16_kernel(c):
    """f32 activations -> f16 in slot order (+ per-window sums for Q4_K), for xin=f16."""
    pairs = [f"kcvt_h2(xv[{SLOT_NAT[2 * i + 1]}], xv[{SLOT_NAT[2 * i]}])" for i in range(16)]
    sums = ""
    if c["weights"] == "q4_K":
        sums = "\n  float sm = 0.f;\n  for (int i = 0; i < 16; i++) sm += kh2f_lo(h[i]) + kh2f_hi(h[i]);\n" \
               "  ((float *)(Q + (size_t)M * K * 2))[(size_t)m * (K / 32) + wdw] = sm;"  # fmt: skip
    return f"""
// f32 X [M][K] -> f16 X16 [M][K] in slot order (one thread per 32-value window){" + window sums [M][K/32]" if sums else ""}
static __global__ void kg_x16(const float *__restrict__ X, uint8_t *__restrict__ Q, int K, int M) {{
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)M * (K / 32)) return;
  const int m = (int)(i / (K / 32)), wdw = (int)(i % (K / 32));
  const float *x = X + (size_t)m * K + (size_t)wdw * 32;
  float xv[32];
  for (int j = 0; j < 32; j++) xv[j] = x[j];
  uint32_t h[16] = {{{", ".join(pairs)}}};
  uint8_t *dst = Q + ((size_t)m * K + (size_t)wdw * 32) * 2;
  for (int j = 0; j < 4; j++) ksts128(dst + 16 * j, h[4 * j], h[4 * j + 1], h[4 * j + 2], h[4 * j + 3]);{sums}
}}
"""


def abi(c, cfg):
    w = c["weights"]
    f16in = c["xin"] == "f16"
    xbytes = ("(size_t)M * K * 2" + (" + (size_t)M * (K / 32) * 4" if w == "q4_K" else "")) if f16in else "0"
    err = "#ifdef KURN_EMU\n  return 0;\n#else\n  return (int)cudaGetLastError();\n#endif"
    quant = (
        "  KURN_LAUNCH(kg_x16, dim3((unsigned)(((long)M * (K / 32) + 127) / 128)), dim3(128), s, X, (uint8_t *)Xq, K, M);\n" + err
        if f16in
        else "  (void)X; (void)Xq; (void)K; (void)M; (void)s;\n  return 0;"
    )
    nt = c["wm"] * c["wn"] * 32
    split = str(c["splitk"]) if c["splitk"] else "kg_auto_split(N, K, M)"
    return f"""
// split-K so the grid covers the GPU about twice (each split gets >= 2 k-steps), at most 16
static int kg_auto_split(int N, int K, int M) {{
  const long tiles = (long)((N + BM - 1) / BM) * ((M + BN - 1) / BN);
  const int nks = (K + BK - 1) / BK;
  long sk = (2L * kurn_sms() + tiles - 1) / tiles;
  if (sk > nks / 2) sk = nks / 2;
  if (sk > 16) sk = 16;
  return sk < 1 ? 1 : (int)sk;
}}

extern "C" {{
const char *kg_config(void) {{ return "{cfg}"; }}
int kg_act(void) {{ return {2 if f16in else 1}; }}  // 1: kg_run reads f32 X directly; 2: kg_quant converts X to f16 first
int kg_check_shape(int N, int K, int M) {{ return (N > 0 && M > 0 && N % 16 == 0 && K > 0 && K % {kmul(c)} == 0) ? 0 : -1; }}
size_t kg_prep_bytes(int N, int K) {{ return {prep_bytes_expr(c)}; }}
size_t kg_xbytes(int K, int M) {{ (void)K; (void)M; return {xbytes}; }}
int kg_prepare(const void *W, void *Wp, int N, int K, cudaStream_t s) {{
  const long n = (long)(N / 16) * (K / {ENGINE[w]["kt"]}) * 128 + (long)(N / 16) * (K / {256 if w == "q4_K" else sblock(w)}) * {16 if w == "q4_K" else 8};
  KURN_LAUNCH(kg_repack, dim3((unsigned)((n + 127) / 128)), dim3(128), s, (const uint8_t *)W, (uint8_t *)Wp, N, K);
{err}
}}
int kg_quant(const float *X, void *Xq, int K, int M, cudaStream_t s) {{
{quant}
}}
size_t kg_xblock_bytes(int K, int M) {{ (void)K; (void)M; return 0; }}
int kg_xblocks(const float *X, void *Xb, int K, int M, cudaStream_t s) {{ (void)X; (void)Xb; (void)K; (void)M; (void)s; return 0; }}
int kg_run(const void *W, const void *Xq, float *Y, int N, int K, int M, cudaStream_t s) {{
  if (kg_check_shape(N, K, M)) return -1;
  const int sk = {split};
#ifndef KURN_EMU
  static int attr = 0;
  if (!attr) {{ cudaFuncSetAttribute(kg_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); attr = 1; }}
#endif
  KURN_LAUNCH_SMEM(kg_gemm, dim3((unsigned)((N + BM - 1) / BM), (unsigned)((M + BN - 1) / BN), (unsigned)sk), dim3({nt}), SMEM_BYTES, s,
                   (const uint8_t *)W, Xq, Y, N, K, M, sk);
{err}
}}
}}
"""
