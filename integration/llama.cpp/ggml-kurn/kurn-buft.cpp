// KURN extra buffer type for ggml-cpu (see kurn/integration/llama.cpp/README.md).
//
// Weights whose type has a kurn kernel (kurn_dispatch.h, generated from the kurn kernel
// registry) are repacked once at load time into kurn's interleaved i16 layout, inside this
// buffer: there is one copy of the weights, owned by ggml like the AMX / CPU_REPACK buffers.
// MUL_MAT and MUL_MAT_ID (MoE experts) on those weights are computed here:
//   1 activation column      -> decode GEMV
//   2..8 columns             -> verify kernel (one weight pass for all columns)
//   more columns (prefill)   -> verify kernel over row chunks x groups of 8 columns, or for
//                               Q8_0 the AMX tile kernel below (opt-in: GGML_KURN_AMX=1)
// The GEMV, the verify kernels and the AMX kernel do the same per-column arithmetic (exact
// int32 dot per 32-value block, then acc = fma(float(isum), d_w * d_x, acc) in block
// order), so a token's result does not depend on how many tokens are computed with it.
//
// Environment: GGML_KURN=0 disables the buffer type; GGML_KURN_TYPES=q8_0,q4_0 restricts
// it to some formats; GGML_KURN_AMX=1 turns the AMX prefill kernel on (off by default: on VMs
// that do not preserve AMX tile state across context switches it returns wrong results) and
// GGML_KURN_AMX_MIN=<cols> sets its threshold (default 16); GGML_KURN_CHUNK_KB sets the
// prefill row-chunk size; GGML_KURN_CHUNKS=<n> hands rows out in n equal chunks per thread
// (0 = one static range per thread; default: guided, see row_chunks); GGML_KURN_VERBOSE=1 logs
// repacks.
//
// Activations are quantized once per op, split across all threads by blocks, and the per-block
// tables the kernels need (scales, block sums) are built in the same pass (the generated
// `_xprep` / `_packed_x` entry points), so the number of kernel calls per op costs nothing extra.
#include "kurn-buft.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "traits.h"
#include "kurn_dispatch.h"

#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__AMX_INT8__) && defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#define KURN_HAVE_AMX 1
#else
#define KURN_HAVE_AMX 0
#endif

namespace {

struct block_q8_0 { uint16_t d; int8_t qs[32]; };  // ggml's block_q8_0
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

bool verbose() {
    static const bool v = env_int("GGML_KURN_VERBOSE", 0) != 0;
    return v;
}

bool type_enabled(const kurn_kernel & k) {
    static const char * list = getenv("GGML_KURN_TYPES");
    if (!list || !*list) {
        return true;
    }
    std::string s = std::string(",") + list + ",";
    return s.find(std::string(",") + k.name + ",") != std::string::npos;
}

const kurn_kernel * find_kernel(enum ggml_type t) {
    for (const kurn_kernel & k : kurn_kernels) {
        if (k.type == (int) t) {
            return type_enabled(k) ? &k : nullptr;
        }
    }
    return nullptr;
}

#if defined(__AMX_INT8__) && defined(__linux__)
bool amx_ok = false;  // ggml's AMX buffer (and kurn's opt-in AMX prefill) can run: XTILEDATA granted
#endif

// Would ggml's own AMX extra buffer take this weight? (mirrors ggml-cpu/amx/amx.cpp supports_op)
bool ggml_amx_takes(const ggml_tensor * w) {
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__) && defined(__linux__)
    if (!amx_ok || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[1] % 32 != 0) {
        return false;
    }
    switch (w->type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q8_0:
            return w->ne[0] % 32 == 0;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS:
            return w->ne[0] % 256 == 0;
        default:
            return false;
    }
#else
    GGML_UNUSED(w);
    return false;
#endif
}

// Per-tensor fallback: weights for which kurn is not measured to beat the ggml path that would get
// them stay with ggml (llama.cpp then places them in the next extra buffer). Measured on Qwen3
// decode shapes, 8 threads (benchmarks/v0.2/q4fix): kurn beats CPU_REPACK and plain vec_dot on
// every shape, and ggml's AMX buffer (whose M = 1 path is a VNNI GEMV) from N >= 1024, K >= 1024;
// at N <= 512 results are mixed (Q8_0 2048 x 512: 0.94x). So when the AMX buffer can take the
// weight and N <= GGML_KURN_FALLBACK_N (512) or K < GGML_KURN_FALLBACK_K (512), ggml keeps it.
// GGML_KURN_FALLBACK=0 turns the rule off (kurn takes every weight it has a kernel for).
bool keep_with_ggml(const ggml_tensor * w) {
    static const bool on = env_int("GGML_KURN_FALLBACK", 1) != 0;
    static const int64_t max_n = env_int("GGML_KURN_FALLBACK_N", 512), min_k = env_int("GGML_KURN_FALLBACK_K", 512);
    return on && ggml_amx_takes(w) && (w->ne[1] <= max_n || w->ne[0] < min_k);
}

// Shapes the kernels handle: 2D or 3D (expert-stacked) contiguous weights, K within the
// generated kernels' limits, minus the per-tensor fallback.
const kurn_kernel * kernel_for(const ggml_tensor * w) {
    const kurn_kernel * k = find_kernel(w->type);
    if (!k || w->ne[3] != 1 || w->ne[0] % k->k_multiple != 0 || w->ne[0] > KURN_MAX_K || w->ne[1] < 1) {
        return nullptr;
    }
    if (keep_with_ggml(w)) {
        if (verbose()) {
            GGML_LOG_INFO("kurn: %s [%lld x %lld] %s stays with ggml (per-tensor fallback)\n", w->name, (long long) w->ne[0],
                          (long long) w->ne[1], ggml_type_name(w->type));
        }
        return nullptr;
    }
    return k;
}

size_t expert_bytes(const kurn_kernel * k, const ggml_tensor * w) {
    return k->bytes(w->ne[0], w->ne[1]);  // multiple of 64
}

int vfy_index(int64_t m) {
    return m <= 2 ? 0 : m <= 4 ? 1 : 2;
}

// Rows are handed out through ggml's shared chunk counter (thread ith starts at chunk ith).
// Default (guided): chunk ith is thread ith's own contiguous share of 3/4 of the rows, so each
// thread streams one long range (several short ranges per thread cost 15-30% of decode
// bandwidth); the last quarter goes out in small chunks so that a preempted or slow thread
// does not hold up the whole op (plain static ranges did, on a busy host).
// GGML_KURN_CHUNKS=0: static ranges only; GGML_KURN_CHUNKS=n > 0: n equal chunks per thread.
// Chunks are whole GEMV passes (row_pass rows: every row group streams) when there are at least as
// many passes as threads, otherwise single records (row_align rows) so that small N still uses
// every thread.
struct row_chunks {
    int64_t align = 1, big = 0, small = 1, nbig = 0, ng = 0, n = 0;

    row_chunks(int64_t N, int64_t row_align, int64_t row_pass, int nth) {
        static const int per_thread = env_int("GGML_KURN_CHUNKS", -1);
        align = N / row_pass >= nth ? row_pass : row_align;
        ng = (N + align - 1) / align;
        if (per_thread >= 0) {
            const int64_t want = per_thread == 0 ? nth : (int64_t) nth * per_thread;
            small = (ng + want - 1) / want;
            n = (ng + small - 1) / small;
            return;
        }
        if (ng < 4 * (int64_t) nth) {  // small op: chunk overhead outweighs balancing, one range per thread
            small = (ng + nth - 1) / nth;
            n = (ng + small - 1) / small;
            return;
        }
        big = ng * 3 / 4 / nth;
        nbig = nth;
        const int64_t tail = ng - nbig * big;
        small = std::max<int64_t>(1, (tail + 2 * nth - 1) / (2 * nth));
        n = nbig + (tail + small - 1) / small;
    }

    void range(int64_t N, int64_t c, int64_t & r0, int64_t & r1) const {
        int64_t g0, g1;
        if (c < nbig) {
            g0 = c * big;
            g1 = g0 + big;
        } else {
            g0 = nbig * big + (c - nbig) * small;
            g1 = g0 + small;
        }
        r0 = std::min(N, std::min(ng, g0) * align);
        r1 = std::min(N, std::min(ng, g1) * align);
    }
};

// ------------------------------------------------------------------------------------------
// AMX prefill for Q8_0 on the i16 layout. A record (16 rows x 32 values) is 16 fp16 scales
// followed by 8 lines of 64 bytes; line kk holds values 4kk..4kk+3 of the 16 rows, biased by
// +128 (u8). That is exactly the B-tile format (K/4 rows x 16 columns x 4 bytes), so a record
// feeds one tdpbsud (signed activations x unsigned weights). The -128 * sum(x) correction
// and the scaling are applied per block, as in the GEMV, which keeps the result identical.
#if KURN_HAVE_AMX
constexpr int KURN_REC = 544;  // must match REC_BYTES of the generated Q8_0 kernels

// Same palette as ggml's AMX code (ggml_tile_config_init), which configures each thread once
// and never again: tmm0/1 B (8 x 64 B), tmm2/3 A (16 x 32 B), tmm4..7 C (16 x 16 int32).
void amx_config() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    c.rows[0] = 8;  c.colsb[0] = 64;
    c.rows[1] = 8;  c.colsb[1] = 64;
    c.rows[2] = 16; c.colsb[2] = 32;
    c.rows[3] = 16; c.colsb[3] = 32;
    for (int t = 4; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

// Per activation row and block: d_x as float and -128 * sum(x) (the u8 weight bias).
void amx_prep_rows(const char * X, size_t xrow, int64_t nk, int64_t row0, int64_t row1, float * dx, int32_t * nc) {
    const __m256i bias = _mm256_set1_epi8((char) 0x80);
    for (int64_t m = row0; m < row1; m++) {
        const block_q8_0 * xb = (const block_q8_0 *) (X + m * xrow);
        for (int64_t k = 0; k < nk; k++) {
            const __m256i v = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) xb[k].qs), bias);
            const __m256i s = _mm256_sad_epu8(v, _mm256_setzero_si256());
            const int64_t t = _mm256_extract_epi64(s, 0) + _mm256_extract_epi64(s, 1) + _mm256_extract_epi64(s, 2) +
                              _mm256_extract_epi64(s, 3);
            dx[m * nk + k] = _cvtsh_ss(xb[k].d);
            nc[m * nk + k] = -128 * (int32_t) (t - 32 * 128);
        }
    }
}

// Y[m*N + n] for n in [r0, r1), 16 columns m. X: 16 rows of K/32 block_q8_0 (row stride xrow),
// dx / nc: their amx_prep_rows values (row stride nk).
void amx_q8_0(const uint8_t * buf, int64_t nrec_k, const char * X, size_t xrow, const float * dx, const int32_t * nc,
              float * Y, int64_t K, int64_t N, int64_t r0, int64_t r1) {
    const int64_t nk = K / 32;
    alignas(64) int32_t cbuf[4][16 * 16];
    amx_config();
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    for (int64_t g = g0; g < g1; g += 4) {
        const int ng = (int) std::min<int64_t>(4, g1 - g);
        __m512 acc[4][16];
        for (int j = 0; j < 4; j++) {
            for (int m = 0; m < 16; m++) {
                acc[j][m] = _mm512_setzero_ps();
            }
        }
        for (int64_t k = 0; k < nk; k++) {
            const uint8_t * rec[4];
            for (int j = 0; j < ng; j++) {
                rec[j] = buf + ((size_t) (g + j) * nrec_k + k) * KURN_REC;
            }
            _tile_loadd(2, X + k * sizeof(block_q8_0) + 2, (long) xrow);
            _tile_zero(4);
            _tile_loadd(0, rec[0] + 32, 64);
            _tile_dpbsud(4, 2, 0);
            if (ng > 1) { _tile_zero(5); _tile_loadd(1, rec[1] + 32, 64); _tile_dpbsud(5, 2, 1); }
            if (ng > 2) { _tile_zero(6); _tile_loadd(0, rec[2] + 32, 64); _tile_dpbsud(6, 2, 0); }
            if (ng > 3) { _tile_zero(7); _tile_loadd(1, rec[3] + 32, 64); _tile_dpbsud(7, 2, 1); }
            _tile_stored(4, cbuf[0], 64);
            if (ng > 1) { _tile_stored(5, cbuf[1], 64); }
            if (ng > 2) { _tile_stored(6, cbuf[2], 64); }
            if (ng > 3) { _tile_stored(7, cbuf[3], 64); }
            for (int j = 0; j < ng; j++) {
                const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec[j]));
                for (int m = 0; m < 16; m++) {
                    const __m512i isum = _mm512_add_epi32(_mm512_load_si512(cbuf[j] + m * 16), _mm512_set1_epi32(nc[m * nk + k]));
                    acc[j][m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), _mm512_mul_ps(dw, _mm512_set1_ps(dx[m * nk + k])), acc[j][m]);
                }
            }
        }
        for (int j = 0; j < ng; j++) {
            const int64_t row0 = (g + j) * 16;
            uint32_t mask = 0xFFFF;
            if (row0 < r0) mask &= 0xFFFFu << (r0 - row0);
            if (row0 + 16 > r1) mask &= (1u << (r1 > row0 ? r1 - row0 : 0)) - 1;
            for (int m = 0; m < 16; m++) {
                _mm512_mask_storeu_ps(Y + m * N + row0, (__mmask16) mask, acc[j][m]);
            }
        }
    }
}
#endif

// ------------------------------------------------------------------------------------------
// GGML_KURN_PROFILE=1: per-phase cycle counts of MUL_MAT (decode-sized ops, M <= 8), printed at exit.
// Phases per thread: quantize + activation prep, wait at the post-quantize barrier, kernels; plus,
// per op, the spread between the first and the last thread to finish its kernels (load imbalance,
// paid as wait time at ggml's barrier after the node).
struct kurn_profile {
    static constexpr int MAXT = 64;
    struct alignas(64) slot { uint64_t quant = 0, wait = 0, kern = 0, ops = 0, done = 0; };
    slot t[MAXT];
    uint64_t spread = 0, nops = 0;
    bool on = env_int("GGML_KURN_PROFILE", 0) != 0;
    ~kurn_profile() {
        if (!on || !nops) {
            return;
        }
        uint64_t q = 0, w = 0, k = 0, n = 0;
        for (const slot & s : t) { q += s.quant; w += s.wait; k += s.kern; n += s.ops; }
        fprintf(stderr, "kurn profile: %llu ops; per op and thread (kcycles): quantize+prep %.2f, barrier wait %.2f, kernels %.2f; "
                "finish spread %.2f kcycles/op\n", (unsigned long long) nops, 1e-3 * q / n, 1e-3 * w / n, 1e-3 * k / n,
                1e-3 * spread / nops);
    }
};
kurn_profile g_prof;

struct tensor_traits : ggml::cpu::tensor_traits {
    const kurn_kernel * k = nullptr;
    std::vector<void *> views;  // packed_t header per expert, pointing into the tensor data
    std::vector<uint8_t *> data;
    size_t stride = 0;
    bool amx = false;

    ~tensor_traits() override {
        for (void * v : views) {
            free(v);
        }
    }

    static int64_t amx_min() {
        static const int64_t v = std::max(16, env_int("GGML_KURN_AMX_MIN", 16));
        return v;
    }

    bool use_amx(int64_t M) const {
        return amx && M >= amx_min();
    }

    // xw: shared activation prep of these columns (k->xprep layout for xc columns, starting at column xc0)
    struct xprep_ref {
        const void * ws = nullptr;
        int64_t C = 0, c0 = 0;
    };

    // one activation column, rows [r0, r1) (r0 a multiple of row_align): full-width passes of `gemv`
    // over the bulk, the one-record `gemv1` over the rest
    void gemv_rows(const void * pk, const char * X, float * Y, int64_t K, int64_t r0, int64_t r1, const xprep_ref & xr,
                   int64_t col) const {
        const int64_t mid = std::min(r1, r0 + (r1 - r0) / k->row_pass * k->row_pass);
        if (xr.ws) {
            if (mid > r0) k->gemv_x(pk, X, xr.ws, xr.C, xr.c0 + col, Y, K, r0, mid);
            if (r1 > mid) k->gemv1_x(pk, X, xr.ws, xr.C, xr.c0 + col, Y, K, mid, r1);
        } else {
            if (mid > r0) k->gemv(pk, X, Y, K, r0, mid);
            if (r1 > mid) k->gemv1(pk, X, Y, K, mid, r1);
        }
    }

    void cols(const void * pk, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
              int64_t r1, const xprep_ref & xr) const {
        for (int64_t c0 = 0; c0 < M; c0 += KURN_VFY_MAX) {
            const int64_t m = std::min<int64_t>(KURN_VFY_MAX, M - c0);
            if (m == 1) {
                gemv_rows(pk, X + c0 * xrow, Y + c0 * N, K, r0, r1, xr, c0);
            } else if (xr.ws) {
                k->vfy_x[vfy_index(m)](pk, X + c0 * xrow, xr.ws, xr.C, xr.c0 + c0, Y + c0 * N, K, N, m, r0, r1);
            } else {
                k->vfy[vfy_index(m)](pk, X + c0 * xrow, Y + c0 * N, K, N, m, r0, r1);
            }
        }
    }

    // M columns of quantized activations X (row stride xrow) -> Y (column stride N), rows [r0, r1).
    // dx / nc: AMX activation prep for the same rows (only when use_amx(M)).
    void run(int e, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1,
             const xprep_ref & xr, const float * dx = nullptr, const int32_t * nc = nullptr) const {
        const void * pk = views[e];
        if (r0 >= r1) {
            return;
        }
        if (M <= KURN_VFY_MAX) {
            cols(pk, X, xrow, Y, K, N, M, r0, r1, xr);
            return;
        }
        // prefill: row chunks that stay in L2 while every column group passes over them
        static const int64_t chunk_bytes = (int64_t) env_int("GGML_KURN_CHUNK_KB", 1024) * 1024;
        const int64_t row_bytes = std::max<int64_t>(1, (int64_t) (stride / (size_t) N));
        int64_t chunk = std::max<int64_t>(64, chunk_bytes / row_bytes);
        chunk = (chunk + 63) / 64 * 64;
        for (int64_t a = r0; a < r1; a += chunk) {
            const int64_t b = std::min(r1, a + chunk);
#if KURN_HAVE_AMX
            if (dx && use_amx(M)) {
                const int64_t nk = K / 32, full = M / 16 * 16;
                for (int64_t c0 = 0; c0 < full; c0 += 16) {
                    amx_q8_0(data[e], nk, X + c0 * xrow, xrow, dx + c0 * nk, nc + c0 * nk, Y + c0 * N, K, N, a, b);
                }
                xprep_ref xt = xr;
                xt.c0 += full;
                cols(pk, X + full * xrow, xrow, Y + full * N, K, N, M - full, a, b, xt);
                continue;
            }
#else
            GGML_UNUSED(dx); GGML_UNUSED(nc);
#endif
            cols(pk, X, xrow, Y, K, N, M, a, b, xr);
        }
    }

    size_t xprep_bytes(int64_t K, int64_t C) const {
        return k->xprep_bytes ? k->xprep_bytes(K, C) + 64 : 0;
    }

    size_t amx_bytes(int64_t rows, int64_t K) const {
        return amx ? (size_t) rows * (K / 32) * 8 + 128 : 0;
    }

    bool work_size(int /*n_threads*/, const ggml_tensor * op, size_t & size) override {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0];
        const size_t xrow = ggml_row_size((ggml_type) k->vec_dot_type, K);
        if (op->op == GGML_OP_MUL_MAT) {
            size = xrow * (size_t) ggml_nrows(src1) + amx_bytes(ggml_nrows(src1), K) + xprep_bytes(K, ggml_nrows(src1)) + 64;
            return true;
        }
        const ggml_tensor * ids = op->src[2];
        const size_t nsel = (size_t) ids->ne[0] * ids->ne[1];
        const size_t E = (size_t) src0->ne[2];
        size = xrow * (size_t) (src1->ne[1] * src1->ne[2]) + xrow * nsel + sizeof(float) * nsel * src0->ne[1] +
               sizeof(int64_t) * (3 * E + 2) + sizeof(int32_t) * nsel + amx_bytes((int64_t) nsel, K) +
               xprep_bytes(K, (int64_t) nsel) + 8 * 64;
        return true;
    }

    // Quantize every row of src1 into wq, split across threads by activation blocks (one decode
    // row is shared by all threads instead of quantized by thread 0 alone), and, when xw is
    // given, build the shared activation prep of the same blocks.
    void quantize_rows(const ggml_compute_params * p, const ggml_tensor * src1, ggml_from_float_t from_float, char * wq,
                       size_t xrow, void * xw) const {
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2], K = src1->ne[0];
        const int64_t n = ggml_nrows(src1);
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const int64_t bs = ggml_blck_size(vdt), ts = (int64_t) ggml_type_size(vdt), nb = K / bs;
        static const bool split = env_int("GGML_KURN_QSPLIT", 0) != 0;  // 1: split rows by blocks
        const int64_t total = n * nb;
        int64_t i0 = total * p->ith / p->nth, i1 = total * (p->ith + 1) / p->nth;
        if (!split) {
            const int64_t r0 = n * p->ith / p->nth, r1 = n * (p->ith + 1) / p->nth;
            i0 = r0 * nb;
            i1 = r1 * nb;
        }
        for (int64_t i = i0; i < i1;) {
            const int64_t c = i / nb, b0 = i % nb, b1 = std::min(nb, b0 + (i1 - i));
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            from_float(src + b0 * bs, wq + c * xrow + b0 * ts, (b1 - b0) * bs);
            if (xw) {
                k->xprep(wq, K, n, c, c + 1, b0 * (bs / 32), b1 * (bs / 32), xw);
            }
            i += b1 - b0;
        }
    }

    // AMX activation prep for rows [0, nrows) of X, split across threads (caller barriers)
    void prep(const ggml_compute_params * p, const char * X, size_t xrow, int64_t nrows, int64_t K, float * dx,
              int32_t * nc) const {
#if KURN_HAVE_AMX
        const int64_t a = nrows * p->ith / p->nth, b = nrows * (p->ith + 1) / p->nth;
        amx_prep_rows(X, xrow, K / 32, a, b, dx, nc);
#else
        GGML_UNUSED(p); GGML_UNUSED(X); GGML_UNUSED(xrow); GGML_UNUSED(nrows); GGML_UNUSED(K); GGML_UNUSED(dx); GGML_UNUSED(nc);
#endif
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            mul_mat(params, op);
            return true;
        }
        if (op->op == GGML_OP_MUL_MAT_ID) {
            mul_mat_id(params, op);
            return true;
        }
        return false;
    }

    static char * align64(void * p) {
        return (char *) (((uintptr_t) p + 63) & ~(uintptr_t) 63);
    }

    void mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0], N = src0->ne[1], M = ggml_nrows(src1);
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const size_t xrow = ggml_row_size(vdt, K);
        char * wq = align64(params->wdata);
        float * dx = (float *) align64(wq + xrow * M);
        int32_t * nc = (int32_t *) (dx + M * (K / 32));
        static const bool use_xprep = env_int("GGML_KURN_XPREP", 1) != 0;
        void * xw = k->xprep && use_xprep ? align64((char *) dx + amx_bytes(M, K)) : nullptr;
        const bool prof = g_prof.on && M <= KURN_VFY_MAX && params->ith < kurn_profile::MAXT;
        const uint64_t t0 = prof ? __rdtsc() : 0;
        quantize_rows(params, src1, ggml_get_type_traits_cpu(vdt)->from_float, wq, xrow, xw);
        const xprep_ref xr = { xw, M, 0 };
        if (params->ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, params->nth);
        }
        const uint64_t t1 = prof ? __rdtsc() : 0;
        ggml_barrier(params->threadpool);
        const uint64_t t2 = prof ? __rdtsc() : 0;
        if (use_amx(M)) {
            prep(params, wq, xrow, M, K, dx, nc);
            ggml_barrier(params->threadpool);
        }
        const row_chunks rc(N, k->row_align, k->row_pass, params->nth);
        for (int64_t c = params->ith; c < rc.n; c = rc.n > params->nth ? ggml_threadpool_chunk_add(params->threadpool, 1) : rc.n) {
            int64_t r0, r1;
            rc.range(N, c, r0, r1);
            run(0, wq, xrow, (float *) op->data, K, N, M, r0, r1, xr, dx, nc);
        }
        if (prof) {
            const uint64_t t3 = __rdtsc();
            kurn_profile::slot & sl = g_prof.t[params->ith];
            sl.quant += t1 - t0; sl.wait += t2 - t1; sl.kern += t3 - t2; sl.ops++; sl.done = t3;
            ggml_barrier(params->threadpool);  // profiling only: read every thread's finish time
            if (params->ith == 0) {
                uint64_t lo = UINT64_MAX, hi = 0;
                for (int i = 0; i < params->nth && i < kurn_profile::MAXT; i++) {
                    lo = std::min(lo, g_prof.t[i].done);
                    hi = std::max(hi, g_prof.t[i].done);
                }
                g_prof.spread += hi - lo;
                g_prof.nops++;
            }
        }
    }

    void mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids = op->src[2];
        const int64_t K = src0->ne[0], N = src0->ne[1], E = src0->ne[2], nk = K / 32;
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
        const int64_t n_ids = ids->ne[0], nsel = n_ids * ids->ne[1];
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const size_t xrow = ggml_row_size(vdt, K);
        char * wq = align64(params->wdata);
        char * gx = align64(wq + xrow * ne11 * ne12);
        float * tmp = (float *) align64(gx + xrow * nsel);
        int64_t * cnt = (int64_t *) align64(tmp + nsel * N);
        int64_t * off = cnt + E;
        int32_t * sel = (int32_t *) align64(off + E + 1);  // sorted by expert: t * n_ids + i
        int64_t * act = (int64_t *) align64(sel + nsel);    // experts with tokens, act[E] = count
        float * dx = (float *) align64(act + E + 1);
        int32_t * nc = (int32_t *) (dx + nsel * nk);
        void * xw = k->xprep ? align64((char *) dx + amx_bytes(nsel, K)) : nullptr;
        const int ith = params->ith, nth = params->nth;

        quantize_rows(params, src1, ggml_get_type_traits_cpu(vdt)->from_float, wq, xrow, nullptr);
        if (ith == 0) {
            std::fill(cnt, cnt + E, 0);
            for (int64_t t = 0; t < ids->ne[1]; t++) {
                for (int64_t i = 0; i < n_ids; i++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0] + t * ids->nb[1]);
                    GGML_ASSERT(e >= 0 && e < E);
                    cnt[e]++;
                }
            }
            off[0] = 0;
            for (int64_t e = 0; e < E; e++) {
                off[e + 1] = off[e] + cnt[e];
            }
            act[E] = 0;
            for (int64_t e = 0; e < E; e++) {
                if (cnt[e] > 0) {
                    act[act[E]++] = e;
                }
            }
            ggml_threadpool_chunk_set(params->threadpool, nth);
            std::vector<int64_t> pos(off, off + E);
            for (int64_t t = 0; t < ids->ne[1]; t++) {
                for (int64_t i = 0; i < n_ids; i++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0] + t * ids->nb[1]);
                    sel[pos[e]++] = (int32_t) (t * n_ids + i);
                }
            }
        }
        ggml_barrier(params->threadpool);
        for (int64_t j = ith; j < nsel; j += nth) {
            const int64_t t = sel[j] / n_ids, i = sel[j] % n_ids;
            memcpy(gx + j * xrow, wq + (t * ne11 + i % ne11) * xrow, xrow);
            if (xw) {
                k->xprep(gx, K, nsel, j, j + 1, 0, nk, xw);
            }
        }
        ggml_barrier(params->threadpool);
        const int64_t max_cnt = *std::max_element(cnt, cnt + E);
        if (use_amx(max_cnt)) {
            prep(params, gx, xrow, nsel, K, dx, nc);
            ggml_barrier(params->threadpool);
        }
        const row_chunks rc(N, k->row_align, k->row_pass, nth);
        const int64_t nchunk = rc.n * act[E];
        for (int64_t c = ith; c < nchunk; c = nchunk > nth ? ggml_threadpool_chunk_add(params->threadpool, 1) : nchunk) {
            const int64_t e = act[c / rc.n];
            int64_t r0, r1;
            rc.range(N, c % rc.n, r0, r1);
            float * Y = tmp + off[e] * N;
            if (cnt[e] == 1) {  // write straight into dst
                const int64_t t = sel[off[e]] / n_ids, i = sel[off[e]] % n_ids;
                Y = (float *) ((char *) op->data + i * op->nb[1] + t * op->nb[2]);
            }
            const xprep_ref xr = { xw, nsel, off[e] };
            run((int) e, gx + off[e] * xrow, xrow, Y, K, N, cnt[e], r0, r1, xr, dx + off[e] * nk, nc + off[e] * nk);
        }
        ggml_barrier(params->threadpool);
        for (int64_t j = ith; j < nsel; j += nth) {
            const int64_t t = sel[j] / n_ids, i = sel[j] % n_ids;
            const int64_t e = std::upper_bound(off, off + E + 1, j) - off - 1;
            if (cnt[e] > 1) {
                memcpy((char *) op->data + i * op->nb[1] + t * op->nb[2], tmp + j * N, sizeof(float) * N);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------
struct buffer_ctx {
    ggml_backend_buffer_t inner;
    std::vector<std::unique_ptr<tensor_traits>> traits;
    std::mutex mu;
};

void buf_free(ggml_backend_buffer_t b) {
    auto * c = (buffer_ctx *) b->context;
    ggml_backend_buffer_free(c->inner);
    delete c;
}

void * buf_base(ggml_backend_buffer_t b) {
    return ggml_backend_buffer_get_base(((buffer_ctx *) b->context)->inner);
}

enum ggml_status buf_init_tensor(ggml_backend_buffer_t b, ggml_tensor * t) {
    auto * c = (buffer_ctx *) b->context;
    t->extra = nullptr;
    if (t->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    const kurn_kernel * k = kernel_for(t);
    if (!k) {
        return GGML_STATUS_SUCCESS;
    }
    auto tt = std::make_unique<tensor_traits>();
    tt->k = k;
    tt->stride = expert_bytes(k, t);
#if KURN_HAVE_AMX
    tt->amx = amx_ok && k->type == GGML_TYPE_Q8_0 && strstr(k->config, "layout=i16") && strstr(k->config, "correction=act") &&
              env_int("GGML_KURN_AMX", 0) != 0;
#endif
    for (int64_t e = 0; e < t->ne[2]; e++) {
        uint8_t * p = (uint8_t *) t->data + e * tt->stride;
        tt->data.push_back(p);
        tt->views.push_back(k->view(p, t->ne[0], t->ne[1]));
    }
    t->extra = tt.get();
    std::lock_guard<std::mutex> lock(c->mu);
    c->traits.push_back(std::move(tt));
    return GGML_STATUS_SUCCESS;
}

void buf_set_tensor(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    auto * tt = (tensor_traits *) t->extra;
    if (!tt) {
        memcpy((char *) t->data + offset, data, size);
        return;
    }
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(t));
    if (verbose()) {
        GGML_LOG_INFO("kurn: repack %s [%lld x %lld x %lld] %s -> %s (%zu -> %zu bytes)\n", t->name, (long long) t->ne[0],
                      (long long) t->ne[1], (long long) t->ne[2], ggml_type_name(t->type), tt->k->config, size,
                      tt->stride * (size_t) t->ne[2]);
    }
    for (int64_t e = 0; e < t->ne[2]; e++) {
        tt->k->pack((const char *) data + e * t->nb[2], t->ne[0], t->ne[1], tt->data[e]);
    }
}

void buf_memset_tensor(ggml_backend_buffer_t, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    memset((char *) t->data + offset, value, size);
}

void buf_clear(ggml_backend_buffer_t b, uint8_t value) {
    ggml_backend_buffer_clear(((buffer_ctx *) b->context)->inner, value);
}

const ggml_backend_buffer_i buf_iface = {
    /* .free_buffer     = */ buf_free,
    /* .get_base        = */ buf_base,
    /* .init_tensor     = */ buf_init_tensor,
    /* .memset_tensor   = */ buf_memset_tensor,
    /* .set_tensor      = */ buf_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ buf_clear,
    /* .reset           = */ nullptr,
};

const char * buft_name(ggml_backend_buffer_type_t) {
    return "KURN";
}

ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t inner = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (!inner) {
        return nullptr;
    }
    auto * c = new buffer_ctx;
    c->inner = inner;
    return ggml_backend_buffer_init(buft, buf_iface, c, size);
}

size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

size_t buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    const kurn_kernel * k = kernel_for(t);
    return k ? expert_bytes(k, t) * (size_t) t->ne[2] : ggml_nbytes(t);
}

ggml_backend_buffer_type_t kurn_buft();

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        if (!src0->buffer || src0->buffer->buft != kurn_buft() || !kernel_for(src0) || !ggml_is_contiguous(src0)) {
            return false;
        }
        if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
            return false;
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;
        }
        if (op->op == GGML_OP_MUL_MAT) {
            return src0->ne[2] == 1;
        }
        return op->src[2]->type == GGML_TYPE_I32 && src1->ne[3] == 1 && op->nb[1] == (size_t) src0->ne[1] * sizeof(float);
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && op->src[0]->buffer &&
            op->src[0]->buffer->buft == kurn_buft()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }
        return nullptr;
    }
};

ggml_backend_buffer_type_t kurn_buft() {
    static ggml_backend_buffer_type t = {
        /* .iface = */ {
            /* .get_name         = */ buft_name,
            /* .alloc_buffer     = */ buft_alloc,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buft_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ buft_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new extra_buffer_type(),
    };
    return &t;
}

}  // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void) {
    static const bool enabled = [] {
        if (env_int("GGML_KURN", 1) == 0) {
            return false;
        }
#if KURN_HAVE_AMX
        // ARCH_REQ_XCOMP_PERM for XTILEDATA (same request as ggml's AMX buffer)
        amx_ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0;
#endif
        return true;
    }();
    return enabled ? kurn_buft() : nullptr;
}

#else

ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void) {
    return nullptr;
}

#endif
