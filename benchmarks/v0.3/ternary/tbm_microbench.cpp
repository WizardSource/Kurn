// Single-thread, L2-resident: the tbm GEMV and decode knock-out variants against kurn's TQ2_0 GEMV, in TSC cycles per
// 64-weight word. Build (L = llama.cpp checkout after apply.sh, here with the tuned TQ2_0 config):
//   gcc -O3 -march=native -mno-avx512fp16 -I$L/ggml/src/ggml-cpu/kurn -c $L/ggml/src/ggml-cpu/kurn/kurn_tq2_0_gemv.c -o tq2.o
//   g++ -O3 -march=native -mno-avx512fp16 -std=c++17 tbm_microbench.cpp tq2.o -o mb && taskset -c 3 ./mb [N] [z]
#include "../../../integration/llama.cpp/ggml-kurn/kurn-tbm.cpp"
#include <vector>
#include <random>
extern "C" {
void *ktq20_gemv_prepare(const void *, int64_t, int64_t);
void ktq20_gemv_packed(const void *, const void *, float *, int64_t, int64_t, int64_t);
}

namespace {
// knock-out variants (V bit flags): 1 no pdep, 2 no popcnt, 4 no sign mask, 8 two chains, 16 kP from memory (asm),
// 32 popcounts per chunk by vpopcntq (stack array), 64 pdep with a memory mask operand (asm)
template <int R, int V>
void rec_v(const uint8_t * const * rec, const uint8_t * xb, __m512 * acc, const uint8_t ** next) {
    constexpr bool TWO = (V & 8) != 0;
    const __m512i ones = _mm512_set1_epi8(1), twos = _mm512_set1_epi8(2);
    __m512 d[R];
    uint64_t off[R];
    const uint8_t * sp[R];
    for (int r = 0; r < R; r++) {
        d[r] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec[r]));
        sp[r] = rec[r] + REC_FIXED;
        off[r] = 0;
    }
    float dx;
    memcpy(&dx, xb, 4);
    for (int kg = 0; kg < 8; kg++) {
        __m512i e[R], o[R];
        int16_t b0, b1;
        memcpy(&b0, xb + 4 + QK + 4 * kg, 2);
        memcpy(&b1, xb + 4 + QK + 4 * kg + 2, 2);
        for (int r = 0; r < R; r++) { e[r] = _mm512_set1_epi32(-(b0 + b1)); o[r] = _mm512_setzero_si512(); }
        alignas(64) uint64_t cnt[R][8];
        if (V & 32)
            for (int r = 0; r < R; r++)
                _mm512_store_si512((__m512i *) cnt[r], _mm512_popcnt_epi64(_mm512_loadu_si512((const __m512i *) (rec[r] + 32 + 64 * kg))));
#pragma GCC unroll 8
        for (int kk = 0; kk < 8; kk++) {
            int32_t q;
            memcpy(&q, xb + 4 + kg * 32 + kk * 4, 4);
            const __m512i xv = _mm512_set1_epi32(q);
            for (int r = 0; r < R; r++) {
                const uint8_t * bm = rec[r] + 32 + 8 * (kg * 8 + kk);
                if (V & 128) {
                    const uint64_t * bq = (const uint64_t *) bm;
                    const uint64_t * wq = (const uint64_t *) (sp[r] + (off[r] >> 3));
                    uint64_t n, w, S2;
                    __mmask64 kp2;
                    __asm__("kmovq %1, %0" : "=k"(kp2) : "m"(*bq));
                    __asm__("popcnt %1, %0" : "=r"(n) : "m"(*bq));
                    __asm__("shrx %2, %1, %0" : "=r"(w) : "m"(*wq), "r"(off[r] & 7));
                    __asm__("pdep %2, %1, %0" : "=r"(S2) : "r"(w), "m"(*bq));
                    off[r] += n;
                    const __m512i u2 = _mm512_mask_blend_epi8(kp2, ones, _mm512_maskz_mov_epi8(_cvtu64_mask64(S2), twos));
                    if (TWO && (kk & 1)) o[r] = _mm512_dpbusd_epi32(o[r], u2, xv);
                    else e[r] = _mm512_dpbusd_epi32(e[r], u2, xv);
                    continue;
                }
                const uint64_t P = ld64(bm);
                uint64_t w = ld64(sp[r] + (off[r] >> 3)) >> (off[r] & 7);
                uint64_t S;
                if (V & 1) S = w;
                else if (V & 64) __asm__("pdep %2, %1, %0" : "=r"(S) : "r"(w), "m"(*(const uint64_t *) bm));
                else S = _pdep_u64(w, P);
                off[r] += (V & 2) ? 37 : (V & 32) ? cnt[r][kk] : (uint64_t) _mm_popcnt_u64(P);
                __mmask64 kp;
                if (V & 16) __asm__("kmovq %1, %0" : "=k"(kp) : "m"(*(const uint64_t *) bm));
                else kp = _cvtu64_mask64(P);
                const __m512i u = (V & 4) ? _mm512_mask_blend_epi8(kp, ones, twos)
                                          : _mm512_mask_blend_epi8(kp, ones, _mm512_maskz_mov_epi8(_cvtu64_mask64(S), twos));
                if (TWO && (kk & 1)) o[r] = _mm512_dpbusd_epi32(o[r], u, xv);
                else e[r] = _mm512_dpbusd_epi32(e[r], u, xv);
            }
        }
        for (int r = 0; r < R; r++)
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(TWO ? _mm512_add_epi32(e[r], o[r]) : e[r]), _mm512_mul_ps(d[r], _mm512_set1_ps(dx)), acc[r]);
    }
    for (int r = 0; r < R; r++) next[r] = sp[r] + ((off[r] + 7) >> 3);
}

template <int R, int V>
void gemv_v(const void * buf, const uint8_t * X, float * Y, int64_t N) {
    const tbm_hdr * h = hdr_of(buf);
    const uint64_t * tab = table_of(buf);
    const uint8_t * data = data_of(buf);
    for (int64_t g = 0; g + R <= (N + 15) / 16; g += R) {
        const uint8_t * rec[R];
        __m512 acc[R];
        for (int r = 0; r < R; r++) { rec[r] = data + tab[g + r]; acc[r] = _mm512_setzero_ps(); }
        for (int64_t p = 0; p < h->nrec_k; p++) {
            const uint8_t * next[R];
            rec_v<R, V>(rec, X + p * XBLK, acc, next);
            for (int r = 0; r < R; r++) rec[r] = next[r];
        }
        for (int r = 0; r < R; r++) _mm512_storeu_ps(Y + (g + r) * 16, acc[r]);
    }
}
}  // namespace

template <class F>
double cyc(F f, int iters) {
    f();
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 5; rep++) {
        const uint64_t t0 = __rdtsc();
        for (int i = 0; i < iters; i++) f();
        best = std::min<uint64_t>(best, (uint64_t) (__rdtsc() - t0));
    }
    return (double) best / iters;
}

int main(int argc, char ** argv) {
    const int64_t K = 2560, N = argc > 1 ? atoll(argv[1]) : 512;
    const double z = argc > 2 ? atof(argv[2]) : 0.42;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> U(0, 1);
    const int64_t nb = K / 256;
    std::vector<uint8_t> W(N * nb * 66);
    for (int64_t b = 0; b < N * nb; b++) {
        uint8_t * o = &W[b * 66];
        memset(o, 0, 64);
        for (int v = 0; v < 256; v++) {
            const int q = U(rng) < z ? 1 : (U(rng) < 0.5 ? 0 : 2);
            o[(v / 128) * 32 + v % 32] |= (uint8_t) (q << (2 * ((v % 128) / 32)));
        }
        const uint16_t d = 0x2e66;
        memcpy(o + 64, &d, 2);
    }
    std::vector<uint8_t> X(nb * XBLK);
    for (int64_t b = 0; b < nb; b++) {
        float d = 0.01f;
        memcpy(&X[b * XBLK], &d, 4);
        int16_t bs[16] = {};
        for (int i = 0; i < 256; i++) { int8_t q = (int8_t) (rng() % 255 - 127); X[b * XBLK + 4 + i] = (uint8_t) q; bs[i / 16] += q; }
        memcpy(&X[b * XBLK + 260], bs, 32);
    }
    std::vector<uint8_t> T(kurn_tbm_bytes(K, N) + 64);
    uint8_t * tb = (uint8_t *) (((uintptr_t) T.data() + 63) & ~(uintptr_t) 63);
    const size_t used = kurn_tbm_pack(35, W.data(), K, N, tb, nullptr);
    void * pk = ktq20_gemv_prepare(W.data(), K, N);
    std::vector<float> Y(N + 64), Y2(N + 64);
    const double words = (double) K * N / 64;
    const int it = 2000;
    if (argc > 3) {
        const int which = atoi(argv[3]);
        for (int i = 0; i < 20000; i++) {
            if (which == 0) ktq20_gemv_packed(pk, X.data(), Y2.data(), K, 0, N);
            else if (which == 1) gemv_v<4, 0>(tb, X.data(), Y.data(), N);
            else if (which == 2) gemv_v<4, 4>(tb, X.data(), Y.data(), N);
            else if (which == 3) gemv_v<4, 2>(tb, X.data(), Y.data(), N);
        }
        return 0;
    }
    printf("K=%lld N=%lld z=%.2f tbm %zu B (%.3f bpw) tq2 %.0f B\n", (long long) K, (long long) N, z, used, 8.0 * used / (K * N), K * N * 2.0625 / 8);
    printf("tq2 tuned (rows=4)     %.3f cyc/word\n", cyc([&] { ktq20_gemv_packed(pk, X.data(), Y2.data(), K, 0, N); }, it) / words);
    printf("tbm shipped (cols_c)   %.3f cyc/word\n", cyc([&] { kurn_tbm_cols(tb, X.data(), 0, Y.data(), K, N, 1, 0, N); }, it) / words);
    printf("  bit-identical: %d\n", memcmp(Y.data(), Y2.data(), N * 4) == 0);
#define V(R, F, name) printf("tbm R=%d %-16s %.3f cyc/word\n", R, name, cyc([&] { gemv_v<R, F>(tb, X.data(), Y.data(), N); }, it) / words);
    V(4, 0, "base") V(4, 8, "two chains") V(2, 8, "two chains") V(8, 0, "base") V(4, 16, "kP mem")
    V(4, 1, "no pdep") V(4, 2, "no popcnt") V(4, 4, "no sign mask") V(4, 1 | 2 | 4, "no decode")
    V(4, 32, "vpopcnt") V(4, 32 | 16, "vpopcnt kPmem") V(4, 32 | 16 | 64, "vpop kPm pdepm") V(4, 32 | 64, "vpop pdepm")
    V(4, 128, "asm mem ops") V(4, 128 | 8, "asm mem two") V(2, 128 | 8, "asm mem two") V(8, 128, "asm mem ops") V(6, 128, "asm mem ops")
    V(4, 32 | 8, "vpopcnt two") V(2, 32 | 8, "vpopcnt two") V(8, 32, "vpopcnt") V(4, 2 | 16, "nopop kPmem")
    return 0;
}
