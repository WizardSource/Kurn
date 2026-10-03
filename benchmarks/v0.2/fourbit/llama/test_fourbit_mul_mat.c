// ggml-cpu MUL_MAT through the kurn hook (GGML_KURN=1) for the fourbit formats, against a
// reference from ggml's own vec_dot on the same quantized activations.
//   test_fourbit_mul_mat TYPE...   (q8_0 q4_0 iq4_nl q4_K mxfp4 nvfp4); exit code = failures
// Build against a llama.cpp patched with ggml-kurn-fourbit.patch:
//   gcc -O2 test_fourbit_mul_mat.c -I$L/ggml/include -L$L/build/bin -lggml -lggml-base -lggml-cpu
//       -Wl,-rpath,$L/build/bin -lm -o test_fourbit_mul_mat
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 88172645463325252ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static int nfail, npass;

static const void *raw_w;  // weights from a file (test_fourbit_mul_mat --raw FILE TYPE K N)
static const float *raw_x;  // optional activation rows (--raw ... XFILE)

static void run_case(ggml_backend_t be, enum ggml_type wt, int64_t K, int64_t N, int64_t M, int threads) {
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(wt);
    const enum ggml_type xt = tt->vec_dot_type;
    const size_t wrow = ggml_row_size(wt, K), xrow = ggml_row_size(xt, K);
    float *wf = malloc(sizeof(float) * K * N), *x = malloc(sizeof(float) * K * M);
    void *wq = malloc(wrow * N), *xq = malloc(xrow * M);
    for (int64_t i = 0; i < K * N; i++) wf[i] = 0.02f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
    if (raw_w) memcpy(wq, raw_w, wrow * N);
    else ggml_quantize_chunk(wt, wf, wq, 0, N, K, NULL);
    for (int64_t i = 0; i < K * M; i++) x[i] = raw_x ? raw_x[i] : frand() * (1 + (i / 64) % 3) * (i % 97 == 0 ? 20 : 1);

    struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(cw, wt, K, N);
    struct ggml_tensor *a = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
    struct ggml_tensor *d = ggml_mul_mat(ca, w, a);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    ggml_backend_tensor_set(w, wq, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, x, 0, ggml_nbytes(a));
    struct ggml_cgraph *gf = ggml_new_graph(ca);
    ggml_build_forward_expand(gf, d);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_graph_compute(be, gf);

    float *got = malloc(sizeof(float) * N * M), *ref = malloc(sizeof(float) * N * M);
    ggml_backend_tensor_get(d, got, 0, sizeof(float) * N * M);
    const struct ggml_type_traits_cpu *xtt = ggml_get_type_traits_cpu(xt);
    for (int64_t m = 0; m < M; m++) xtt->from_float(x + m * K, (char *)xq + m * xrow, K);
    double err = 0, mag = 1e-30;
    int64_t worst = 0;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            tt->vec_dot(K, &ref[m * N + n], 0, (const char *)wq + n * wrow, 0, (char *)xq + m * xrow, 0, 1);
            const double e = isfinite(got[m * N + n]) ? fabs((double)got[m * N + n] - ref[m * N + n]) : INFINITY;
            if (e > err) { err = e; worst = m * N + n; }
            mag = fmax(mag, fabs(ref[m * N + n]));
        }
    const double rel = err / mag;
    const int ok = rel < 1e-4;
    ok ? npass++ : nfail++;
    if (!ok || getenv("VERBOSE"))
        printf("%s %-7s K=%ld N=%ld M=%ld T=%d relerr=%.2e (worst row %ld col %ld: got %g ref %g)\n", ok ? "ok  " : "FAIL",
               ggml_type_name(wt), (long)K, (long)N, (long)M, threads, rel, (long)(worst % N), (long)(worst / N),
               got[worst], ref[worst]);
    free(got); free(ref);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_free(ca); ggml_free(cw);
    free(wf); free(x); free(wq); free(xq);
}

static enum ggml_type type_of(const char *s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++)
        if (ggml_type_name((enum ggml_type)t) && !strcmp(ggml_type_name((enum ggml_type)t), s)) return (enum ggml_type)t;
    fprintf(stderr, "unknown type %s\n", s);
    exit(2);
}

static void *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(n);
    if (fread(p, 1, n, f) != (size_t)n) exit(2);
    fclose(f);
    return p;
}

int main(int argc, char **argv) {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (argc >= 6 && !strcmp(argv[1], "--raw")) {
        const enum ggml_type wt = type_of(argv[3]);
        const int64_t K = atoll(argv[4]), N = atoll(argv[5]);
        raw_w = slurp(argv[2]);
        if (argc > 6) raw_x = slurp(argv[6]);
        for (int T = 1; T <= 8; T++) run_case(be, wt, K, N, 1, T);
        printf("%d passed, %d failed (kurn hook %s)\n", npass, nfail, getenv("GGML_KURN") ? "ON" : "off");
        return nfail;
    }
    for (int i = 1; i < argc; i++) {
        const enum ggml_type wt = type_of(argv[i]);
        const int64_t shapes[][2] = {{2048, 2048}, {2048, 1024}, {2048, 6144}, {6144, 2048}, {256, 32}, {512, 96}};
        for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++)
            for (int T = 1; T <= 8; T += 3)
                for (int64_t M = 1; M <= 8; M += M < 2 ? 1 : M < 4 ? 2 : 4)
                    run_case(be, wt, shapes[s][0], shapes[s][1], M, T);
    }
    printf("%d passed, %d failed (kurn hook %s)\n", npass, nfail, getenv("GGML_KURN") ? "ON" : "off");
    ggml_backend_free(be);
    return nfail;
}
