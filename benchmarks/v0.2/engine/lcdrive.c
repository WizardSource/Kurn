// llama.cpp reference driver with the same protocol as the kurn engine (src/kurn/model/engine.c):
// every token (prompt included) goes through llama_decode one at a time, per-token wall time is
// recorded, greedy argmax picks the next token, and `ppl` scores the 2nd half of each CTX chunk.
//
//   lcdrive MODEL.gguf gen T N_GEN tok,tok,...     -> "gen: ..." + decode_tok_s / med_ms / cpu_s_per_tok
//   lcdrive MODEL.gguf ppl T CTX tokens.txt        -> "ppl X over N tokens; ..."
// LC_REPACK=0 disables extra buffer types (ggml's plain vec_dot path; default 1 = AMX repack).
// KURN_PERF_CTL=fifo: enable/disable `perf record -D -1 --control fifo:FIFO` around the decode loop.
// Build: L=~/src/llama.cpp; gcc -O2 lcdrive.c -I $L/include -I $L/ggml/include -L $L/build/bin -lllama -Wl,-rpath,$L/build/bin -lm
#define _GNU_SOURCE
#include "llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static double cpu_s(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static void quiet(enum ggml_log_level l, const char *t, void *u) { (void)l; (void)t; (void)u; }

static void perf_ctl(const char *cmd) {  // same KURN_PERF_CTL protocol as the engine
    const char *p = getenv("KURN_PERF_CTL");
    if (!p) return;
    int fd = open(p, O_WRONLY);
    if (fd >= 0) { if (write(fd, cmd, strlen(cmd)) < 0) perror("perf ctl"); close(fd); }
}

// summed time all threads of this process were runnable but not running (schedstat field 2), s
static double runq_s(void) {
    double s = 0;
    DIR *d = opendir("/proc/self/task");
    for (struct dirent *e; d && (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char p[300]; snprintf(p, sizeof p, "/proc/self/task/%s/schedstat", e->d_name);
        FILE *f = fopen(p, "r");
        unsigned long long run, wait;
        if (f && fscanf(f, "%llu %llu", &run, &wait) == 2) s += wait * 1e-9;
        if (f) fclose(f);
    }
    if (d) closedir(d);
    return s;
}

static struct llama_context *ctx;
static int n_vocab;

static const float *decode1(int tok, int pos) {
    llama_token t = tok;
    struct llama_batch b = llama_batch_get_one(&t, 1);
    (void)pos;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
    return llama_get_logits_ith(ctx, -1);
}
static int argmax(const float *l) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) if (l[i] > l[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model.gguf gen|ppl THREADS N_GEN|CTX tokens\n", argv[0]); return 2; }
    const int T = atoi(argv[3]);
    if (!getenv("LC_VERBOSE")) llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = !(getenv("LC_REPACK") && !strcmp(getenv("LC_REPACK"), "0"));
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = T; cp.n_threads_batch = T;
    ctx = llama_init_from_model(model, cp);
    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (!strcmp(argv[2], "gen")) {
        const int ngen = atoi(argv[4]);
        int toks[2048], n = 0;
        for (char *p = strtok(argv[5], ","); p; p = strtok(NULL, ",")) toks[n++] = atoi(p);
        int next = 0;
        for (int i = 0; i < n; i++) next = argmax(decode1(toks[i], i));
        double *dt = malloc(sizeof(double) * ngen);
        perf_ctl("enable\n");
        double w0 = now(), c0 = cpu_s(), rq0 = runq_s();
        printf("gen:");
        for (int i = 0; i < ngen; i++) {
            printf(" %d", next);
            double t0 = now();
            next = argmax(decode1(next, n + i));
            dt[i] = now() - t0;
        }
        double wall = now() - w0, cpu = cpu_s() - c0, rq = runq_s() - rq0;
        perf_ctl("disable\n");
        qsort(dt, ngen, sizeof(double), cmpd);
        printf("\ndecode_tok_s %.2f cpu_s_per_tok %.4f wall_s %.2f med_ms %.3f p10_ms %.3f p90_ms %.3f runq_share %.4f\n", ngen / wall,
               cpu / ngen, wall, dt[ngen / 2] * 1e3, dt[ngen / 10] * 1e3, dt[ngen * 9 / 10] * 1e3, rq / (wall * T));
    } else {
        const int c_ctx = atoi(argv[4]);
        FILE *f = fopen(argv[5], "r");
        int *toks = malloc(sizeof(int) << 20), n = 0, x;
        while (n < (1 << 20) && fscanf(f, "%d", &x) == 1) toks[n++] = x;
        int nchunks = getenv("PPL_CHUNKS") ? atoi(getenv("PPL_CHUNKS")) : 4;
        double nll = 0; long cnt = 0, steps = 0;
        double w0 = now(), c0 = cpu_s();
        for (int c = 0; (c + 1) * c_ctx <= n && c < nchunks; c++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            for (int i = 0; i < c_ctx - 1; i++) {
                const float *lg = decode1(toks[c * c_ctx + i], i);
                steps++;
                if (i >= c_ctx / 2) {
                    double mx = -INFINITY, den = 0;
                    for (int v = 0; v < n_vocab; v++) mx = fmax(mx, lg[v]);
                    for (int v = 0; v < n_vocab; v++) den += exp(lg[v] - mx);
                    nll += -(lg[toks[c * c_ctx + i + 1]] - mx - log(den));
                    cnt++;
                }
            }
        }
        printf("ppl %.4f over %ld tokens; %.2f tok/s, cpu_s/tok %.4f\n", exp(nll / cnt), cnt, steps / (now() - w0),
               (cpu_s() - c0) / steps);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
