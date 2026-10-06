#define _GNU_SOURCE
// AMX state survival across preemption: load tiles, yield/spin, then use them.
#include <immintrin.h>
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
static int iters = 200000, yield_mode = 1, pin = 0;
static void cfg(void) {
    struct __attribute__((aligned(64))) { uint8_t p, s, r[14]; uint16_t c[16]; uint8_t rows[16]; } t = {0};
    t.p = 1; for (int i = 0; i < 3; i++) { t.rows[i] = 16; t.c[i] = 64; }
    _tile_loadconfig(&t);
}
static void *run(void *arg) {
    long bad = 0, badcfg = 0;
    if (pin) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET((int)(long)arg % 8, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
    int8_t a[16 * 64], b[16 * 64]; int32_t c[256], ref[256];
    for (int i = 0; i < 1024; i++) { a[i] = (int8_t)(i * 7 + 3); b[i] = (int8_t)(i * 13 + 1); }
    cfg();
    _tile_zero(0); _tile_loadd(1, a, 64); _tile_loadd(2, b, 64); _tile_dpbssd(0, 1, 2); _tile_stored(0, ref, 64);
    for (int it = 0; it < iters; it++) {
        cfg();
        _tile_zero(0); _tile_loadd(1, a, 64); _tile_loadd(2, b, 64);
        if (yield_mode) sched_yield(); else { volatile int x = 0; for (int i = 0; i < 2000; i++) x += i; }
        _tile_dpbssd(0, 1, 2); _tile_stored(0, c, 64);
        if (memcmp(c, ref, sizeof c)) bad++;
        // config survival without reloading
        struct __attribute__((aligned(64))) { uint8_t p, s, r[14]; uint16_t c[16]; uint8_t rows[16]; } st;
        _tile_storeconfig(&st); if (st.p != 1 || st.rows[0] != 16) badcfg++;
    }
    printf("thread %ld: %ld/%d wrong results, %ld lost configs\n", (long)arg, bad, iters, badcfg);
    return NULL;
}
int main(int argc, char **argv) {
    if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("arch_prctl"); return 1; }
    int nt = argc > 1 ? atoi(argv[1]) : 16; yield_mode = argc > 2 ? atoi(argv[2]) : 1; pin = argc > 3 ? atoi(argv[3]) : 0;
    pthread_t th[64]; for (long i = 0; i < nt; i++) pthread_create(&th[i], 0, run, (void *)i);
    for (int i = 0; i < nt; i++) pthread_join(th[i], 0);
}
