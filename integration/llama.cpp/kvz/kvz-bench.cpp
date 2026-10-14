// kvz-bench SEQ_FILE [THREADS] [ROWS] [ZSTD_LEVEL]
// Round-trips every K and V block of a llama_state_seq_save_file blob (one stream, f16 / q8_0 / q4_0 / k4c-as-f16
// rows) through kvz and checks the bytes; prints sizes and encode / decode throughput (and zstd's, if built with
// -DKVZ_BENCH_ZSTD -lzstd).
#include "llama-kvz.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef KVZ_BENCH_ZSTD
#include <zstd.h>
#endif

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: kvz-bench SEQ_FILE [THREADS] [ROWS] [ZSTD_LEVEL]\n");
        return 2;
    }
    const int nt = argc > 2 ? atoi(argv[2]) : kvz_default_threads();
    const size_t rows_max = argc > 3 ? strtoull(argv[3], nullptr, 10) : SIZE_MAX;
    const int zl = argc > 4 ? atoi(argv[4]) : 0;
    const int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        return 2;
    }
    const uint8_t * buf = (const uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) {
        return 2;
    }
    size_t off = 0;
    auto u32 = [&]() { uint32_t v; memcpy(&v, &buf[off], 4); off += 4; return v; };
    auto u64 = [&]() { uint64_t v; memcpy(&v, &buf[off], 8); off += 8; return v; };
    u32(); u32();
    off += 4 * (size_t) u32();
    u32();
    const uint32_t n = u32();
    for (uint32_t i = 0; i < n; ++i) {
        u32();
        off += 4 * (size_t) u32();
    }
    u32();
    const uint32_t nl = u32();
    const size_t rows = std::min<size_t>(n, rows_max);
    struct agg { size_t raw = 0, z = 0, zs = 0; double te = 0, td = 0, tz = 0; } A[2];
    bool all_ok = true;
    for (int kv = 0; kv < 2; ++kv) {
        for (uint32_t l = 0; l < nl; ++l) {
            const int32_t type = (int32_t) u32();
            const uint64_t rb = u64();
            const uint8_t * src = &buf[off];
            off += n * rb;
            const kvz_layout lay = (type == 1 || type == 43) ? KVZ_PLANES2 : type == 8 ? KVZ_Q8_0 : type == 2 ? KVZ_Q4_0 : KVZ_RAW;
            const size_t raw = rows * rb;
            std::vector<uint8_t> blk(src, src + raw), enc(kvz_bound(raw)), dec(raw);  // block in RAM, buffers faulted in
            src = blk.data();
            memset(enc.data(), 0, enc.size());
            memset(dec.data(), 0, dec.size());
            double t0 = now();
            const size_t ne = kvz_encode(lay, src, rows, rb, enc.data(), nt);
            double t1 = now();
            const bool ok = kvz_decode(enc.data(), ne, dec.data(), rows, rb, nt) && !memcmp(dec.data(), src, raw);
            double t2 = now();
            all_ok &= ok;
            if (!ok) {
                fprintf(stderr, "MISMATCH %s layer %u\n", kv ? "V" : "K", l);
            }
            A[kv].raw += raw;
            A[kv].z += ne;
            A[kv].te += t1 - t0;
            A[kv].td += t2 - t1;
#ifdef KVZ_BENCH_ZSTD
            if (zl) {
                std::vector<uint8_t> zb(ZSTD_compressBound(raw));
                ZSTD_CCtx * cc = ZSTD_createCCtx();
                ZSTD_CCtx_setParameter(cc, ZSTD_c_compressionLevel, zl);
                ZSTD_CCtx_setParameter(cc, ZSTD_c_nbWorkers, nt > 1 ? nt : 0);
                double z0 = now();
                const size_t zn = ZSTD_compress2(cc, zb.data(), zb.size(), src, raw);
                A[kv].tz += now() - z0;
                A[kv].zs += zn;
                ZSTD_freeCCtx(cc);
            }
#endif
        }
    }
    for (int kv = 0; kv < 2; ++kv) {
        printf("%s: raw %.1f MiB -> kvz %.1f MiB (%.3fx), encode %.2f GB/s, decode %.2f GB/s", kv ? "V" : "K", A[kv].raw / 1048576.0,
               A[kv].z / 1048576.0, (double) A[kv].raw / A[kv].z, A[kv].raw / A[kv].te / 1e9, A[kv].raw / A[kv].td / 1e9);
        if (zl) {
            printf(", zstd-%d %.3fx at %.2f GB/s", zl, (double) A[kv].raw / A[kv].zs, A[kv].raw / A[kv].tz / 1e9);
        }
        printf("\n");
    }
    printf("total: %.3fx, threads %d, rows %zu, %s\n", (double) (A[0].raw + A[1].raw) / (A[0].z + A[1].z), nt, rows, all_ok ? "bit-exact" : "MISMATCH");
    return all_ok ? 0 : 1;
}
