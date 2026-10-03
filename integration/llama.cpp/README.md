# llama.cpp integration: the KURN extra buffer type

`ggml-kurn/kurn-buft.cpp` adds a ggml-cpu *extra buffer type* named `KURN`, the same mechanism as ggml's own `AMX`
and `CPU_REPACK` buffers. At model load, every weight whose type has a kurn kernel is repacked once into kurn's
interleaved i16 layout inside that buffer, so there is one copy of the weights, owned by ggml. `MUL_MAT` and
`MUL_MAT_ID` (MoE experts) on those weights then run kurn kernels:

| activation columns (tokens) | kernel |
|---|---|
| 1 (decode) | kurn GEMV |
| 2..8 (speculative verify, small batches) | kurn verify kernel: one pass over the weights for all columns |
| more (prefill) | verify kernel over L2-sized row chunks, or for Q8_0 (opt-in) an AMX tile kernel on the same layout |

All of them do the same per-column arithmetic (exact int32 dot per block, then `fma(float(isum), d_w * d_x, acc)` in
block order), so a token's logits do not depend on how many tokens are computed with it (batch invariance; checked by
`test_kurn_buft.c`). Rows are handed out in dynamic chunks through ggml's shared chunk counter.

The kernels come from the kurn registry: `gen_ggml_sources.py` emits one GEMV and three verify kernels (2, 4, 8
columns) per format that has an AVX-512 VNNI GEMV + verify kernel, a ggml-compatible activation type (`q8_0` /
`q8_K`) and a `GGML_TYPE_*` in the target's `ggml.h`, plus the packing glue and the dispatch table
(`kurn_dispatch.h`). A format added to the registry is picked up by re-running `apply.sh`. Today that is Q8_0, Q4_0,
IQ4_NL, Q4_K, Q2_0, TQ2_0 and Q1_0.

## Build

```sh
git -C ~/src/llama.cpp worktree add ~/src/llama-kurn 4ebdf2c     # any recent llama.cpp should do
kurn/integration/llama.cpp/apply.sh ~/src/llama-kurn            # copies sources, generates kernels, hooks CMake
cd ~/src/llama-kurn && CC=gcc CXX=g++ cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16 && cmake --build build -j
```

`apply.sh LLAMA_DIR [--config JSON] [--only q8_0,q4_0] [--patch OUT.diff]` is idempotent. It copies `ggml-kurn/` and
the generated kernels to `ggml/src/ggml-cpu/kurn/`, adds them to the ggml-cpu CMake target and registers the KURN
buffer type first in `ggml_backend_cpu_get_extra_buffer_types()`, so it wins over AMX / CPU_REPACK for the formats it
covers; other formats keep going to those buffers. `--config` overrides kernel keys per format (for example
`{"q4_0": {"rows": 4, "prefetch": 1}}`). The buffer type needs AVX-512 F/BW/VNNI at build time; without it,
`ggml_backend_cpu_kurn_buffer_type()` returns NULL and nothing changes.

The model log shows `KURN model buffer size = ...` when weights were repacked. No llama.cpp flag is needed;
`--repack 0` / `--no-repack` (no extra buffers) turns it off along with AMX and CPU_REPACK.

## Environment

| variable | effect |
|---|---|
| `GGML_KURN=0` | buffer type disabled (stock behaviour) |
| `GGML_KURN_TYPES=q8_0,q4_0` | only these formats |
| `GGML_KURN_AMX=1` | Q8_0 prefill through the AMX tile kernel (default off, see the AMX caveat) |
| `GGML_KURN_AMX_MIN=N` | AMX for N or more columns (default and minimum 16) |
| `GGML_KURN_CHUNK_KB=N` | prefill row-chunk size in KiB (default 1024) |
| `GGML_KURN_CHUNKS=N` | N equal row chunks per thread, handed out dynamically (0 = one static range per thread). Default: guided (each thread first streams its own contiguous 3/4 share in whole GEMV passes, the last quarter goes out in single passes); ops with fewer than 4 passes per thread use static ranges |
| `GGML_KURN_FALLBACK=0` | turn off the per-tensor fallback (below); `GGML_KURN_FALLBACK_N` / `_K` set its thresholds (512 / 512) |
| `GGML_KURN_PROFILE=1` | per-phase cycle counts of decode-sized MUL_MAT (quantize + activation prep, barrier wait, kernels, finish spread), printed at exit |
| `GGML_KURN_XPREP=0`, `GGML_KURN_QSPLIT=1` | A/B switches: per-call activation prep instead of the shared one; activation quantization split by blocks instead of whole rows |
| `GGML_KURN_VERBOSE=1` | log every repacked tensor and every fallback decision |

**Per-tensor fallback.** kurn declines a weight (llama.cpp then gives it to the next extra buffer) when ggml's AMX
buffer can take it and it is in the shape range where kurn is not measured to win: N <= 512 or K < 512 (Q8_0 2048 x 512
measured 0.94x). Everywhere else kurn was measured at or above ggml's best path: AMX buffer (its M = 1 path is a VNNI
GEMV), CPU_REPACK and plain `vec_dot` (`benchmarks/v0.2/q4fix`). Builds without AMX keep every weight on kurn.

**Kernel defaults** (`TUNED_KEYS` in `gen_ggml_sources.py`): Q8_0 `rows=8` (v0.1 vnni16's pass width), Q4_0
`unpack=pair rows=4`, Q4_K `unpack=pair correction=dpmin rows=4`, IQ4_NL `unpack=perm rows=4`. Four or more row groups
per pass give each core four or more concurrent weight streams, which the DRAM prefetchers need (one stream per core:
42 GB/s, four: 80 GB/s on the dev VM). Each format also gets a one-row-group GEMV on the same packing for range
remainders, and `_xprep` / `_packed_x` entry points so activations are quantized and prepared once per op.

## Tests

`test_kurn_buft.c` is a test-backend-ops-style checker for `MUL_MAT` and `MUL_MAT_ID` on KURN-buffer weights against
a reference built from ggml's own `vec_dot` (relative error < 1e-4), with batch invariance (column 0 bit-identical
for every batch size), buffer reuse, and shapes the buffer type must refuse (K above the kernel limit):

```sh
L=~/src/llama-kurn
gcc -O2 -I$L/ggml/include test_kurn_buft.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin -o t
./t smoke | ./t quick [q8_0,q4_0] | ./t full        # exit code = number of failing cases
./t case q8_0 4096 4096 16 8                       # one case: TYPE K N M THREADS [REPS]
TEST_BUFT=AMX ./t quick q8_0                         # same checks against another extra buffer type
```

`tests/test_llama_integration.py` runs format discovery, the packing glue and GEMV / verify kernels against a Python
reference, and the checker's smoke mode when a patched llama.cpp is found (`KURN_LLAMA_CPP`, default
`~/src/llama-kurn`); it skips cleanly otherwise.

## AMX caveat

On the KVM guest used for development, AMX tile registers are not reliably preserved across context switches: with
other processes competing for the cores, both this AMX kernel and ggml's own AMX kernels intermittently return wrong
results (`tools/amx_ctx.c` reproduces it without ggml). `tools/amx_check.sh` (20 repeated runs per shape, threads
pinned, under the bench lock) still saw 1-6 of 19 repeats differ for kurn's AMX kernel and up to 19 of 19 for ggml's AMX
buffer. The AMX kernel is therefore opt-in (`GGML_KURN_AMX=1`); it should be safe on bare metal or on hypervisors that
switch AMX state correctly.

## Legacy prototype

`ggml-q8_0-gemv-kurn-vnni16.patch` + `test_mul_mat_hook.c` are the v0.1 prototype (a hook inside ggml's `mul_mat`
with lazily repacked, duplicated Q8_0 weights). They are kept for reference only; the buffer type replaces them.
