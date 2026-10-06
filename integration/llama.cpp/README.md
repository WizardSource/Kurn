# llama.cpp integration: the KURN extra buffer type and kurn attention

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

## Attention: kurn as `FLASH_ATTN_EXT`

`apply.sh` also makes kurn's attention kernel (`kurn.attention`, `data/attn_kernel.c`) the first implementation of
ggml-cpu's `FLASH_ATTN_EXT`: `ggml-kurn/kurn-attn.cpp` is called at the top of `ggml_compute_forward_flash_attn_ext`
and ggml's own kernel runs for every node it does not take. `gen_ggml_attn.py` generates the kernels (one renamed
instance per configuration, compiled only when the build has the ISA it needs) and `kattn_dispatch.h`.

| | kurn takes | falls back to ggml |
|---|---|---|
| KV | K and V both F16, BF16 or Q8_0, rows contiguous (`-fa on` cache layout) | other or mixed types, transposed V |
| heads | dk = dv = 64, 128, 256; dk 576 / dv 512 (MLA, V a view of K); any GQA ratio | other head dims |
| mask | F16 mask shared by all heads (llama.cpp's KQ mask, causal / SWA / multi-sequence) | per-head masks |
| other | streams (`ne[3]`, llama.cpp's per-sequence KV streams), dim-3 broadcast | ALiBi, logit softcap, attention sinks |

The mask carries causality, so kurn runs with `causal = 0`; KV tiles that the mask hides for every row of a tile are
skipped. Decode (n_q x G <= 8) uses kurn's f32 row engine with automatic KV splits (flash-decoding); larger batches
use the tile engine. Prefill-sized calls of the bf16 / AMX engines pack K/V once per call (`kattn_pack`) and read
every query tile from that copy (`kattn_packed`) instead of re-packing each KV tile for every query tile.

Two modes:

- **fast** (default): tile engine `f32` (default), `amx` (AMX-BF16, `GGML_KURN_AMX=1`, see the AMX caveat; the kernel
  times each tile block and recomputes blocks that spanned a preemption) or `bf16` (AVX512-BF16).
- **exact** (`GGML_KURN_FA_MODE=exact`): the f32 tile engine for every batch size and a single KV split. A token's
  attention output then does not depend on how many tokens are computed with it, the (padded) KV length or the thread
  count; together with the buffer type's batch-invariant matmuls, a speculative verify batch reproduces one-token
  decoding bit for bit. Decode is slower than fast mode at long context (the tile engine computes 32 query rows).

| variable | effect |
|---|---|
| `GGML_KURN_FA=0` (or `GGML_KURN=0`) | ggml's flash attention only |
| `GGML_KURN_FA_MODE=fast\|exact` | see above (default fast) |
| `GGML_KURN_FA_ENGINE=f32\|amx\|bf16` | tile engine of fast mode (default f32, amx with `GGML_KURN_AMX=1`) |
| `GGML_KURN_FA_PACK=0` | no per-call K/V packing (bf16 / AMX engines) |
| `GGML_KURN_VERBOSE=1` | log the configuration, every new node shape with its kernel or fallback reason, and node counts at exit |

Checks: `test-backend-ops -o FLASH_ATTN_EXT -b CPU` compares the CPU backend against itself in reference mode, where
kurn steps aside, so every supported case is kurn against ggml's vec kernel (all 5,314 cases pass in every mode and
engine; 545 of them run on kurn). `tests/test_llama_attn.py` checks the generator, runs the renamed kernels through
kurn's harness, checks that `exact` is batch-invariant, and runs a test-backend-ops subset when a built checkout is
found (`KURN_LLAMA_CPP`).

## k4c KV cache (`k4c/`)

`k4c/apply.sh LLAMA_DIR` (after `apply.sh`) adds `GGML_TYPE_K4C`, kurn's k4c key format, as a llama.cpp K cache type:
`-ctk k4c -ctv q4_0` is kurn's `k4c_q4`, `-ctv q8_0` its `k4c_q8` (`-ctk k4c_q4` / `-ctk k4c_q8` set both). Keys are
quantized to 4 bits per channel over groups of 32 cache cells (f16 scale and min per channel and group, 5 bits per value
on average); kurn's attention reads the groups directly (`kurn-attn.cpp`, kernels `kattn_*_k4c_q4/q8_*`).

- **After RoPE.** llama.cpp caches rotated keys, so the kernels run with `rope_dim 0`; kurn's own engine stores keys
  before RoPE. On Qwen3-1.7B the per-channel 4-bit keys are still accurate (WikiText-2, ctx 2048, KL vs F16 KV with
  ggml's FA: k4c_q4 0.016 at +0.10% perplexity, k4c_q8 0.010; llama.cpp's own Q4_0 KV 0.32 at +29%, 1.15 at +107%
  without its Hadamard rotation).
- **Writes** (`ggml-k4c.c`, SET_ROWS into a K4C cache): each touched group is re-encoded from the exact values of its
  valid rows, kept as f16 in a bounded table (`GGML_K4C_EXACT_MB`, default 256) for recently written groups. A group
  filled one token at a time, rolled back (rejected drafts) or partly cleared holds the same bytes as the same rows
  written at once; freed cells are cleared, so stale keys never widen a group's range. Older groups that left the table
  are re-encoded from their 4-bit values if they are ever rewritten.
- **Session state** (prompt cache, `--slot-save-path` save / restore, `llama_state_seq_*`): K4C keys are written as f16
  rows and re-quantized into whatever cells they are restored to; with the f16 table this restores the same codes. With
  `GGML_KURN_FA_MODE=exact`, llama-server's cached, restored and recomputed continuations are bit-identical.
- **Not supported:** K-shift (context shift, `--cache-reuse`: `get_can_shift()` is false), the non-FA path, V types
  other than Q4_0 / Q8_0, MLA models, cache sizes that are not a multiple of 32, non-CPU buffers.

`k4c/test_k4c.c` checks writes in llama.cpp's patterns (prefill ending inside a group, appends, rollback, out-of-order
rows, clears), byte-identical groups for prefill vs appends + rollback, FA on K4C against ggml on the dequantized keys,
and batch invariance in exact mode; `tests/test_llama_k4c.py` runs it against a built checkout.

## Speculative decoding: cost-aware verify width (`spec-width/`)

Verify cost on this buffer type is a staircase: 3 columns cost as much as 4 and 5-7 as much as 8 (one kernel per
2 / 4 / 8 columns), and every 8 more columns add a group pass. `spec-width/` sizes the draft to it:

- `kurn-spec-calib` (same flags as `llama-speculative-simple`; `KURN_CALIB_OUT=prefix`) measures the whole-forward
  cost table (`verify M ms`: target, M tokens with logits for all; `draft M ms`) and a greedy acceptance trace.
  `kurn specwidth show prefix.cost` prints it; `kurn specwidth simulate prefix.cost *.trace` replays traces against
  fixed widths and the policy.
- `kurn-spec-width.h` (header-only, the twin of `kurn.specwidth.WidthPolicy`) picks the draft length that maximises
  expected accepted tokens minus lambda x time (lambda = running tokens/ms), with acceptance learned online per
  draft-confidence bin. `KURN_SPEC_WIDTH=prefix.cost llama-speculative-simple ... --spec-draft-n-max 15` caps,
  stops and truncates every draft through it (`KURN_SPEC_WIDTH_MODE=cap`: rate-only cap). The patch adds draft
  confidences and a keep-drafting callback to draft-simple; without `KURN_SPEC_WIDTH` behaviour is unchanged.

```sh
kurn/integration/llama.cpp/spec-width/apply.sh ~/src/llama-kurn   # after apply.sh; idempotent
cmake --build build -j --target llama-speculative-simple kurn-spec-calib
```

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
