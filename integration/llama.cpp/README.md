# llama.cpp integration: the KURN extra buffer type and kurn attention

`ggml-kurn/kurn-buft.cpp` adds a ggml-cpu *extra buffer type* named `KURN`, the same mechanism as ggml's own `AMX`
and `CPU_REPACK` buffers. At model load, every weight whose type has a kurn kernel is repacked once into kurn's
interleaved i16 layout inside that buffer, so there is one copy of the weights, owned by ggml. `MUL_MAT` and
`MUL_MAT_ID` (MoE experts) on those weights then run kurn kernels:

| activation columns (tokens) | kernel |
|---|---|
| 1 (decode) | kurn GEMV |
| 2..8 (speculative verify, small batches) | kurn verify kernel of exactly that width: one pass over the weights for all columns |
| 9..16, Q8_0 on AMX CPUs | `kurn-amx-vfy.cpp`: all columns in one pass over the weights (M-row AMX-INT8 tiles), bit-identical to the GEMV per column |
| 9..31 (other formats; Q8_0 17..31) | verify kernels over L2-sized row chunks |
| 32+ (prefill), AMX CPU (default) | AMX-BF16 GEMM: weights dequantized per panel from the same buffer, bf16 activations, fp32 accumulation |
| 32+, exact mode or no AMX | Q8_0: AMX-INT8 kernel on the same layout (AMX CPUs); other formats: verify kernels over row chunks |

The GEMV, the verify kernels and the AMX-INT8 kernel do the same per-column arithmetic (exact int32 dot per block,
then `fma(float(isum), d_w * d_x, acc)` in block order), so a token's logits do not depend on how many tokens are
computed with it (batch invariance; checked by `test_kurn_buft.c`). The AMX-BF16 prefill path does not have that
property: it rounds activations to bf16 instead of quantizing them to Q8, which is closer to f32 than ggml's own
int8-activation dot (`test_kurn_buft bf16`), but not bit-identical to the GEMV. `GGML_KURN_EXACT=1`, or kurn
attention's exact mode (`GGML_KURN_FA_MODE=exact`), keeps every matmul batch invariant. Prefill-sized ops hand rows
out in dynamic chunks through ggml's shared chunk counter.

**Decode-sized ops (1-8 columns): work stealing.** These are memory-bound, and each thread streams at its own
~15 GB/s, so a thread that finishes late is pure loss: the others cannot absorb its tail. With guided chunks the last
thread finished ~60 us after the first on 250 us Qwen3-8B decode matmuls, and 11.7% of decode thread time was barrier
wait after matmuls; finer static chunks cost bandwidth instead (restarted streams). Now each thread starts on its own
contiguous share of whole GEMV passes and claims from its front in shrinking batches (a quarter of what is left, then
single passes, then single 16-row records); a thread that runs out steals the back half of the unclaimed rest of the
thread with the most left (one 64-bit CAS per claim or steal; slots at the end of the op's work buffer). Results do
not depend on which thread computes a row. `opbench`, Qwen3-8B Q8_0 shapes, paired CI over 12 pairs: 1 / 4 / 8
columns 0.966x / 0.969x / 0.971x time (CIs 0.920-0.981, 0.953-0.992, 0.957-0.984); finish spread 65 -> 32-40
kcycles per op. Also used by the native Q6_K / Q5_K kernels and by fused groups. `GGML_KURN_STEAL=0` restores the
guided chunks.

**Formats without repacked kurn kernels (Q6_K, Q5_K).** On AMX-BF16 CPUs the buffer also takes 2D Q6_K and Q5_K
weights, unrepacked in ggml's block layout: AMX-BF16 for 16 or more columns, and below that `kurn-native-vfy.cpp`, which
unpacks each super-block once for up to 8 columns (`GGML_KURN_NATIVE_VFY=0`: ggml's `vec_dot` per row and column).
Without this, the Q6_K tensors of a Q4_K_M file go to ggml's AMX buffer, whose prefill ran 2.8x slower per
multiply-accumulate than kurn's Q4_K kernel on Qwen3-8B. Qwen3-8B Q6_K shapes (`ffn_down` 12288 x 4096 and
`attn_v` 4096 x 1024, 8 layers): 1 / 8 columns 3.0 / 6.4 ms, against 4.5 / 15.6 ms for `vec_dot` and 5.9 / 15.5 ms for
ggml's AMX buffer; Q5_K 2.7 / 6.8 ms (`vec_dot` 3.6 / 22.4, AMX buffer 3.0 / 7.5). In exact mode the buffer takes them
too, without AMX-BF16: the native kernels compute every width with the same per-column arithmetic, where ggml's AMX
buffer (VNNI at one column, tiles at more) is not batch invariant, so Q4_K_M models are batch invariant end to end.

**Wide verify (9-16 columns) in one weight pass.** The VNNI verify kernels stop at 8 columns, so 9-16 columns took
two passes (the second from L2) and cost 1.4-1.8x a decode step. On AMX CPUs, Q8_0 at 9-16 columns now runs
`kurn-amx-vfy.cpp`: a kurn i16 record is an AMX B tile, so one `tdpbsud` per record and row group gives the exact int32
block dots of all M columns (M-row A and C tiles), scaled per block in block order like the GEMV, which keeps every
column bit-identical to one-token decoding (exact mode included). Three row groups per pass keep three weight streams
per core (one group per pass was 30-45% slower); accumulators beyond the register file spill to L1. Every block step
is timed and recomputed after a preemption-length gap, as in the other AMX paths (stress next to AMX interlopers:
0/59 runs differ; with the guard disabled 1-2/59 differ). The kernel was first written for 2-8 columns, where it does
not beat the prefetching VNNI kernels. `opbench`, Qwen3-8B Q8_0 shapes, paired CI over 12 pairs: 9 / 12 / 16 columns
0.930x / 0.877x / 0.813x time (CIs 0.922-0.950, 0.871-0.882, 0.802-0.820). `GGML_KURN_AMX_VFY=0` turns it off,
`GGML_KURN_AMX_VFY_NG` sets row groups per pass (3).

**Fused FFN gate + up + SwiGLU.** `apply.sh` hooks `ggml_kurn_fused_mul_mat` into ggml-cpu's graph fusion
(`ggml_cpu_try_fuse_ops`): when `MUL_MAT(gate)`, `MUL_MAT(up)` on the same activations and their `SWIGLU` are
consecutive nodes, gate and up are used only by the SwiGLU, and both weights are in the KURN buffer with the same
path, the three nodes run as one op. The activations are quantized (or converted to bf16) once, one row schedule
covers both weights (one load-imbalance tail and one set of barriers instead of two), and each row chunk computes the
same rows of gate and up into thread-local tiles and writes only `silu(gate) * up`, with ggml's own
`ggml_vec_swiglu_f32`. Gate and up never reach memory and the GLU node disappears. Every row is computed by the same
kernel and arithmetic as unfused, so outputs are bit-identical to the unfused graph in both modes (`test_kurn_buft
fused`; whole-model logits `cmp`-identical). Qwen3-8B Q8_0 transformer-block graph (`tools/blockbench.c`, 8 layers,
paired CI over 10 alternating pairs): 512 columns 0.954x time (CI 0.912-0.992), 1 column 0.982x (0.941-0.993),
8 columns 0.995x (unresolved). `GGML_KURN_FUSE=0` turns it off.

**Fused attention q + k + v.** Two to four consecutive `MUL_MAT`s on the same activations also run as one op (one
quantization, one row schedule over the weights' rows laid end to end, outputs written in place). llama.cpp emits
q, k and v as consecutive nodes only when `apply.sh` has patched `build_qkv` (`src/llama-graph.cpp`, shared by every
model with separate q/k/v weights), and only for weights in the KURN buffer: other backends keep llama.cpp's node
order. Bit-identical to unfused (`test_kurn_buft fused`). Block graph, paired CI: 512 columns 0.977x (CI
0.940-0.994); 1 and 8 columns 0.985x / 0.996x, unresolved. `GGML_KURN_FUSE_QKV=0` keeps only the gate + up fusion.

The kernels come from the kurn registry: `gen_ggml_sources.py` emits one GEMV and seven verify kernels (2..8
columns, one per exact width) per format that has an AVX-512 VNNI GEMV + verify kernel, a ggml-compatible activation type (`q8_0` /
`q8_K`) and a `GGML_TYPE_*` in the target's `ggml.h`, plus the packing glue and the dispatch table
(`kurn_dispatch.h`). A format added to the registry is picked up by re-running `apply.sh`. Today that is Q8_0, Q4_0,
IQ4_NL, Q4_K, Q2_0, TQ2_0 and Q1_0.

**Verify schedules.** The verify kernels share the GEMV's packed layout and arithmetic; only their schedule differs:
row groups (16 rows each) per pass, limited by the register budget to rows x cols <= 8, and software prefetch of every
line of the record P records ahead (`pfgran=line`). Without the prefetch, one weight stream per core ran at 45-65 GB/s
against ~110 for the 8-stream GEMV; with it, 2..8 columns of Qwen3-8B Q8_0 cost 0.9-1.3x a GEMV pass instead of
1.7-2.4x (`VFY_KEYS` in `gen_ggml_sources.py` holds the measured values, `--config '{"q8_0": {"vfy": {"5": {"rows":
1, "prefetch": 16}}}}'` overrides one width). The Q8_0 and Q4_K GEMVs use the same whole-record prefetch.

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
| `GGML_KURN_AMX=0` | no AMX at all: prefill on the verify kernels, f32 attention engine. `GGML_KURN_AMX=1` additionally selects kurn attention's AMX engine |
| `GGML_KURN_AMX_MM=0` | AMX prefill matmuls off (attention engine unchanged) |
| `GGML_KURN_AMX_MIN=N` | AMX prefill for N or more columns (default 32, minimum 16) |
| `GGML_KURN_EXACT=1` | batch-invariant matmuls only: Q8_0 prefill on the AMX-INT8 kernel, other formats on the verify kernels, Q6_K / Q5_K on the native kernels at every width (also implied by `GGML_KURN_FA_MODE=exact`) |
| `GGML_KURN_NATIVE_VFY=0` | Q6_K / Q5_K below the AMX-BF16 threshold through ggml's `vec_dot` instead of `kurn-native-vfy.cpp` (A/B switch) |
| `GGML_KURN_Q8_BF16=0` | Q8_0 prefill on the AMX-INT8 kernel instead of AMX-BF16 |
| `GGML_KURN_NATIVE=0` | leave Q6_K / Q5_K weights to ggml |
| `GGML_KURN_FUSE=0` | no fused MUL_MAT groups (FFN gate + up + SwiGLU run as three nodes) |
| `GGML_KURN_AMX_VFY=0` | Q8_0 at 9-16 columns on two 8-column VNNI verify passes instead of the one-pass AMX verify kernel; `GGML_KURN_AMX_VFY_NG` row groups per pass (1-3, default 3) |
| `GGML_KURN_FUSE_QKV=0` | no q + k + v groups (gate + up + SwiGLU stays fused) |
| `GGML_KURN_STEAL=0` | decode-sized ops (1-8 columns) use the guided row chunks below instead of work stealing |
| `GGML_KURN_BF16_ROWS`, `_KC`, `_KCH`, `_PASS` | AMX-BF16 blocking: panel rows (128), K chunk (4096), guarded sub-step in 32-value steps (32), row-chunk granularity (64) |
| `GGML_KURN_AMX_GUARD=N` | preemption guard threshold in TSC cycles (8192, plus 128 per k-step of the guarded step) |
| `GGML_KURN_CHUNK_KB=N` | prefill row-chunk size in KiB (default 1024) |
| `GGML_KURN_CHUNKS=N` | prefill-sized ops (and decode-sized ones with `GGML_KURN_STEAL=0`): N equal row chunks per thread, handed out dynamically (0 = one static range per thread). Default: guided (each thread first streams its own contiguous 3/4 share in whole GEMV passes, the last quarter goes out in single passes); ops with fewer than 4 passes per thread use static ranges |
| `GGML_KURN_FALLBACK=0` | turn off the per-tensor fallback (below); `GGML_KURN_FALLBACK_N` / `_K` set its thresholds (512 / 512) |
| `GGML_KURN_PROFILE=1` | per-phase cycle counts of decode-sized MUL_MAT (quantize + activation prep, barrier wait, kernels, finish spread), printed at exit |
| `GGML_KURN_XPREP=0`, `GGML_KURN_QSPLIT=1` | A/B switches: per-call activation prep instead of the shared one; activation quantization split by blocks instead of whole rows |
| `GGML_KURN_TBM=0\|1` | TQ1_0 / TQ2_0 as bitmap + signs never / always (default: when the tensor's sampled zero fraction is >= `GGML_KURN_TBM_Z` / 1000, default 300) |
| `GGML_KURN_TBM_AMX_MIN=N` | bitmap + signs prefill on AMX-INT8 from N columns (default 32, minimum 16; batch invariant, also in exact mode) |
| `GGML_KURN_TBM_PF=0`, `GGML_KURN_TBM_ROWS=N` | bitmap + signs GEMV without software prefetch; N row groups per pass (2, 4 (default), 6, 8) |
| `GGML_KURN_VERBOSE=1` | log every repacked tensor, every fallback decision and each weight's prefill path; at exit, the number of AMX steps recomputed by the preemption guard |

**Per-tensor fallback.** kurn declines a weight (llama.cpp then gives it to the next extra buffer) when ggml's AMX
buffer can take it and it is in the shape range where kurn is not measured to win: N <= 512 or K < 512 (Q8_0 2048 x 512
measured 0.94x). Everywhere else kurn was measured at or above ggml's best path: AMX buffer (its M = 1 path is a VNNI
GEMV), CPU_REPACK and plain `vec_dot` (`benchmarks/v0.2/q4fix`). Builds without AMX keep every weight on kurn.

**Kernel defaults** (`TUNED_KEYS` in `gen_ggml_sources.py`): Q8_0 `rows=8` (v0.1 vnni16's pass width), Q4_0
`unpack=pair rows=4`, Q4_K `unpack=pair correction=dpmin rows=4`, IQ4_NL `unpack=perm rows=4`, TQ2_0 `rows=4 prefetch=4
pfgran=line`. The TQ2_0 entry replaced `rows=2` without prefetch. Paired A/B with alternating order, 95% CIs
(`benchmarks/v0.3/ternary/paired_tq2.sh`, `paired_ci.py`):
  - Decode layers of five ternary models: 1.29-1.47x (10/10 pairs each).
  - `llama-bench` decode: 1.15x BitNet 2B-4T, 1.36x TriLM 3.9B, 1.29x Ternary-Bonsai 8B, 1.35x Llama3-8B-1.58.
  - 2-column verify: 1.11x.
  - 4 / 8 columns and prefill: unchanged within +-2%, with one small measured loss: Llama3-8B prefill 0.98x.
  - The arithmetic is unchanged: 351 MUL_MAT cases (`dump_tq2.c`) and perplexity are bit-identical to the old schedule. Four or more row groups
per pass give each core four or more concurrent weight streams, which the DRAM prefetchers need (one stream per core:
42 GB/s, four: 80 GB/s on the dev VM). Each format also gets a one-row-group GEMV on the same packing for range
remainders, and `_xprep` / `_packed_x` entry points so activations are quantized and prepared once per op.

## Ternary weights: presence bitmap + signs (`kurn-tbm.cpp`)

TQ1_0 and TQ2_0 tensors can be repacked as one presence bit per weight plus one sign bit per nonzero ("tbm", the
BITCOS layout): 2.0625 - z bits per weight for zero fraction z, instead of TQ2_0's 2.0625. Measured zero fractions
(`kurn ternary stats`): BitNet b1.58 2B-4T 0.422, TriLM 3.9B 0.387, Ternary-Bonsai 8B 0.383, Llama3-8B-1.58 0.357,
Falcon3-7B-1.58bit 0.313. That is 15-20% fewer bytes than TQ2_0 on all five, and from -2.8% to +3.7% against
TQ1_0's 1.6875 (break-even z = 0.375). TQ1_0 tensors, which kurn had no kernel for, use the same path. Below the
threshold they go to the TQ2_0 kernels, converted without changing any value.

- **Layout** (spec and reference encoder: `kurn.ternary`). Records of 16 rows x 256 values (one Q8_K block): 16 fp16 row
  scales, 64 bitmap words, a flag byte, and ceil(nnz / 8) sign bytes. A table gives each 16-row group's start. Word
  `kg * 8 + kk`, bit `b` is row `b / 4`, value `32 kg + 4 kk + b % 4`: the bytes of the vpdpbusd operand that the
  TQ2_0 kernels build from 2-bit codes.
- **Decode** per 64 weights: `S = pdep(next popcount(P) sign bits, P)`, `u = P ? (S ? 2 : 0) : 1`, which is TQ2_0's code.
  The integer dot per 32 values (seeded with -sum(x)) and the epilogue `fma(float(isum), d_w * d_x, acc)` are TQ2_0's,
  so results are **bit-identical to the TQ2_0 kernels at every width** (`test_kurn_buft tbm`), and batch invariance
  and exact mode hold.
- **Kernels:** a GEMV with 4 row-group streams and 1 KB software prefetch per stream; verify kernels for 2..8 columns;
  and from 32 columns an AMX-INT8 prefill. The AMX prefill decodes s8 tiles once per panel and runs tdpbssd per 32
  values with the same epilogue, so it is batch invariant and also used in exact mode. Without AMX, prefill uses the
  verify kernels over row chunks.
- **Cost:** the decode is front-end bound at about 1.8 cycles per 64 weights, against 1.15 for TQ2_0 (L2-resident,
  one core). It therefore only wins when the weights stream from DRAM. There it matches TQ2_0's effective bandwidth
  and wins by the byte ratio; in L3 it ties.
- **Allocation:** `get_alloc_size` runs before the data is known, so a tensor gets the layout's worst case (about
  TQ2_0's size). Only the used bytes are written, and the tail pages are never touched.

Decode matmuls of whole decoder layers (`tools/opbench`, real shapes, 8 threads, medians over alternating rounds,
`OPBENCH_ZERO` = the model's zero fraction), against the TQ2_0 kernel with its current default schedule (`rows=4
prefetch=4 pfgran=line`, "tuned") and with the earlier `rows=2` default ("old"):

| shapes (z) | tbm vs TQ2_0 tuned | vs TQ2_0 old | vs ggml TQ2_0 / TQ1_0 |
|---|---|---|---|
| BitNet 2B-4T (0.422) | 1.13x | 1.51x | 1.96x / 2.11x |
| TriLM 3.9B (0.387) | 1.10x | 1.55x | 2.03x / 2.34x |
| Ternary-Bonsai 8B (0.383) | 1.06x | 1.57x | 2.04x / 2.25x |
| Llama3-8B-1.58 (0.357) | 1.02x | 1.48x | 2.02x / 2.23x |
| Falcon3-7B-1.58bit (0.313) | 1.07x | 1.59x | 2.12x / 2.32x |

The same against tuned TQ2_0 at other widths: 2 / 3 / 4 / 8 columns 1.16 / 1.20 / 1.13 / 0.86x (BitNet shapes; Bonsai
8B 1.08x at 2 and 4, 0.94x at 8), and
prefill at 128 / 512 columns (AMX-INT8, against TQ2_0's verify-kernel prefill) 1.17 / 1.20x. Zero-fraction sweep:
0.94x at z = 0.25, 1.03x at 0.31, 1.0-1.2x above, which is why the default threshold is 0.30. End-to-end numbers:
`docs/kurn-ternary-results.md` in the project store.

## Tests

`test_kurn_buft.c` is a test-backend-ops-style checker for `MUL_MAT` and `MUL_MAT_ID` on KURN-buffer weights against
a reference built from ggml's own `vec_dot` (relative error < 1e-4), with batch invariance (column 0 bit-identical
for every batch size), buffer reuse, and shapes the buffer type must refuse (K above the kernel limit):

```sh
L=~/src/llama-kurn
gcc -O2 -I$L/ggml/include test_kurn_buft.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin -o t
./t smoke | ./t quick [q8_0,q4_0] | ./t full        # exit code = number of failing cases
./t native                                          # native-layout Q6_K / Q5_K (also with GGML_KURN_EXACT=1)
./t tbm [smoke]                                     # TQ1_0 / TQ2_0 as bitmap + signs: bit-identical to TQ2_0 records
./t case q8_0 4096 4096 16 8                       # one case: TYPE K N M THREADS [REPS]
./t bf16                                           # AMX-BF16 prefill path: error vs an f32 reference <= 2x ggml's vec_dot
./t fused                                          # fused gate + up + SwiGLU == unfused, bit for bit (also with GGML_KURN_EXACT=1)
./t freivalds                                      # Freivalds oracle on the AMX-BF16 path: accepts real outputs, rejects faults
TEST_BUFT=AMX ./t quick q8_0                         # same checks against another extra buffer type
```

Every mode except `bf16`, `fused` and `freivalds` checks exact results and runs with `GGML_KURN_EXACT=1` unless the
variable is set, so on AMX CPUs it exercises the AMX-INT8 Q8_0 kernel (bit-identical to the GEMV) rather than the
AMX-BF16 path.

**Freivalds oracle for KL-mode kernels.** Kernels that are not bit-exact (today the AMX-BF16 path) are checked the way
github.com/WizardSource/bigVerify checks float matmuls: secret +-1 probes per 64 output columns, `C r` against
`X (W r)` in double from the f32 inputs, with a threshold from the path's error model (bf16 inputs, 2^-8 per product;
fp32 accumulation) in statistical mode and Hoeffding over the probe signs (false alarm <= 2^-40 per check). It costs
O(t N/64) of the product. The detection floor is about 0.57x the output RMS; real outputs peak at ~0.3 of the
threshold. `./t freivalds` checks that real outputs pass and that a 1x-RMS entry fault and a swapped 16 x 16 tile
(what lost AMX tile state looks like) are rejected; `./t fused` runs the AMX-BF16 members of the fused op through it.
Test-only: nothing in llama.cpp calls it.

**Per-node profile.** `tools/opprof.patch` (`git apply` in the llama.cpp checkout) adds `GGML_OPPROF=1` to ggml's
graph executor (`GGML_OPPROF=2`: every graph, decode included): for every prefill graph it records, per node type (weight type, buffer and shape for matmuls), the
cycles each thread spends computing and waiting at the barrier after the node, and prints the table at exit.

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

- **fast** (default): tile engine `amx` (AMX-BF16; default where AMX is usable; the kernel times each tile block and
  recomputes blocks that spanned a preemption, see the AMX caveat), `f32` (default without AMX or with
  `GGML_KURN_AMX=0`) or `bf16` (AVX512-BF16). On Qwen3-8B Q8_0 (WikiText-2, ctx 4096, AMX prefill matmuls) KL vs F16
  is 0.00047 with `amx` and 0.00040 with `f32`; the shipped build before AMX prefill measured 0.00117.
- **exact** (`GGML_KURN_FA_MODE=exact`): the f32 tile engine for every batch size and a single KV split. A token's
  attention output then does not depend on how many tokens are computed with it, the (padded) KV length or the thread
  count; together with the buffer type's batch-invariant matmuls, a speculative verify batch reproduces one-token
  decoding bit for bit. Decode is slower than fast mode at long context: the tile engine computes the decode rows in
  blocks of 4 (n_q x G, padded to 4; AMX pads to 32), Qwen3-1.7B tg32 0.80x fast mode at 4K, 0.49x at 32K.

| variable | effect |
|---|---|
| `GGML_KURN_FA=0` (or `GGML_KURN=0`) | ggml's flash attention only |
| `GGML_KURN_FA_MODE=fast\|exact` | see above (default fast) |
| `GGML_KURN_FA_ENGINE=f32\|amx\|bf16` | tile engine of fast mode (default amx where AMX is usable, f32 otherwise or with `GGML_KURN_AMX=0`) |
| `GGML_KURN_FA_PACK=0` | no per-call K/V packing (bf16 / AMX engines) |
| `GGML_KURN_VERBOSE=1` | log the configuration, every new node shape with its kernel or fallback reason, and node counts at exit |

Checks: `test-backend-ops -o FLASH_ATTN_EXT -b CPU` compares the CPU backend against itself in reference mode, where
kurn steps aside, so every supported case is kurn against ggml's vec kernel (all 5,314 cases pass in every mode and
engine; 545 of them run on kurn). `tests/test_llama_attn.py` checks the generator, runs the renamed kernels through
kurn's harness, checks that `exact` is batch-invariant, and runs a test-backend-ops subset when a built checkout is
found (`KURN_LLAMA_CPP`).

## k4c KV cache (`k4c/`)

`k4c/apply.sh LLAMA_DIR` (after `apply.sh`) adds `GGML_TYPE_K4C`, kurn's k4c key format, as a llama.cpp K cache type:
`-ctk k4c` is kurn's `k4c_q8` (V q8_0, the long-context default), `-ctk k4c -ctv q4_0` its `k4c_q4` (`-ctk k4c_q4` /
`-ctk k4c_q8` set both). Keys are
quantized to 4 bits per channel over groups of 32 cache cells (f16 scale and min per channel and group, 5 bits per value
on average); kurn's attention reads the groups directly (`kurn-attn.cpp`, kernels `kattn_*_k4c_q4/q8_*`).

- **After RoPE.** llama.cpp caches rotated keys, so the kernels run with `rope_dim 0`; kurn's own engine stores keys
  before RoPE. On Qwen3-1.7B the per-channel 4-bit keys are still accurate (WikiText-2, ctx 2048, KL vs F16 KV with
  ggml's FA: k4c_q4 0.016 at +0.06% perplexity, k4c_q8 0.010; llama.cpp's own Q4_0 KV 0.32 at +29%, 1.15 at +107%
  without its Hadamard rotation).
- **Writes** (`ggml-k4c.c`, SET_ROWS into a K4C cache): each touched group is re-encoded from the exact values of its
  valid rows, kept as f16 in a bounded table (`GGML_K4C_EXACT_MB`, default 256) for recently written groups. A group
  filled one token at a time, rolled back (rejected drafts) or partly cleared holds the same bytes as the same rows
  written at once; freed cells are cleared, so stale keys never widen a group's range. Older groups that left the table
  are re-encoded from their 4-bit values if they are ever rewritten.
- **Session state** (prompt cache, `--slot-save-path` save / restore, `llama_state_seq_*`): K4C keys are stored as native
  group records, the size of the live cache (Qwen3-1.7B, 32K tokens: k4c_q4 1,065 MiB, k4c_q8 1,513 MiB; f16 rows were
  2,296 / 2,744). A group that holds only the saved rows is stored as its bytes; the partial last group as the codec's
  exact f16 values (re-encoding them gives the same bytes and keeps later appends exact); a group shared with other
  sequences as f16 rows. Restored at the same in-group offsets the state is byte-exact and the logits that follow are
  bit-identical; at other offsets the records are re-encoded (as close to F16 KV as the live run in `test_k4c_state.cpp`).
  The row-size field marks the form: older f16-row files still load, and `LLAMA_K4C_STATE=f16` writes them for older
  readers. With `GGML_KURN_FA_MODE=exact`, llama-server's cached, restored and recomputed continuations are bit-identical.
- **K-shift** (llama-server's context shift, `--cache-reuse`, `llama_memory_seq_add` / `seq_div`): the shifted cells are
  read as floats (`ggml_get_rows_k4c`: the exact f16 values while the codec keeps them, else the 4-bit ones), rotated
  by their position delta with the f16 cache's RoPE and written back (`ggml_set_rows_k4c`), which re-encodes their
  groups. `test_kshift.cpp` runs it through llama.cpp: KL to F16 KV after a shift is within noise of the KL without one.
- **Which V:** k4c_q8 holds the 2% perplexity budget at 8-32K tokens of depth; k4c_q4 is marginal there (see
  `docs/kurn-kv-backlog-results.md` in the project). V is Hadamard-rotated by llama.cpp's `attn_rot_v` for both.
- **Not supported:** the non-FA path, V types other than Q4_0 / Q8_0, MLA models, cache sizes that are not a multiple
  of 32, non-CPU buffers.

`k4c/test_k4c.c` checks writes in llama.cpp's patterns (prefill ending inside a group, appends, rollback, out-of-order
rows, clears), byte-identical groups for prefill vs appends + rollback, FA on K4C against ggml on the dequantized keys,
batch invariance in exact mode and K-shift against keys written at the shifted positions; `k4c/test_kshift.cpp` checks
K-shift end to end through libllama; `k4c/test_k4c_state.cpp` checks session state (byte-exact restore, older
files, restores at other cell offsets, sizes); `tests/test_llama_k4c.py` runs them against a built checkout.

## Speculative decoding: cost-aware verify width (`spec-width/`)

Verify cost on this buffer type is not linear in the width: there is one kernel per width 2..8, and every 8 more
columns add a group pass (before the exact-width kernels it was a staircase: 3 columns cost as much as 4 and 5-7 as
much as 8). `spec-width/` sizes the draft to the measured costs:

- `kurn-spec-calib` (same flags as `llama-speculative-simple`; `KURN_CALIB_OUT=prefix`) measures the whole-forward
  cost table (`verify M ms`: target, M tokens with logits for all; `draft M ms`) and a greedy acceptance trace.
  `kurn specwidth show prefix.cost` prints it; `kurn specwidth simulate prefix.cost *.trace` replays traces against
  fixed widths and the policy.
- `kurn-spec-width.h` (header-only, the twin of `kurn.specwidth.WidthPolicy`) picks the draft length that maximises
  expected accepted tokens minus lambda x time (lambda = running tokens/ms), with acceptance learned online per
  draft-confidence bin. `KURN_SPEC_WIDTH=prefix.cost llama-speculative-simple ... --spec-draft-n-max 15` caps,
  stops and truncates every draft through it (`KURN_SPEC_WIDTH_MODE=cap`: rate-only cap). The patch adds draft
  confidences and a keep-drafting callback to draft-simple; without `KURN_SPEC_WIDTH` behaviour is unchanged.

- **llama-server** (`llama-server-spec-width.patch`): `--spec-width TABLE` (or `KURN_SPEC_WIDTH=TABLE`) with a draft
  model (`-md DRAFT --spec-type draft-simple --spec-draft-n-max 15`) gives every slot its own policy, which keeps
  learning across the slot's requests: the draft length is capped per step, drafting stops per token, and drafts are
  truncated before verification (`--spec-width-mode cap`: rate-only cap). A step the policy declines runs without a
  draft. Each finished request logs `kurn spec width: verify widths (M:steps) = ...`.

```sh
kurn/integration/llama.cpp/spec-width/apply.sh ~/src/llama-kurn   # after apply.sh; idempotent
cmake --build build -j --target llama-server llama-speculative-simple kurn-spec-calib
llama-server -m TARGET.gguf -md DRAFT.gguf --spec-type draft-simple --spec-draft-n-max 15 --spec-width target.cost
```

With `GGML_KURN_FA_MODE=exact` (kurn attention, see above) speculative output equals the no-draft output: Qwen3-1.7B /
Qwen3-0.6B draft in llama-server, 8 prompts x 4 draft configs, 0 of 32 runs differ (ggml's FA: 22, kurn fast mode: 21).

## RSI build: retrieval drafts, speculative probabilities, replay (`rsi/`)

`rsi/apply.sh` (after `apply.sh`, `spec-width/apply.sh` and `k4c/apply.sh`) adds four things to llama-server.

**Retrieval drafts as an arm of the verify selector.** With `--spec-width TABLE` and an n-gram `--spec-type`
(`ngram-simple`, `ngram-map-k`, `ngram-mod`, `ngram-cache`), each slot gets a `kurn::arm_selector`
(`kurn-spec-width.h`, twin `kurn.specwidth.ArmSelector`). Retrieval drafts come from the prompt and the generated
history and cost no draft time; the draft model (`-md`, `draft-simple`) is the second arm. Every step:
- a lookup candidate is cut to the prefix worth verifying, and taken only if its value (expected accepted tokens
  - lambda x time, acceptance learned online) is at least the model arm's;
- otherwise the draft model drafts, if that is worth it now, including the decode it must catch up on first;
- otherwise the step runs without a draft.

The draft model syncs lazily: it no longer mirrors every target batch, and catches up from the committed tokens
only when it drafts (`KURN_SPEC_LAZY=0` restores mirroring). Each request logs
`kurn spec arms: steps none N, retrieval N (accepted/drafted), model N (...)`.

```sh
llama-server -m TARGET.gguf -md DRAFT.gguf --spec-type ngram-simple,draft-simple --spec-ngram-simple-size-n 3 \
  --spec-ngram-simple-size-m 16 --spec-draft-n-max 15 --spec-width target.cost      # -md optional
```

**Probabilities of accepted speculative tokens.** With `n_probs`, accepted draft tokens used to come back with
`logprob 0` and empty `top_logprobs`. They now carry what plain decoding reports: pre-sampling from the verify batch's
logits, post-sampling (`post_sampling_probs`) by replaying the positions on a copy of the sampler.

**Replay primitives** (per request, off by default):
- `"kurn_replay": 1` returns `kurn_replay` with a run manifest and a sha256 hash chain over the generated tokens and
  their probabilities (`kurn-replay.h` documents the format); `2` adds per-token hashes, which locate the first
  divergent token. The manifest has the build (`llama_build_info`, `kurn_integration_sha256`), model and draft paths,
  sizes and, with `KURN_MANIFEST_SHA256=1`, sha256 of the files, every `GGML_KURN*` / `KURN_*` variable, the CPU
  model and a hash of its flags, threads, context and batch sizes, KV types, speculative settings with the cost
  table's hash, and the request's sampling parameters with the resolved seed.
- `"kurn_rng_position": true` replaces the sequential sampling RNG with a uniform derived from (seed, position):
  a request resumed from its own prefix continues with the same tokens, and batching or speculation cannot shift
  the stream.

`rsi/rsi_check.py --server ... --model ... --draft ... --table ...` checks all of this end to end in exact mode.
`tests/test_llama_rsi.py` checks the patch, the checkout and the hash-chain format.

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
