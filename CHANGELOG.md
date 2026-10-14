# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased] - 0.3.0.dev3

### Changed: AMX preemption guard threshold 8192 -> 4096 cycles
- Matmuls and the ternary kernel (`GGML_KURN_AMX_GUARD`, plus 128 per k-step), attention (`KA_AMX_GUARD`), the AMX
  verify kernel (`KURN_AMX_GUARD`) and the hypercomplex layers. `tools/amx_selftest.c` saw preemptions return in ~12K cycles, under
  the old 12,288-cycle threshold of a 32-step step: 5 wrong results in 2.47B computations at 8192, 0 in 2.13B at 4096.
- No measurable cost (paired CI, Qwen3-8B Q8_0): pp512 0.986 (0.931-1.013), pp2048 0.999 (0.973-1.027), exact pp512
  1.002 (0.991-1.027); opbench at M = 512 0.994 / 1.003 / 1.003. `benchmarks/v0.3/amx-guard/`.
- About 2% of AMX-BF16 steps are now redone (0.25% before). After a failed self-test, `GGML_KURN_AMX_GUARD` up to
  8192 is still honoured, as margin for machines with slower memory.

### Added: AMX startup self-test
- `kurn-amx-selftest.cpp`: at load, about 10-20 ms of int8 and bf16 tile matmuls against a scalar reference, plain
  and across forced context switches. On a machine that loses AMX tile data across context switches it warns, keeps
  the AMX kernels with the preemption guard forced on and stops handing small weights to ggml's unguarded AMX buffer
  (`GGML_KURN_AMX_SELFTEST_ACTION=off`: non-AMX kernels instead). Wrong results without a context switch turn AMX
  off. `GGML_KURN_AMX_SELFTEST=0` skips it.
- `tools/amx_selftest.c`: the standalone test, for mismatch rates under load, with and without the guard.

### Added: fused gate+up+SwiGLU epilogue (`epi=swiglu`), GEMV and tensor-core GEMM
- W is the stacked [gate; up] matrix (N = 2F rows, ggml blocks or one KTBM buffer). The kernel writes Y[m][f] =
  silu(gate f · x_m) · (up f · x_m), [M][F] only, so gate and up outputs never round-trip to memory.
- One activation quantize / convert serves both matrices:
  - GEMV: each output row runs the gate and the up row against the same q8 activation blocks. SASS confirms the up
    row reuses the gate row's activation loads: in the q4_0 hot loop the weight loads double and the activation loads
    stay the same.
  - GEMM: the repack interleaves gate tile j and up tile j as packed row tiles 2j and 2j+1 (`KG_SRT`, also in the tbm
    device walk). Each warp holds a matching pair, which reads the same staged activation tile. Rules: bm / wm a
    multiple of 32, no split-K, N % 32 == 0.
- GEMV register limits: a SwiGLU GEMV with c columns uses the measured limits for 2c columns.

### Added: fused Freivalds silent-data-corruption check (`epi=fv`, `fvb` 64 / 256 / 1024)
- This is bigVerify's float "statistical" mode fused into the GEMM epilogue, so the check adds no extra pass over Y,
  W or X.
  - For each block of `fvb` output rows, column slot (columns folded into 16 slots with secret signs) and probe
    (2 secret ±1 row probes), every CTA adds Σ r·acc from its accumulators (C side). It also adds its share of
    Σ_k F_b[k]·x[m][k] from a precomputed fingerprint F_b = W_bᵀ r (X side; the block's CTAs split the k-windows).
  - The block's last CTA compares the two against τ = λ·K·2⁻²⁴·sqrt(Σ_k x² G2_b) (G2 = Σ W² over the block, bf16
    rounded up). τ comes from trusted inputs only, so a fault cannot inflate it, and NaN fails. The last CTA then
    counts {checks, fails, worst passing ratio} and resets for the next call.
  - Split-K is fine, because both sides are linear.
- ABI (every kernel): `kg_epi`, `kg_fv_block`, `kg_fv_setup(Wp, N, K, F, G2, signs, key, lambda)`,
  `kg_fv_status(Wp, N, K, st[3])`. The region sits after the prepared weights in `kg_prep_bytes`. The host
  fingerprint is `kref_fv_fingerprint` in `kurn_gpu_ref.h`.
- Fingerprint traffic per call is 10 bytes per K per block: 1.7% / 7% / 28% of Q4_0 weight bytes at fvb 1024 / 256 /
  64. bigVerify's sensitivity, in contrast, grows with smaller blocks.

### Added: single-pass wide verify (GEMV `cols=16`)
- The dp4a GEMV takes `cols=16`, so 9-16 columns stream the weights once (`cols=8` made two passes). The engine's bn=16
  tiles were already single-pass.
- Legal at unroll 1 for Q4_0, IQ4_NL, Q2_0, Q1_0, MXFP4 and NVFP4 (256-512 threads). The register limits are measured
  by ptxas on sm_80 / sm_90 / sm_100 / sm_120 and merged into gemv_threads.json.
  - `tools/gemv_threads.py --cols` measures and merges only those column counts.
  - It now skips combinations the static cols × unroll cap forbids and records them as 0: 1176 instead of 4788
    configurations.
- Exact on the emulator for all ten GEMV formats × 3 layouts at 9 / 12 / 16 columns.
- The new limits also make SwiGLU GEMVs with up to 8 columns legal: 594 such variants at their largest block sizes show
  0 spills on 4 archs. The exception is Q1_0 with sub=1: it expands 2 × 128 values per lane, so it is capped at
  cols × unroll ≤ 4 (measured spills otherwise).

### Kit: `kurn gpu ablate` (run by run_gpu_check.sh; DRYRUN builds the kernels only)
- Three experiments:
  - ffn: plain kernel + separate SwiGLU kernel (`bench_gpu run --swiglu-after 1`, one graph) vs fused, Llama-3-8B
    gate+up;
  - width: GEMV cols 8 / 16 and the bn=16 engine tiles at M = 1, 8, 9, 12, 16;
  - fv: overhead per `fvb`, false alarms over every timed call, worst |C − X| / τ, and detection of a corrupted weight
    tile (`--corrupt 1`, `--lambda`).
- Rendered as "Fused epilogues and wide verify (ablations)" in report.md and the kit v2 report.

### Added: compact quaternion and complex linear layers (`integration/hypercomplex/`)
- `kurn-hc.{h,cpp}`: layers that keep D = 4 (quaternion) or D = 2 (complex) components per weight instead of the
  D^2 reals of the expansion, in f32 or Q8_0-per-component (`KURN_HC_Q8`, kurn i16 records per component).
- GEMV and 1-8 column verify kernels compute the Hamilton / complex product from shared loads, input-component-outer
  (`p = o ^ q`); 9-31 columns (and exact mode) run them over 8-column groups; 32+ columns use AMX-BF16 on the expansion
  built per L2 panel.
- Every path is bit-identical to kurn's real Q8_0 kernels on the expanded matrix (block-planar layout), fast and
  exact mode; activations quantized with ggml's `quantize_row_q8_0`, reproduced bit for bit.
- Qwen3-8B-width layers (4096 x 12288 real), paired CI: quaternion decode 3.1x, MLP decode 4.15x, 8 columns 2.0x,
  prefill parity; complex decode 2.1x, MLP decode 1.9x.
- `hc_bench` (checks, benchmarks, synthetic quaternion MLP), `kurn.hypercomplex` references, `tests/test_hypercomplex.py`.

### Added: kvz - lossless entropy-coded KV cache in llama.cpp session state
- `integration/llama.cpp/kvz/apply.sh` (after `k4c/apply.sh`): `llama-server --cache-kvz` entropy-codes the KV cache
  of RAM prompt cache entries (`--cache-ram`, parked idle slots) and `--slot-save-path` files. In the API it is
  `LLAMA_STATE_SEQ_FLAGS_KVZ` and `llama_state_seq_save_file_ext`; readers detect it.
- Restores are byte-exact: the restored state equals the plain one, and the next tokens' logits are bit-identical.
- Coding: per-channel static rANS over f16 byte planes and Q8_0 / Q4_0 scales and codes (`llama-kvz.{h,cpp}`).
- Sizes on real Qwen3 caches: F16 1.19x, Q8_0 1.12x, k4c_q4 state 1.17x (zstd -3: 1.02-1.11x).
- Speed: ~1.5 / 1.9 GB/s encode / decode for F16 on 8 threads.
- Tests: `test_kvz.cpp` (codec, ASan / UBSan), `test_kvz_state.cpp` (every cache type on a real model),
  `server_kvz_test.py` and `tests/test_llama_kvz.py`.

### Changed: k4c session state stores native group records
- K4C keys in session state (slot files, `--cache-ram`, `llama_state_seq_*`) are no longer f16 rows: groups owned by the
  saved sequence are stored as their bytes, the partial last group as the codec's exact f16 values, groups shared with
  other sequences as f16 rows. Qwen3-1.7B at 32K tokens: k4c_q4 2,296 -> 1,065 MiB (2.16x), k4c_q8 2,744 -> 1,513 MiB.
- Restore at the same in-group offsets is byte-exact with bit-identical logits afterwards (`k4c/test_k4c_state.cpp`);
  at other offsets records are re-encoded. The row-size field tells the forms apart, so older f16-row files load;
  `LLAMA_K4C_STATE=f16` writes the old form.

### Changed: k4c_q8 is the long-context k4c default
- Position-bucketed KL / ΔPPL vs F16 weights + F16 KV on three 32K-token windows of released math papers (Qwen3-1.7B Q8_0
  weights): k4c_q8 stays within the 2% ΔPPL budget in every paper and 8K band (worst +1.83%, mean +0.76 / +1.16 / +0.78 /
  +0.72% for 0-8 / 8-16 / 16-24 / 24-32K; F16 KV +0.50 / +0.60 / +0.30 / +0.14%); k4c_q4 is marginal (worst +2.10%).
  KL rises with depth for both (k4c_q8 0.0073 at 0-4K -> 0.0124 at 28-32K).
- `-ctk k4c` without `-ctv` now means k4c_q8 (V q8_0) instead of an error; `-ctv q4_0` still gives k4c_q4.

### Added: K-shift for k4c KV caches (llama-server context shift and --cache-reuse)
- `get_can_shift()` is no longer false for K4C. The K-shift graph reads the shifted cells (`ggml_get_rows_k4c`: the
  codec's exact f16 values for recently written groups, else the 4-bit values), applies the f16 cache's RoPE by the
  position delta and writes them back (`ggml_set_rows_k4c`). Session-state writes read whole groups at once.
- Qwen3-1.7B through libllama (`k4c/test_kshift.cpp`, A + B + C with B removed and C shifted, 200 scored tokens), KL to
  F16 KV under the same operation: k4c_q4 0.024 fresh / 0.028 after the shift, k4c_q8 0.014 / 0.020, q8_0 0.0009 /
  0.0011 (the shift itself moves F16 by 0.068). llama-server: context shift in a 1024-token context and `--cache-reuse 64`
  (1 prompt token processed instead of 1,061) behave as with F16 KV.

### Added: Hadamard-rotated V in kurn's engine and attention harness
- Engine (`KURN_KV` quantized formats): `KURN_VROT=n` (default 64, 0 = off) rotates each V row by the normalized
  Walsh-Hadamard matrix in n-blocks before Q4_0 / Q8_0 quantization and rotates each head's attention output back,
  as llama.cpp's `attn_rot_v` does. Qwen3-1.7B, WikiText-2 ctx 2048, KL vs F16 KV: k4c_q4 0.0112 -> 0.0091 (-19%),
  k4c_q8 0.0075 -> 0.0041 (-45%), q8_0 0.0088 -> 0.0055 (-37%); q4_0 stays dominated by its unrotated post-RoPE K.
- Harness (`bench_attn`, spec keys `vrot` / `vout`): `--v-rot N` stores V rotated and rotates the output back;
  `--v-outlier S` scales 4 V channels; new CSV column `vqerr` = relative RMS error against the unquantized V. With
  outlier channels (x4-16) the rotation cuts vqerr 1.5-3x for q8_0 / k4c_q4 / k4c_q8; on iid V it is neutral.

### Changed: attention tile engine pads query rows to 4, not 32, on the f32 and AVX512-BF16 engines
- `ka_tile` computed every tile for 32 query rows (the AMX tile height) on every engine. The f32 and AVX512-BF16 engines
  work in 4-row register blocks, so decode-sized tiles (n_q x G = 1-8 rows) now compute 4-8 rows. Rows are independent:
  outputs are bit-identical to before (42 cases, three engines), and batch invariance of the `exact` llama.cpp mode holds.
- llama.cpp exact mode (`GGML_KURN_FA_MODE=exact`), Qwen3-1.7B Q8_0, tg32: 1.22x / 1.42x / 1.50x at depth 4K / 16K / 32K
  (29.7 -> 36.2, 10.5 -> 15.0, 6.2 -> 9.2 tok/s); 8-token batches 1.14-1.23x. Fast mode's decode engine is unchanged.

### Changed: TQ2_0 kernels default to 4 row groups + whole-record prefetch
- `TUNED_KEYS["tq2_0"] = rows=4 prefetch=4 pfgran=line` (was `rows=2 prefetch=0`; `kurn.specwidth` mirrors it). The
  shipped schedule streamed ~60 GB/s from DRAM, against 88 GB/s now.
- Paired A/B with alternating order and 95% CIs (`benchmarks/v0.3/ternary/paired_tq2.sh`, `paired_ci.py`):
  - decode layers of five ternary models: 1.29-1.47x;
  - `llama-bench` decode: BitNet 2B-4T 1.15x, TriLM 3.9B 1.36x, Ternary-Bonsai 8B 1.29x, Llama3-8B-1.58 1.35x;
  - 2-column verify: 1.11x;
  - 4 / 8 columns and prefill: within +-2%, with Llama3-8B prefill 0.98x.
- Bit-identical to the old schedule: 351 raw MUL_MAT outputs (`dump_tq2.c`, exact on / off, AMX on / off) and
  perplexity on three models.

### Added: ternary weights as presence bitmap + signs (TQ1_0 / TQ2_0 in the KURN buffer type)
- `kurn.ternary` defines the layout ("tbm", the BITCOS idea: one presence bit per weight and one sign bit per
  nonzero, 2.0625 - z bpw at zero fraction z). It has the reference encoder / decoder and the size model.
  `kurn ternary stats FILE...` measures zero fractions of GGUF TQ1_0 / TQ2_0 tensors and of packed 2-bit safetensors.
  Real models: BitNet b1.58 2B-4T 0.422, TriLM 3.9B 0.387, Ternary-Bonsai 8B 0.383, Llama3-8B-1.58 0.357,
  Falcon3-7B-1.58bit 0.313. That is 15-20% smaller than TQ2_0 on all five; against TQ1_0 it ranges from -2.8% to
  +3.7%, not the ~9% the preprint reports at z = 0.515.
- `integration/llama.cpp/ggml-kurn/kurn-tbm.cpp`: the KURN buffer type repacks TQ1_0 and TQ2_0 tensors into this
  layout at load time when their sampled zero fraction is >= 0.30 (`GGML_KURN_TBM`, `GGML_KURN_TBM_Z`). TQ1_0
  tensors were stock ggml before; below the threshold they now go to the TQ2_0 kernels.
  - Kernels: GEMV, verify kernels for 2..8 columns, and AMX-INT8 prefill from 32 columns.
  - The decoded code is TQ2_0's, so results are bit-identical to the TQ2_0 kernels at every width, and batch
    invariance and exact mode hold (the AMX-INT8 prefill is exact and is used in exact mode too).
- Decode matmuls of whole layers (opbench, 8 threads) against TQ2_0 with a tuned `rows=4 prefetch=4 line` schedule:
  1.02-1.13x on five models' shapes, 1.5x against the earlier `rows=2` TQ2_0 schedule, and 2x against ggml's TQ2_0 / TQ1_0.
  Prefill (512 columns) is 1.20x TQ2_0's verify-kernel prefill. The decode costs 1.8 cycles per 64 weights against
  1.15 for TQ2_0: it wins when weights stream from DRAM and ties in L3.
- `test_kurn_buft tbm [smoke]` checks bit-identity with the TQ2_0 records (TQ1_0 and TQ2_0; z = 0, 0.42 and 0.9;
  1..129 columns), batch invariance, buffer reuse and MUL_MAT_ID. `tests/test_ternary.py` covers the reference
  encoder / decoder, decoding against gguf-py, the scans, and the C packer byte for byte against the Python
  reference. `tools/opbench` takes `OPBENCH_ZERO=z` (exact ternary weights at zero fraction z).

### Added: one-pass AMX verify kernel for Q8_0 at 9-16 columns
- `kurn-amx-vfy.cpp`: M-row AMX-INT8 tiles on kurn's i16 records, block-order scaling as in the GEMV, so every column
  is bit-identical to one-token decoding; three row groups per pass; preemption guard as in the other AMX paths.
- Qwen3-8B Q8_0 `opbench`, paired CI: 9 / 12 / 16 columns -7% / -12% / -19%. `GGML_KURN_AMX_VFY=0` turns it off.

### Added: fused attention q + k + v on KURN weights
- 2-4 consecutive MUL_MATs on the same activations run as one op; `apply.sh` makes llama.cpp's `build_qkv` emit
  q, k, v as consecutive nodes when the weights are in the KURN buffer.
- Bit-identical; block graph prefill -2.3% (paired CI), decode trend -1.5% (unresolved). `GGML_KURN_FUSE_QKV=0`.

### Changed: work stealing for decode-sized matmuls (1-8 columns)
- Each thread streams its own contiguous share in shrinking batches; finished threads steal the back half of the
  unclaimed rest of the slowest thread. Same per-row arithmetic, bit-identical outputs.
- Qwen3-8B Q8_0 `opbench`, paired CI: 1 / 4 / 8 columns -3.4% / -3.2% / -2.9%; `GGML_KURN_STEAL=0` restores the
  guided chunks.

### Added: fused FFN gate + up + SwiGLU on KURN weights
- `ggml_kurn_fused_mul_mat`, hooked into ggml-cpu's graph fusion by `apply.sh`: one activation quantization, one row
  schedule for both weights, SwiGLU applied to thread-local row tiles; gate and up never reach memory.
- Bit-identical to the unfused graph, exact mode included (`test_kurn_buft fused`, whole-model logits).
- Qwen3-8B Q8_0 block graph, paired CI: prefill (512 columns) -4.6%, decode -1.8%; `GGML_KURN_FUSE=0` turns it off.
- `test_kurn_buft freivalds`: bigVerify-style Freivalds oracle for KL-mode (AMX-BF16) kernels, test-only.
- `tools/blockbench.c`, `tools/logits_dump.cpp`, `GGML_OPPROF=2`, `benchmarks/v0.3/matmul-opt/`.

### Added: RSI build - retrieval drafts in the verify selector, speculative probabilities, replay primitives
- `integration/llama.cpp/rsi/` (`apply.sh`, `llama-rsi.patch`, `kurn-replay.h`, `rsi_check.py`):
  - n-gram / trajectory retrieval drafts and the draft model are two arms of the cost-aware selector
    (`kurn::arm_selector`, Python twin `kurn.specwidth.ArmSelector`), chosen per step;
  - lazy draft-model sync;
  - correct `completion_probabilities` for accepted speculative tokens (were `logprob 0`, empty top);
  - per-request `kurn_replay` (run manifest + sha256 hash chain over tokens and probabilities) and
    `kurn_rng_position` (position-keyed sampling).
- Qwen3-1.7B Q8_0 on 8 vCPUs, exact mode, outputs bit-identical to plain decoding:
  - repair turn 175 vs 44 tok/s (4.0x);
  - long thinking trace +18%.
- Qwen3-8B + 0.6B draft: 2.09x, unchanged from the draft-only selector.

### Changed: autotuner confirmation is a paired-CI gate
- `tune._confirm` measures 6-14 alternating pairs and admits a move only if the whole bootstrap CI of the median
  ratio is below `1 - min_gain` (alpha 0.05 split over the looks), with a three-valued decision.
- The median-of-3 rule it replaces admitted 2 of 10 real A/A trials (identical code "winning" by up to 9%).

### Fixed: `kurn verify` check shapes reach every kernel's main loop
- `harness.check` runs a second shape where each check thread gets a full pass of the widest row blocking
  (`check_shapes`, `main_loop_rows`).
- Before, N=200 over 3 threads never ran the main loop of a `rows=8` vnni16 GEMV, and a planted bug there passed
  (`tests/test_verify_coverage.py`).

### Added: ternary bitmap + signs (`tbm`, KTBM layout v1) on the GPU tensor-core engine
- The new weight format `tbm` takes the CPU side's settled KTBM v1 buffer: 16-row × 256-value variable-length records
  (fp16 d[16], a 64-word presence bitmap, sign bits for the nonzeros only) and a per-group offset table. It costs
  2.0625 − z bpw (z = zero fraction) against TQ2_0's 2.0625. `kurn.gpu.tbm` is an independent v1 encoder and decoder,
  built from the spec. The C side has the same in `kurn_gpu_ref.h`: `kref_tbm_encode` / `kref_tbm_row`, byte-identical
  to the Python one.
- `kg_prepare` repacks v1 on the device at load, in three kernels:
  - one thread per group walks its records;
  - one thread scans the per-tile sign sizes;
  - one warp per record writes the outputs.
  The outputs, at the same byte budget:
  - presence bits in mma fragment order (16 bytes per lane per 16×256 tile);
  - TQ2_0's scale plane;
  - each lane's signs compacted bit-continuously per tile;
  - a u32 sign offset per tile.
  v1's CPU word interleave already maps one 4-value nibble of one row onto two f16x2 A registers, so no layout v2 is
  needed.
- The kernel is packed-direct only (`wload=reg`, default `pf 2`):
  - phase 1 of each ring slot loads presence words and the tile's sign base;
  - one k-step later, phase 2 computes each lane's sign offset with a warp popcount scan and loads the (at most 5)
    sign words its signs span;
  - decode is one 2 KB shared-table lookup per nibble, (mask << 4 | next 4 sign bits) → two finished f16x2 registers.
  The values and scales are TQ2_0's, so results are **bit-identical to the TQ2_0 packed kernel on the same weights**.
- Verified on the emulator against the C decoder's exact reference:
  - 7 tile configs × random / dense / randomized-schedule data;
  - bit-identity with TQ2_0;
  - 7 injected bugs caught (scan, nibble placement, sign advance, window offset, repack nibble order, sign prefix,
    word spill).
  - ptxas: no spills and no local memory on sm_80 / sm_100 / sm_120; SASS shows HMMA, LDG.128, LDS.64 table lookups,
    POPC and SHFL.
  - Hot loop at bk 256: 815 instructions vs TQ2_0's 537 for the same 16 MMAs, against about 0.8× the bytes at
    40% zeros.
- Kit v2: the matmul matrix gains `tbm` (harness weights at 40% zeros; GB/s counts its encoded bytes). The reports get
  a "Ternary bitmap + signs (tbm) vs TQ2_0" ablation (same packed tile, speedup, weight-bytes ratio, 2σ verdict).
  `formats_for(op)` keeps GEMV-only paths off it.

### Added: packed-direct weights for the tensor-core GEMM/GEMV engine (`wload=reg`)
- This applies the "decode packed weights directly into tensor-core operands" transformation (AI lowering, 3.34x on
  BitDelta) to every engine format, including MXFP4, NVFP4, Q4_K, Q8_0 and ternary TQ2_0.
  - With `wload=reg`, each lane loads its own 16 fragment-order bytes per (row tile, k-tile), plus its scale words
    (Q4_K: the meta records of rows g and g+8). The loads are predicated 128-bit / 32-bit read-only `ld.global.nc`,
    straight into a register ring that runs `pf` (1-3) k-steps ahead of the MMAs.
  - The existing per-format decoders turn the ring into the mma.sync A fragments in place.
  - Weights get no cp.async, no shared-memory tile, no shared loads and no stage barrier; shared memory holds only
    the activations. `wload=smem` (the cp.async pipeline) stays the default, and its code is unchanged.
- SASS (sm_80, the batch-1-8 default tile): weight cp.async and LDS go to 0, with the same HMMA count. Hot-loop
  instructions: Q4_K -7% per MMA, NVFP4 -5%, Q8_0 -13%, MXFP4 -16%, Q4_0 -24%, TQ2_0 -31%. Whole-kernel instructions
  roughly halve.
- Correctness, on the emulator and ptxas:
  - every GEMM covering configuration passes on the emulator against the exact reference (539, about 230 of them
    packed), with random data, extreme data and randomized schedules;
  - 5 new mutants on the ring (wrong slot, ring not advancing, short issue distance, wrong scale block, Q4_K meta row)
    are caught;
  - 300 random legal packed configs show 0 spills on sm_80, sm_100 and sm_120;
  - the register estimate charges 3/4 of the ring, fitted to ptxas.
- Kit v2's matmul section races `default-mma8-reg` / `default-mma16-reg` (the mma8 / mma16 tiles with `wload=reg`),
  and the batch-1-16 tune spaces include `wload` and `pf`. The matmul report and the single kit report get a
  "Packed-direct weights vs shared-memory staging" ablation table (2-sigma verdicts).
- Covering sets now also probe keys whose legal values depend on a mode key (`pf` under `wload=reg`).

### Added: GPU kit v2 - one command, one report, like-for-like attention baselines
- `contrib/gpu-check/run_kit.sh` (QUICK=1 / full / DRYRUN=1) runs every GPU check from a copy of the kit:
  - the toolchain preflight (split installs included);
  - a llama.cpp ggml-cuda build;
  - attention fatbins for all three tiers, GPU correctness, the HBM roofline, and the decode matrix with baselines
    and ablations;
  - the GEMM/GEMV matmul kit (`run_gpu_check.sh`, all formats incl. MXFP4 / NVFP4);
  - the GPU pytest subset.

  It writes a single `report.md` (`kurn gpu kit-report`, `kurn.gpu.kitreport`) and a tarball. It writes nothing outside
  the kit folder: TMPDIR, kurn's cache, the ggml build and the CUDA / Python / Triton / FlashInfer caches all live
  there, and `containment.txt` checks it. It needs no network (ggml sources and pytest wheels are bundled), no root
  and no pip installs.
- Attention baselines on the same problems:
  - llama.cpp's CUDA flash attention (`data/bench_ggml_attn.cpp`). It builds the graph llama.cpp builds for decode:
    strided K/V cache views, F16 causal mask, F32 accumulation, MLA V as a view of K. It uses kurn's data generator,
    float64 reference, cold-KV rotation and byte count, and runs on every matrix cell
    (`kurn gpu attn matrix --llama DIR`).
  - FlashInfer / FlashAttention-2 decode (`bench_attn_torch.py`), if the given interpreter already has them.
- `--ablate 1024,16384` runs each phase-2 change set back to its run-1 value, the fused merge, and the whole run-1
  config. The report states the target before the numbers and checks it:
  - F16/BF16 >= 75% of measured HBM read bandwidth at >= 4K;
  - Q8_0 >= 1.5x kurn F16 at >= 8K;
  - Q8_0 >= 1.3x the best F16 baseline (the plan's stop-or-continue rule).
- The report lists explicitly what the box cannot test. On an A100 that is FP8 KV and sm_100 / sm_120 runtime
  (compile + emulator only).
- `kurn gpu ggml-build` builds ggml + ggml-cuda with cmake and kurn's CUDA resolution: preflight -I/-L, NVCC_*_FLAGS
  dirs, build rpath. A split install (runtime in a third-party include_no_implicit/ + lib/) therefore builds, and
  libggml-cuda finds libcublas at run time. ccache is off.
- `make_kit.sh` adds `KIT_GGML` / `KIT_WHEELS` / `KIT_NAME`. Every zip and every file stays under 1 MB: the main zip,
  ggml parts and wheels unzip into one folder. The kit now carries the GPU test subset. `run_gpu_check.sh` takes
  `KIT_OUT`, `NO_ATTN`, `NO_TAR`, `NO_CLONE` and `NO_KERNELS` (used by run_kit.sh).
- `kurn gpu attn verify` on the emulator runs configurations in parallel (`--jobs`, default all CPUs). The GPU still
  runs one at a time.
- Dry run from an unzipped copy, on a simulated split install (nvcc with no headers, runtime in a third-party
  include_no_implicit/ + lib/), QUICK, 23 min:
  - ggml-cuda built in 4 min;
  - 586 attention fatbin configs on 3 tiers, 0 failures; emulator 161 + 44 configs, 0 failures;
  - harnesses and the llama.cpp bench compiled;
  - matmul kit: 500 configs ptxas + emulator, 0 failures; ggml CPU check 3/3;
  - GPU pytest subset passed.

### Changed: GPU attention, A100 phase 2 (driven by the first A100 run; correctness on the emulator, speed to be measured)
- Q8_0 KV dequantizes in registers (`deq 1`, the new default). K and V MMA fragments are built from the raw 34-byte blocks
  with u16 / byte shared loads, `__byte_perm` into 0x64XX f16 and one f16x2 multiply by the block scale. The result is
  bit-identical to the old shared-memory pass (`deq 0`, kept for the kit's ablation), which cost a second pass over
  every tile, an extra barrier and an f16 copy of K and V. Tiles land with 8-byte cp.async when rows and strides are
  8-byte aligned (4-byte before). The Q8_0 tile drops from 76.5 to 41.7 KB. Run 1: Q8_0 2.2x slower than F16 at 16K.
- Splits: one wave of resident CTAs (1-2 per SM, by shared memory), with at least max(tk, 64) tokens per split
  (256 before), at most 128 splits (64 before). At 1K context a CTA streams 1-2 tiles instead of 4-8 (Llama-3-8B: 16
  splits instead of 4). The first KV tiles are issued before q is read, so those two DRAM round trips overlap. The merge
  kernel runs one block per (row, kv head, 128 columns), reading split weights from shared memory. The opt-in
  `merge 1` lets the last CTA of each (row tile, kv head) merge, with no second launch; its self-resetting counters
  assume one call of a library at a time per device. Run 1: ~21 us at 1K.
- MLA defaults to 4 warps on sm_80 / sm_120 (tk 32 / wn 4; was wn 2). That needs 8-token QK^T warp slices (ldmatrix x2;
  `tk / wn` must now be a multiple of 8, was 16). O takes 64 registers per thread instead of 128, at 128 threads per CTA.
  ptxas: ~164 registers, spill-free on all three archs. Run 1: ~242 registers, 64 threads, ~200 GB/s.
- BF16 q is split hi + lo (`qsplit 1`, now also for bf16; default where it fits), two QK^T MMAs per step, about 16 bits of
  q. Qwen3-1.7B at 16K: relerr 1.3e-2 -> 1.8e-3 on the emulator (run 1 on the A100: 1.4e-2). The bf16 tolerance tightens
  to 6e-3; `qsplit 0` keeps 1.5e-2.
- `splits_for()` mirrors the kernel's split choice (checked against the emulator at 108 SMs). The emulator driver takes
  `KGA_RUNS` (repeated calls on one workspace) and `KEMU_SMS`; tests add mutants for every new path.

### Fixed: GPU builds with the CUDA runtime outside nvcc's toolkit (split installs); fail-fast preflight
- On a box with nvcc in /usr/local/cuda-12.8 but cuda_runtime.h / cuda_fp16.h / libcudart in a monorepo third-party dir
  (`include_no_implicit/`, `lib/`), all 537 fatbin builds and the harness failed the same way. `kurn.gpu.cudaenv` now
  locates the runtime and checks it once before any build:
  - It honors KURN_CUDA_INCLUDE / KURN_CUDA_LIB (':'-separated lists), CUDA_HOME / CUDA_PATH, and the -I/-L in
    NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS. Those flags are passed through unchanged; kurn only adds an rpath.
  - It probes nvcc's own targets/*/{include,lib}, /usr/local/cuda and sibling cuda-* installs (same version as nvcc
    first), and `include_no_implicit`-style layouts. KURN_CUDA_NO_PROBE=1 limits it to explicit settings.
- Preflight (`kurn gpu doctor`, the first step of both kit scripts): compiles and links a tiny .cu (cuda_runtime.h,
  cuda_fp16.h, -lcudart; with `--cublas`, cublas_v2.h and -lcublas) with the exact nvcc and arch flags the builds use,
  and runs it when a GPU is present.
  - On failure it stops with one message listing what is missing, the paths searched and the variables that fix it.
  - Builds raise the same error at once, and `build_many` aborts on it instead of collecting one failure per config.
- Fatbin, cubin and harness builds and cuobjdump all use the same resolution. nvcc lookup also honors CUDA_HOME /
  CUDA_PATH, and the kit no longer overwrites KURN_NVCC.
- Warnings when the headers' CUDART_VERSION differs from nvcc's version, and when the driver is older than the runtime.
- The resolved toolchain (nvcc, header and library dirs, versions, how each was found) goes to run.log,
  toolchain.json and archs.json.

### Added: Blackwell tiers (sm_100 B200/GB200, sm_120 RTX 50), compile-only plus emulator
- GPU attention has `arch` tiers sm_80 / sm_100 / sm_120 with per-tier shared-memory limits (163 / 227 / 99 KB per block)
  and default tiles. dk=256 on sm_120 now defaults to 32-token tiles (78 KB; the 64-token tile did not fit).
  `kernel_for(kv, dk, arch)` dispatches, and `tier_for()` maps other GPUs.
- Fatbins carry SASS for sm_80, sm_100 and sm_120 (nvcc >= 12.8) plus PTX. The kit builds all three archs and reports
  which arch a run used, and whether that was native SASS or PTX JIT.
- The existing `mma.sync` GEMM/GEMV kernels compile spill-free for sm_100 (398-config covering set, 40 defaults).

### Added: MXFP4 and NVFP4 weights on the GPU (all archs, A100 included)
- GEMV: dp4a over the 2 * E2M1 int8 codebook with Q8_0 activations. MXFP4 uses the E8M0 scale per 32; NVFP4 uses
  UE4M3 per 16, two scales per 32-value unit. Split and native layouts, including 17-byte MXFP4 blocks with byte loads.
- Tensor-core engine, both exact against the double reference:
  - MXFP4: codes dequantized in registers, with the E8M0 scale stored as an exact bf16 record and applied after the MMA.
  - NVFP4: codes times UE4M3/2 multiplied into the f16 fragment before the MMA. The product is exact in f16: at most
    6 significant bits, 2^-10 .. 2688.
- `gemv_threads.json` entries for both, ptxas-measured on sm_80/90/100/120. `tools/gemv_threads.py --formats` measures
  and merges selected formats.

### Added: FP8 (e4m3) KV attention on sm_100 / sm_120
- `kv fp8` runs QK^T on FP8 `mma.sync` m16n8k32 with q per-row scaled. By default (`qsplit 1`) q is split into hi + lo
  e4m3, for about 8 bits. PV runs in f16 with V converted e4m3 -> f16 (exact).
- Emulator, end to end vs float64: 2e-3 with `qsplit 1`, up to 9.6e-2 with `qsplit 0` on the peaked test data. Every
  check also runs against a reference with q rounded exactly as the kernel does: <= 4.2e-4.
- Emulator: FP8 `mma.sync` and e4m3 conversions.

### Added: decode / verify kernels for native-layout Q6_K and Q5_K (`integration/llama.cpp/ggml-kurn/kurn-native-vfy.cpp`)
- The Q6_K / Q5_K weights the KURN buffer keeps in ggml's layout (AMX-BF16 prefill) ran ggml's `vec_dot` per row and
  column below 16 columns. The new kernels unpack each super-block once for up to 8 columns: Qwen3-8B Q6_K shapes at
  8 columns 6.4 ms vs 15.6 (`vec_dot`) and 15.5 (ggml's AMX buffer); 1 column 3.0 vs 4.5. Q5_K 6.8 vs 22.4 / 7.5.
- In exact mode the buffer now takes Q6_K / Q5_K too and runs every width on these kernels, which are batch invariant
  (ggml's AMX buffer, where they went before, is not). `test_kurn_buft native` (also with `GGML_KURN_EXACT=1`).

### Changed: exact-width, prefetching verify kernels in the KURN buffer type (`integration/llama.cpp`)
- One verify kernel per width 2..8 instead of 2 / 4 / 8 (3 columns no longer pay for 4, 5-7 for 8). Same packed layout
  and per-column arithmetic as the GEMV, so every column stays bit-identical to one-token decoding; `test_kurn_buft`
  now checks every column of every batch width against its own GEMV (`batch invariance, all columns`).
- Verify schedules measured per width (`VFY_KEYS`): whole-record software prefetch (`pfgran=line`, new for the
  i16 / i8 lowerings) and up to 4 row groups per pass. Qwen3-8B shapes, 8 threads, matmuls of 8 layers: Q8_0 2..8
  columns 25-36 ms -> 13.8-18.3 ms (stock ggml's AMX buffer 26-31 ms), Q4_K 15-25 ms -> 8.3-10.9 ms. The Q8_0 and
  Q4_K decode GEMVs take the same prefetch: 15.0 -> 13.4 ms and 9.7 -> 8.4 ms.

### Changed: the model engine's context is sized at run time
- `engine.c` no longer has a 2048-token compile-time `MAX_CTX`: the KV cache, RoPE table and score rows are sized from
  the request (gen: prompt + generated tokens, ppl: CTX, rounded up to 256); `KURN_CTX=n` asks for more and
  `-DMAX_CTX=n` is now a minimum. Same results as a fixed-capacity build (Qwen3-1.7B, ppl at CTX 4096, F16 and k4c_q4
  KV: identical). ppl mode no longer overruns the cache when CTX exceeds the old limit.

### Added: cost-aware verify width in llama-server (`integration/llama.cpp/spec-width`)
- `llama-server-spec-width.patch` (applied by `spec-width/apply.sh`): `--spec-width TABLE [--spec-width-mode cap]`
  (or `KURN_SPEC_WIDTH`) runs `kurn-spec-width.h`'s policy per slot with a draft model: per-step cap, per-token stop,
  truncation before verification, online acceptance learned across the slot's requests.
- Qwen3-1.7B target / Qwen3-0.6B draft (a pair where speculation does not pay: the draft costs ~40% of a target step),
  8 prompts x 256 tokens: no draft 55.5 tok/s, fixed k = 3 / 7 / 15 50.4 / 45.3 / 33.2, policy 54.7.

### Added: k4c as a llama.cpp KV cache type (`integration/llama.cpp/k4c`)
- `k4c/apply.sh` adds `GGML_TYPE_K4C` (per-channel 4-bit keys in 32-cell groups, read by kurn's attention):
  `-ctk k4c -ctv q4_0|q8_0` (or `-ctk k4c_q4|k4c_q8`) in llama-cli / llama-server / llama-perplexity. Keys are cached
  after RoPE (kernel `rope_dim 0`). Qwen3-1.7B, WikiText-2 ctx 2048, KL vs F16 KV: k4c_q4 0.016 (+0.06% PPL), k4c_q8
  0.010; llama.cpp's Q4_0 KV 0.32 (+29%).
- Group writes re-encode from exact (f16) values of recently written groups, so appends, rollbacks and clears give the
  same bytes as a prefill; session state (prompt cache, slot save / restore) stores f16 rows and restores the same
  codes. No K-shift. `k4c/test_k4c.c`, `tests/test_llama_k4c.py`.

### Added: kurn attention as llama.cpp's `FLASH_ATTN_EXT` (`integration/llama.cpp`)
- `apply.sh` generates kurn attention kernels (`gen_ggml_attn.py`: f32 / AMX-BF16 / AVX512-BF16 tile engines plus a
  batch-invariant `exact` configuration, F16 / BF16 / Q8_0 KV, head dims 64 / 128 / 256 / 576-512) and makes
  `ggml-kurn/kurn-attn.cpp` the first `FLASH_ATTN_EXT` implementation; ggml's kernel stays as the fallback (ALiBi,
  softcap, sinks, other types or head dims, per-head masks). GQA, llama.cpp's KQ mask, KV streams, kurn's flash-decoding
  splits and per-call K/V packing (`kattn_pack`) for the bf16 / AMX engines.
- Qwen3-1.7B Q8_0, F16 KV, 8 threads, against ggml's FA in the same build: pp512 at depth 4096 1.81x (f32) / 2.62x
  (AMX), tg64 at depth 4096 2.1x; WikiText-2 (ctx 2048) KL vs ggml's FA 0.0018, the same as ggml's non-FA path.
- `GGML_KURN_FA_MODE=exact`: a token's attention output does not depend on the batch, the padded KV length or the
  thread count (`tests/test_llama_attn.py`).

### Added: cost-aware speculative verify width (`kurn specwidth`, `integration/llama.cpp/spec-width`)
- Verify cost on the KURN buffer type is a staircase: one kernel per 2 / 4 / 8 columns (3 and 5-7 columns pay for 4
  and 8), plus a group pass per further 8. `kurn specwidth kernels` measures it per kernel; `kurn-spec-calib` measures
  the whole forward inside llama.cpp (Qwen3-8B Q8_0, 8 threads: M = 1 / 2 / 4 / 8 / 16 cost 1.00 / 1.53 / 1.86 / 2.19 /
  3.04x of 74.8 ms; M = 3 costs as much as 4, M = 5-6 more than 8, M = 13-15 more than 16).
- `WidthPolicy` (Python) / `kurn-spec-width.h` (C++ twin) pick the draft length that maximises expected accepted
  tokens minus lambda x time from that table, with acceptance learned online per draft-confidence bin.
  `KURN_SPEC_WIDTH=table llama-speculative-simple` uses it (patch: draft confidences + a keep-drafting callback in
  draft-simple). Qwen3-8B / Qwen3-0.6B draft, 8 prompts, 256 tokens: see `benchmarks/v0.2/specwidth/results/`.
- `kurn specwidth simulate` replays greedy acceptance traces (exact for greedy chain drafting) against fixed widths,
  p_min cutoffs and the policy.

### Added: GPU attention (`op attn`, `target cuda`), built for A100 (sm_80) and RTX 50 (sm_120)
- Flash attention / flash-decoding with the CPU op's semantics: F16/BF16/Q8_0 KV, GQA, MLA (v aliases k), causal plus
  optional fp16 mask, split-KV with an LSE merge kernel. `kga_args` (`kurn_gpu_attn.h`) has `kattn_args`'s layout.
- A100 design: `cp.async` double-buffered KV tiles (4-byte copies for 34-byte Q8_0 blocks), f16 or bf16 `mma.sync`
  m16n8k16 with `ldmatrix`, log2-domain online softmax with lazy rescale, auto split from the SM count.
- Every build is a fatbin: sm_80 and sm_120 SASS plus compute_80 PTX. sm_120 allows 99 KB of shared memory per block,
  so `kga_run` returns -4 when a tile doesn't fit the device.
- Verified without a GPU: CPU warp emulator against a float64 reference and against the CPU op on identical arguments,
  injected-bug tests, 0 spills on both archs for the 150-config covering set (CUDA 12.9). **Not run on a GPU yet.**
- `kurn gpu attn ...`, `examples/gpu/attn_q8_0_decode_cuda.kurn`, a GPU harness (cold KV larger than L2, CUDA graphs,
  NVML energy), and an attention stage in the gpu-check kit.

### Changed: CPU warp emulator
- bf16 `mma.sync` and 4-byte `cp.async`. Seeded schedules (`KEMU_SEED`) also hold back whole warps and land half of the
  `cp.async` copies at issue, which exposes cross-warp races and refilling a stage that other warps still read.

### Added: `tools/sm120_probe.py`
- Which PTX features ptxas accepts for sm_80 / sm_90a / sm_100a / sm_120 / sm_120a / sm_120f.

### Added: aligned activation layout for the dp4a GEMV (`xlayout split`), contributed by the user
- `xlayout blocks|split` for the q8_0-activation formats (Q8_0, Q4_0, IQ4_NL, Q2_0, Q1_0). `split` reads the activations as an
  aligned int8 plane [M][K] plus a float scale plane [M][K/32], with 16-byte vector loads instead of 16-bit loads from ggml's
  34-byte blocks. The q8_K formats (Q4_K, TQ2_0, E8P) stay on `blocks`.
- Ported from 0.3.0.dev0 onto dev2's restructured GEMV. The per-column activation pointers became dev2's 32-bit column offsets,
  and the per-column scale pointers are derived from them (scale offset = plane offset / 32), so no registers were added.
- `split` is the default for those formats: on the default kernels it cuts global loads 3-5x and hot-loop instructions
  14-40% (SASS). `xlayout` is in the kit's GEMV tune space.

### Fixed: spilling kernels in the search space
- **GEMV.** The block-size rule allowed 512 threads (a 128-register cap) for every 2-7 column kernel and 1024 threads (a
  64-register cap) for most 1-column ones, and `cols * unroll` up to 16 for Q1_0/Q2_0/Q4_0. Under those caps ptxas spills
  Q1_0 at 4 columns, and split activations raise pressure further (16-byte loads hold four registers each). Adding the
  `xlayout` key changed which random configs the covering sets sample, and that surfaced a Q1_0 4-column kernel with
  `xlayout blocks` at 512 threads (32 bytes of spill). Some of the other spilling configs the old rule allowed are older
  than `xlayout`.
- The GEMV limit is now measured, not estimated. Uncapped register counts swing by +-90 with `tpr` alone, so no estimate
  predicts where ptxas spills. `kurn/gpu/data/gemv_threads.json` holds, for every (weights, xlayout, sub, cols, unroll), the
  largest `tpr * rpb * max(1, minb)` that compiles without spill or stack on sm_80/90/100. Each entry is the worst case
  over every layout, mins, unpack, `tpr` (8-128) and `minb` (1, 2, 4); `minb` matters because ptxas schedules
  `__launch_bounds__(128, 4)` differently from `(512, 1)`. `tools/gemv_threads.py` regenerates the table. The old rules
  stay, so this only removes configs: 2,452 of 77,418 (3.2%), mostly Q1_0 (1,872), Q2_0 (354) and Q8_0 (166). Every
  default keeps at least 4x headroom.
- **GEMM.** Fresh covering seeds found engine tiles the register estimate let through. All of them had 64-row warp tiles
  with 2+ n8 tiles and 2+ k-tiles per stage, and each thread staged 12+ 16-byte `cp.async` chunks per stage: Q4_0, Q2_0,
  TQ2_0 and E8P, with 1-4 warps. This class sits at the 255-register ceiling and the estimate cannot separate its spills
  from clean tiles, so it is now an explicit rule. The rule removes 2,484 of 88,668 tiles (2.8%), and every default and
  matrix tile is unchanged.
- Validation: ptxas covering sweeps on sm_80/90/100 over 40 seeds, about 8,800 GEMV and 11,200 GEMM configs, with 0 spills;
  `kurn gpu ptxas --defaults --strict` is clean.

### Changed
- Compile identical source/target requests once per parallel search batch while benchmarking every runtime configuration separately.

### Fixed: optimizer selection, plan reuse, and sampling
- Cache repeated legality-probe validations locally without changing draw weights or the random stream.
- Aggregate energy-delay products per repetition and reject incomplete or invalid measurements during tuning.
- Preserve confirmed refinement winners, validate plan reuse, and publish compiler outputs atomically.
- Bound search parameters and avoid redundant sampling work without merging runtime settings.

## 0.3.0.dev2

### Changed (first A100 run: correctness held, speed below the competition)
- **`op gemm` is now a tensor-core engine for every batch size and all 8 formats.** It replaces the int8 GEMM, which reached at
  most 5% of tensor-core peak and 0.11-0.41x of ggml-cuda.
  - Arithmetic: f16 `mma.sync.m16n8k16` with f32 accumulation. Weights are repacked once into fragment order, and each lane's
    16 bytes dequantize in registers to small integers that are exact in f16:
    - Q4 via the 0x6400 magic number;
    - IQ4_NL via a `prmt` lookup;
    - Q8 via byte permutes;
    - 2-bit and 1-bit via shifted masks and a power-of-two FMA;
    - E8P via a shared-memory codebook with sign flips.
  - Block scales are applied in f32 after the MMA.
  - Data movement:
    - weights and scales go through a multi-stage `cp.async` pipeline;
    - activations sit in XOR-swizzled shared memory and are read with `ldmatrix.x4`;
    - f32 activations are rounded to f16 inside the kernel (`xin=f32`, one launch per matmul, no quantization kernel) or
      converted once and streamed (`xin=f16`).
  - Tiles go up to 128x128 with 64x32 warp tiles.
  - Split-K uses a deterministic serial fixup and is sized from the SM count by default.
  - Exact against the f16-rounded activations; `kg_act()` tells harnesses which activation path a kernel uses.
- **dp4a GEMV defaults:**
  - the split layout with lane loads sized to 16 bytes and 4 units in flight (2 for Q8_0/Q4_K/TQ2_0);
  - an unrolled main loop without per-unit bounds checks, so loads issue ahead of the math;
  - 32-bit column offsets. The default Q4_K multi-column kernel no longer sits at 255 registers (138-168, no spills).
- **Spilling configs are not legal.** A ptxas-calibrated register estimate and launch-bounds checks prune them; every covering
  config compiles without spills or stack on sm_80/90/100. Kernels emit `__launch_bounds__(threads, >= 1)` (ptxas otherwise
  targets 128 registers and spills).
- **Benchmarks and kit:**
  - the matrix races `default-*` and `tuned-*` KURN kernels per batch range;
  - the kit tunes every format before the matrix;
  - `report.md` shows default vs tuned side by side;
  - `kurn gpu sass` reports the SASS instruction mix (tensor-core MMA, ldmatrix, cp.async, local memory, hot loop).
- **Release:** `tools/make_release.sh` fails if `contrib/gpu-check` is missing.

## 0.3.0.dev1

Includes everything in 0.2.2 (CPU release: clang `--strict` helper pruning, optional numpy, toolchain probe, self-contained
release), merged into the CUDA development line. The CUDA modules (`kurn.gpu`) do not need numpy.

### Changed (measurement)
- **`kurn roofline` is now a peak-read probe.** Before, `bench --bw` timed one pass of a loop that compiled to mixed
  64/128/256-bit loads, with one stream per thread, and under-reported bandwidth: about 2x on a Xeon 8339HC, 25% on this VM.
  - It now uses the widest vector loads (AVX-512, AVX2, NEON; else 64-bit) into 4 independent accumulators per thread, with
    all threads pinned.
  - Each pass is barrier-timed; the result is the best pass, with the median also shown. Without `--streams`, it measures 1, 2,
    4 and 8 interleaved streams per thread and reports the best.
  - This VM, 8 threads, same session: DRAM 140–144 GB/s (old default probe 103–108); L2 1.40–1.61 TB/s (old 0.75–1.09).
- **`kurn tune` ranks on medians of interleaved rounds** (`--rounds`, default 3). It re-measures the leaders (`--keep`,
  `--budget`) until their order is stable, reports each result's spread, and warns ("ranking: NOT resolved", suggesting
  `--secs` / `--rounds`) when the spread is too large to rank. `--rounds 1 --keep 1` gives the old single-run behaviour.

### Fixed (cross targets without a cross toolchain)
- `kurn verify` no longer counts NEON configurations as failures when the AArch64 cross compiler is missing. Like the
  assembler probe, it skips the target once, says why, and names the package to install
  (`apt install gcc-aarch64-linux-gnu`, or set `KURN_CROSS_CC`).
- If the cross compiler is present but qemu is missing, configurations are still compiled (so `--strict` warnings are caught)
  and are reported as "compiled, not run" with the qemu hint (`apt install qemu-user`, or set `KURN_QEMU`). They are counted
  separately from passes.
- `run_mode` reasons now name the missing piece and how to install it.

### Fixed (CUDA emulator portability, found by the 0.2.2 clang runs)
- The CPU emulator now probes the host C++ compiler once (`kurn.gpu.toolchain.cxx_problem`). If it can't build a C++17
  program, it reports why (typically clang++ on Ubuntu selecting a GCC installation without libstdc++ headers) and the
  emulator checks are skipped rather than failing. `kurn gpu targets` shows the emulator status.
- Emulator builds with clang++ no longer fail `-Werror` on `#pragma unroll` loops clang can't unroll at -O1
  (`-Wno-pass-failed`, clang only).

### Added: CUDA backend (`target cuda`, `kurn.gpu`), stages 0-2 of the GPU plan; no GPU measurements yet
- **Code generation:** CUDA C++ for the following, every file implementing `kurn_gpu.h`:
  - decode GEMV (dp4a) on Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0, Q1_0 and E8P, in `native` and `split` layouts, with
    multi-column small-batch variants;
  - an int8 tensor-core GEMM (`mma.sync.m16n8k32`, sync / register double-buffer / 2- and 3-stage `cp.async` pipelines) for
    Q8_0, Q4_0 and IQ4_NL;
  - activation quantizers matching ggml's reference rounding;
  - device repack kernels.
- **CPU warp emulator** (`kurn_cuemu.h`): fibers with real barrier, warp-shuffle and `mma.sync` fragment semantics, deferred
  `cp.async`, alignment and guard-page checks, randomized schedules. Every generated kernel runs on it against an exact C
  reference (`kurn_gpu_ref.h`, tied to `kurn.formats`).
- **nvcc/ptxas checks** for sm_80 / sm_90 / sm_100: registers, shared memory, spills and static occupancy. SASS is checked for
  IMMA and LDGSTS.
- **GPU harness** (`bench_gpu.cu`): measured HBM and tensor-core roofline, CUDA-graph timing with cold weights, NVML board
  energy, cuBLAS FP16/INT8 baselines, and llama.cpp ggml-cuda MUL_MAT as an in-process competitor on identical bytes.
- **Benchmark matrix:** formats × batch 1/4/16/64/256 on Llama-3-8B layer shapes, with interleaved rounds.
- **Report:** a 2-sigma win rule, wins/ties/losses, and `dispatch.json`. `kernel_for()` falls back to the competitor's kernel
  wherever KURN does not win.
- **Tuning:** energy-ranked GPU tuning (NVML joules).
- **Kit:** the hand-run kit `contrib/gpu-check/` (`run_gpu_check.sh`, `make_kit.sh`, optional Marlin script).
- **Integration:** `kurn.hooks.TARGET_BACKENDS` routes `kurn check|gen|build|verify|tune` to a backend by the spec's target.
- **CLI:** `kurn gpu ...` commands.

## [0.2.2] - 2026-10-03

Portability release, from a run of 0.2.1 on a second Linux machine. No kernel performance changes.

### Fixed
- `kurn verify --all --strict` with clang: the generators emitted every `static inline` helper (`f16f`, `bits8x4`, `ld64`,
  `ld16`, ...) whether or not a kernel used it. Clang's `-Wunused-function` (GCC stays quiet) turned that into errors on
  most AVX-VNNI and many AVX-512 configs. `generate()` now drops the helpers a kernel doesn't call.
- Q4_K AVX-512 native-layout GEMV (the `act`/`accum`/`scales`/`correction` algorithm space and the v2 kernel) summed the
  min-correction vector over the undefined upper half of `_mm512_castps256_ps512`. GCC happened to zero it. Clang didn't
  at `rows=4`, which gave relative errors of 0.08-0.1. The kernel now zero-extends explicitly.
- numpy was a hidden dependency: five modules (`codebook`, `entropy`, `latent`, `lowrank`, `mixed`) imported it at load time,
  so 9 test files failed to collect without it. It is now the optional extra `kurn[compress]`, with `kurn[gguf]` for GGUF
  input. The modules import numpy lazily, and their tests use `pytest.importorskip`.
- A toolchain that can't build a target (binutils < 2.36 can't assemble AVX-VNNI) failed every config of that target. kurn
  now probes each target once. It reports "toolchain can't assemble target X; set KURN_CC" and skips the target in
  `kurn verify`, `kurn targets` and the tests. The harness falls back from `-march=native` to portable flags.
- Two more clang-only build errors, found by running the full suite with `KURN_CC=clang`. First, the attention kernel's
  shared helpers are now marked unused, because each engine uses only some of them (`kurn attn verify --strict`). Second,
  the fused engine epilogue now passes `_mm512_roundscale_ps` a literal rounding mode, which clang requires.
- Parallel builds of identical kernel sources could compile a half-written file. Sources are now written atomically.
- Tests that import benchmark harnesses (`benchmarks/v0.2/e2e`) or the llama.cpp integration skip cleanly when those
  directories are absent, instead of failing to load.

### Added
- `tools/offline-check/`: the one-command host check (verify, roofline, tune sweeps, AMX test), shipped in the package. It runs
  on the enclosing tree and falls back to the source tree when offline.
- `tools/make_release.sh`: builds the release zip from the committed tree. It checks that the archive holds every tracked
  file, that the docs-path test passes, and that pytest collects with no errors inside the unpacked copy.
- Tests: clang `-Werror` over every golden file and every legal config (`test_codegen_golden.py`), every path README.md and
  CONTRIBUTING.md reference (`test_docs_paths.py`), and the toolchain probe (`test_toolchain_probe.py`).
- CI: `kurn verify --all --strict` with clang, and a test job without numpy.

### Changed
- The code base is `ruff format`-clean, as CONTRIBUTING.md requires.

## [0.2.1] - 2026-10-03

### Fixed (weak 4-bit / Q8_0 end to end in the llama.cpp buffer type)
- Q4_0 decode through the KURN buffer type was 0.83-0.92x stock llama.cpp while its kernel tied ggml. Causes, measured
  per op: 4 dynamic row chunks per thread (15-30% of op bandwidth), two weight streams per core (rows=2), and the older
  `mask` unpack. Now `pair` with 4 row groups, guided partitioning in whole passes, shared activation prep and a
  one-record GEMV for remainders. Qwen3-1.7B Q4_0 decode 75.8 vs 71.1 tok/s stock (AMX buffer), prefill 1.53x.
- Q8_0 decode (0.97x stock): `rows=8` on the i16 records (v0.1 vnni16's pass width) plus the partitioning fix.
- `gen_ggml_sources.py --config` overrides of a tuned key no longer raise a duplicate-keyword error.
- Q4_K_M end to end is now a resolved win: decode 1.32x stock (AMX buffer), 1.39x stock built without AMX; prefill 1.28x.
- 8-core DRAM roofline on a Qwen3-8B-shaped graph: Q4_0 79%, Q4_K 73%, Q8_0 91% of 128.7 GB/s (the 4-bit >= 90% goal is
  still not met).

### Added
- `pair` verify kernels up to 8 columns (one accumulator chain beyond 4 row-group x column pairs).
- `ilv` schedule key (record interleave across row groups, `kurn.ext.q4fix`); measured slower than separate streams and
  not used by default.
- Generated `_xprep` / `_xprep_bytes` / `_packed_x` entry points (`xprep` codegen flag) for every generic lowering.
- Per-tensor fallback in the buffer type (`GGML_KURN_FALLBACK*`), `GGML_KURN_PROFILE` phase timing,
  `integration/llama.cpp/tools/opbench.c` (model-shaped MUL_MAT graphs per buffer type) and
  `benchmarks/v0.2/q4fix/cmp.py` (interleaved A/B rounds with 2-sigma verdicts).

## [0.2.0] - 2026-10-03

Built as eight parallel workstreams on one integration branch. Every kernel is checked against an exact reference; lossy
formats carry a stated tolerance plus perplexity/KLD checks. Measured results, including what did not pay off, are in the
project's v0.2 results document; per-workstream scripts and CSVs are under `benchmarks/v0.2/`.

### Added
- **Extension mechanism** (`kurn.hooks`, `kurn.ext`): modules register formats, recipes, kernels, layouts, schedule keys,
  ops, CLI commands, test generators and golden cases without editing `spec.py` or `kernels.py`. 1,590 x86 + 8 NEON legal
  configurations (v0.1: 102 + 8).
- **Recipe lowering ("what" vs "how")** (`generic.py`): one recipe per format, lowered to interleaved lane-per-row layouts
  (`i16` AVX-512, `i8` AVX2-VNNI), with algorithm keys `unpack` (mask, LUT, `mask16`, `pair`, `perm`), `correction`
  (activation-sum seed, weight sums, Q4_K `dpmin` min precompute), `scales` and `accum`.
- **CuTe-style layout algebra** (`layout.py`) and a layout-driven lowering (`sched.py`, `layout=composed`): shape/stride
  composition, complement, divide/product tiling, swizzles. Repack code and kernel addressing are generated from the same
  layout expression; vnni16/i16/i8 are reproduced bit for bit. Tiling (`kpanel`, `rpanel`) and pipelining (`stages`,
  `pfhint`, `pfgran`) keys.
- **Staged schedule search** (`kurn search`): search spaces derived from the registry; rediscovers the hand-designed Q4_K v2
  algorithm with no hints.
- **Plan cache** (`kurn plan build/show/lookup`): tuned winners per machine fingerprint x GGUF tensor shape, reused at load.
- **Formats:** Q4_0, IQ4_NL, MXFP4, NVFP4 (with quantizers), Q2_K, Q2_0, TQ1_0, TQ2_0, Q1_0 (Bonsai), E8P lattice codebook.
- **Low-bit kernels:** exact int16 lookup-table GEMVs (T-MAC style: group size, bit-serial planes, mirror consolidation,
  base-3 TQ1_0), add/subtract-only ternary kernels, Q2_K `k16`.
- **Multi-token verify op** (`op verify`, 2–8 activation columns, same per-column arithmetic as the GEMV).
- **Attention op** (`kurn attn`): tiled online-softmax attention (f32 FMA, AVX-512 BF16, AMX-BF16 with a guard against VMs
  that lose AMX tile state), split-KV decode with log-sum-exp merge, causal tile skipping, F16/BF16/Q8_0 KV, MLA latent KV.
- **Compressed weights:** `kurn mix` (sensitivity-predictor-guided per-tensor mixed precision with stock ggml types), E8P
  codebook with LUT GEMV, Huffman entropy-coded GEMV, LoRC low-rank residual kernels.
- **Runtime** (`src/kurn/runtime/`): persistent pinned pool; static, balanced, split-K and work-stealing partitioners; spin,
  spin-then-futex and yield waits; cache-cliff and expert-prefetch benchmarks.
- **Whole-model engine v2** (`kurn model`): one compiled decode step per fixed model (Qwen3 dense, OLMoE MoE; Q8_0 or
  Q4_0), schedule fixed at load and replayed per token, fused GEMV epilogues, per-sync-kind wait policies.
- **llama.cpp integration:** a KURN extra buffer type (`integration/llama.cpp/apply.sh`) that repacks once at load (no
  second weight copy), covers `MUL_MAT` and `MUL_MAT_ID`, is batch-invariant through the verify kernels, and leaves AMX
  prefill opt-in (`GGML_KURN_AMX=1`). An end-to-end harness (`benchmarks/v0.2/e2e/`) for decode/prefill tok/s, J/token
  proxy, perplexity and RSS.
- `benchmarks/v0.2/benchlock.sh`: exclusive lock for timed runs on shared machines.

### Changed
- `legal_configs` enumerates extension keys; open-ended composed layouts register covering sets instead.
- The v0.1 llama.cpp hook patch is kept for reference; the buffer type replaces it.

### Known limitations
- Measured on one 8-vCPU Emerald Rapids VM (Linux). ARM kernels only run under qemu; macOS, Windows and AMD are untested.
- AMX is not reliable on that VM (tile data is not preserved across context switches), so AMX paths are opt-in.
- 4-bit decode is compute-bound when cache-resident (two `vpdpbusd` per 64-byte load); see the results document for DRAM.

## [0.1.0] - 2026-10-02

First public release, extracted from an internal research prototype (`kdsl`).

### Added
- Registries for formats (`formats.py`), targets (`targets.py`) and kernels (`kernels.py`), with a table-driven harness, so new
  formats and kernels (v0.2: 4-bit repacked layouts, 2-bit/ternary LUT kernels) drop in without touching the CLI, validation or
  tests; `tests/test_registry.py` checks that Python and C sides agree.
- Spec language (`.kurn`): flat `key value` files with closed, target-dependent value sets; `tune` lines declaring search spaces;
  errors that list the legal values; detection of tune values that are illegal in every combination.
- Formats `q8_0`, `q4_K` (weights) and `q8_0`, `q8_K` (activations), byte-compatible with ggml, with a pure-Python reference.
- Lowering rules: Q8_0 GEMV (scalar, AVX2, AVX-VNNI, AVX-512 VNNI incl. the `vnni16` interleaved layout, NEON), Q4_K GEMV
  (scalar, AVX2, AVX-VNNI, AVX-512 VNNI), Q8_0 GEMM (scalar, AVX-512 VNNI register tiles, AMX tiles). 110 legal configurations.
- `kurn` CLI: `check`, `gen` (with `--embed PREFIX` for pasting into ggml), `build`, `verify` (`--space`, `--all`, `--strict`),
  `tune` (energy proxy / speed / EDP objectives, static platform power, Pareto front, CSV), `roofline`, `targets`.
- Standalone benchmark harness (no ggml dependency): random valid ggml blocks, exact double-precision reference, NaN-filled outputs,
  pinned thread pool with spin or futex barriers, hot/cold regimes, serial-gap modelling, bandwidth roofline. Builds on x86-64 and
  AArch64.
- NEON kernels cross-compiled and run under `qemu-user` for correctness on x86 hosts.
- Content-addressed build cache keyed by generated source, flags and compiler.
- Optional ggml-linked harness (`contrib/ggml-harness`) and a llama.cpp integration patch (`integration/llama.cpp`).
- Tests: spec validation, golden codegen, every legal config vs the Python reference (random and extreme data), harness failure
  detection, tuner, CLI. CI: lint, tests on x86-64 (Python 3.9 and 3.12), NEON cross-compile + qemu.

### Changed (relative to the prototype)
- `threads` is validated as 1..256 instead of 1..host CPU count, so specs validate the same on every machine; the tuner skips
  points with more threads than CPUs.
- The vnni16 kernel's inner broadcast variable is renamed (`xv` → `xk`) so generated code is clean under `-Wshadow`, as llama.cpp
  builds it. No change in behaviour.
- NEON is compiled with `-mcpu=neoverse-n1` (GCC and Clang spelling) and the cross compiler is auto-detected or set with
  `KURN_CROSS_CC`.
