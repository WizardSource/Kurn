# AMX preemption guard: base 4096 instead of 8192 cycles

The guard redoes an AMX step whose `rdtsc` span exceeds base + 128 cycles per k-step (matmuls), or base (attention,
AMX verify kernel). Lowering the base from 8192 to 4096 closed the residual wrong results of
`tools/amx_selftest.c`: one process with 8 threads on the 8 vCPUs gave 5 wrong in 2.47B computations at 8192, and 0
in 2.13B at 4096. Some preemptions came back in about 12K cycles, just under the old 12,288-cycle threshold of a
32-step step.

## Cost (Emerald Rapids 8-vCPU KVM guest, Qwen3-8B Q8_0, 8 threads)

Paired runs alternate order. The value is the median cost ratio new/old with its 95% bootstrap CI, where < 1 means the
new version is faster.

| Benchmark | Pairs | Cost ratio (95% CI) |
|---|---|---|
| llama-bench pp512 | 12 | 0.986 (0.931-1.013) |
| llama-bench pp2048 | 10 | 0.999 (0.973-1.027) |
| llama-bench pp512, `GGML_KURN_EXACT=1` (AMX-INT8) | 10 | 1.002 (0.991-1.027) |
| opbench q8_0, every matmul of 4 layers, M = 512 (AMX-BF16) | 16 | 0.994 (0.974-1.007) |
| opbench q8_0, M = 512, exact (AMX-INT8) | 16 | 1.003 (0.995-1.009) |
| opbench q4_K, M = 512 (AMX-BF16) | 16 | 1.003 (0.993-1.022) |

## Redo rate and margin

The test was `test_kurn_buft stress`, 4096 x 4096 x 512, 8 threads, 200 repeats (about 1.65M guarded AMX-BF16 steps),
with `GGML_KURN_AMX_SELFTEST=0` so that the variable is honoured. It shows steps recomputed and wall time per
threshold base:

| Base | q8_0 AMX-BF16 | q4_K AMX-BF16 | q8_0 exact AMX-INT8 |
|---|---|---|---|
| 2048 | 253,756 redone, 1.61 s | 124,025, 2.11 s | 16,311, 1.53 s |
| 3072 | 60,821, 1.28 s | 109,635, 2.39 s | 15,126, 1.61 s |
| 4096 | 34,442 (~2%), 1.32 s | 28,494, 2.12 s | 14,328, 1.55 s |
| 6144 | 10,557, 1.26 s | 10,431, 2.09 s | 12,718, 1.57 s |
| 8192 | 4,173 (~0.25%), 1.29 s | 2,064, 2.11 s | 1,767, 1.58 s |

Normal steps take about half of the new 8,192-cycle threshold of a 32-step step. On a machine with slower memory,
`GGML_KURN_VERBOSE=1` shows the count at exit; raise `GGML_KURN_AMX_GUARD` toward 8192 if it is high. After a failed
self-test, values up to 8192 are honoured.

None resolves a change. The whole-model runs compare two builds, because attention and the verify kernel use a
compile-time threshold. The opbench runs use one binary with `GGML_KURN_AMX_GUARD` set to 8192 or 4096.

- `paired_guard.sh`: whole model, comparing two llama.cpp build directories (`MODEL=`, `OLD=`, `NEW=`). Results in
  `results/whole-model.jsonl`.
- `paired_op.sh`: opbench (`OPBENCH=`). Results in `results/opbench.jsonl`.
