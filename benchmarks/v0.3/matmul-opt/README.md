# Matmul optimizations (CPU): tools, measurements, raw results

Results: `docs/kurn-matmul-opt.md` in the project store. Host: 8-vCPU Emerald Rapids KVM guest, Qwen3-8B shapes,
8 threads. Run-to-run noise on this host is ±15-25 % for whole-model runs, so every speed claim is a paired A/B with
alternating order and kurn's paired-CI gate (`kurn.tune.paired_gate`, bootstrap CI of the median ratio).

| file | what |
|---|---|
| `paired.py` | paired A/B driver: `paired.py --pairs N --metric ms|tps --regex R -- 'cmd A' -- 'cmd B'`; prints JSON with the median cost ratio B/A, its 95 % CI and the gate (1 = B wins, 0 = cannot win, null = unresolved) |
| `../../../integration/llama.cpp/tools/blockbench.c` | Qwen3-8B-shaped transformer-block matmul graph (rms norm, q/k/v, o, gate/up/SwiGLU, down, adds) on KURN weights; quieter than whole-model runs |
| `../../../integration/llama.cpp/tools/logits_dump.cpp` | prefill + decode steps + one 5-token batch, every logit row dumped raw: `cmp` of two dumps is the bit-exactness check |
| `amxblk.c` | AMX-BF16 2x2-tile microbenchmark: row-major vs tile-packed activations, panel shapes, guard sub-steps |
| `bw.c` | 8-thread streaming read bandwidth (the decode roofline) |
| `phase-instrument.diff` | rdtsc phase timers for the AMX-BF16 path (`KURN_PH=1`): conversion, pack, tiles, stores |
| `results/` | paired-CI JSON per item, profiles |

`GGML_OPPROF=2` (tools/opprof.patch) profiles every graph, decode included; `GGML_OPPROF=1` prefill graphs only.
