# Expert streaming (KXS): big MoE models in a small RAM budget

Routed MoE experts stay on SSD in kurn's repacked layout and are read on demand into a slab cache;
everything else (attention, router, shared / dense FFN, embeddings, head) loads as usual and stays
in RAM. Output is bit-identical to the all-in-RAM kurn run: the expert matmuls run the same kurn
kernels on the same bytes, only from a slab instead of the KURN buffer. Linux x86 with AVX-512 VNNI
(the KURN buffer's requirement); O_DIRECT reads.

## Build

After the six base scripts (`apply.sh`, `spec-width/apply.sh`, `k4c/apply.sh`, `kvz/apply.sh`, `rsi/apply.sh`,
`memsys/apply.sh`; `apply.sh` copies `ggml-kurn/kurn-xs.{cpp,h}` with the rest of the KURN buffer), or on a
checkout with `llama.cpp-kurn.patch` applied:

```sh
Kurn-gpu/integration/llama.cpp/expert-streaming/apply.sh llama-kurn
cmake --build llama-kurn/build -j
```

This adds the proc-address entry points in `ggml-cpu.cpp`, the loader / graph hooks in `src/`,
`--expert-stream` / `--expert-cache-mb` in `common/arg.cpp`, and two tools in `tools/kurn-xs/`.

## Use

```sh
B=llama-kurn/build/bin
$B/llama-kurn-xs-convert model.gguf model.kxs            # once per model; ~10 s for Gemma 4 26B-A4B
$B/llama-kurn-xs-convert model.gguf model.kxs --verify   # optional: re-read and compare every record
$B/llama-cli -m model.gguf --expert-stream model.kxs --expert-cache-mb 2000 ...
$B/llama-server -m model.gguf --expert-stream model.kxs --expert-cache-mb 2000 ...
```

Equivalent environment: `LLAMA_EXPERT_STREAM=model.kxs LLAMA_EXPERT_CACHE_MB=2000`.

- `--expert-cache-mb` is the slab cache only. Total RAM is about the resident core (Gemma 4 26B-A4B
  Q4_0: 1.49 GB of weights; Qwen3-30B-A3B Q4_0: 0.6 GB) + KV + compute buffers + this. Pick it as
  RAM target minus the footprint `llama-kurn-xs-bench` reports at `--expert-cache-mb 1`, minus
  ~250 MB for page cache of the small mmapped tensors.
- Converted files carry the kurn kernel config of their layout; a build whose kernels changed
  refuses the file (re-run the converter).
- One streamed model per process (a draft model with its own `*_exps` tensors would collide by name).
- On VMs whose AMX state is not trusted, use `GGML_KURN_FALLBACK=0` (or exact mode): weights the per-tensor fallback
  leaves to ggml's AMX buffer (e.g. Qwen3's [2048 x 512] K/V projections) have no preemption guard there, which made
  runs nondeterministic with or without streaming.
- Prefill: a larger ubatch reads each expert fewer times (1.8k-token prompt, Gemma 4 at 4 GB: TTFT 29 s at
  ubatch 128, 13.6 s at 512, 8.8 s at 2048) but needs more RAM for compute buffers; take it out of the cache.

## Knobs (environment)

| Variable | Default | Effect |
|---|---|---|
| `GGML_KXS_IO_THREADS` | 4 | parallel O_DIRECT reads |
| `GGML_KXS_IO_CHUNK_KB` | 1024 | bytes per read request |
| `GGML_KXS_OVERLAP` | 1 | 0: read every expert of an op before computing any (no overlap) |
| `GGML_KXS_PREFETCH` | 0 | 1: speculative reads of the next layer's predicted experts. Pays only with spare SSD bandwidth (+6.5% on Gemma 4 at 4 GB on ~4 GB/s storage; -13% under a 2 GB/s cap) |
| `GGML_KXS_PREFETCH_K` | 4 | predicted experts per layer (top-4 measured best; top-8 loses) |
| `GGML_KXS_SPEC_INFLIGHT` | 4 | speculative reads in flight (1 loses) |
| `GGML_KXS_SPEC_MAX_FREQ` | unlimited | speculative reads only evict slabs used at most this often |
| `GGML_KXS_POLICY` | lfu | `lru` for plain LRU |
| `GGML_KXS_PRED_MODE` | ratio | `raw`: no per-channel router-scale correction |
| `GGML_KXS_MEASURE` | 0 | 1: record next-layer prediction recall (decode), no reads |
| `GGML_KXS_FROM_GGUF` | 0 | ablation: read GGUF bytes and repack after each read |
| `GGML_KXS_FREQ_MAX_TOKENS` | 16 | ops with more tokens (prefill) do not add to LFU counts |
| `GGML_KXS_MIN_SLOTS` | 1 | minimum slots per tensor |
| `GGML_KXS_DIRECT` | 1 | 0: buffered reads |

## Design

- **File** (`.kxs`, `llama-kurn-xs-convert`): a header with one record per expert weight (name,
  kurn config, shape, bytes), then every expert's repacked bytes, 4 KiB aligned. kurn's records end
  in all-zero padding row groups; only bytes up to the last nonzero 4 KiB block are stored (for Q4_0
  that is exactly the GGUF's size) and the slab keeps the zero tail from allocation.
- **Cache** (`kurn-xs.cpp`): per streamed tensor a pool of slabs (all tensors of a layer get the same
  slot count; `--expert-cache-mb` is split evenly over the MoE layers). LFU with a recency
  tie-break; slabs in use by the running op are pinned.
- **Reads**: an I/O thread pool, two FIFO queues. Demand reads always go first; speculative reads
  only when no demand read is queued and fewer than `GGML_KXS_SPEC_INFLIGHT` are in flight; queued
  speculative reads are promoted when demanded and dropped when their layer is planned.
- **Plan node**: `build_moe_ffn` adds a `GGML_OP_CUSTOM` node right after the top-k
  (`ffn_moe_xs_plan-L`) and expands it at once, which also orders it ahead of a shared expert the
  model builds afterwards (Gemma 4). It pins cached experts and queues the misses' reads for all
  of the layer's streamed tensors, then returns.
- **MUL_MAT_ID** (`kurn-buft.cpp`): for a streamed weight, cached experts are computed first; each
  miss is computed when its slab is ready (`kxs::wait`), and a slab is unpinned (and recycled for a
  pending miss when the op needs more experts than its pool holds, e.g. prefill) after its last
  row chunk. The row chunks are handed out in that order through the existing chunk counter.
- **Streamed MUL_MAT_ID items** are handed out in compute order on ggml's chunk counter (not the work-stealing
  schedule or the NUMA split): a pending miss only gets a slab when an earlier expert finishes.
- **Prediction**: the plan node of layer L applies layer L+1's router to layer L's router input,
  with L+1's per-channel router scale in place of L's (Pre-gated MoE / ProMoE), and queues
  speculative reads of the top `GGML_KXS_PREFETCH_K` experts not already cached.

## Tests

- `llama-kurn-xs-convert --verify`: every record against a fresh repack.
- Bitwise gates: `llama-kurn-xs-bench` hashes every logit row of a run; streamed runs must match the
  all-in-RAM run (and, in exact mode, across ubatch sizes). See `benchmarks/v0.3/expert-streaming/`.
