# Expert streaming (KXS) benchmarks

Harness and raw results for [the results doc](../../../integration/llama.cpp/expert-streaming/README.md)
(Project write-up: `docs/kurn-expert-streaming.md`). One fresh `llama-kurn-xs-bench` process per run,
inside its own cgroup v2 (`memory.max` = the RAM budget, optional `io.max` read cap), after
`echo 3 > /proc/sys/vm/drop_caches`. Needs sudo for the cgroup and the cache drop.

- `xsrun.py`: one run (modes `ram` = all-in-RAM kurn, `mmap` = stock ggml with mmap and no repack, `xs` = streaming);
  adds cgroup `memory.peak`, disk bytes (`io.stat`) and OOM kills to the bench's JSON.
- `xsexp.py`: interleaved paired experiments (every repetition runs every configuration on the same prompt, in rotated
  order) and summaries: mean ± 95% t-interval, paired ratios against a baseline.
- `main_decode.py` (decode matrix: budgets 2/4/8 GB x stock mmap / cache only / + overlap / + prefetch, plus RAM
  references), `predacc.py` (offline next-layer prediction recall), `tune_io.py`, `tune_pf.py`, `ablate.py`
  (LFU/LRU, from-GGUF repack), `prefill.py` / `prefill2.py` (ubatch sweep on a ~1.8k-token prompt), `gates.py` /
  `gates2.py` (exact-mode bitwise gates), `batch3.py` (fast-mode gate, LFU prefill counting), `batch4.py` (2 GB/s
  read cap), `mmapbase.py` (load + prefill bytes for per-token mmap bytes), `amxtest.py` / `dettest*.py`
  (the AMX nondeterminism investigation), `analyze.py` (tables), `rebase_check.py` (merged-tree check).
- `prompts/`: p1-p5 (short chat prompts) and `long.txt` (~1.8k tokens).
- `results/*.jsonl.gz`: every run (one JSON object per line).

Models: Gemma 4 26B-A4B IT (Google QAT Q4_0 GGUF) and Qwen3-30B-A3B Q4_0 (unsloth; its six Q4_1 `ffn_down_exps`
requantized to Q4_0 with `llama-quantize --allow-requantize --output-tensor-type q6_k --token-embedding-type q4_0`).
