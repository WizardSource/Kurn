#!/usr/bin/env python3
"""Optional attention decode baselines: FlashInfer (single_decode_with_kv_cache) and FlashAttention-2
(flash_attn_with_kvcache), on the kurn decode-matrix shapes with F16 / BF16 KV.

Runs only with an interpreter that already has torch plus flashinfer and/or flash_attn (nothing is installed; no
network). Same problem shapes, the same data distribution (q ~ N(0, 9), k and v ~ N(0, 1)), the same cold-KV rotation
(the cache replicated into layers > --cold-bytes, CUDA-graph replay of one call per layer) and the same byte count as
the kurn harness, plus a float64 reference on the GPU for relerr. MLA and Q8_0 KV are not covered (no comparable
single-request decode API); FP8 is untestable on A100. Appends JSON lines to --out:
{"impl": "flashinfer" | "flash-attn-2", "model", "kv", "nkv", "nq", "relerr", "status", "us", "GBps", ...} or a
{"status": "not installed"} line per missing library.
"""

import argparse
import json
import math
import sys

MODELS = {"llama3-8b": (32, 8, 128), "qwen3-1.7b": (16, 8, 128)}
TOL = {"f16": 4e-3, "bf16": 1.5e-2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="1024,4096,16384,32768")
    ap.add_argument("--secs", type=float, default=0.4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cold-bytes", type=float, default=3e8)
    a = ap.parse_args()
    out = open(a.out, "a")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row))

    try:
        import torch
    except ImportError as e:
        for impl in ("flashinfer", "flash-attn-2"):
            emit({"impl": impl, "status": "not installed", "error": f"torch: {e}"})
        return 0
    if not torch.cuda.is_available():
        for impl in ("flashinfer", "flash-attn-2"):
            emit({"impl": impl, "status": "not installed", "error": "torch has no CUDA device"})
        return 0
    impls = {}
    try:
        import flashinfer

        impls["flashinfer"] = lambda q, k, v, sc: flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD", sm_scale=sc)
        ver_fi = getattr(flashinfer, "__version__", "?")
    except Exception as e:  # noqa: BLE001 - any import failure means "not available"
        emit({"impl": "flashinfer", "status": "not installed", "error": str(e).splitlines()[0][:200]})
    try:
        import flash_attn
        from flash_attn import flash_attn_with_kvcache

        def fa2(q, k, v, sc):
            return flash_attn_with_kvcache(q.view(1, 1, *q.shape), k.unsqueeze(0), v.unsqueeze(0), softmax_scale=sc).view(q.shape)

        impls["flash-attn-2"] = fa2
        ver_fa = getattr(flash_attn, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        emit({"impl": "flash-attn-2", "status": "not installed", "error": str(e).splitlines()[0][:200]})
    dev = torch.cuda.get_device_name(0)
    for impl, fn in impls.items():
        ver = ver_fi if impl == "flashinfer" else ver_fa
        for (model, (nh, nkvh, d)), kv, ctx in (
            (m, k, c) for m in MODELS.items() for k in ("f16", "bf16") for c in map(int, a.contexts.split(","))
        ):
            row = {"impl": impl, "version": ver, "model": model, "kv": kv, "nkv": ctx, "nq": 1, "device": dev}
            try:
                dt = torch.float16 if kv == "f16" else torch.bfloat16
                g = torch.Generator(device="cuda").manual_seed(1)
                q = (3 * torch.randn(nh, d, device="cuda", generator=g)).to(dt)
                kv_bytes = 2 * ctx * nkvh * d * 2
                nl = int(min(64, max(2, math.ceil(a.cold_bytes / kv_bytes)))) if a.cold_bytes > 0 else 1
                ks = [torch.randn(ctx, nkvh, d, device="cuda", generator=g).to(dt) for _ in range(nl)]
                vs = [torch.randn(ctx, nkvh, d, device="cuda", generator=g).to(dt) for _ in range(nl)]
                sc = 1.0 / math.sqrt(d)
                o = fn(q, ks[0], vs[0], sc)
                rep = nh // nkvh
                kd, vd = ks[0].double().repeat_interleave(rep, 1), vs[0].double().repeat_interleave(rep, 1)
                s = torch.einsum("hd,nhd->hn", q.double(), kd) * sc
                ref = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vd)
                err = ((o.double() - ref).abs().max() / ref.abs().max()).item()
                outs = [None] * nl
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    for i in range(nl):  # warm-up outside capture (JIT, workspace)
                        outs[i] = fn(q, ks[i], vs[i], sc)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph, stream=st):
                        for i in range(nl):
                            outs[i] = fn(q, ks[i], vs[i], sc)
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                us = []
                import time

                for _ in range(a.reps):
                    n = 0
                    t0 = time.time()
                    e0.record()
                    while time.time() - t0 < a.secs:
                        for _ in range(8):
                            graph.replay()
                        n += 8
                        torch.cuda.synchronize()
                    e1.record()
                    e1.synchronize()
                    us.append(e0.elapsed_time(e1) * 1e3 / (n * nl))
                m = sum(us) / len(us)
                byt = ctx * nkvh * d * 2 * 2 + 4.0 * (2 * nh * d)
                row.update(
                    relerr=err,
                    status="ok" if err <= TOL[kv] else "FAIL",
                    us=m,
                    us_sd=(sum((u - m) ** 2 for u in us) / max(1, len(us) - 1)) ** 0.5,
                    GBps=byt / (min(us) * 1e3),
                    layers=nl,
                )
                del ks, vs, outs, graph
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - one failing cell must not stop the others
                row.update(status="error", error=str(e).splitlines()[0][:200] if str(e) else type(e).__name__)
            emit(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
