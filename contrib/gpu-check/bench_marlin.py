"""Optional competitor for the KURN GPU kit: Marlin (FP16 x INT4, group size 128).

Runs only if PyTorch with CUDA and one of these is importable:
  - vLLM (its Marlin kernels: vllm._custom_ops.gptq_marlin_gemm), or
  - the standalone `marlin` package (github.com/IST-DASLab/marlin).
Writes JSON lines in the harness schema (impl "marlin-int4-g128", fmt "int4-g128"); the KURN report
compares them with the 4-bit formats as a cross-format competitor. If nothing usable is installed,
writes one {"kind": "skip"} row with the reason. Never installs anything.

    python3 bench_marlin.py --out marlin.jsonl [--quick]
"""

import argparse
import ctypes
import json
import sys
import time

SHAPES = ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))
BATCHES = (1, 4, 16, 64, 256)
IMPL = "marlin-int4-g128"


class Nvml:
    def __init__(self, bus_id=None):
        self.ok, self.dev = False, ctypes.c_void_p()
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                return
            if bus_id:
                r = self.lib.nvmlDeviceGetHandleByPciBusId_v2(bus_id.encode(), ctypes.byref(self.dev))
            else:
                r = self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.dev))
            self.ok = r == 0
        except OSError:
            pass

    def joules(self):
        if not self.ok:
            return float("nan")
        e = ctypes.c_ulonglong()
        return e.value * 1e-3 if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.dev, ctypes.byref(e)) == 0 else float("nan")


def emit(out, row):
    out.write(json.dumps(row) + "\n")
    out.flush()


def make_backend(torch):
    """-> (name, prepare(w_fp16 [N,K]) -> (state, w_ref [N,K]), run(state, x [M,K]) -> y [M,N])."""
    errors = []
    try:  # vLLM
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.scalar_type import scalar_types

        qt = scalar_types.uint4b8

        def prep(w):
            w_ref, q, s, g_idx, sort_idx, _ = marlin_quantize(w.t().contiguous(), qt, 128, False)
            ws = marlin_make_workspace_new(w.device)
            return (q, s, g_idx, sort_idx, ws, w.shape[0], w.shape[1]), w_ref.t()

        def run(st, x):
            q, s, g_idx, sort_idx, ws, n, k = st
            return ops.gptq_marlin_gemm(
                x,
                None,
                q,
                None,
                s,
                None,
                None,
                None,
                g_idx,
                sort_idx,
                ws,
                qt,
                x.shape[0],
                n,
                k,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        return "vllm.gptq_marlin_gemm", prep, run
    except Exception as e:  # noqa: BLE001
        errors.append(f"vllm: {type(e).__name__}: {e}")
    try:  # standalone marlin
        import marlin

        def prep(w):
            n, k = w.shape
            layer = marlin.Layer(k, n, groupsize=128).to(w.device)
            wq = w.float().reshape(n, k // 128, 128)
            s = wq.abs().amax(-1, keepdim=True) / 7
            q = torch.clamp(torch.round(wq / s), -8, 7)
            w_ref = (q * s).reshape(n, k).half()
            lin = torch.nn.Linear(k, n, bias=False).to(w.device).half()
            lin.weight.data = w_ref
            layer.pack(lin, s.reshape(n, k // 128).t().half())
            return layer, w_ref

        def run(layer, x):
            return layer(x)

        return "marlin.Layer", prep, run
    except Exception as e:  # noqa: BLE001
        errors.append(f"marlin: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--secs", type=float, default=0.4)
    a = ap.parse_args()
    out = open(a.out, "w")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch has no CUDA device")
        name, prep, run = make_backend(torch)
    except Exception as e:  # noqa: BLE001
        emit(out, {"kind": "skip", "impl": IMPL, "fmt": "int4-g128", "N": 0, "K": 0, "M": 0, "reason": f"not installed: {e}"})
        print(f"marlin: skipped ({e})", file=sys.stderr)
        return 0
    props = torch.cuda.get_device_properties(0)
    nv = Nvml(getattr(props, "pci_bus_id", None))
    l2 = getattr(props, "L2_cache_size", 50 << 20)
    batches = (1, 16) if a.quick else BATCHES
    reps, secs = (3, 0.2) if a.quick else (a.reps, a.secs)
    torch.manual_seed(0)
    for n, k in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).half()
        st, w_ref = prep(w)
        wbytes = n * k // 2 + n * (k // 128) * 2
        copies = max(2, min(64, (4 * l2 + wbytes - 1) // wbytes))
        states = [st] + [prep(w)[0] for _ in range(copies - 1)]
        for m in batches:
            x = torch.randn(m, k, device="cuda").half()
            y = run(states[0], x).float()
            ref = x.float() @ w_ref.float().t()
            rel = ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
            emit(out, {"kind": "check", "fmt": "int4-g128", "N": n, "K": k, "M": m, "impl": IMPL, "config": name, "status": "ok",
                       "relerr_exact": None, "relerr_model": rel, "copies": copies})  # fmt: skip
            graph = None
            try:  # replay from a CUDA graph, as KURN and llama.cpp do, so Python launch cost is not timed
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for s in states:
                        run(s, x)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for s in states:
                        run(s, x)
            except Exception:  # noqa: BLE001
                graph = None

            def once(graph=graph, states=states, x=x):
                if graph is not None:
                    graph.replay()
                else:
                    for s in states:
                        run(s, x)

            for r in range(reps):
                once()
                torch.cuda.synchronize()
                e0, t0, calls = nv.joules(), time.perf_counter(), 0
                while time.perf_counter() - t0 < secs:
                    once()
                    calls += copies
                    torch.cuda.synchronize()
                t1, e1 = time.perf_counter(), nv.joules()
                emit(out, {"kind": "sample", "fmt": "int4-g128", "N": n, "K": k, "M": m, "impl": IMPL, "round": r,
                           "us": (t1 - t0) / calls * 1e6, "joules": (e1 - e0) / calls, "watts": float("nan"), "calls": calls,
                           "bytes": wbytes + 2 * m * k + 2 * m * n, "ops": 2.0 * n * k * m})  # fmt: skip
        del states
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
