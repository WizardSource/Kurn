"""Pick CPU, GPU, or both for quantized matmuls.

Modes (also `KURN_DEVICE`):
  cpu      always the CPU path (kurn CPU kernels / ggml-cpu)
  gpu      always the CUDA path when a GPU is present; else CPU with reason
  auto     one device per op: GPU when `kernel_for()` says kurn wins, else CPU
  hybrid   CPU and GPU in the same step: GPU takes formats/batches it wins on;
           CPU takes the rest (and can overlap when ops are independent)

Environment:
  KURN_DEVICE          default mode (cpu|gpu|auto|hybrid)
  KURN_GPU_DISPATCH    path to dispatch.json from `kurn gpu report`
  KURN_HYBRID_MIN_K    refuse GPU below this K (default 512; PCIe/overhead floor)
"""

from __future__ import annotations

import os
from typing import Any

MODES = ("cpu", "gpu", "auto", "hybrid")


def _gpu_present():
    try:
        from kurn.gpu.harness import gpu_present

        return bool(gpu_present())
    except Exception:
        return False


def _kernel_for(fmt, batch, arch=None, table=None):
    from kurn.gpu.dispatch import kernel_for

    return kernel_for(fmt, batch, arch=arch, table=table)


def _min_k():
    try:
        return int(os.environ.get("KURN_HYBRID_MIN_K", "512"))
    except ValueError:
        return 512


def route(fmt, batch, mode=None, *, arch=None, table=None, K=None, gpu_ok=None):
    """Decide where one matmul runs.

    Returns dict:
      device: "cpu" | "gpu"
      impl:   "kurn" | "stock"   (which kernel family on that device)
      mode:   resolved mode
      reason: short explanation
      config: resolved CUDA config when device=gpu and impl=kurn
    """
    mode = (mode or os.environ.get("KURN_DEVICE") or "auto").lower()
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: expected one of {list(MODES)}")
    if gpu_ok is None:
        gpu_ok = _gpu_present()

    if mode == "cpu" or not gpu_ok:
        why = "mode=cpu" if mode == "cpu" else "no CUDA device"
        return {"device": "cpu", "impl": "kurn", "mode": mode, "reason": why}

    if mode == "gpu":
        pick = _kernel_for(fmt, batch, arch=arch, table=table)
        if pick.get("impl") == "kurn":
            if K is not None and K < _min_k():
                return {"device": "cpu", "impl": "kurn", "mode": mode,
                        "reason": f"K={K} < KURN_HYBRID_MIN_K={_min_k()} (GPU overhead)"}
            return {"device": "gpu", "impl": "kurn", "mode": mode, "reason": "mode=gpu",
                    "config": pick.get("config"), "measured_batch": pick.get("measured_batch")}
        return {"device": "gpu", "impl": "stock", "mode": mode,
                "reason": pick.get("reason", "stock ggml-cuda")}

    # auto / hybrid: use the measured GPU dispatch table; fall back to CPU
    pick = _kernel_for(fmt, batch, arch=arch, table=table)
    if pick.get("impl") != "kurn":
        return {"device": "cpu", "impl": "kurn", "mode": mode,
                "reason": pick.get("reason", "GPU does not win; use CPU")}
    if K is not None and K < _min_k():
        return {"device": "cpu", "impl": "kurn", "mode": mode,
                "reason": f"K={K} < KURN_HYBRID_MIN_K={_min_k()} (GPU overhead)"}
    return {"device": "gpu", "impl": "kurn", "mode": mode, "reason": "GPU win in dispatch table",
            "config": pick.get("config"), "measured_batch": pick.get("measured_batch")}


def plan(ops, mode=None, *, arch=None, table=None, gpu_ok=None):
    """Assign a list of matmul descriptors to CPU and/or GPU.

    Each op is a dict with at least `fmt` and `batch`, optional `K`, `N`, `name`.
    In hybrid mode, independent ops may land on different devices so both can run
    in the same step; auto/cpu/gpu collapse to a single device preference per op
    without requiring overlap.

    Returns {"mode", "cpu": [...], "gpu": [...], "overlap": bool}.
    """
    mode = (mode or os.environ.get("KURN_DEVICE") or "auto").lower()
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: expected one of {list(MODES)}")
    if gpu_ok is None:
        gpu_ok = _gpu_present()

    cpu, gpu = [], []
    for i, op in enumerate(ops):
        fmt = op["fmt"]
        batch = int(op.get("batch", op.get("M", 1)))
        decision = route(fmt, batch, mode=mode, arch=arch, table=table, K=op.get("K"), gpu_ok=gpu_ok)
        entry = {**op, "index": i, **decision}
        (gpu if decision["device"] == "gpu" else cpu).append(entry)

    overlap = mode == "hybrid" and bool(cpu) and bool(gpu)
    return {"mode": mode, "cpu": cpu, "gpu": gpu, "overlap": overlap,
            "summary": f"{len(cpu)} on cpu, {len(gpu)} on gpu"
                       + (" (overlap)" if overlap else "")}
