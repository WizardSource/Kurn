#!/usr/bin/env python3
"""WS-B (fourbit) quality: perplexity / KL divergence of 4-bit formats on a llama.cpp model.

  quality.py swap --src BF16.gguf --template Q4_0-pure.gguf --format mxfp4|nvfp4 --method ggml|mse --out X.gguf
      Every tensor the template stores as Q4_0 is re-quantized from the BF16 source into
      the target format with kurn.mx; all other tensors and metadata are copied from the
      template byte for byte, so models differ only in the 4-bit format. NVFP4 `mse` adds
      llama.cpp's optional per-tensor `<name>.scale` (F32 [1]) tensors.
  quality.py ppl MODEL.gguf [--label L] [--threads 8]
      llama-perplexity on wikitext-2 test (-c 512, 32 chunks, batch 512), with KL divergence
      against the BF16 logits when the base file exists. Appends to results/quality.csv.

Model loads use > 2 GB RAM: run both through benchmarks/v0.2/benchlock.sh.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

from kurn import mx

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.expanduser(os.environ.get("LLAMA_BIN", "~/src/llama.cpp/build/bin"))
TEXT = os.path.expanduser("~/data/wiki.test.raw")
KLD_BASE = os.path.expanduser("~/models/compress/kld-base-bf16-c32.bin")
SCALED = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def _gguf():
    sys.path.insert(0, os.path.expanduser("~/src/llama.cpp/gguf-py"))
    import gguf

    return gguf


def _f32(gguf, t):
    """Reader tensor (F32 / F16 / BF16) -> float32 array (rows, K)."""
    k = int(t.shape[0])
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        u = np.frombuffer(t.data.tobytes(), dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(-1, k)
    if t.tensor_type == gguf.GGMLQuantizationType.F16:
        return np.frombuffer(t.data.tobytes(), dtype=np.float16).astype(np.float32).reshape(-1, k)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data.tobytes(), dtype=np.float32).reshape(-1, k)
    raise SystemExit(f"{t.name}: source must be F32/F16/BF16, got {t.tensor_type.name}")


def swap(a):
    gguf = _gguf()
    src = {t.name: t for t in gguf.GGUFReader(a.src).tensors}
    tpl = gguf.GGUFReader(a.template)
    qt = gguf.GGMLQuantizationType.MXFP4 if a.format == "mxfp4" else gguf.GGMLQuantizationType.NVFP4
    bpb, blk = (17, 32) if a.format == "mxfp4" else (36, 64)
    arch = tpl.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(a.out, arch)
    for f in tpl.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    w.add_string("kurn.fourbit.swap", f"{a.format}/{a.method} from {os.path.basename(a.src)}")
    plan = []
    for t in tpl.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            k, rows = int(t.shape[0]), int(np.prod(t.shape[1:]))
            if k % blk:
                raise SystemExit(f"{t.name}: K={k} not a multiple of {blk}")
            w.add_tensor_info(t.name, (rows, k // blk * bpb), np.dtype(np.uint8), rows * k // blk * bpb, raw_dtype=qt)
            stem = t.name.rsplit(".", 1)[0]
            scaled = a.format == "nvfp4" and a.method == "mse" and stem.split(".")[-1] in SCALED
            plan.append(("q", t, scaled))
            if scaled:
                w.add_tensor_info(stem + ".scale", (1,), np.dtype(np.float32), 4)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            plan.append(("copy", t, False))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    for kind, t, scaled in plan:
        if kind == "copy":
            w.write_tensor_data(t.data, tensor_endianess=tpl.endianess)
            continue
        x = _f32(gguf, src[t.name])
        if a.format == "mxfp4":
            q, s = mx.quantize_mxfp4(x, a.method), 1.0
        else:
            q, s = mx.quantize_nvfp4(x, a.method)
            if not scaled and s != 1.0:
                raise AssertionError(t.name)
        w.write_tensor_data(np.frombuffer(q, dtype=np.uint8).reshape(x.shape[0], -1))
        if scaled:
            w.write_tensor_data(np.array([s], dtype=np.float32))
        print(f"{t.name:32s} {x.shape} scale={s:.3e} {time.time() - t0:6.1f}s", flush=True)
    w.close()


def ppl(a):
    cmd = [os.path.join(LLAMA, "llama-perplexity"), "-m", a.model, "-f", TEXT, "-c", "512", "--chunks", "32",
           "-t", str(a.threads), "-b", "512"]  # fmt: skip
    if not a.repack:
        # ggml's default extra buffers send Q4_0/Q4_K/Q6_K to AMX, whose tile data this VM does not context-switch
        cmd += ["--no-repack"]
    if os.path.exists(KLD_BASE) and not a.no_kld:
        cmd += ["--kl-divergence-base", KLD_BASE, "--kl-divergence"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode:
        raise SystemExit(out[-3000:])

    def grab(pat):
        m = re.findall(pat, out)
        return m[-1] if m else ""

    row = {
        "label": a.label or os.path.basename(a.model),
        "model": a.model,
        "size_MiB": f"{os.path.getsize(a.model) / 2**20:.1f}",
        "ppl": grab(r"Final estimate: PPL = ([\d.]+)") or grab(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"),
        "ppl_err": grab(r"Final estimate: PPL = [\d.]+ \+/- ([\d.]+)") or grab(r"Mean PPL\(Q\)\s*:\s*[\d.]+ ± ([\d.]+)"),
        "kld": grab(r"Mean\s+KLD:\s+([-\d.]+)"),
        "kld_err": grab(r"Mean\s+KLD:\s+[-\d.]+ ± ([\d.]+)"),
        "same_top": grab(r"Same top p:\s+([\d.]+)"),
        "ppl_ratio": grab(r"Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)"),
        "wall_s": f"{time.time() - t0:.0f}",
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(row, flush=True)
    path = os.path.join(HERE, "results", "quality.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    with open(os.path.join(HERE, "results", f"ppl_{row['label']}.log"), "w") as fh:
        fh.write(out[-20000:])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--src", required=True)
    s.add_argument("--template", required=True)
    s.add_argument("--format", choices=("mxfp4", "nvfp4"), required=True)
    s.add_argument("--method", choices=("ggml", "mse"), default="ggml")
    s.add_argument("--out", required=True)
    p = sub.add_parser("ppl")
    p.add_argument("model")
    p.add_argument("--label")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-kld", action="store_true")
    p.add_argument("--repack", action="store_true", help="ggml default extra buffers (AMX); run under benchlock.sh")
    a = ap.parse_args()
    return swap(a) if a.cmd == "swap" else ppl(a)


if __name__ == "__main__":
    sys.exit(main())
