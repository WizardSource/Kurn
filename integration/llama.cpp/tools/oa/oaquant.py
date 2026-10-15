"""Outlier-aware (OA) weight quantization for kurn.

For every linear weight W (N x K) with calibration input statistics (E[x^2], mean over non-first tokens):
  - idx = the k input channels with the largest E[x^2] (the giant-activation channels);
  - main = W with columns idx zeroed, quantized to the base file's type for that tensor, with importance
    E[x^2] (0 on idx) so block scales are chosen without the outlier columns;
  - side = W[:, idx] kept exactly (F16), applied as W[:, idx] @ x[idx] in f32 by the kernel, which also zeroes
    x[idx] before quantizing the activations for the main product;
  - optional (--bias): bias = (W - main)[:, not idx] @ mean[not idx], the quantization error's response to the
    near-constant part of the input (the dark dimensions), folded into one f32 per output row. Off by default: about
    neutral at 4 bits and harmful at 2 bits (Llama-3.2-3B IQ2_XS: KL 0.52 without, 0.68 with), see
    benchmarks/v0.3/outlier-compression.
  y = main @ x_(idx zeroed) + side @ x[idx] (+ bias)

The main file is a byte-for-byte copy of a llama-quantize output (same tensor types and offsets) with the linear
weights rewritten in place, so it keeps llama-quantize's per-tensor type mix. The side data goes to OUT.oa.gguf.

  python oaquant.py --f16 F16.gguf --base BASE.gguf --stats calib.pt --out OUT.gguf [--k 16] [--bias]
"""
import argparse, ctypes, os, re, shutil, sys
import numpy as np
import torch
import gguf
from gguf import GGUFReader, GGMLQuantizationType as QT

LIB = os.environ.get("GGML_LIB", "/tmp/llama-kurn/build/bin/libggml-base.so")
_ggml = None


def ggml():
    global _ggml
    if _ggml is None:
        _ggml = ctypes.CDLL(LIB)
        f = _ggml.ggml_quantize_chunk
        f.restype = ctypes.c_size_t
        f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                      ctypes.c_void_p]
        _ggml.ggml_row_size.restype = ctypes.c_size_t
        _ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        _ggml.ggml_quantize_init.restype = None
        _ggml.ggml_quantize_init.argtypes = [ctypes.c_int]
    return _ggml


Q2_0 = 42


def quant_q2_0(W, imp):
    """Q2_0 (64 values, fp16 d, codes {-1,0,1,2} * d). Weighted search over d, both signs, then a weighted
    least-squares refit of d for the chosen codes. Returns bytes in ggml's block_q2_0 layout."""
    N, K = W.shape
    w = W.reshape(N, K // 64, 64).astype(np.float32)
    im = np.broadcast_to(imp.reshape(1, K // 64, 64), w.shape).astype(np.float32) + 1e-12
    amax = np.abs(w).max(-1, keepdims=True) + 1e-20
    best_e = np.full(amax.shape, np.inf, np.float32)
    best_d = np.zeros_like(amax)
    for s in np.linspace(0.25, 1.6, 28):
        for sg in (1.0, -1.0):
            d = sg * amax * s / 2.0
            q = np.clip(np.rint(w / d), -1, 2)
            # refit d for these codes
            num = (im * w * q).sum(-1, keepdims=True)
            den = (im * q * q).sum(-1, keepdims=True) + 1e-30
            d2 = np.where(den > 1e-30, num / den, d)
            d2 = np.where(np.sign(d2) == sg, d2, d)
            q2 = np.clip(np.rint(w / d2), -1, 2)
            e = (im * (w - q2 * d2) ** 2).sum(-1, keepdims=True)
            better = e < best_e
            best_e = np.where(better, e, best_e)
            best_d = np.where(better, d2, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)
    q = (np.clip(np.rint(w / np.where(d == 0, 1, d)), -1, 2) + 1).astype(np.uint8)  # 0..3
    q = q.reshape(N, K // 64, 16, 4)
    qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    out = np.empty((N, K // 64, 18), np.uint8)
    out[..., :2] = d16.view(np.uint8).reshape(N, K // 64, 2)
    out[..., 2:] = qs
    return out.reshape(N, -1)


def deq_q2_0(raw, N, K):
    b = raw.reshape(N, K // 64, 18)
    d = b[..., :2].copy().view(np.float16).astype(np.float32)
    qs = b[..., 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(N, K // 64, 64).astype(np.float32) - 1
    return (q * d).reshape(N, K)


_q2 = None


def quantize(t, W, imp):
    """W f32 (N x K) -> raw bytes of ggml type t (via ggml_quantize_chunk, imatrix = imp)."""
    global _q2
    if t == Q2_0:
        so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liboa_q2_0.so")
        if not os.path.exists(so):
            return quant_q2_0(W, imp).tobytes()
        if _q2 is None:
            _q2 = ctypes.CDLL(so)
            _q2.oa_quant_q2_0.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2
        N, K = W.shape
        W = np.ascontiguousarray(W, np.float32)
        im = np.ascontiguousarray(imp, np.float32)
        out = np.empty(N * (K // 64) * 18, np.uint8)
        _q2.oa_quant_q2_0(W.ctypes.data, im.ctypes.data, out.ctypes.data, N, K)
        return out.tobytes()
    N, K = W.shape
    L = ggml()
    rs = L.ggml_row_size(t, K)
    dst = np.empty(N * rs, np.uint8)
    W = np.ascontiguousarray(W, np.float32)
    im = np.ascontiguousarray(imp, np.float32)
    L.ggml_quantize_init(t)  # grids for the IQ types, before the threads
    nth = int(os.environ.get("OA_THREADS", os.cpu_count() or 1))
    step = max(1, (N + nth - 1) // nth)
    from concurrent.futures import ThreadPoolExecutor

    def part(r0):  # ggml_quantize_chunk releases the GIL (ctypes call)
        return L.ggml_quantize_chunk(t, W.ctypes.data, dst.ctypes.data, r0 * K, min(step, N - r0), K, im.ctypes.data)

    with ThreadPoolExecutor(nth) as ex:
        n = sum(ex.map(part, range(0, N, step)))
    assert n == N * rs, (n, N * rs)
    return dst.tobytes()


def dequantize(t, raw, N, K):
    if t == Q2_0:
        return deq_q2_0(np.frombuffer(raw, np.uint8), N, K)
    return gguf.quants.dequantize(np.frombuffer(raw, np.uint8).reshape(N, -1), QT(t)).reshape(N, K).astype(np.float32)


SITE = {"attn_q": "qkv", "attn_k": "qkv", "attn_v": "qkv", "attn_output": "o", "ffn_gate": "gu", "ffn_up": "gu",
        "ffn_down": "down"}


def site_of(name):
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", name)
    if not m or m.group(2) not in SITE:
        return None
    return f"{SITE[m.group(2)]}:{m.group(1)}"


def f16_tensor(rd, name):
    for t in rd.tensors:
        if t.name == name:
            a = np.asarray(t.data)
            if t.tensor_type == QT.F16:
                return a.astype(np.float32).reshape(t.shape[::-1])
            if t.tensor_type == QT.F32:
                return a.reshape(t.shape[::-1]).astype(np.float32)
            if t.tensor_type == QT.BF16:
                return gguf.quants.dequantize(a, QT.BF16).reshape(t.shape[::-1]).astype(np.float32)
            raise ValueError(f"{name}: source type {t.tensor_type}")
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--stats", required=True, help="qstream.py --mode calib output (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-down", type=int, default=-1, help="salient channels for ffn_down (default: --k)")
    ap.add_argument("--bias", action="store_true", help="add the mean-fold bias (ablation; off by default)")
    ap.add_argument("--no-bias", action="store_true", help="(default; kept for old command lines)")
    ap.add_argument("--no-side", action="store_true", help="imatrix-only control: same quantizer, no side/bias")
    ap.add_argument("--main-type", default="", help="override: requantize linear weights of this base type, e.g. q2_0")
    args = ap.parse_args()

    st = torch.load(args.stats)["sites"]
    src = GGUFReader(args.f16)
    base = GGUFReader(args.base)
    shutil.copyfile(args.base, args.out)
    mm = np.memmap(args.out, dtype=np.uint8, mode="r+")
    side_w = gguf.GGUFWriter(args.out + ".oa.gguf" if not args.no_side else os.devnull, "kurn-oa")
    side_w.add_uint32("kurn.oa.version", 1)
    nside = 0
    tot_main = tot_side = 0
    for t in base.tensors:
        site = site_of(t.name)
        if site is None or site not in st:
            continue
        N, K = int(t.shape[1]), int(t.shape[0])
        ty = int(t.tensor_type)
        W = f16_tensor(src, t.name)
        assert W.shape == (N, K), (t.name, W.shape, N, K)
        ms = st[site]["ms"].numpy().astype(np.float64)
        mu = st[site]["mean"].numpy().astype(np.float64)
        k = args.k_down if (args.k_down >= 0 and "ffn_down" in t.name) else args.k
        if args.no_side:
            k = 0
        idx = np.sort(np.argsort(-ms)[:k]).astype(np.int32)
        Wm = W.copy()
        imp = ms.copy()
        if k:
            Wm[:, idx] = 0
            imp[idx] = 0
        imp = np.maximum(imp, 1e-8 * imp.max())
        raw = quantize(ty, Wm, imp.astype(np.float32))
        assert len(raw) == t.n_bytes, (t.name, len(raw), t.n_bytes)
        mm[t.data_offset:t.data_offset + t.n_bytes] = np.frombuffer(raw, np.uint8)
        tot_main += t.n_bytes
        if k:
            Wq = dequantize(ty, raw, N, K)
            err = (W - Wq).astype(np.float64)
            err[:, idx] = 0
            bias = (err @ mu).astype(np.float32) if args.bias and not args.no_bias else np.zeros(N, np.float32)
            side_w.add_tensor(t.name + ".oa_idx", idx)
            side_w.add_tensor(t.name + ".oa_side", np.ascontiguousarray(W[:, idx].astype(np.float16)))  # [N, k]
            side_w.add_tensor(t.name + ".oa_bias", bias)
            tot_side += idx.nbytes + N * k * 2 + N * 4
            nside += 1
        print(f"{t.name:28s} {QT(ty).name:7s} N={N} K={K} k={k}", flush=True)
    mm.flush()
    if not args.no_side:
        side_w.write_header_to_file()
        side_w.write_kv_data_to_file()
        side_w.write_tensors_to_file()
        side_w.close()
    print(f"linear main {tot_main / 2**20:.1f} MiB, side {tot_side / 2**20:.2f} MiB ({nside} tensors), "
          f"file {os.path.getsize(args.out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
