"""Absorbed-MLA attention on real PLM-1.8B activations vs llama.cpp's own (decompressed) attention.

    tensor_dump MODEL TEXT --layer 5 --names q_states,kv_compressed,k_pe,kqv_out --out DIR
    check_plm.py MODEL DIR [target ...]       (targets: avx512, amx_bf16; amx needs benchlock)

Prints, for 512 real tokens of one layer: float64 absorbed reference vs llama.cpp kqv_out (does
the absorption reproduce the model?), and each kurn kernel vs the float64 reference and vs kqv_out.
"""

import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from kurn import attention as A
from kurn import latent


def load(path):
    with open(path, "rb") as fh:
        ne = np.frombuffer(fh.read(32), np.int64)
        data = np.frombuffer(fh.read(), np.float32)
    return data.reshape(tuple(int(x) for x in ne[::-1])).squeeze()


def main(model, d, targets=("avx512",), layer=5, n_head=16, d_nope=128, d_r=64, d_v=128):
    q = load(f"{d}/q_states.bin")  # [T, H, 192]: nope dims then rope dims
    c = load(f"{d}/kv_compressed.bin")  # [T, 512]
    kpe = load(f"{d}/k_pe.bin")  # [T, 64]
    ref_llama = load(f"{d}/kqv_out.bin").reshape(q.shape[0], n_head, d_v)
    w = next(t for t in GGUFReader(model).tensors if t.name == f"blk.{layer}.attn_kv_b.weight")
    w_uk, w_uv = latent.split_kv_b(dequantize(w.data, w.tensor_type), n_head, d_nope, d_v)
    scale = 1.0 / np.sqrt(d_nope + d_r)
    qt = latent.absorb_queries(q[..., :d_nope], q[..., d_nope:], w_uk)
    cache = latent.latent_cache(c, kpe)
    o64 = latent.reference(qt, cache, c.shape[1], scale)
    ref64 = latent.up_project(o64, w_uv)

    def rel(x, y):
        return float(np.abs(x - y).max() / np.abs(y).max())

    print(f"tokens {q.shape[0]}, latent row {cache.shape[-1]} x f16 = {cache.shape[-1] * 2} B/token/layer "
          f"(llama.cpp caches {n_head * (d_nope + d_r + d_v) * 2} B decompressed)")  # fmt: skip
    print(f"float64 absorbed vs llama.cpp kqv_out: relerr {rel(ref64, ref_llama):.2e}")
    for tgt in targets:
        c_ = A.resolve({"target": tgt, "kv": "f16", "dk": 576, "mla": 1, "heads": n_head, "kv_heads": 1})
        lib = latent.load(A.build(c_))
        o = latent.run_kattn(lib, qt, cache, None, c.shape[1], scale, threads=4)
        out = latent.up_project(o, w_uv)
        print(f"kurn {tgt}: latent output vs float64 {rel(o, o64):.2e}; after W_UV vs float64 {rel(out, ref64):.2e}, "
              f"vs llama.cpp {rel(out, ref_llama):.2e}")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3:]) or ("avx512",))
