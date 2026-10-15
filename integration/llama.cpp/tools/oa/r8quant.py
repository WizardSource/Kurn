"""R8 weights for kurn (kurn-r8.h): every linear weight W [N x K] becomes I8 codes of W H (orthonormal block
Hadamard along K, block hb = the largest power of two dividing K, at most 4096) with one f32 scale per row; other
tensors are copied from a base GGUF (e.g. the Q8_0 file: token embedding and output head keep their types). The
scales and rotation blocks go to OUT.oa.gguf (the sidecar llama.cpp picks up next to the model).

  python r8quant.py --f16 F16.gguf --base Q8_0.gguf --out OUT.gguf [--clip mse|absmax]
"""
import argparse, os, re
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType as QT

LINEAR = re.compile(r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")


def hblock(K):
    b = 1
    while K % (b * 2) == 0 and b < 4096:
        b *= 2
    return b


def fwht_rows(W, hb):
    """orthonormal block Walsh-Hadamard of each row (float64)"""
    N, K = W.shape
    x = W.reshape(N, K // hb, hb).astype(np.float64).copy()
    h = 1
    while h < hb:
        x = x.reshape(N, K // hb, hb // (2 * h), 2, h)
        a, b = x[:, :, :, 0, :].copy(), x[:, :, :, 1, :].copy()
        x[:, :, :, 0, :], x[:, :, :, 1, :] = a + b, a - b
        h *= 2
    return x.reshape(N, K) / np.sqrt(hb)


def quant_rows(R, clip):
    amax = np.abs(R).max(1, keepdims=True) + 1e-30
    if clip == "absmax":
        s = amax / 127
        return np.clip(np.rint(R / s), -127, 127).astype(np.int8), s[:, 0].astype(np.float32)
    best_q, best_s, best_e = None, None, None
    for c in np.linspace(0.70, 1.0, 13):
        s = amax * c / 127
        q = np.clip(np.rint(R / s), -127, 127)
        e = ((q * s - R) ** 2).sum(1, keepdims=True)
        if best_q is None:
            best_q, best_s, best_e = q, s, e
        else:
            m = e < best_e
            best_q = np.where(m, q, best_q)
            best_s = np.where(m, s, best_s)
            best_e = np.where(m, e, best_e)
    return best_q.astype(np.int8), best_s[:, 0].astype(np.float32)


def f16_tensor(rd, name):
    t = rd[name]
    a = np.asarray(t.data)
    if t.tensor_type in (QT.F16, QT.F32):
        return a.astype(np.float32).reshape(t.shape[::-1])
    return gguf.quants.dequantize(a, t.tensor_type).reshape(t.shape[::-1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="mse", choices=["mse", "absmax"])
    args = ap.parse_args()
    src = {t.name: t for t in GGUFReader(args.f16).tensors}
    base = GGUFReader(args.base)
    arch = base.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(args.out, arch)
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    for t in base.tensors:
        if LINEAR.match(t.name):
            N, K = int(t.shape[1]), int(t.shape[0])
            w.add_tensor_info(t.name, (N, K), np.dtype(np.int8), N * K, raw_dtype=QT.I8)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    side = gguf.GGUFWriter(args.out + ".oa.gguf", "kurn-oa")
    side.add_uint32("kurn.oa.version", 1)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    nl = 0
    for t in base.tensors:
        if LINEAR.match(t.name):
            Wf = f16_tensor(src, t.name)
            hb = hblock(Wf.shape[1])
            q, s = quant_rows(fwht_rows(Wf, hb), args.clip)
            w.write_tensor_data(q)
            side.add_tensor(t.name + ".r8_scale", s)
            side.add_tensor(t.name + ".r8_rot", np.array([hb], np.int32))
            nl += 1
        else:
            w.write_tensor_data(t.data, tensor_endianess=base.endianess)
    w.close()
    side.write_header_to_file()
    side.write_kv_data_to_file()
    side.write_tensors_to_file()
    side.close()
    print(f"{nl} R8 weights, file {os.path.getsize(args.out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
