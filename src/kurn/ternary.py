"""Ternary weights as a presence bitmap plus packed signs ("tbm", the BITCOS layout).

A ternary weight w in {-1, 0, +1} costs log2(3) = 1.585 bits only if the three values are equally
common. Real ternary LLMs are not: zeros are 30-50% of the weights. Storing one presence bit per
weight plus one sign bit per nonzero costs 2 - z bits per weight (z = zero fraction), which beats
TQ1_0's five-trits-per-byte packing once z > 0.375 (the per-256 fp16 scale adds 1/16 to both).

The KURN buffer type (integration/llama.cpp, kurn-tern.cpp) repacks TQ1_0 and TQ2_0 tensors into
this layout at load time when it is smaller, and computes them with the same per-column arithmetic
as the TQ2_0 kernels (bit-identical results). This module holds the layout's reference encoder and
decoder (the tests check the C packer against it), the size model, and `kurn ternary stats`, which
measures zero fractions of real models (GGUF TQ1_0 / TQ2_0, and packed 2-bit safetensors).

Layout of one tensor [K x N] (K a multiple of 256), all integers little-endian:

  header   64 bytes: "KTBM", u32 version, i64 K, i64 N, i64 ngroups, i64 nrec_k, i64 used bytes,
           i64 dense records
  table    (ngroups + 1) u64 byte offsets of each 16-row group's record stream (relative to the
           start of the data), padded to 64 bytes
  data     per group of 16 rows, nrec_k records (256 values each = one Q8_K activation block):
             d      16 x fp16, row scales (padding rows: 0)
             bitmap 64 x u64: word i = kg * 8 + kk covers values kg*32 + kk*4 + j (j < 4) of the 16
                    rows; bit b <-> row b // 4, j = b % 4 (the byte order of a vpdpbusd operand
                    whose lane is a row); set = nonzero
             flag   u8: bit 0 set when some word's signs do not fit the decoder's one-load window
                    (64 - (offset % 8) bits; see DENSE), the kernels then take the two-load path
             signs  one bit per nonzero (1 = +1), word by word, set bits in ascending order,
                    LSB-first and continuous across words, ceil(nnz / 8) bytes
           records follow each other without padding; 64 bytes of slack end the buffer (the
           decoder reads 16 bytes past a word's first sign byte)
"""

import argparse
import json
import os
import struct
import sys

from ._numpy import np

MAGIC = b"KTBM"
VERSION = 1
HDR = 64
REC_FIXED = 32 + 512 + 1  # d, bitmap, flag
SLACK = 64
TQ1_BPW = 54 * 8 / 256  # 1.6875
TQ2_BPW = 66 * 8 / 256  # 2.0625


def tbm_bpw(z):
    """Bits per weight of the layout at zero fraction z (records only; header, table and the
    per-record flag byte and byte padding add ~0.003 bpw)."""
    return 1 + 1 / 16 + (1 - z)


def breakeven_z():
    """Zero fraction above which tbm is smaller than TQ1_0 (0.375)."""
    return 1 + 1 / 16 + 1 - TQ1_BPW


# --------------------------------------------------------------------------- native blocks -> trits
def tq2_0_trits(raw, nblocks):
    """TQ2_0 blocks (66 bytes: qs[64], fp16 d) -> (trits int8 [nblocks, 256] in value order, d uint16)."""
    b = np.frombuffer(raw, dtype=np.uint8, count=nblocks * 66).reshape(nblocks, 66)
    qs = b[:, :64].reshape(nblocks, 2, 32)  # value v: byte (v // 128) * 32 + v % 32, bits 2 * ((v % 128) // 32)
    sh = np.array([0, 2, 4, 6], dtype=np.uint8)
    q = (qs[:, :, None, :] >> sh[None, None, :, None]) & 3  # [blk, half, shift, 32]
    trits = q.reshape(nblocks, 256).astype(np.int8) - 1
    d = b[:, 64:66].copy().view(np.uint16).reshape(nblocks)
    return trits, d


def _tq1_digits(bytes_, n):
    """ggml's base-3 fixed point: digit n of each byte (0..2)."""
    pow3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint8)
    q = (bytes_.astype(np.uint16) * pow3[n]) & 0xFF
    return ((q * 3) >> 8).astype(np.int8)


def tq1_0_trits(raw, nblocks):
    """TQ1_0 blocks (54 bytes: qs[48], qh[4], fp16 d) -> (trits int8 [nblocks, 256], d uint16), in
    the value order of ggml's dequantize_row_tq1_0."""
    b = np.frombuffer(raw, dtype=np.uint8, count=nblocks * 54).reshape(nblocks, 54)
    parts = []
    for j0, w in ((0, 32), (32, 16)):
        for n in range(5):
            parts.append(_tq1_digits(b[:, j0 : j0 + w], n))
    for n in range(4):
        parts.append(_tq1_digits(b[:, 48:52], n))
    trits = np.concatenate(parts, axis=1) - 1
    d = b[:, 52:54].copy().view(np.uint16).reshape(nblocks)
    return trits.astype(np.int8), d


def zero_count_tq2_0(raw, nblocks):
    t, _ = tq2_0_trits(raw, nblocks)
    return int(np.count_nonzero(t == 0)), nblocks * 256


def zero_count_tq1_0(raw, nblocks):
    t, _ = tq1_0_trits(raw, nblocks)
    return int(np.count_nonzero(t == 0)), nblocks * 256


# --------------------------------------------------------------------------- reference encoder / decoder
def _word_positions():
    """pos[i, b] = (row, value-in-record) of bit b of bitmap word i."""
    i = np.arange(64)[:, None]
    b = np.arange(64)[None, :]
    kg, kk = i // 8, i % 8
    return b // 4, kg * 32 + kk * 4 + b % 4


def encode(trits, d):
    """trits int8 [N, K] in {-1, 0, 1}, d uint16 [N, K // 256] (fp16 bits) -> tbm bytes."""
    trits = np.asarray(trits, dtype=np.int8)
    N, K = trits.shape
    if K % 256:
        raise ValueError(f"K={K} is not a multiple of 256")
    ng, nrk = (N + 15) // 16, K // 256
    rows, vals = _word_positions()
    tpad = np.zeros((ng * 16, K), dtype=np.int8)
    tpad[:N] = trits
    dpad = np.zeros((ng * 16, nrk), dtype=np.uint16)
    dpad[:N] = d
    bitw = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    streams, offs, dense = [], [0], 0
    for g in range(ng):
        out = bytearray()
        for p in range(nrk):
            blk = tpad[g * 16 : g * 16 + 16, p * 256 : p * 256 + 256]
            w = blk[rows, vals]  # [64 words, 64 bits]
            nz = w != 0
            words = (nz.astype(np.uint64) * bitw).sum(axis=1, dtype=np.uint64)
            signs, off, flag = [], 0, 0
            for i in range(64):
                n = int(nz[i].sum())
                if n > 64 - (off % 8):
                    flag = 1
                signs.extend((w[i][nz[i]] > 0).tolist())
                off += n
            dense += flag
            sb = np.packbits(np.array(signs, dtype=np.uint8), bitorder="little").tobytes() if signs else b""
            out += dpad[g * 16 : g * 16 + 16, p].astype("<u2").tobytes()
            out += words.astype("<u8").tobytes()
            out += bytes([flag])
            out += sb
        streams.append(bytes(out))
        offs.append(offs[-1] + len(out))
    table = struct.pack(f"<{ng + 1}Q", *offs)
    table += b"\0" * (-len(table) % 64)
    data = b"".join(streams)
    used = HDR + len(table) + len(data) + SLACK
    hdr = MAGIC + struct.pack("<I6q", VERSION, K, N, ng, nrk, used, dense)
    return hdr + b"\0" * (HDR - len(hdr)) + table + data + b"\0" * SLACK


def header(buf):
    if bytes(buf[:4]) != MAGIC:
        raise ValueError("not a tbm buffer")
    v, K, N, ng, nrk, used, dense = struct.unpack_from("<I6q", buf, 4)
    return {"version": v, "K": K, "N": N, "ngroups": ng, "nrec_k": nrk, "used": used, "dense": dense}


def decode(buf):
    """tbm bytes -> (trits int8 [N, K], d uint16 [N, K // 256])."""
    h = header(buf)
    K, N, ng, nrk = h["K"], h["N"], h["ngroups"], h["nrec_k"]
    tab = HDR
    offs = struct.unpack_from(f"<{ng + 1}Q", buf, tab)
    data = tab + (8 * (ng + 1) + 63) // 64 * 64
    rows, vals = _word_positions()
    trits = np.zeros((ng * 16, K), dtype=np.int8)
    d = np.zeros((ng * 16, nrk), dtype=np.uint16)
    for g in range(ng):
        o = data + offs[g]
        for p in range(nrk):
            d[g * 16 : g * 16 + 16, p] = np.frombuffer(bytes(buf[o : o + 32]), dtype="<u2")
            words = np.frombuffer(bytes(buf[o + 32 : o + 544]), dtype="<u8")
            nz = ((words[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
            nnz = int(nz.sum())
            sb = np.frombuffer(bytes(buf[o + REC_FIXED : o + REC_FIXED + (nnz + 7) // 8]), dtype=np.uint8)
            signs = np.unpackbits(sb, bitorder="little")[:nnz].astype(bool)
            w = np.zeros((64, 64), dtype=np.int8)
            w[nz] = np.where(signs, 1, -1)
            blk = np.zeros((16, 256), dtype=np.int8)
            blk[rows, vals] = w
            trits[g * 16 : g * 16 + 16, p * 256 : p * 256 + 256] = blk
            o += REC_FIXED + (nnz + 7) // 8
        assert o == data + offs[g + 1], "record stream length mismatch"
    return trits[:N], d[:N]


# --------------------------------------------------------------------------- model scans
def _gguf_tensors(path):
    from gguf import GGUFReader

    r = GGUFReader(path)
    for t in r.tensors:
        yield t.name, t.tensor_type.name, int(np.prod(t.shape)), t.data


def _safetensors_packed(path):
    """Packed 2-bit ternary tensors of a safetensors file (BitNet b1.58 / Falcon3-1.58bit /
    Llama3-8B-1.58: uint8, four codes per byte, value = code - 1)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
        base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, m in meta.items():
        if name == "__metadata__" or m["dtype"] != "U8":
            continue
        a, b = m["data_offsets"]
        yield name, "packed2", 4 * (b - a), mm[base + a : base + b]


def scan(path, chunk_blocks=1 << 16):
    """Per-tensor zero counts of the ternary tensors in one model file."""
    out = []
    if path.endswith(".safetensors"):
        for name, kind, n, data in _safetensors_packed(path):
            z = 0
            for i in range(0, len(data), 1 << 22):
                c = np.asarray(data[i : i + (1 << 22)])
                z += sum(int(np.count_nonzero(((c >> s) & 3) == 1)) for s in (0, 2, 4, 6))
            out.append({"name": name, "type": kind, "n": n, "zeros": z})
        return out
    for name, typ, n, data in _gguf_tensors(path):
        if typ not in ("TQ1_0", "TQ2_0"):
            continue
        raw = np.asarray(data).reshape(-1).view(np.uint8)
        bs = 54 if typ == "TQ1_0" else 66
        fn = zero_count_tq1_0 if typ == "TQ1_0" else zero_count_tq2_0
        z = tot = 0
        nb = len(raw) // bs
        for b0 in range(0, nb, chunk_blocks):
            b1 = min(nb, b0 + chunk_blocks)
            zz, tt = fn(raw[b0 * bs : b1 * bs].tobytes(), b1 - b0)
            z += zz
            tot += tt
        out.append({"name": name, "type": typ, "n": tot, "zeros": z})
    return out


def summarize(rows):
    n = sum(r["n"] for r in rows)
    z = sum(r["zeros"] for r in rows)
    zs = [r["zeros"] / r["n"] for r in rows if r["n"]]
    zf = z / n if n else 0.0
    win = sum(1 for r in rows if r["n"] and r["zeros"] / r["n"] > breakeven_z())
    return {
        "tensors": len(rows),
        "weights": n,
        "zero_fraction": zf,
        "z_min": min(zs) if zs else 0.0,
        "z_max": max(zs) if zs else 0.0,
        "tensors_beating_tq1": win,
        "bpw_tq2_0": TQ2_BPW,
        "bpw_tq1_0": TQ1_BPW,
        "bpw_tbm": tbm_bpw(zf),
        "bpw_tbm_per_tensor_best": sum(min(TQ2_BPW, tbm_bpw(r["zeros"] / r["n"])) * r["n"] for r in rows if r["n"]) / n if n else 0.0,
    }


def cli_main(argv):
    ap = argparse.ArgumentParser(prog="kurn ternary", description="ternary weights: zero fractions and the tbm size model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="zero fraction of the ternary tensors of model files (GGUF TQ1_0/TQ2_0, packed safetensors)")
    p.add_argument("files", nargs="+")
    p.add_argument("--per-tensor", action="store_true")
    p.add_argument("--json", help="write the results here")
    a = ap.parse_args(argv)
    res = {}
    for f in a.files:
        rows = scan(f)
        s = summarize(rows)
        res[f] = {"summary": s, "tensors": rows}
        print(f"{os.path.basename(f)}: {s['tensors']} ternary tensors, {s['weights'] / 1e9:.3f} G weights, "
              f"z = {s['zero_fraction']:.4f} (per tensor {s['z_min']:.3f}..{s['z_max']:.3f}); bpw TQ2_0 {TQ2_BPW:.4f}, "
              f"TQ1_0 {TQ1_BPW:.4f}, tbm {s['bpw_tbm']:.4f} ({s['tensors_beating_tq1']}/{s['tensors']} tensors below TQ1_0)")  # fmt: skip
        if a.per_tensor:
            for r in rows:
                print(f"  {r['name']:48} {r['type']:6} {r['n']:>12} z={r['zeros'] / max(1, r['n']):.4f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0
