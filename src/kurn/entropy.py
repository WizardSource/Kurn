"""Lossless entropy coding of quantized weight codes (Q8_0 / Q4_0 / Q4_K) with a fused decode GEMV.

Analysis: per-tensor order-0 entropy of the stored codes (the Shannon bound of any static
per-tensor code), a length-limited Huffman code (max 12 bits, so one 4096-entry table decodes a
symbol per lookup) and its average length.

Coder `q8h`: Q8_0 codes Huffman-coded per row in S interleaved streams (the row's K codes split
into S contiguous segments, one bit stream each, so S table lookups are in flight); per-block
fp16 scales stay raw. The fused GEMV decodes 256 codes per stream-step into an L1 buffer and
dots them with Q8_0 activations (AVX-512 VNNI), so decoded weights never touch memory. Encoder
here; decoder and benchmark harness in benchmarks/v0.2/compress/entropy_bench.c.
"""

import heapq

import numpy as np

# ---------------------------------------------------------------- codes from GGUF tensors


def q8_0_split(raw):
    """Q8_0 raw bytes (rows, nb*34) -> (d fp16 (rows, nb), qs int8 (rows, nb*32))."""
    rows = raw.shape[0]
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, -1, 34)
    d = b[:, :, :2].copy().view(np.float16)[..., 0]
    qs = b[:, :, 2:].copy().view(np.int8).reshape(rows, -1)
    return d, qs


def q4_0_codes(raw):
    """Q4_0 raw (rows, nb*18) -> nibble codes 0..15 (rows, nb*32) in weight order."""
    rows = raw.shape[0]
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, -1, 18)[:, :, 2:]
    return np.concatenate([b & 0xF, b >> 4], axis=2).reshape(rows, -1)


def q4_k_codes(raw):
    """Q4_K raw (rows, nb*144) -> nibble codes 0..15 (rows, nb*256) (order within block irrelevant
    for order-0 statistics)."""
    rows = raw.shape[0]
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, -1, 144)[:, :, 16:]
    return np.concatenate([b & 0xF, b >> 4], axis=2).reshape(rows, -1)


def q6_k_codes(raw):
    """Q6_K raw (rows, nb*210) -> 6-bit codes 0..63 (rows, nb*256) in ggml's weight order."""
    rows = raw.shape[0]
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, -1, 210)
    out = []
    for h in range(2):
        ql, qh = b[:, :, 64 * h : 64 * h + 64], b[:, :, 128 + 32 * h : 160 + 32 * h]
        a, c = ql[:, :, :32], ql[:, :, 32:]
        out += [(a & 0xF) | ((qh & 3) << 4), (c & 0xF) | (((qh >> 2) & 3) << 4),
                (a >> 4) | (((qh >> 4) & 3) << 4), (c >> 4) | (((qh >> 6) & 3) << 4)]  # fmt: skip
    return np.concatenate(out, axis=2).reshape(rows, -1)


CODES = {"Q8_0": lambda r: q8_0_split(r)[1].view(np.uint8), "Q4_0": q4_0_codes, "Q4_K": q4_k_codes, "Q6_K": q6_k_codes}
BITS = {"Q8_0": 8, "Q4_0": 4, "Q4_K": 4, "Q6_K": 6}
BPW = {"Q8_0": 8.5, "Q4_0": 4.5, "Q4_K": 4.5, "Q6_K": 6.5625}


# ---------------------------------------------------------------- entropy / Huffman


def histogram(codes, nsym=256):
    return np.bincount(np.asarray(codes, dtype=np.uint8).reshape(-1), minlength=nsym).astype(np.int64)


def entropy(hist):
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_lengths(hist, max_len=12):
    """Code lengths of a Huffman code limited to `max_len` bits (heuristic limit: lengths above
    the limit are clamped and the Kraft sum repaired by lengthening the cheapest short codes)."""
    n = len(hist)
    f = np.maximum(hist.astype(np.float64), 0)
    used = np.nonzero(f)[0]
    lens = np.zeros(n, dtype=np.int64)
    if len(used) == 1:
        lens[used] = 1
        return lens
    heap = [(float(f[i]), int(i), (int(i),)) for i in used]
    heapq.heapify(heap)
    uid = n
    while len(heap) > 1:
        fa, _, a = heapq.heappop(heap)
        fb, _, b = heapq.heappop(heap)
        for s in a + b:
            lens[s] += 1
        heapq.heappush(heap, (fa + fb, uid, a + b))
        uid += 1
    if lens.max() > max_len:
        lens[used] = np.minimum(lens[used], max_len)
        kraft = lambda: sum(2.0 ** -lens[i] for i in used)  # noqa: E731
        order = sorted(used, key=lambda i: f[i])  # rarest first
        while kraft() > 1.0 + 1e-12:
            for i in order:  # lengthen the rarest code still below the limit
                if lens[i] < max_len:
                    lens[i] += 1
                    break
    return lens


def canonical_codes(lens):
    """Canonical Huffman codes (MSB-first values) for code lengths."""
    codes = np.zeros(len(lens), dtype=np.int64)
    code, prev = 0, 0
    for i in sorted(np.nonzero(lens)[0], key=lambda i: (lens[i], i)):
        code <<= int(lens[i]) - prev
        prev = int(lens[i])
        codes[i] = code
        code += 1
    return codes


def avg_len(hist, lens):
    return float((hist * lens).sum() / hist.sum())


def decode_table(lens, max_len=12):
    """4096-entry LSB-first decode table: entry = symbol | length << 8 (bit-reversed canonical codes,
    so the decoder peeks the low `max_len` bits of a little-endian bit buffer)."""
    codes = canonical_codes(lens)
    tab = np.zeros(1 << max_len, dtype=np.uint16)
    for s in np.nonzero(lens)[0]:
        ln = int(lens[s])
        rev = int(f"{int(codes[s]):0{ln}b}"[::-1], 2)
        for hi in range(1 << (max_len - ln)):
            tab[rev | (hi << ln)] = s | (ln << 8)
    return tab


def encode_stream(sym, lens):
    """Symbols -> LSB-first bit stream (bytes, padded with 8 zero bytes for the decoder's reads)."""
    codes = canonical_codes(lens)
    ln = lens[sym]
    rev = np.zeros(len(lens), dtype=np.uint64)
    for s in np.nonzero(lens)[0]:
        rev[s] = int(f"{int(codes[s]):0{int(lens[s])}b}"[::-1], 2)
    vals = rev[sym]
    pos = np.concatenate([[0], np.cumsum(ln)[:-1]]).astype(np.int64)
    nbits = int(ln.sum())
    out = np.zeros((nbits + 7) // 8 + 16, dtype=np.uint8)
    # scatter each code (<= 12 bits) into at most 3 bytes
    for k in range(3):
        byte = (pos >> 3) + k
        sh = (pos & 7) - 8 * k
        part = np.where(sh >= 0, vals << sh.clip(0).astype(np.uint64), vals >> (-sh).clip(0).astype(np.uint64))
        np.bitwise_or.at(out, byte, (part & 0xFF).astype(np.uint8))
    return out.tobytes(), nbits


def decode_stream(buf, n, tab, max_len=12):
    """Reference decoder (Python) for encode_stream."""
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    for k in range(n):
        w = int.from_bytes(b[pos >> 3 : (pos >> 3) + 3].tobytes(), "little") >> (pos & 7)
        e = int(tab[w & ((1 << max_len) - 1)])
        out[k] = e & 0xFF
        pos += e >> 8
    return out


def encode_q8h(qs, lens, streams=4):
    """Q8_0 codes (rows, K) uint8 -> (offsets u64 (rows*streams + 1,) in bytes, stream bytes).
    Each row is split into `streams` contiguous segments of K/streams codes, one bit stream each;
    every stream starts on an 8-byte boundary."""
    rows, k = qs.shape
    seg = k // streams
    parts, offs, pos = [], [0], 0
    for r in range(rows):
        for s in range(streams):
            b, _ = encode_stream(qs[r, s * seg : (s + 1) * seg], lens)
            b = b[: len(b) - 16]
            b += bytes((-len(b)) % 8)
            parts.append(b)
            pos += len(b)
            offs.append(pos)
    return np.array(offs, dtype=np.uint64), b"".join(parts) + bytes(16)


def tensor_report(t, fmt):
    """{entropy, huffman12 avg length, bits, coded bpw (codes + raw scales), saving} of one GGUF tensor."""
    raw = t.data
    h = np.zeros(256, dtype=np.int64)
    for a in range(0, raw.shape[0], 2048):
        h += histogram(CODES[fmt](np.asarray(raw[a : a + 2048])))
    h = h[: 1 << BITS[fmt]]
    H = entropy(h)
    L = avg_len(h, huffman_lengths(h, 12))
    side = BPW[fmt] - BITS[fmt]
    return {"fmt": fmt, "n": int(h.sum()), "entropy": H, "huff12": L, "bits": BITS[fmt],
            "bpw": BPW[fmt], "bpw_entropy": H + side, "bpw_huff12": L + side}  # fmt: skip
