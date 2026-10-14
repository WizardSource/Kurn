"""Ternary bitmap + signs ("tbm", KTBM layout version 1): encoder / decoder for the GPU kernels' tests and harnesses.

Implements the settled v1 byte layout from the CPU side's spec, independently of the CPU code (kurn.ternary), so the
two can be checked against each other once both are in one tree:

    header (64 B: "KTBM", version 1, K, N, ngroups, nrec_k, used bytes, dense records)
    group table ((ngroups + 1) x u64, offsets of each 16-row group's first record relative to `data`, padded to 64 B)
    records, per (16-row group, 256-value block), in K order within a group:
        d[16] fp16 | bitmap[64] u64 | flag u8 (bit 0: dense) | signs ceil(nnz / 8) bytes
    64 bytes of zero slack

Bitmap word i = kg * 8 + kk covers values v = 32 kg + 4 kk + j (j = 0..3) of the 16 rows: bit b is row b / 4, j = b % 4.
Signs: for words 0..63 in order, one bit per set bitmap bit in ascending bit order (1 = +1, 0 = -1), packed LSB-first,
continuous across words; each record's sign stream starts on a byte boundary. Values are w * d with w in {-1, 0, +1}
and d one fp16 per (row, 256 values): exactly TQ2_0's values and scale.

The GPU kernel does not read v1 in its inner loop: kg_prepare repacks it on the device into fragment-ordered presence
bits plus per-lane compacted signs (see kurn.gpu.mma, "tbm").
"""

import struct

MAGIC, VERSION = b"KTBM", 1
REC_FIXED = 545  # d[16] + bitmap[64 x u64] + flag


def _pc(x):
    return bin(x).count("1")


def data_offset(ngroups):
    return 64 + (8 * (ngroups + 1) + 63) // 64 * 64


def encode(w, d, n, k):
    """w: n rows of k values in {-1, 0, 1}; d: n rows of k / 256 fp16 bit patterns -> the v1 buffer (bytes)."""
    assert k % 256 == 0
    ngroups, nrec = (n + 15) // 16, k // 256
    data, table, dense = bytearray(), [], 0
    for g in range(ngroups):
        table.append(len(data))
        for p in range(nrec):
            rec = bytearray()
            for r in range(16):
                row = 16 * g + r
                rec += struct.pack("<H", d[row][p] if row < n else 0)
            words, bits, off, flag = [], [], 0, 0
            for i in range(64):
                kg, kk = divmod(i, 8)
                word, sg = 0, []
                for b in range(64):
                    row, v = 16 * g + b // 4, 256 * p + 32 * kg + 4 * kk + b % 4
                    if row < n and w[row][v]:
                        word |= 1 << b
                        sg.append(1 if w[row][v] > 0 else 0)
                if _pc(word) > 64 - off % 8:
                    flag = 1
                words.append(word)
                bits += sg
                off += len(sg)
            rec += struct.pack("<64Q", *words) + bytes([flag])
            sb = bytearray((len(bits) + 7) // 8)
            for i, bt in enumerate(bits):
                sb[i // 8] |= bt << (i % 8)
            rec += sb
            dense += flag
            data += rec
    table.append(len(data))
    tb = data_offset(ngroups) - 64
    used = 64 + tb + len(data) + 64
    hdr = MAGIC + struct.pack("<I6q", VERSION, k, n, ngroups, nrec, used, dense)
    hdr += bytes(64 - len(hdr))
    tab = struct.pack(f"<{len(table)}Q", *table)
    return bytes(hdr + tab + bytes(tb - len(tab)) + data + bytes(64))


def header(buf):
    if buf[:4] != MAGIC:
        raise ValueError("not a KTBM buffer")
    ver, k, n, ngroups, nrec, used, dense = struct.unpack_from("<I6q", buf, 4)
    return {"version": ver, "K": k, "N": n, "ngroups": ngroups, "nrec_k": nrec, "used": used, "dense": dense}


def decode(buf):
    """-> (w, d): the reference decoder of the spec (pdep of the sign window into each bitmap word)."""
    h = header(buf)
    if h["version"] != VERSION:
        raise ValueError(f"KTBM version {h['version']}")
    n, k, ngroups, nrec = h["N"], h["K"], h["ngroups"], h["nrec_k"]
    base = data_offset(ngroups)
    w = [[0] * k for _ in range(n)]
    d = [[0] * nrec for _ in range(n)]
    for g in range(ngroups):
        pos = base + struct.unpack_from("<Q", buf, 64 + 8 * g)[0]
        for p in range(nrec):
            ds = struct.unpack_from("<16H", buf, pos)
            words = struct.unpack_from("<64Q", buf, pos + 32)
            signs = pos + REC_FIXED
            off = 0
            for i, P in enumerate(words):
                for b in range(64):
                    if not (P >> b) & 1:
                        continue
                    s = (buf[signs + off // 8] >> (off % 8)) & 1
                    off += 1
                    row, v = 16 * g + b // 4, 256 * p + 32 * (i // 8) + 4 * (i % 8) + b % 4
                    if row < n:
                        w[row][v] = 1 if s else -1
            for r in range(16):
                if 16 * g + r < n:
                    d[16 * g + r][p] = ds[r]
            pos = signs + (off + 7) // 8
    return w, d


def random_ternary(rng, n, k, zero=0.4):
    """Ternary values with zero fraction `zero` and TQ2_0-like fp16 scales (positive, 2^-6 .. 2^-1)."""
    w = [[0 if rng.random() < zero else rng.choice((-1, 1)) for _ in range(k)] for _ in range(n)]
    d = [[((9 + rng.randrange(5)) << 10) | rng.randrange(1024) for _ in range(k // 256)] for _ in range(n)]
    return w, d


def to_tq2_0(w, d, n, k):
    """The same values as ggml TQ2_0 blocks (qs[64] then fp16 d; value v at byte (v/128)*32 + v%32, bits 2*((v%128)/32))."""
    out = bytearray()
    for r in range(n):
        for b in range(k // 256):
            qs = bytearray(64)
            for v in range(256):
                qs[(v // 128) * 32 + v % 32] |= (w[r][256 * b + v] + 1) << (2 * ((v % 128) // 32))
            out += qs + struct.pack("<H", d[r][b])
    return bytes(out)


def problem(rng, n, k, extreme=False):
    """Random tbm weights; `extreme`: almost no zeros (long sign streams, dense records)."""
    w, d = random_ternary(rng, n, k, 0.01 if extreme else rng.choice((0.25, 0.4, 0.6)))
    return encode(w, d, n, k)
