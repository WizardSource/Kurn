"""kurn.entropy: entropy, length-limited Huffman, bit-stream roundtrip, code extraction."""

import pytest

from kurn import entropy as en

np = pytest.importorskip("numpy")


def test_entropy_bounds():
    assert en.entropy(np.ones(256, dtype=np.int64)) == pytest.approx(8.0)
    assert en.entropy(np.array([5, 0, 0])) == 0.0


@pytest.mark.parametrize("scale", [3.0, 40.0, 200.0])
def test_huffman_limited_and_kraft(scale):
    rng = np.random.default_rng(0)
    sym = np.clip(np.round(rng.standard_normal(200000) * scale), -128, 127).astype(np.int64) & 0xFF
    h = en.histogram(sym.astype(np.uint8))
    lens = en.huffman_lengths(h, 12)
    used = h > 0
    assert lens[used].max() <= 12 and (lens[~used] == 0).all()
    assert sum(2.0 ** -lens[used]) <= 1.0 + 1e-12
    H, L = en.entropy(h), en.avg_len(h, lens)
    assert H <= L < H + 1.0


def test_stream_roundtrip():
    rng = np.random.default_rng(1)
    sym = (np.round(rng.standard_normal(5000) * 30).astype(np.int64) & 0xFF).astype(np.uint8)
    h = en.histogram(sym)
    lens = en.huffman_lengths(h, 12)
    buf, nbits = en.encode_stream(sym.astype(np.int64), lens)
    assert nbits == int(lens[sym].sum())
    np.testing.assert_array_equal(en.decode_stream(buf, len(sym), en.decode_table(lens)), sym)
    offs, data = en.encode_q8h(sym[:4096].reshape(4, 1024).astype(np.int64), lens, streams=4)
    assert len(offs) == 17 and all(o % 8 == 0 for o in offs)


def test_q8_0_split_and_q4_codes():
    rng = np.random.default_rng(2)
    d = rng.standard_normal((3, 2)).astype(np.float16)
    qs = rng.integers(-127, 128, (3, 64)).astype(np.int8)
    raw = np.concatenate([np.concatenate([d[:, b : b + 1].view(np.uint8).reshape(3, 2), qs[:, 32 * b : 32 * b + 32].view(np.uint8)], 1)
                          for b in range(2)], 1)  # fmt: skip
    d2, q2 = en.q8_0_split(raw)
    np.testing.assert_array_equal(d2, d)
    np.testing.assert_array_equal(q2, qs)
    nib = rng.integers(0, 16, (2, 32)).astype(np.uint8)
    blk = np.concatenate([np.zeros((2, 2), np.uint8), nib[:, :16] | (nib[:, 16:] << 4)], 1)
    np.testing.assert_array_equal(en.q4_0_codes(blk), nib)


def test_q6_k_codes_match_ggml_dequant():
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, (2, 210), dtype=np.uint8)
    raw[:, 208:210] = np.array([0x00, 0x3C], dtype=np.uint8)  # d = 1.0
    raw[:, 192:208] = 1  # all sub-block scales 1 -> value = q - 32
    w = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.Q6_K)
    np.testing.assert_array_equal(en.q6_k_codes(raw).astype(np.int64) - 32, w.astype(np.int64))
