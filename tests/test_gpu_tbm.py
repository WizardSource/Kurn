"""Ternary bitmap + signs ("tbm", KTBM layout v1) on the GPU tensor-core engine: the v1 encoder / decoder against the
spec, the device repack at load plus the packed-direct decoder on the CPU warp emulator (exact vs the independent C
decoder's reference, bit-identical with the TQ2_0 kernel on the same values), bug mutants, ptxas and SASS."""

import os
import random
import shutil
import struct
import subprocess
import tempfile

import pytest

from kurn.gpu import emu, mma, ref, tbm, toolchain
from kurn.gpu import spec as gs
from kurn.spec import SpecError

emu_ok = pytest.mark.skipif(bool(toolchain.cxx_problem()), reason=f"CPU emulator unavailable: {toolchain.cxx_problem()}")
needs_nvcc = pytest.mark.skipif(not toolchain.nvcc(), reason="nvcc not installed")


def test_v1_bytes_match_the_spec_by_hand():
    # 16 rows x 256: row 0 value 36 = -1, row 3 values 37 = +1 and 38 = -1. Value v = 32 kg + 4 kk + j -> word 8 kg + kk,
    # bit 4 row + j: all three land in word 9 (bits 0, 13, 14); signs in bit order: -1, +1, -1 -> 0b010
    w = [[0] * 256 for _ in range(16)]
    w[0][36], w[3][37], w[3][38] = -1, 1, -1
    d = [[0x3C00 + r] for r in range(16)]
    b = tbm.encode(w, d, 16, 256)
    data = 128  # 64 + round_up(8 * 2, 64)
    assert b[:4] == b"KTBM" and struct.unpack_from("<I6q", b, 4) == (1, 256, 16, 1, 1, data + 546 + 64, 0)
    assert struct.unpack_from("<2Q", b, 64) == (0, 546) and len(b) == data + 546 + 64
    assert struct.unpack_from("<16H", b, data) == tuple(0x3C00 + r for r in range(16))
    words = struct.unpack_from("<64Q", b, data + 32)
    assert words[9] == (1 << 0) | (1 << 13) | (1 << 14) and sum(words) == words[9]
    assert b[data + 544] == 0 and b[data + 545] == 0b010 and b[data + 546 :] == bytes(64)


def test_v1_roundtrip_dense_and_tails():
    rng = random.Random(3)
    for zero, n, k in ((0.4, 37, 512), (0.0, 16, 256), (1.0, 32, 256), (0.01, 48, 512)):
        w, d = tbm.random_ternary(rng, n, k, zero)
        b = tbm.encode(w, d, n, k)
        assert tbm.decode(b) == (w, d)
        h = tbm.header(b)
        assert (h["N"], h["K"], h["ngroups"], h["used"]) == (n, k, (n + 15) // 16, len(b))
        if zero == 0.01:
            assert h["dense"] > 0  # some word starts with fewer window bits than it has nonzeros
    with pytest.raises(ValueError):
        tbm.header(b"XXXX" + bytes(60))


@emu_ok
def test_c_encoder_and_decoder_match_python(tmp_path):
    """The GPU harness generates tbm weights with kref_tbm_encode (kurn_gpu_ref.h) and checks kernels with kref_tbm_row:
    both must agree with kurn.gpu.tbm byte for byte / value for value."""
    rng = random.Random(9)
    n, k = 40, 512
    w, d = tbm.random_ternary(rng, n, k, 0.01)  # dense records included
    w[3] = [0] * k  # an all-zero row
    src = tmp_path / "t.cpp"
    src.write_text(
        '#include <cstdio>\n#include "kurn_gpu_ref.h"\nint main() {\n'
        f"  const int N = {n}, K = {k};\n  std::vector<int8_t> w(N * K); std::vector<uint16_t> d(N * (K / 256));\n"
        "  for (auto &v : w) { int x; if (scanf(\"%d\", &x) != 1) return 1; v = (int8_t)x; }\n"
        "  for (auto &v : d) { int x; if (scanf(\"%d\", &x) != 1) return 1; v = (uint16_t)x; }\n"
        "  std::vector<uint8_t> out(kref_tbm_max_bytes(N, K));\n  size_t used = kref_tbm_encode(w.data(), d.data(), N, K, out.data());\n"
        "  fwrite(out.data(), 1, used, stdout);\n  std::vector<double> row(K); int bad = 0;\n"
        "  for (int r = 0; r < N; r++) { kref_tbm_row(out.data(), K, r, row.data());\n"
        "    for (int i = 0; i < K; i++) bad += row[i] != (w[r * K + i] ? w[r * K + i] * kref_h2d(d[r * (K / 256) + i / 256]) : 0.0); }\n"
        "  fprintf(stderr, \"%d\", bad);\n  return 0;\n}\n"
    )  # fmt: skip
    exe = tmp_path / "t"
    inc = os.path.dirname(toolchain.data_path("kurn_gpu_ref.h"))
    subprocess.run([*toolchain.cxx(), "-std=c++17", "-O1", "-I", inc, str(src), "-o", str(exe)], check=True)
    data = " ".join(str(v) for row in w for v in row) + " " + " ".join(str(v) for row in d for v in row)
    r = subprocess.run([str(exe)], input=data.encode(), capture_output=True, check=True)
    assert r.stdout == tbm.encode(w, d, n, k) and r.stderr == b"0"


def test_spec_rules():
    c = gs.resolve({"op": "gemm", "weights": "tbm"})
    assert (c["wload"], c["pf"], c["bk"]) == ("reg", 2, 256)
    with pytest.raises(SpecError, match="gemv"):
        gs.resolve({"op": "gemv", "weights": "tbm"})
    with pytest.raises(SpecError, match="wload"):
        gs.resolve({"op": "gemm", "weights": "tbm", "wload": "smem"})
    assert "tbm" in gs.formats_for("gemm") and "tbm" not in gs.formats_for("gemv")
    from kurn.gpu.matrix import default_kernels

    names = set(default_kernels("tbm", "sm_80"))
    assert "default-gemv" not in names and {"default-mma8-reg", "default-mma16-reg"} <= names
    assert not names & {"default-mma8", "default-mma16"}  # the staged names would duplicate the packed tiles


CASES = [{}, {"pf": 1}, {"pf": 3}, {"xin": "f16", "stages": 3}, {"bm": 32, "bn": 16, "wm": 2, "wn": 2, "splitk": 4, "xin": "f16"},
         {"bm": 128, "wm": 4, "splitk": 1}, {"bm": 16, "wm": 1, "bn": 32, "wn": 4, "xin": "f16", "splitk": 2}]  # fmt: skip


@emu_ok
@pytest.mark.parametrize("ov", CASES, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "default")
def test_kernel_matches_reference(ov):
    c = gs.resolve({"op": "gemm", "weights": "tbm", **ov})
    for kw in ({"seed": 1}, {"seed": 2, "extreme": True, "sched": 7}, {"seed": 3, "sched": 13}):
        r = emu.run(c, **kw)
        assert r["ok"], (gs.label(c), kw, r)


def _y(c, W, x, n, k, m, src=None):
    exe = toolchain.emu_build(c, src)
    d = tempfile.mkdtemp(prefix="kurn-tbm-")
    try:
        with open(os.path.join(d, "params.txt"), "w") as fh:
            fh.write(f"{n} {k} {m} {c['weights']}\n")
        with open(os.path.join(d, "W.bin"), "wb") as fh:
            fh.write(W)
        with open(os.path.join(d, "X.bin"), "wb") as fh:
            fh.write(ref.pack_f32(x))
        r = subprocess.run([exe, d], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        with open(os.path.join(d, "Y.bin"), "rb") as fh:
            return fh.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@emu_ok
def test_bit_identical_with_the_tq2_0_kernel():
    """Same values and scales, same tile and split: the tbm kernel must reproduce the TQ2_0 packed kernel bit for bit
    (its decoder builds the same {-1, 0, +1} f16 fragments, and the epilogue is the same)."""
    rng = random.Random(5)
    n, k, m = 80, 1024, 7
    w, d = tbm.random_ternary(rng, n, k, 0.4)
    x = ref.activations(rng, k, m)
    for ov in ({"splitk": 1}, {"splitk": 2, "xin": "f16", "pf": 3}):
        ct = gs.resolve({"op": "gemm", "weights": "tbm", **ov})
        cq = gs.resolve({"op": "gemm", "weights": "tq2_0", "wload": "reg", **{"pf": 2, **ov}})
        assert _y(ct, tbm.encode(w, d, n, k), x, n, k, m) == _y(cq, tbm.to_tq2_0(w, d, n, k), x, n, k, m), ov


@emu_ok
@pytest.mark.parametrize("old, new, why", [
    ("(uint32_t)(x - cnt);", "(uint32_t)x;", "lane sign offset uses the inclusive scan"),
    ("a[nb & 1][nb >> 1] = e.x; a[nb & 1][2 + (nb >> 1)] = e.y;", "a[nb >> 1][nb & 1] = e.x; a[nb >> 1][2 + (nb & 1)] = e.y;",
     "nibbles placed in the wrong A registers"),
    ("sg >>= KURN_POPC(mn);", "sg >>= 4;", "sign stream advanced by 4 per nibble instead of its popcount"),
    ("if (wl & 1) pre += KURN_POPC(q[wl >> 1] & 0xFFFFu);", "", "odd windows skip the even window's signs"),
    ("const int j = q & 1, row = g + 8 * (q >> 1)", "const int j = q >> 1, row = g + 8 * (q & 1)", "repack nibble order"),
    ("kpre[wp][2 * lane + 1] = x - c1;", "kpre[wp][2 * lane + 1] = x - c0 - c1;", "repack: odd word's sign prefix misses the even word"),
    ("    if (sh) atomicOr(dst + k + 1, bits[k] >> (32u - sh));\n", "", "repack drops the bits that spill into the next word"),
])  # fmt: skip
def test_emulator_catches_tbm_bugs(old, new, why):
    c = gs.resolve({"op": "gemm", "weights": "tbm", "splitk": 1})
    from kurn.gpu.codegen import generate

    src = generate(c)
    assert old in src, old
    bad = src.replace(old, new, 1)
    orig = toolchain.emu_build
    emu.emu_build = lambda cc, s=None: orig(cc, bad)
    try:
        res = [emu.run(c, seed=s, sched=s) for s in (1, 4)]
        caught = not all(r["ok"] for r in res)
    except emu.EmuError:
        caught = True
    finally:
        emu.emu_build = orig
    assert caught, f"emulator missed: {why}"


@needs_nvcc
def test_covering_configs_do_not_spill_on_three_archs():
    from kurn.gpu.toolchain import build_many, ptxas_report, resource_rows

    cs = gs.covering_configs("gemm", "tbm")
    bad = []
    for c, rep in build_many(cs, lambda c: ptxas_report(c, ("sm_80", "sm_100", "sm_120"))):
        assert not isinstance(rep, Exception), rep
        bad += [(gs.label(c), r) for r in resource_rows(c, rep) if r["spill"]]
    assert len(cs) > 10 and not bad, bad[:5]


@needs_nvcc
def test_sass_packed_with_a_shared_table(tmp_path):
    from kurn.gpu.toolchain import nvcc_build

    c = gs.resolve({"op": "gemm", "weights": "tbm"})
    so, _ = nvcc_build(c, ("sm_80",), out_dir=str(tmp_path))
    dump = subprocess.run([os.path.join(os.path.dirname(toolchain.nvcc()), "cuobjdump"), "-sass", so], capture_output=True, text=True)
    if dump.returncode:
        pytest.skip("cuobjdump / nvdisasm unavailable")
    s = dump.stdout
    assert "HMMA.16816.F32" in s and "LDG.E.128.CONSTANT" in s and "LDS.64" in s and "POPC" in s and "SHFL.IDX" in s
    assert "LDL" not in s and "STL" not in s
    assert mma.smem_bytes(c) == c["stages"] * c["bn"] * c["bk"] * 2 + 2048  # activations + the nibble table
