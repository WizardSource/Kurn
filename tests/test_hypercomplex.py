"""Hypercomplex layers (integration/hypercomplex): the algebra and expansion references, and (with a llama.cpp
checkout carrying the KURN buffer type, KURN_LLAMA_CPP) the compiled kernels against kurn's real kernels."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from kurn import hypercomplex as hc
from kurn.targets import TARGETS
from kurn.toolchain import cpu_flags

from conftest import ROOT

HC = ROOT / "integration" / "hypercomplex"
rng = np.random.default_rng(7)


def test_hamilton_is_the_quaternion_product():
    i, j, k, one = (np.eye(4)[n] for n in (1, 2, 3, 0))
    assert np.allclose(hc.hamilton(i, j), k) and np.allclose(hc.hamilton(j, k), i) and np.allclose(hc.hamilton(k, i), j)
    assert np.allclose(hc.hamilton(i, i), -one) and np.allclose(hc.hamilton(i, j), -hc.hamilton(j, i))
    a, b, c = rng.standard_normal((3, 4))
    assert np.allclose(hc.hamilton(hc.hamilton(a, b), c), hc.hamilton(a, hc.hamilton(b, c)))  # associative
    assert np.isclose(np.linalg.norm(hc.hamilton(a, b)), np.linalg.norm(a) * np.linalg.norm(b))  # normed


@pytest.mark.parametrize("D", [2, 4])
def test_left_matrix_and_component_table(D):
    for _ in range(5):
        w, x = rng.standard_normal((2, D))
        ref = hc.hamilton(w, x) if D == 4 else np.array([w[0] * x[0] - w[1] * x[1], w[0] * x[1] + w[1] * x[0]])
        assert np.allclose(hc.left_matrix(w) @ x, ref)
    # every row and column of the component table is a permutation (a Latin square): p = o ^ q
    T = np.array([[hc.component(o, q) for q in range(D)] for o in range(D)])
    assert all(sorted(r) == list(range(D)) for r in T) and all(sorted(c) == list(range(D)) for c in T.T)


@pytest.mark.parametrize("D", [2, 4])
def test_block_planar_expansion_is_the_layer(D):
    Nq, Kq = 64, 96
    W = rng.standard_normal((D, Nq, Kq))
    x = rng.standard_normal((Kq, D))
    y = hc.layer(W, x)
    E = hc.expand(W)
    assert E.shape == (D * Nq, D * Kq)
    assert np.allclose(hc.from_block_planar(E @ hc.to_block_planar(x), D), y)
    # every 32-value block of an expanded row is one weight component of one row, possibly negated
    for R in (0, 33, D * 32 + 5):
        for b in range(D * Kq // 32):
            blk = E[R, 32 * b:32 * b + 32]
            n = (R // (32 * D)) * 32 + R % 32
            o, q, kb = (R % (32 * D)) // 32, b % D, b // D
            assert np.array_equal(blk, hc.sign_table(D)[o, q] * W[hc.component(o, q), n, 32 * kb:32 * kb + 32])


def test_quantized_compact_expands_to_a_q8_0_matrix():
    D, Nq, Kq = 4, 32, 64
    W = rng.standard_normal((D, Nq, Kq)).astype(np.float32)
    q, d = hc.quantize_q8_0(W)
    E = hc.expand(hc.dequantize_q8_0(q, d).astype(np.float64))
    # re-quantizing the expansion per 32-value block reproduces it exactly: the expansion is a Q8_0 matrix
    q2, d2 = hc.quantize_q8_0(E.astype(np.float32))
    assert np.array_equal(hc.dequantize_q8_0(q2, d2), E.astype(np.float32))
    c, e = hc.param_bytes(D, Nq, Kq, "q8")
    assert e == 4 * c


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    ok = (d / "ggml" / "src" / "ggml-cpu" / "kurn" / "kurn-buft.cpp").exists() and (d / "build" / "bin" / "libggml-cpu.so").exists()
    return d if ok else None


HAVE_VNNI512 = TARGETS["avx512_vnni"].requires <= cpu_flags()


@pytest.mark.skipif(_llama_dir() is None, reason="no llama.cpp checkout with the kurn buffer type (KURN_LLAMA_CPP)")
@pytest.mark.skipif(not HAVE_VNNI512 or shutil.which("g++") is None, reason="needs g++ and an AVX-512 VNNI CPU")
def test_compact_kernels_match_kurns_real_kernels(tmp_path):
    d = _llama_dir()
    exe = tmp_path / "hc_bench"
    flags = ["-O2", "-march=native", "-mno-avx512fp16", "-fopenmp"]
    subprocess.run(["g++", *flags, "-c", str(HC / "kurn-hc.cpp"), "-o", str(tmp_path / "kurn-hc.o")], check=True)
    subprocess.run(
        ["gcc", *flags, f"-I{HC}", f"-I{d}/ggml/include", str(HC / "hc_bench.c"), str(tmp_path / "kurn-hc.o"), f"-L{d}/build/bin",
         "-lggml", "-lggml-base", "-lggml-cpu", "-lm", "-lstdc++", f"-Wl,-rpath,{d}/build/bin", "-o", str(exe)],
        check=True,
    )
    env = dict(os.environ, OMP_WAIT_POLICY="passive")
    r = subprocess.run([str(exe), "quant"], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout
    ms = "1 2 3 4 5 6 7 8 9 16 31 32 64"
    for D, kq in ((4, "512"), (2, "1024")):
        for exact in ("1", "0"):
            r = subprocess.run([str(exe), "check", str(D), "q8", "512", kq, ms, "8"], capture_output=True, text=True, timeout=1200,
                               env=dict(env, GGML_KURN_EXACT=exact))
            assert r.returncode == 0, (D, exact, r.stdout[-3000:])
    r = subprocess.run([str(exe), "check", "4", "f32", "256", "256", "1 8 64", "8"], capture_output=True, text=True, timeout=1200, env=env)
    assert r.returncode == 0, r.stdout[-3000:]
    r = subprocess.run([str(exe), "mlp", "4", "q8", "256", "768", "3", "8", "8", "0", "check"], capture_output=True, text=True, timeout=1200,
                       env=dict(env, GGML_KURN_EXACT="1"))
    assert r.returncode == 0 and "==" in r.stdout, r.stdout[-3000:]
