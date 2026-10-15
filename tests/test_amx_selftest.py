"""AMX startup self-test (integration/llama.cpp/ggml-kurn/kurn-amx-selftest.cpp) on a built llama.cpp checkout.

Needs a checkout with apply.sh applied and built (KURN_LLAMA_CPP, default ~/src/llama-kurn) and an AMX CPU. The
outcomes are read from kurn's log (GGML_KURN_VERBOSE=1); GGML_KURN_AMX_SELFTEST_FAULT injects a wrong tile into the
plain phase (test-only)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/ggml-kurn/kurn-amx-selftest.cpp")
INTEG = ROOT / "integration" / "llama.cpp"


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    if not (d / "ggml" / "src" / "ggml-cpu" / "kurn" / "kurn-amx-selftest.cpp").exists():
        return None
    if not (d / "build" / "bin" / "libggml-cpu.so").exists():
        return None
    return d


def _have_amx():
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return all(f in flags for f in ("amx_tile", "amx_int8", "amx_bf16"))


pytestmark = [
    pytest.mark.skipif(_llama_dir() is None, reason="no built llama.cpp checkout with the AMX self-test (KURN_LLAMA_CPP)"),
    pytest.mark.skipif(not _have_amx() or shutil.which("gcc") is None, reason="needs gcc and an AMX CPU"),
]


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    d = _llama_dir()
    exe = tmp_path_factory.mktemp("amx") / "test_kurn_buft"
    subprocess.run(
        ["gcc", "-O2", f"-I{d}/ggml/include", str(INTEG / "test_kurn_buft.c"), f"-L{d}/build/bin", "-lggml", "-lggml-base",
         "-lggml-cpu", "-lm", f"-Wl,-rpath,{d}/build/bin", "-o", str(exe)],
        check=True,
    )
    return exe


def _run(exe, *args, **env):
    e = dict(os.environ, GGML_KURN_VERBOSE="1", VERBOSE="1")
    for k in ("GGML_KURN_AMX", "GGML_KURN_AMX_SELFTEST", "GGML_KURN_AMX_SELFTEST_ACTION", "GGML_KURN_AMX_SELFTEST_FAULT"):
        e.pop(k, None)
    e.update(env)
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=600, env=e)
    return r.returncode, r.stdout + r.stderr


def _prefill(exe, **env):
    return _run(exe, "stress", "q8_0", "1024", "256", "64", "2", "3", **env)


def test_plain_phase_wrong_once_is_not_broken_amx(checker):
    # a wrong tile that is right on recompute is a lost tile (preemption), not broken AMX: AMX stays on, guarded
    rc, out = _prefill(checker, GGML_KURN_AMX_SELFTEST_FAULT="plain-transient")
    assert rc == 0, out
    assert "AMX kernels off" not in out, out
    assert "AMX self-test" in out and "this machine does not keep AMX tile data" in out, out
    assert "prefill: AMX-BF16" in out, out


def test_plain_phase_repeating_wrong_result_turns_amx_off(checker):
    rc, out = _prefill(checker, GGML_KURN_AMX_SELFTEST_FAULT="plain-persistent")
    assert rc == 0, out
    assert "returned wrong results again on recompute" in out and "AMX kernels off" in out, out
    assert "prefill: verify kernels" in out, out


def test_no_fault_never_turns_amx_off(checker):
    for _ in range(20):
        rc, out = _prefill(checker)
        assert rc == 0, out
        assert "AMX kernels off" not in out, out


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
@pytest.mark.parametrize("env", [{"GGML_KURN_AMX_SELFTEST_FAULT": "plain-persistent"}, {"GGML_KURN_AMX": "0"}])
def test_hypercomplex_follows_kurns_amx_decision(tmp_path, env):
    # kurn-hc must not run AMX-BF16 when the KURN buffer type does not: compact != expanded at 32 / 64 columns otherwise
    d = _llama_dir()
    hc = ROOT / "integration" / "hypercomplex"
    flags = ["-O2", "-march=native", "-mno-avx512fp16", "-fopenmp"]
    subprocess.run(["g++", *flags, "-c", str(hc / "kurn-hc.cpp"), "-o", str(tmp_path / "kurn-hc.o")], check=True)
    exe = tmp_path / "hc_bench"
    subprocess.run(
        ["gcc", *flags, f"-I{hc}", f"-I{d}/ggml/include", str(hc / "hc_bench.c"), str(tmp_path / "kurn-hc.o"), f"-L{d}/build/bin",
         "-lggml", "-lggml-base", "-lggml-cpu", "-lm", "-lstdc++", f"-Wl,-rpath,{d}/build/bin", "-o", str(exe)],
        check=True,
    )
    e = dict(os.environ, OMP_WAIT_POLICY="passive", GGML_KURN_EXACT="0", **env)
    r = subprocess.run([str(exe), "check", "4", "q8", "512", "512", "16 32 64", "8"], capture_output=True, text=True, timeout=600, env=e)
    assert r.returncode == 0 and "0 failed" in r.stdout, r.stdout[-3000:]
