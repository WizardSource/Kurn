# ruff: noqa: E501  (the fake nvcc and the program it writes are kept on single lines)
"""CUDA runtime discovery and the fail-fast preflight (kurn.gpu.cudaenv), on fake toolkit layouts.

The fake nvcc behaves like the real one where it matters: it applies NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS itself, its
implicit include / library paths are its own toolkit's targets/<arch>/{include,lib}, a missing header or -lcudart fails
with nvcc's message, and the program it "builds" reports CUDART_VERSION from the headers it found. Every invocation is
logged, so the tests can count builds and check what nvcc received.
"""

import json
import os
import sys
import textwrap

import pytest

from kurn.gpu import cudaenv as E
from kurn.gpu import toolchain

FAKE_NVCC = r"""#!{python}
import json, os, re, shlex, sys
top = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
ver = os.environ.get("FAKE_NVCC_VERSION", "{version}")
argv = sys.argv[1:]
with open(os.environ.get("FAKE_NVCC_LOG", os.devnull), "a") as fh:
    fh.write(json.dumps({{"argv": argv, "append": os.environ.get("NVCC_APPEND_FLAGS"), "prepend": os.environ.get("NVCC_PREPEND_FLAGS")}}) + "\n")
if argv[:1] == ["--version"]:
    print(f"Cuda compilation tools, release {{ver}}, V{{ver}}.1"); sys.exit(0)
if argv[:1] == ["--list-gpu-code"]:
    print("sm_80\nsm_86\nsm_89\nsm_90\nsm_100\nsm_120"); sys.exit(0)
args = shlex.split(os.environ.get("NVCC_PREPEND_FLAGS", "")) + argv + shlex.split(os.environ.get("NVCC_APPEND_FLAGS", ""))
incs, libs, i = [os.path.join(top, "targets", "{target}", "include")], [os.path.join(top, "targets", "{target}", "lib")], 0
while i < len(args):
    a = args[i]
    for pre, out in (("-I", incs), ("-isystem", incs), ("-L", libs)):
        if a == pre: out.append(args[i + 1]); i += 1; break
        if a.startswith(pre) and len(a) > len(pre): out.append(a[len(pre):]); break
    i += 1
src = next(a for a in args if a.endswith(".cu"))
hdr = 0
for h in re.findall(r"#include <(\w+\.h)>", open(src).read()):
    d = next((d for d in incs if os.path.isfile(os.path.join(d, h))), None)
    if d is None:
        print(f"<command-line>: fatal error: {{h}}: No such file or directory", file=sys.stderr); sys.exit(1)
    if h == "cuda_runtime.h":
        m = re.search(r"CUDART_VERSION\s+(\d+)", open(os.path.join(d, "cuda_runtime_api.h")).read())
        hdr = int(m.group(1))
for lib in ("cudart", "cublas"):
    if "-l" + lib in args and not any(os.path.exists(os.path.join(d, f"lib{{lib}}.so")) for d in libs):
        print(f"/usr/bin/ld: cannot find -l{{lib}}", file=sys.stderr); sys.exit(1)
out = args[args.index("-o") + 1]
drv = int(os.environ.get("FAKE_DRIVER", "0"))
fail = int(os.environ.get("FAKE_GPU_FAIL", "0"))
with open(out, "w") as fh:
    fh.write("#!/bin/sh\n")
    fh.write(f"if [ \"$1\" = run ] && [ {{fail}} = 1 ]; then echo '{{{{\"runtime\": {{hdr}}, \"driver\": {{drv}}, \"header\": {{hdr}}, \"ok\": 0, \"run_error\": \"no kernel image is available\"}}}}'; exit 1; fi\n")
    fh.write(f"echo '{{{{\"runtime\": {{hdr}}, \"driver\": {{drv}}, \"header\": {{hdr}}, \"ok\": 1, \"run_error\": \"\", \"cublas_ok\": 1}}}}'\n")
os.chmod(out, 0o755)
"""


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def runtime(d, cudart=12080, cublas=True):
    """Headers in d/include-ish and libraries in a sibling lib dir: returns (include dir, lib dir)."""
    inc, lib = d["inc"], d["lib"]
    for h in ("cuda_runtime.h", "cuda_fp16.h"):
        _write(os.path.join(inc, h), "/* fake */\n")
    _write(os.path.join(inc, "cuda_runtime_api.h"), f"#define CUDART_VERSION {cudart}\n")
    _write(os.path.join(lib, "libcudart.so"))
    _write(os.path.join(lib, "libcudart_static.a"))
    if cublas:
        _write(os.path.join(inc, "cublas_v2.h"))
        _write(os.path.join(lib, "libcublas.so"))
    return inc, lib


def toolkit(root, version="12.8", with_runtime=True, cudart=None):
    """A fake toolkit at root (bin/nvcc), with or without its own targets/<arch>/{include,lib}."""
    nv = os.path.join(root, "bin", "nvcc")
    _write(nv, FAKE_NVCC.format(python=sys.executable, version=version, target=E._target()))
    os.chmod(nv, 0o755)
    if with_runtime:
        major, minor = version.split(".")
        runtime({"inc": os.path.join(root, "targets", E._target(), "include"), "lib": os.path.join(root, "targets", E._target(), "lib")},
                cudart=cudart or int(major) * 1000 + int(minor) * 10)  # fmt: skip
    return nv


def monorepo(base, cudart=12080):
    """A third-party dir with include_no_implicit/ and lib/ (the split install)."""
    root = os.path.join(base, "mono", "third_party", "cuda")
    inc, lib = runtime({"inc": os.path.join(root, "include_no_implicit"), "lib": os.path.join(root, "lib")}, cudart=cudart)
    return root, inc, lib


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Isolate from the real toolkit: no CUDA env vars, no system roots, no GPU, a nvcc invocation log."""
    for v in E.ENV_VARS + ("KURN_GPU_ARCHS", "FAKE_DRIVER", "FAKE_GPU_FAIL", "FAKE_NVCC_VERSION"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(E, "SYSTEM_ROOTS", [])
    monkeypatch.setattr(E, "_gpu_present", lambda: False)
    log = tmp_path / "nvcc.log"
    monkeypatch.setenv("FAKE_NVCC_LOG", str(log))
    monkeypatch.setenv("KURN_CACHE_DIR", str(tmp_path / "cache"))
    toolchain.WARNED.clear()
    E.reset()

    def calls(compiles_only=True):
        if not log.exists():
            return []
        rows = [json.loads(ln) for ln in log.read_text().splitlines()]
        return [r for r in rows if not compiles_only or r["argv"][:1] not in (["--version"], ["--list-gpu-code"])]

    yield tmp_path, calls
    monkeypatch.undo()
    E.reset()


def test_flag_dirs_parsing():
    assert E.flag_dirs("-I/a -I /b -isystem /c --include-path=/d -L/e -L /f -O3") == (["/a", "/b", "/c", "/d"], ["/e", "/f"])
    assert E.flag_dirs("") == ([], [])


def test_header_version(tmp_path):
    _write(str(tmp_path / "cuda_runtime_api.h"), "#define CUDART_VERSION 12080\n")
    assert E.header_version(str(tmp_path)) == 12080 and E.cuda_version_str(12080) == "12.8" and E.cuda_version_str(13000) == "13.0"
    assert E.header_version(str(tmp_path / "nope")) is None


def test_normal_toolkit_needs_no_extra_flags(fake, monkeypatch):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8")))
    rep = E.preflight()
    assert rep["flags"] == [] and rep["how"].startswith("as configured") and rep["header"] == 12080
    res = E.resolve()
    assert res["include_how"] == "nvcc's toolkit" and res["lib_how"] == "nvcc's toolkit" and res["nvcc_how"] == "KURN_NVCC"


def test_split_install_fails_fast_with_one_actionable_message(fake, monkeypatch):
    tmp, calls = fake
    nv = toolkit(str(tmp / "cuda-12.8"), with_runtime=False)
    monorepo(str(tmp))  # exists, but nothing points at it
    monkeypatch.setenv("KURN_NVCC", nv)
    with pytest.raises(E.CudaToolchainError) as ei:
        E.preflight()
    msg = str(ei.value)
    assert "missing: cuda_runtime.h + cuda_fp16.h, libcudart" in msg and "fatal error: cuda_runtime.h" in msg
    assert os.path.join(str(tmp / "cuda-12.8"), "targets", E._target(), "include") in msg  # the paths searched
    for var in ("KURN_CUDA_INCLUDE", "KURN_CUDA_LIB", "CUDA_HOME", "NVCC_APPEND_FLAGS"):
        assert var in msg
    n = len(calls())
    assert n == 1  # nothing discovered: one compile attempt
    with pytest.raises(E.CudaToolchainError):  # later builds fail at once, without compiling again
        E.build_flags()
    assert len(calls()) == n


@pytest.mark.parametrize("bogus_first", [False, True])
def test_kurn_cuda_include_and_lib(fake, monkeypatch, capsys, bogus_first):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    _, inc, lib = monorepo(str(tmp))
    monkeypatch.setenv("KURN_CUDA_INCLUDE", (str(tmp / "nonexistent") + os.pathsep if bogus_first else "") + inc)
    monkeypatch.setenv("KURN_CUDA_LIB", lib)
    rep = E.preflight()
    assert rep["flags"] == ["-I", inc, "-L", lib, f"-Xlinker=-rpath={lib}"]
    assert E.resolve()["include_how"] == "KURN_CUDA_INCLUDE" and E.resolve()["lib_how"] == "KURN_CUDA_LIB"
    if bogus_first:
        assert "is not a directory (ignored)" in capsys.readouterr().err
    assert len(calls()) == 1


@pytest.mark.parametrize("var", ["CUDA_HOME", "CUDA_PATH"])
def test_cuda_home_with_include_no_implicit(fake, monkeypatch, capsys, var):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    root, inc, lib = monorepo(str(tmp))
    monkeypatch.setenv(var, root)
    rep = E.preflight()
    assert rep["flags"] == ["-I", inc, "-L", lib, f"-Xlinker=-rpath={lib}"] and rep["include"] == inc
    assert var in rep["how"] and "include_no_implicit" in rep["how"]
    assert "not on nvcc's default paths" in capsys.readouterr().err  # discovered: say so


@pytest.mark.parametrize("var", ["NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"])
def test_nvcc_flags_env_is_passed_through_unchanged(fake, monkeypatch, var):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    _, inc, lib = monorepo(str(tmp))
    value = f"-I{inc} -L {lib}"
    monkeypatch.setenv(var, value)
    rep = E.preflight()
    assert rep["flags"] == [f"-Xlinker=-rpath={lib}"]  # kurn adds only an rpath; nvcc applies the variable itself
    (call,) = calls()
    assert call["append" if var == "NVCC_APPEND_FLAGS" else "prepend"] == value
    assert inc not in " ".join(call["argv"]) and f"-I{inc}" not in call["argv"]
    assert E.resolve()["include_how"] == var


def test_sibling_install_is_probed_and_a_version_mismatch_warns(fake, monkeypatch, capsys):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    toolkit(str(tmp / "cuda-13.0"), version="13.0")  # another install next to it has headers
    rep = E.preflight()
    assert os.path.join(str(tmp / "cuda-13.0"), "targets") in rep["include"] and "sibling install" in rep["how"]
    assert "headers are 13.0 but nvcc is 12.8" in capsys.readouterr().err


def test_no_probe_skips_sibling_installs(fake, monkeypatch):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    toolkit(str(tmp / "cuda-13.0"), version="13.0")
    monkeypatch.setenv("KURN_CUDA_NO_PROBE", "1")
    with pytest.raises(E.CudaToolchainError, match="missing: cuda_runtime.h"):
        E.preflight()
    assert not any("cuda-13.0" in d for d in E.resolve()["searched_include"])


def test_same_version_runtime_is_tried_before_others(tmp_path):
    a, b = str(tmp_path / "a" / "include"), str(tmp_path / "b" / "include")
    runtime({"inc": a, "lib": str(tmp_path / "a" / "lib")}, cudart=13000)
    runtime({"inc": b, "lib": str(tmp_path / "b" / "lib")}, cudart=12080)
    pairs = E._pairs([(a, "sibling install"), (b, "sibling install")], [(str(tmp_path / x / "lib"), "s") for x in "ab"], (12, 8))
    assert [p["include"] for p in pairs] == [b, a] and pairs[0]["lib"] == str(tmp_path / "b" / "lib")


def test_driver_older_than_runtime_warns(fake, monkeypatch, capsys):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8")))
    monkeypatch.setenv("FAKE_DRIVER", "12040")
    rep = E.preflight(run=True)
    assert rep["ran"] and rep["driver"] == 12040
    assert "driver supports CUDA 12.4 but the runtime is CUDA 12.8" in capsys.readouterr().err


def test_gpu_run_failure_fails_fast(fake, monkeypatch):
    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8")))
    monkeypatch.setenv("FAKE_GPU_FAIL", "1")
    with pytest.raises(E.CudaToolchainError, match="failed on the GPU: no kernel image is available"):
        E.preflight(run=True)


def test_cublas_missing_is_reported(fake, monkeypatch):
    tmp, calls = fake
    nv = toolkit(str(tmp / "cuda-12.8"), with_runtime=False)
    runtime({"inc": os.path.join(str(tmp / "cuda-12.8"), "targets", E._target(), "include"),
             "lib": os.path.join(str(tmp / "cuda-12.8"), "targets", E._target(), "lib")}, cublas=False)  # fmt: skip
    monkeypatch.setenv("KURN_NVCC", nv)
    assert E.preflight()["flags"] == []
    with pytest.raises(E.CudaToolchainError, match="missing: cublas_v2.h, libcublas"):
        E.preflight(cublas=True)


def test_build_commands_stop_at_the_first_preflight_failure(fake, monkeypatch, capsys):
    """537 builds must not produce 537 identical failures: the CLI reports the preflight once and exits 2."""
    from kurn.gpu.attn_cli import main

    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    assert main(["ptxas", "--all", "--tier", "sm_80"]) == 2
    err = capsys.readouterr()
    assert (err.out + err.err).count("CUDA preflight failed") == 1 and "FAIL " not in err.out
    assert len(calls()) == 1


def test_doctor_writes_the_toolchain_and_exits_nonzero_on_failure(fake, monkeypatch, capsys):
    from kurn.gpu.cli import main

    tmp, calls = fake
    monkeypatch.setenv("KURN_NVCC", toolkit(str(tmp / "cuda-12.8"), with_runtime=False))
    out = tmp / "toolchain.json"
    assert main(["doctor", "--no-run", "--json", str(out)]) == 2
    assert "CUDA preflight failed" in json.loads(out.read_text())["preflight"]["error"]
    _, inc, lib = monorepo(str(tmp))
    monkeypatch.setenv("KURN_CUDA_INCLUDE", inc)
    monkeypatch.setenv("KURN_CUDA_LIB", lib)
    E.reset()
    assert main(["doctor", "--no-run", "--cublas", "--json", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["include"] == inc and rep["include_how"] == "KURN_CUDA_INCLUDE" and rep["preflight"]["cublas"]
    assert rep["env"]["KURN_CUDA_INCLUDE"] == inc and rep["arch_flags"][:2] == ["-gencode", "arch=compute_80,code=[sm_80,compute_80]"]
    text = capsys.readouterr().out
    assert "headers:" in text and "via KURN_CUDA_INCLUDE" in text and "preflight:  ok" in text


def test_nvcc_lookup_order(fake, monkeypatch):
    tmp, _ = fake
    a = toolkit(str(tmp / "home"))
    b = toolkit(str(tmp / "explicit"))
    monkeypatch.setenv("CUDA_HOME", str(tmp / "home"))
    E.reset()
    assert toolchain.nvcc() == a and E.resolve()["nvcc_how"] == "CUDA_HOME"
    monkeypatch.setenv("KURN_NVCC", b)
    E.reset()
    assert toolchain.nvcc() == b and E.resolve()["nvcc_how"] == "KURN_NVCC"


@pytest.mark.skipif(not toolchain.nvcc(), reason="nvcc not installed")
def test_real_toolkit_preflight():
    E.reset()
    rep = E.preflight(run=False)
    assert rep["header"] and textwrap.dedent(rep["command"]).startswith(toolchain.nvcc())
