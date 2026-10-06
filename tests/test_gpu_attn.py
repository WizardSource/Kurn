"""GPU attention (`op attn`, target cuda, sm_80) without a GPU: spec rules, numerics on the CPU warp emulator against
the float64 reference, bug classes the emulator must catch, and nvcc/ptxas (no spills) when nvcc is installed."""

import pytest

from kurn.gpu import attn as A
from kurn.gpu import toolchain
from kurn.spec import SpecError

emu = pytest.mark.skipif(bool(toolchain.cxx_problem()), reason=f"CPU emulator unavailable: {toolchain.cxx_problem()}")


def test_defaults_and_mla_defaults():
    c = A.resolve({"op": "attn", "target": "cuda"})
    assert (c["kv"], c["dk"], c["dv"], c["mla"], c["tk"], c["wm"], c["wn"], c["split"]) == ("f16", 128, 128, 0, 64, 1, 4, 0)
    m = A.resolve({"kv": "q8_0", "dk": 576})
    assert (m["dv"], m["mla"]) == (512, 1) and A.smem_bytes(m) <= A.SMEM_MAX


@pytest.mark.parametrize("ov, msg", [
    ({"arch": "sm_90"}, "A100"),
    ({"tk": 32, "wn": 4}, "tk / wn"),
    ({"dk": 576, "mla": 0, "tk": 64}, "shared memory"),
    ({"wm": 4, "wn": 2, "tk": 128}, "shared memory"),
    ({"dk": 256, "wn": 1}, "registers"),
    ({"dk": 128, "dv": 64}, "dv=64"),
    ({"heads": 6, "kv_heads": 4}, "multiple"),
    ({"bogus": 1}, "unknown key"),
])  # fmt: skip
def test_illegal_configs_name_the_rule(ov, msg):
    with pytest.raises(SpecError, match=msg):
        A.resolve(ov)


def test_covering_set_is_legal_and_varies_every_key():
    cs = A.covering_configs()
    for k in ("tk", "wm", "wn", "split", "mla", "kv", "dk"):
        assert len({c[k] for c in cs}) > 1, k
    assert all(A.smem_bytes(c) <= A.SMEM_MAX for c in cs)


def test_generated_source_carries_the_config():
    src = A.generate(A.resolve({"kv": "bf16", "dk": 64, "wn": 2}))
    assert "#define KGA_KV 1" in src and "#define KGA_WN 2" in src and "op=attn arch=sm_80" in src
    assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in src


@emu
@pytest.mark.parametrize("kv", sorted(A.KV_FORMATS))
def test_default_kernel_matches_reference_on_awkward_shapes(kv):
    w = A.emu_check(A.resolve({"kv": kv}))
    assert w["ok"], w


@emu
@pytest.mark.parametrize("ov", [
    {"kv": "f16", "dk": 576},
    {"kv": "q8_0", "dk": 576},
    {"kv": "bf16", "dk": 576},
    {"kv": "f16", "dk": 64, "wn": 1, "split": 16},
    {"kv": "q8_0", "dk": 256, "tk": 32, "wn": 2},
    {"kv": "bf16", "dk": 128, "wm": 4, "wn": 2, "tk": 32},
    {"kv": "q8_0", "dk": 128, "mla": 1, "split": 4},
], ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))  # fmt: skip
def test_variants_match_reference_under_random_schedules(ov):
    w = A.emu_check(A.resolve(ov), shapes=A.QUICK_SHAPES, scheds=(0, 9))
    assert w["ok"], w


@emu
def test_forced_splits_exercise_the_merge():
    c = A.resolve({"kv": "f16", "split": 16})
    r = A.emu_run(c, {"nq": 1, "nkv": 4100, "heads": 16, "kv_heads": 8})
    assert r["splits"] == 13 and r["ok"], r  # chunks of whole tiles: ceil(4100 / 16) -> 320 tokens
    r = A.emu_run(c, {"nq": 1, "nkv": 100, "heads": 4, "kv_heads": 2})  # fewer tiles than splits: no empty splits
    assert r["splits"] == 2 and r["ok"], r


def _mutant(c, old, new, shapes, scheds=(0, 3)):
    src = A.generate(c)
    assert old in src, old
    exe = A.emu_build(c, src.replace(old, new, 1))
    try:
        return [A.emu_run(c, sh, seed=1 + i, sched=s, exe=exe) for i, sh in enumerate(shapes) for s in scheds]
    except A.EmuError as e:
        return str(e)


DEC = {"nq": 1, "nkv": 1000, "heads": 8, "kv_heads": 2}
PRE = {"nq": 37, "nkv": 300, "heads": 6, "kv_heads": 3}


@emu
@pytest.mark.parametrize("ov, old, new, shapes, why", [
    ({}, "KCP_WAIT(KGA_STAGES - 1);", "KCP_WAIT(KGA_STAGES);", (DEC,), "tile read before its cp.async group completes"),
    ({}, "#if KGA_WN > 1\n    __syncthreads();\n#else\n    __syncwarp();", "#if 0\n#else\n    __syncwarp();", (DEC, PRE),
     "P read by other warps before it is written"),
    ({}, "__syncthreads();  // the next iteration", "  // the next iteration", (DEC,), "stage refilled while still read"),
    ({}, "o[n][0] *= alpha[0];", "", (PRE,), "output not rescaled when the running max moves"),
    ({}, "j <= (hb ? limb : lima)", "j <= (hb ? limb : lima) + 1", (PRE,), "causal bound off by one"),
    ({"split": 8}, "L += kga_exp2(base[s * rs] - M) * base[s * rs + 1];", "L += base[s * rs + 1];", (DEC,),
     "split merge ignores the partial maxima"),
    ({"kv": "q8_0"}, "sc * (float)q1, sc * (float)q0", "sc * (float)q0, sc * (float)q1", (DEC,), "Q8_0 values swapped"),
    ({"kv": "bf16"}, "kga_pack(x1, x0)", "kcvt_h2(x1, x0)", (DEC,), "q packed as f16 for a bf16 MMA"),
])  # fmt: skip
def test_emulator_catches_attention_bugs(ov, old, new, shapes, why):
    res = _mutant(A.resolve(ov), old, new, shapes)
    assert isinstance(res, str) or not all(r["ok"] for r in res), f"emulator missed: {why}"


@pytest.mark.skipif(not toolchain.nvcc(), reason="nvcc not installed")
@pytest.mark.parametrize("ov", [{"kv": "f16"}, {"kv": "bf16"}, {"kv": "q8_0"}, {"kv": "q8_0", "dk": 576}, {"kv": "f16", "dk": 256},
                                {"kv": "f16", "dk": 64, "wn": 1}])  # fmt: skip
def test_fatbin_builds_for_sm80_and_sm120_without_spills(ov):
    c = A.resolve(ov)
    rep = A.ptxas(c)
    for arch in A.fatbin_archs():
        assert rep[arch]["kga_main"]["regs"] > 0 and rep[arch]["kga_main"]["spill"] == 0 and rep[arch]["kga_merge"]["spill"] == 0, rep
    sass, ptx = A.fatbin_contents(A.nvcc_build(c)[0])
    assert sass == sorted(A.fatbin_archs()) and ptx == ["sm_80"]
    assert ("sm_120" in A.fatbin_archs()) == (tuple(int(x) for x in A.nvcc_version().split(".")) >= (12, 8))


def test_old_nvcc_builds_sm80_only(monkeypatch):
    monkeypatch.setattr(A, "nvcc_version", lambda: "12.2")
    assert A.fatbin_archs() == ["sm_80"] and "arch=compute_120,code=sm_120" not in A.fatbin_flags()
    monkeypatch.setattr(A, "nvcc_version", lambda: "12.9")
    assert A.fatbin_flags() == ["-gencode", "arch=compute_80,code=[sm_80,compute_80]", "-gencode", "arch=compute_120,code=sm_120"]


@pytest.mark.skipif(not toolchain.nvcc(), reason="nvcc not installed")
@pytest.mark.parametrize("arch", A.FATBIN_ARCHS)
def test_gpu_harness_compiles(arch, tmp_path):
    if arch not in A.fatbin_archs():
        pytest.skip(f"nvcc {A.nvcc_version()} can't target {arch}")
    assert A.build_harness(arch, out_dir=str(tmp_path))


def test_cli_routes_attn_specs(tmp_path, capsys):
    from kurn.cli import main

    sp = tmp_path / "a.kurn"
    sp.write_text("op attn\ntarget cuda\nkv q8_0\ndk 576\n")
    assert main(["check", str(sp)]) == 0
    out = capsys.readouterr().out
    assert '"mla": 1' in out and "fits: sm_80, sm_120" in out
    assert main(["gpu", "attn", "check", "-", "kv=bf16", "tk=32", "wn=4"]) == 2


def test_shared_memory_decides_where_a_config_runs():
    assert A.archs_for(A.resolve({"kv": "q8_0"})) == ["sm_80", "sm_120"]
    assert A.archs_for(A.resolve({"kv": "q8_0", "dk": 576})) == ["sm_80", "sm_120"]
    assert A.archs_for(A.resolve({"kv": "f16", "dk": 256})) == ["sm_80"]  # 146 KB tile: A100 only
    assert A.archs_for(A.resolve({"kv": "f16", "dk": 256, "tk": 32, "wn": 2})) == ["sm_80", "sm_120"]


_LAYOUT = """#include <cstddef>
#include "kurn_attn.h"
#include "kurn_gpu_attn.h"
#define SAME(f) static_assert(offsetof(kattn_args, f) == offsetof(kga_args, f), #f);
SAME(n_q) SAME(n_kv) SAME(q_pos0) SAME(n_head) SAME(n_head_kv) SAME(causal) SAME(scale) SAME(q) SAME(q_s_tok) SAME(q_s_head)
SAME(k) SAME(k_s_tok) SAME(k_s_head) SAME(v) SAME(v_s_tok) SAME(v_s_head) SAME(mask) SAME(mask_s_tok) SAME(out) SAME(o_s_tok)
SAME(o_s_head) SAME(k_tail) SAME(kt_s_tok) SAME(kt_s_head) SAME(rope_freq) SAME(k_pos0) SAME(rope_dim) SAME(rope_mode)
static_assert(sizeof(kattn_args) == sizeof(kga_args), "size");
static_assert(KATTN_KV_F16 == KGA_KV_F16 && KATTN_KV_BF16 == KGA_KV_BF16 && KATTN_KV_Q8_0 == KGA_KV_Q8_0, "formats");
int main() {}
"""


@emu
def test_kga_args_has_the_cpu_abi_layout(tmp_path):
    """kga_args must stay field-for-field identical to the CPU op's kattn_args (hosts copy one into the other)."""
    import os
    import subprocess

    from kurn import toolchain as cpu_toolchain

    src = tmp_path / "layout.cpp"
    src.write_text(_LAYOUT)
    inc = [os.path.dirname(cpu_toolchain.data_path("kurn_attn.h")), os.path.dirname(toolchain.data_path("kurn_gpu_attn.h"))]
    r = subprocess.run([*toolchain.cxx(), "-std=c++17", "-DKURN_EMU", "-I", inc[0], "-I", inc[1], str(src), "-o", str(tmp_path / "x")],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]


def _cpu_lib(kv, dk):
    from kurn import attention as CPU

    c = CPU.resolve({"op": "attn", "target": "avx512", "kv": kv, "dk": dk})
    ok, missing = CPU.runnable("avx512")
    if not ok:
        pytest.skip(f"CPU attention op needs AVX-512 here (missing {missing})")
    return CPU.build(c)


@emu
@pytest.mark.parametrize("kv, dk", [("f16", 128), ("bf16", 128), ("q8_0", 128), ("q8_0", 576), ("f16", 64)])
def test_matches_the_cpu_attention_op_on_the_same_arguments(kv, dk):
    """The GPU kernel (emulated) against kurn's CPU op (f32 engine) on identical inputs and kattn_args: decode, prefill,
    explicit mask, non-causal, head-major cache, fully masked rows."""
    lib = _cpu_lib(kv, dk)
    c = A.resolve({"kv": kv, "dk": dk})
    exe = A.emu_build(c)
    for i, sh in enumerate(A.QUICK_SHAPES + (A.CHECK_SHAPES[6],)):
        if c["mla"]:
            sh = {**sh, "heads": A.MLA_HEADS if sh["heads"] >= 4 else sh["heads"], "kv_heads": 1}
        r = A.emu_run(c, sh, seed=3 + i, exe=exe, cpu_lib=lib)
        assert r["ok"] and r["vs_cpu"] <= A.TOL[kv] and r["cpu_relerr"] <= 1e-4, (sh, r)


def test_report_renders_the_matrix(tmp_path, capsys):
    from kurn.gpu.attn_cli import main

    rows = [{"model": "llama3-8b", "kv": "q8_0", "nkv": 1024, "nq": 1, "config": "x", "relerr": 1e-3, "status": "ok", "splits": 8,
             "us": 12.5, "us_sd": 0.1, "GBps": 900.0, "TFLOPs": 1.0, "uJ": 3.0, "layers": 64},
            {"model": "mla-dsv2-lite", "kv": "f16", "nkv": 1024, "nq": 1, "status": "error", "error": "kga_run failed (-4)"}]  # fmt: skip
    (tmp_path / "attn_matrix.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
    assert main(["report", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "2 cells, 1 correct, 1 not ok" in out and "| llama3-8b | q8_0 | 1024 | 1 | ok | 1.0e-03 | 8 | 12.5 ± 0.1 | 900 |" in out


@pytest.mark.skipif(not toolchain.nvcc(), reason="nvcc not installed")
def test_gpu_plumbing_with_a_stub_harness(tmp_path):
    """gpu_check / matrix / report parse the harness's JSON lines (the stub stands in for bench_gpu_attn on a GPU)."""
    import json

    from kurn.gpu.attn_cli import main

    stub = tmp_path / "bench_gpu_attn"
    stub.write_text(
        "#!/bin/sh\n"
        'echo \'{"kind": "check", "config": "c", "device": "NVIDIA A100-SXM4-80GB", "sm": 80, "sms": 108, "relerr": 1.0e-03, '
        '"tol": 4e-3, "status": "ok", "splits": 8, "workspace": 0}\'\n'
        'echo \'{"kind": "sample", "round": 0, "us": 10.0, "GBps": 1500.0, "TFLOPs": 0.5, "uJ": 2.0, "calls": 10, "layers": 64, '
        '"kv_bytes_per_layer": 1, "sm_mhz": 1410, "mem_mhz": 1593}\'\n'
    )
    stub.chmod(0o755)
    c = A.resolve({"kv": "q8_0"})
    w = A.gpu_check(str(stub), c, A.QUICK_SHAPES)
    assert w["status"] == "ok" and w["device"].startswith("NVIDIA A100")
    rows = A.matrix(str(stub), str(tmp_path), kvs=("q8_0",), contexts=(1024,), log=lambda *_: None)
    assert len(rows) == len(A.MATRIX_MODELS) and all(r["status"] == "ok" and r["us"] == 10.0 for r in rows)
    assert json.loads((tmp_path / "attn_matrix.jsonl").read_text().splitlines()[0])["model"] == "llama3-8b"
    assert main(["report", str(tmp_path)]) == 0
    stub.write_text('#!/bin/sh\necho \'{"kind": "error", "error": "kga_run failed (-4): tile exceeds"}\'\nexit 1\n')
    with pytest.raises(A.HarnessError, match="-4"):
        A.gpu_run(str(stub), "x.so", c)
