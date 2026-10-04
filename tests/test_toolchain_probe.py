"""Toolchain probe: a target the compiler or assembler cannot build (binutils < 2.36 has no AVX-VNNI) is reported once
and skipped, instead of failing every configuration."""

import shutil
import stat

import pytest

from kurn import toolchain
from kurn.cli import main

from conftest import EXAMPLES

pytestmark = pytest.mark.skipif(toolchain.host_arch() != "x86_64" or not shutil.which("gcc"), reason="needs gcc on x86-64")

OLD_AS = """#!/bin/sh
for a in "$@"; do case "$a" in
  -mavxvnni) echo "probe.s:12: Error: no such instruction: \\`{vex} vpdpbusd %ymm2,%ymm1,%ymm0'" >&2; exit 1;;
esac; done
exec gcc "$@"
"""


def _clear():
    for f in (toolchain.host_cc, toolchain.toolchain_problem):
        f.cache_clear()


@pytest.fixture
def old_binutils(tmp_path, monkeypatch):
    cc = tmp_path / "oldgcc"
    cc.write_text(OLD_AS)
    cc.chmod(cc.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KURN_CC", str(cc))
    monkeypatch.setenv("KURN_CACHE_DIR", str(tmp_path / "cache"))
    _clear()
    yield
    _clear()


def test_probe_names_target_and_env_var(old_binutils):
    why = toolchain.toolchain_problem("avx2_vnni")
    assert why.startswith("toolchain can't assemble target avx2_vnni; set KURN_CC") and "vpdpbusd" in why
    assert toolchain.toolchain_problem("avx2") is None
    with pytest.raises(toolchain.ToolchainError, match="avx2_vnni"):
        toolchain.cc_for("avx2_vnni")
    assert toolchain.run_mode("avx2_vnni") == (None, why)


def test_verify_skips_instead_of_failing(old_binutils, capsys):
    assert main(["verify", str(EXAMPLES / "q8_0_gemv_vnni16.kurn"), "target=avx2_vnni", "layout=native"]) == 0
    out = capsys.readouterr().out
    assert "skip   target avx2_vnni: toolchain can't assemble target avx2_vnni" in out
    assert "0 failures, 1 skipped" in out


def test_working_toolchain_passes_probe():
    _clear()
    assert toolchain.toolchain_problem("scalar") is None


def test_verify_skips_neon_without_cross_compiler(monkeypatch, capsys):
    monkeypatch.setattr(toolchain, "cross_cc", lambda arch: None)
    _clear()
    try:
        why = toolchain.toolchain_problem("neon")
        assert "aarch64 cross compiler" in why and "apt install gcc-aarch64-linux-gnu" in why and "KURN_CROSS_CC" in why
        assert toolchain.run_mode("neon") == (None, why)
        with pytest.raises(toolchain.ToolchainError):
            toolchain.cc_for("neon")
        assert main(["verify", str(EXAMPLES / "q8_0_gemv_vnni16.kurn"), "target=neon", "layout=native", "--strict"]) == 0
        out = capsys.readouterr().out
        assert "skip   target neon: target neon needs an aarch64 cross compiler" in out
        assert "0 failures, 1 skipped (no usable toolchain for neon" in out
    finally:
        _clear()


@pytest.mark.skipif(not toolchain.cross_cc("aarch64"), reason="needs an aarch64 cross compiler")
def test_verify_compiles_but_does_not_run_neon_without_qemu(monkeypatch, capsys):
    monkeypatch.setattr(toolchain, "qemu", lambda arch="aarch64": None)
    _clear()
    try:
        mode, why = toolchain.run_mode("neon")
        assert mode is None and "no qemu-aarch64" in why and "apt install qemu-user" in why and "KURN_QEMU" in why
        assert main(["verify", str(EXAMPLES / "q8_0_gemv_vnni16.kurn"), "target=neon", "layout=native", "--strict"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("built ") and "compiled, not run: no qemu-aarch64" in out
        assert "0 failures, 1 compiled but not run (neon: no qemu-aarch64" in out
    finally:
        _clear()
