"""Kernel registry: which (op, weight format) pairs exist, which targets each one
lowers to, and how each is called through `kurn.h` and the harness.

Adding a kernel (e.g. a 4-bit repacked GEMV or a 2-bit LUT GEMV) means:
  1. a Format in formats.py (layout, activation format, Python reference),
  2. a lowering function in codegen.py,
  3. a Kernel entry below,
  4. its entry points in data/kurn.h and a row in the harness tables in data/bench.c,
  5. legal schedule values in spec.SCHEDULE if the defaults do not fit.
Everything else (spec validation, CLI, verify, tune, tests) is driven by these
registries.
"""

from dataclasses import dataclass
from typing import Callable

from . import codegen, generic, hooks
from .formats import FORMATS


def _dispatch(native):
    """Native/vnni16 layouts use the hand-structured lowerings in codegen.py;
    interleaved layouts (i16 / i8) and new formats use the generic recipe lowering."""

    def lower(target, c):
        if c.get("layout") in hooks.LOWERINGS:
            return hooks.LOWERINGS[c["layout"]](target, c)
        if c.get("layout") in ("i16", "i8", "l32") or native is None:
            return generic.lower(target, c)
        return native(target, c)

    return lower


_GENERIC_TARGETS = ("scalar", "avx2_vnni", "avx512_vnni")
_VERIFY_TARGETS = ("avx2_vnni", "avx512_vnni")


@dataclass(frozen=True)
class Kernel:
    op: str  # "gemv" (decode: one activation row) | "gemm" (prefill: M rows)
    weights: str  # key into FORMATS; the activation format is FORMATS[weights].act
    targets: tuple  # keys into targets.TARGETS
    lower: Callable  # (target, config) -> C source
    entry: str  # kurn.h entry point, e.g. "kq8_gemv"; `<entry>_prepare` / `<entry>_packed` are optional
    bench: str  # harness --kernel name
    doc: str = ""

    @property
    def act(self):
        return FORMATS[self.weights].act

    @property
    def entry_points(self):
        return (self.entry, f"{self.entry}_prepare", f"{self.entry}_packed")


KERNELS = {
    (k.op, k.weights): k
    for k in (
        Kernel(
            "gemv",
            "q8_0",
            ("scalar", "avx2", "avx2_vnni", "avx512_vnni", "neon"),
            _dispatch(codegen.q8_0_gemv),
            "kq8_gemv",
            "q8gemv",
            "Q8_0 weights x Q8_0 activation vector (decode)",
        ),  # fmt: skip
        Kernel(
            "gemv",
            "q4_K",
            ("scalar", "avx2", "avx2_vnni", "avx512_vnni"),
            _dispatch(codegen.q4_K_gemv),
            "kq4k_gemv",
            "q4kgemv",
            "Q4_K weights x Q8_K activation vector (decode)",
        ),  # fmt: skip
        Kernel(
            "gemm",
            "q8_0",
            ("scalar", "avx512_vnni", "amx"),
            codegen.q8_0_gemm,
            "kq8_gemm",
            "q8gemm",
            "Q8_0 weights x M Q8_0 activation rows (prefill)",
        ),  # fmt: skip
    )
}
# v0.2: formats lowered only through the generic recipe lowering (decode GEMV), and a
# multi-column `verify` op (2-8 activation columns: speculative decoding / small batches)
# for every recipe.
for _w, _entry, _bench, _doc in (
    ("q4_0", "kq40_gemv", "q40gemv", "Q4_0 weights x Q8_0 activation vector"),
    ("iq4_nl", "kiq4nl_gemv", "iq4nlgemv", "IQ4_NL (non-linear 4-bit codebook) x Q8_0"),
    ("q2_0", "kq20_gemv", "q20gemv", "Q2_0 (2-bit, ternary Bonsai) x Q8_0"),
    ("tq2_0", "ktq20_gemv", "tq20gemv", "TQ2_0 (ternary, 2.06 bpw) x Q8_K"),
    ("q1_0", "kq10_gemv", "q10gemv", "Q1_0 (1-bit Bonsai) x Q8_0"),
):
    KERNELS[("gemv", _w)] = Kernel("gemv", _w, _GENERIC_TARGETS, _dispatch(None), _entry, _bench, _doc)
for _w, _short in (
    ("q8_0", "q8"),
    ("q4_K", "q4k"),
    ("q4_0", "q40"),
    ("iq4_nl", "iq4nl"),
    ("q2_0", "q20"),
    ("tq2_0", "tq20"),
    ("q1_0", "q10"),
):
    KERNELS[("verify", _w)] = Kernel(
        "verify",
        _w,
        _VERIFY_TARGETS,
        _dispatch(None),
        f"k{_short}_vfy",
        f"{_short}vfy",
        f"{_w} weights x 2-8 activation columns (multi-token verify)",
    )

from . import ext  # noqa: E402,F401  (workstream extensions register into the dicts above)

ENTRY_POINTS = tuple(sym for k in KERNELS.values() for sym in k.entry_points)


def kernel(c):
    return KERNELS[(c["op"], c["weights"])]


def generate(c):
    """Resolved config (see spec.resolve) -> C source implementing the kurn.h ABI."""
    k = kernel(c)
    return codegen.prune_helpers(k.lower(c["target"], dict(c, entry=k.entry)))


def embed(src, prefix):
    """Static, `prefix`-ed version of generated C, without the kurn.h include (see codegen.embed)."""
    return codegen.embed(src, prefix, ENTRY_POINTS)
