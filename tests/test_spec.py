import collections

import pytest

from kurn.spec import (
    DEFAULTS,
    SpecError,
    iter_space,
    legal_configs,
    load,
    parse,
    parse_overrides,
    resolve,
    validate_space,
)

from conftest import EXAMPLES

BASE = "op gemv\nweights q8_0\ntarget avx512_vnni\n"


def test_parse_comments_and_tune():
    spec, tune = parse("# header\nop gemv   # decode\n\nweights q8_0\ntune rows=1,2 wait=spin,sleep\n")
    assert spec == {"op": "gemv", "weights": "q8_0"}
    assert tune == {"rows": [1, 2], "wait": ["spin", "sleep"]}


@pytest.mark.parametrize(
    "text,msg",
    [
        ("op gemv\nop gemm", "duplicate key 'op'"),
        ("op", "has no value"),
        ("op gemv gemm", "takes one value"),
        ("tune rows", "tune entries look like key=v1,v2"),
        ("tune rows=1 rows=2", "duplicate tune key"),
    ],
)
def test_parse_errors(text, msg):
    with pytest.raises(SpecError, match=msg):
        parse(text)


def test_resolve_fills_defaults():
    c = resolve(parse(BASE)[0])
    assert c["layout"] == "native" and c["rows"] == 4 and c["act"] == "once" and c["prefetch"] == 0
    assert c["threads"] == DEFAULTS["threads"] and c["wait"] == "spin"
    assert c["act_format"] == "q8_0" and c["kernel"] == "q8_0_gemv_avx512_vnni"
    assert resolve(parse("op gemv\nweights q4_K\ntarget avx2")[0])["act_format"] == "q8_K"


@pytest.mark.parametrize(
    "extra,msg",
    [
        ("", None),
        ("op conv", "op 'conv': expected one of \\['gemv', 'gemm', 'verify'\\]"),
        ("weights q5_K", "weights 'q5_K' not supported for op gemv"),
        ("target sve", "target 'sve' not supported for gemv/q8_0"),
        ("rows 3", "rows=3 not allowed for gemv/q8_0/avx512_vnni: expected one of \\[1, 2, 4, 8\\]"),
        ("prefetch 5", "expected one of \\[0, 2, 4, 8, 16, 32\\]"),
        ("threads 0", "expected an integer in 1..256"),
        ("colour blue", "unknown key 'colour'"),
        ("layout vnni16\nact inline", "act applies to layout=native only"),
        ("align 64", "align applies to layout=vnni16 only"),
    ],
)
def test_resolve_errors_list_legal_values(extra, msg):
    spec = {**parse(BASE)[0], **parse(extra)[0]}
    if msg is None:
        resolve(spec)
    else:
        with pytest.raises(SpecError, match=msg):
            resolve(spec)


def test_missing_required_key():
    with pytest.raises(SpecError, match="missing required key 'target'"):
        resolve({"op": "gemv", "weights": "q8_0"})


def test_gemm_register_budget():
    with pytest.raises(SpecError, match="exceeds 24 accumulators"):
        resolve({"op": "gemm", "weights": "q8_0", "target": "avx512_vnni", "rows": 6, "cols": 6})
    resolve({"op": "gemm", "weights": "q8_0", "target": "avx512_vnni", "rows": 6, "cols": 4})


def test_target_specific_values():
    # vnni16 and align=64 exist only for q8_0 gemv on avx512_vnni; avx2 q8_0 has no `act once`
    with pytest.raises(SpecError, match="layout='vnni16' not allowed"):
        resolve({"op": "gemv", "weights": "q8_0", "target": "avx2", "layout": "vnni16"})
    assert resolve({"op": "gemv", "weights": "q8_0", "target": "avx2"})["act"] == "inline"
    assert resolve({"op": "gemm", "weights": "q8_0", "target": "amx"})["rows"] == 1


def test_overrides():
    single, lists = parse_overrides(["rows=8", "layout=vnni16", "threads=2,4"])
    assert single == {"rows": 8, "layout": "vnni16"} and lists == {"threads": [2, 4]}
    with pytest.raises(SpecError, match="expected key=value"):
        parse_overrides(["rows"])


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.kurn")), ids=lambda p: p.name)
def test_examples_resolve_and_tune_spaces_are_legal(path):
    spec, space = load(path)
    resolve(spec)
    assert space, "every example declares a tune space"
    assert validate_space(spec, space) > 0


def test_tune_space_rejects_typos():
    spec = parse(BASE)[0]
    with pytest.raises(SpecError, match=r"tune rows: \[3\] are not legal"):
        validate_space(spec, {"rows": [1, 3]})
    with pytest.raises(SpecError, match="tune key 'colour'"):
        validate_space(spec, {"colour": [1]})
    assert validate_space(spec, {"layout": ["native", "vnni16"], "act": ["once", "inline"]}) == 3


def test_iter_space_skips_illegal_points():
    spec = parse(BASE)[0]
    pts = list(iter_space(spec, {"layout": ["native", "vnni16"], "align": ["packed", 64]}))
    assert [ov for ov, _ in pts] == [{"layout": "native", "align": "packed"}, {"layout": "vnni16", "align": "packed"},
                                     {"layout": "vnni16", "align": 64}]  # fmt: skip


def test_legal_config_count():
    # The closed search space is part of the language; changing it should be deliberate.
    counts = collections.Counter(c["target"] == "neon" for c in legal_configs())
    # + lowbit (tq1_0, q2_K, layouts lut/addsub/k16) + fourbit (mxfp4, nvfp4, unpack mask16/perm/pair, dpmin)
    # + compress (e8p gemv)
    # + 60: unpack=pair verify kernels with rows * cols up to 8 (single accumulator chain, WS-I)
    # + 4: rows=8 for the Q8_0 i16 GEMV (v0.1 vnni16's pass width; prefetch 0/8 x act/weight correction)
    assert counts[False] == 912 + 178 + 488 + 12 + 60 + 4 and counts[True] == 8
