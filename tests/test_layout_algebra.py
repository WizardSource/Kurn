"""CuTe-style layout algebra (kurn.layout): functional identities, the published CuTe
examples, C emission, and the packed record layouts that reproduce kurn's fixed layouts."""

import itertools
import random

import pytest

from kurn import generic
from kurn.layout import (
    Layout,
    Swizzle,
    blocked_product,
    coalesce,
    complement,
    composition,
    left_inverse,
    logical_divide,
    logical_product,
    make_layout,
    make_ordered_layout,
    preset,
    raked_product,
    recipe_meta,
    record_layout,
    right_inverse,
    zipped_divide,
)


def same_function(a, b, n=None):
    n = a.size() if n is None else n
    return all(a(i) == b(i) for i in range(n))


def rand_layout(rng, maxrank=3):
    shape = tuple(rng.choice((1, 2, 3, 4)) for _ in range(rng.randint(1, maxrank)))
    # an injective layout: a random permutation of compact strides, with gaps
    order = list(range(len(shape)))
    rng.shuffle(order)
    stride, cur = [0] * len(shape), rng.choice((1, 2))
    for i in order:
        stride[i] = cur
        cur *= shape[i] * rng.choice((1, 1, 2))
    return Layout(shape, tuple(stride))


def test_basic_eval_and_coords():
    a = Layout((2, (2, 2)), (4, (2, 1)))
    assert a.size() == 8 and a.cosize() == 8
    assert [a(i) for i in range(8)] == [0, 4, 2, 6, 1, 5, 3, 7]
    assert a((1, (0, 1))) == 5
    assert a(1, (1, 1)) == 7
    assert Layout((4, 3)).stride == (1, 4)


def test_coalesce():
    assert coalesce(Layout((2, (1, 6)), (1, (6, 2)))) == Layout(12, 1)
    a = Layout((2, 4), (1, 3))
    assert same_function(coalesce(a), a)


def test_composition_doc_example():
    a = Layout((6, 2), (8, 2))
    b = Layout((4, 3), (3, 1))
    r = composition(a, b)
    assert same_function(r, Layout(b.shape, b.stride).__class__(r.shape, r.stride))
    assert all(r(i) == a(b(i)) for i in range(b.size()))
    assert r == Layout(((2, 2), 3), ((24, 2), 8))


@pytest.mark.parametrize("seed", range(60))
def test_composition_is_function_composition(seed):
    rng = random.Random(seed)
    a = rand_layout(rng)
    # B must be admissible: a tiler whose strides divide A's shapes (CuTe's condition)
    n = a.size()
    divs = [d for d in range(1, n + 1) if n % d == 0]
    s = rng.choice(divs)
    rest = n // s
    d = rng.choice([x for x in range(1, rest + 1) if rest % x == 0])
    b = Layout(s, d) if s * d <= n else Layout(s, 1)
    try:
        r = composition(a, b)
    except ValueError:
        pytest.skip("inadmissible pair")
    assert all(r(i) == a(b(i)) for i in range(b.size())), (a, b, r)


@pytest.mark.parametrize("seed", range(60))
def test_complement_fills_holes(seed):
    rng = random.Random(seed)
    a = rand_layout(rng)
    m = a.cosize() * rng.choice((1, 2))
    c = complement(a, m)
    whole = make_layout(a, c)
    img = sorted(whole(i) for i in range(whole.size()))
    assert len(set(img)) == len(img), "complement overlaps the layout"
    assert img[: m] == list(range(m)) if whole.size() >= m else True


def test_complement_doc_examples():
    assert complement(Layout(4, 2), 24) == Layout((2, 3), (1, 8))
    assert complement(Layout((2, 2), (1, 6)), 24) == Layout((3, 2), (2, 12))


@pytest.mark.parametrize("seed", range(40))
def test_inverses(seed):
    rng = random.Random(seed)
    a = rand_layout(rng)
    li = left_inverse(a)
    assert all(li(a(i)) == i for i in range(a.size()))
    b = make_ordered_layout(a.shape, tuple(rng.sample(range(len(a)), len(a))))  # bijective
    ri = right_inverse(b)
    assert all(b(ri(i)) == i for i in range(ri.size()))
    assert ri.size() == b.size()


def test_logical_divide_doc_example():
    a = Layout((4, 2, 3), (2, 1, 8))
    r = logical_divide(a, Layout(4, 2))
    assert r == Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
    assert same_function(coalesce(r), coalesce(r))


def test_logical_divide_tiles_k_into_records():
    k = Layout(256)
    t = logical_divide(k, 32)  # (32 values per K-group, 8 K-groups)
    assert t.shape == (32, 8) and t.stride == (1, 32)
    t2 = logical_divide(Layout((16, 64)), (Layout(4), Layout(8)))  # mode-wise tiler
    assert t2.shape == ((4, 4), (8, 8))
    z = zipped_divide(Layout((16, 64)), (Layout(4), Layout(8)))
    assert z[0].size() == 32 and z[1].size() == 32


def test_logical_product_doc_example():
    a = Layout((2, 2), (4, 1))
    r = logical_product(a, Layout(6, 1))
    assert r == Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))


def test_blocked_and_raked_products():
    a = Layout((2, 2), (1, 2))
    b = Layout((2, 3), (1, 2))
    bp, rp = blocked_product(a, b), raked_product(a, b)
    assert bp.size() == rp.size() == 24
    assert sorted(bp(i) for i in range(24)) == list(range(24))
    assert sorted(rp(i) for i in range(24)) == list(range(24))


def test_swizzle_is_an_involution():
    for b, m, s in ((1, 6, 3), (2, 6, 4), (3, 4, 3)):
        sw = Swizzle(b, m, s)
        vals = [sw(i) for i in range(1 << (m + s + b))]
        assert sorted(vals) == list(range(1 << (m + s + b)))
        assert all(sw(sw(i)) == i for i in range(1 << (m + s + b)))


def _eval_c(expr, **env):
    return eval(expr.replace(" / ", " // ").replace(") / ", ") // ").replace("/", "//").replace("////", "//"), {}, env)


@pytest.mark.parametrize("seed", range(20))
def test_c_expr_matches_python(seed):
    rng = random.Random(seed)
    a = make_layout(rand_layout(rng, 2), rand_layout(rng, 3))
    e = a.c_expr(("r", "k"))
    for r, k in itertools.product(range(a[0].size()), range(a[1].size())):
        assert _eval_c(e, r=r, k=k) == a((r, k)), e


def test_swizzle_c_expr():
    sw = Swizzle(2, 6, 4)
    for i in range(4096):
        assert _eval_c(sw.c_expr("x"), x=i) == sw(i)


# --------------------------------------------------------------------------- packed records


def _elem_generic(r, bits, L, hdr, row, k, kg_bytes):
    """generic.lower's repack address for (row, k) as (byte, bit shift)."""
    kg, kin = divmod(k, 32)
    kk, j = divmod(kin, 4)
    base = hdr + kg * kg_bytes
    vb = 4 * L
    if bits == 8:
        return base + kk * vb + row * 4 + j, 0
    if bits == 4:
        return base + (kk // 2) * vb + row * 4 + j, 4 * (kk & 1)
    if bits == 2:
        return base + (kk // 4) * vb + row * 4 + j, 2 * (kk & 3)
    return base + kk * (vb // 8) + (row * 4 + j) // 8, (row * 4 + j) % 8


@pytest.mark.parametrize("fmt", sorted(generic.RECIPES))
@pytest.mark.parametrize("target", ["avx512_vnni", "avx2_vnni"])
@pytest.mark.parametrize("corr", ["act", "weight"])
def test_i16_i8_presets_match_generic_addresses(fmt, target, corr):
    r = generic.RECIPES[fmt]
    if corr not in generic.legal_keys(fmt, target)["correction"]:
        pytest.skip("correction not legal")
    lay = preset("i16" if target == "avx512_vnni" else "i8", r, target, correction=corr)
    L = 16 if target == "avx512_vnni" else 8
    hdr, cbytes, corrb, groups, rec = generic._record_layout(r, {"scales": "unpacked", "correction": corr}, L)
    assert lay.rec_bytes == rec and lay.code_base == hdr and lay.kg_bytes == cbytes + corrb
    for row in range(L):
        for k in range(r.period):
            e = lay.code(row, k)
            got = (e * r.bits // 8, (e % (8 // r.bits)) * r.bits)
            assert got == _elem_generic(r, r.bits, L, hdr, row, k, cbytes + corrb), (row, k)


def test_vnni16_preset_is_codes_then_scales():
    lay = preset("vnni16", generic.RECIPES["q8_0"])
    assert lay.rec_bytes == 544 and lay.code_base == 0 and lay.field("d").layout(0, 0, 0) == 512
    assert preset("vnni16", generic.RECIPES["q8_0"], align=64).rec_bytes == 576
    for row, k in itertools.product(range(16), range(32)):
        assert lay.code(row, k) == (k // 4) * 64 + row * 4 + k % 4


def test_l32_preset_matches_lut_lowering():
    for fmt in ("q2_0", "q1_0", "tq2_0"):
        r = generic.RECIPES[fmt]
        lay = preset("l32", r)
        g = 4 // r.bits
        chunks = 32 // g
        idx_bytes = 32 * chunks // 2
        for row in range(32):
            for k in range(r.period):
                kg, kin = divmod(k, 32)
                ch, j = divmod(kin, g)
                byte = 64 + kg * idx_bytes + (ch // 2) * 32 + row
                bit = 4 * (ch & 1) + r.bits * j
                e = lay.code(row, k)
                assert (e * r.bits // 8, (e * r.bits) % 8) == (byte, bit)


@pytest.mark.parametrize("bits,plane", [(4, "kstep"), (4, "khalf"), (4, "rows"), (2, "kstep"), (2, "khalf"), (2, "rows"),
                                        (1, "atom"), (1, "kstep"), (8, "none")])
@pytest.mark.parametrize("rgroup", [1, 2])
@pytest.mark.parametrize("place", ["head", "tail"])
def test_record_layouts_are_injective_and_in_bounds(bits, plane, rgroup, place):
    r = generic.RECIPES["q4_K"]
    lay = record_layout(bits, recipe_meta(r), lanes=16, plane=plane, kblock=256, period=256, rgroup=rgroup, place=place,
                        corr_fields=(("wsum", "i16"),), align=64)
    seen = set()
    epb = 8 // bits
    code_bytes = set()
    for row in range(lay.rows):
        for k in range(lay.kblock):
            e = lay.code(row, k)
            assert e not in seen
            seen.add(e)
            code_bytes.add(e // epb)
    assert len(seen) == lay.rows * lay.kblock
    meta = set()
    for f in lay.fields:
        size = {"f16": 2, "f32": 4, "u8": 1, "i16": 2}[f.ctype]
        for i in range(f.layout.size()):
            for b in range(size):
                meta.add(f.layout(i) + b)
    assert not (meta & code_bytes), "metadata overlaps codes"
    assert max(meta | code_bytes) < lay.rec_bytes and lay.rec_bytes % 64 == 0
