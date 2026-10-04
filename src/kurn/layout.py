"""CuTe-style composable layout algebra, adapted to CPU weight packing.

Part 1 is the algebra (after NVIDIA CuTe; written from its published semantics):

  * an IntTuple is an int or a (nested) tuple of IntTuples;
  * a Layout is a pair (shape, stride) of congruent IntTuples and is a function
    from coordinates to offsets: crd -> sum_i crd_i * stride_i, where an integer
    coordinate is first unpacked colexicographically into a natural coordinate;
  * operations: coalesce, composition (A o B), complement, right/left inverse,
    logical_divide (tiling), logical/blocked/raked product, zipped/tiled divide,
    make_ordered_layout, and Swizzle / ComposedLayout (XOR swizzles).

Every Layout can emit itself as a C index expression (`c_expr`), so generated
code computes addresses with the same function the Python side reasons about.

Part 2 adapts it to CPU kernels: a packed weight *record* (L rows x kblock K
values of one weight matrix) is described by a code layout (logical (row, k) ->
code element, element = `bits` wide, packed little-endian in bytes) plus
metadata field layouts (scales, corrections) and an outer record layout. The
code layout is built from primitives:

  lanes_atom(L, D)       the vpdpbusd atom: L rows into L 32-bit lanes, D=4
                         consecutive K values per lane (rows-into-lanes interleave)
  logical_product        stack atoms into vectors and vectors into K-groups
  planes(EPB)            sub-byte bit planes: EPB = 8 / bits codes per byte;
                         which logical mode the planes bind to (`plane`):
                           kstep  consecutive K steps share a byte (mask/shift unpack)
                           khalf  the first / second half of a K-group share a byte
                           rows   2 row groups share one vector (lo / hi nibbles)
                           atom   1-bit codes laid along the atom (bitmask unpack)
  regroup                bind storage leaves to the logical (row, k) modes
  Swizzle                XOR swizzle of vectors by row-group index (set conflicts)
  padding / alignment    record rounding to 64 B, extra bytes per row-group stride

The fixed v0.1/v0.2 layouts are compositions of these primitives (PRESETS):
vnni16 (Q8_0, metadata after the codes), i16 / i8 (every recipe, metadata first,
kstep planes) and l32 (T-MAC LUT indices, 32 int16 lanes). sched.py lowers any
record layout to plain C: the repack (`_prepare`) loops evaluate the emitted
code-layout expression, and the kernel's load offsets, unpack shifts and
metadata addresses are obtained by evaluating the same layout.
"""

from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Optional

# --------------------------------------------------------------------------- IntTuple helpers


def is_int(x):
    return isinstance(x, int)


def is_tuple(x):
    return isinstance(x, tuple)


def flatten(t):
    if is_tuple(t):
        return tuple(chain.from_iterable(flatten(x) for x in t)) if t else ()
    return (t,)


def product(t):
    if is_tuple(t):
        return reduce(lambda a, b: a * b, (product(x) for x in t), 1)
    return t


def prefix_product(shape, init=1):
    """Exclusive prefix product, congruent with `shape` (compact column-major strides)."""
    if is_tuple(shape):
        out = []
        for s in shape:
            out.append(prefix_product(s, init))
            init *= product(s)
        return tuple(out)
    return init


def shape_div(a, b):
    """CuTe shape_div: a / b if b divides a, else 1 if a divides b (static shapes)."""
    if is_tuple(a):
        out = []
        for x in a:
            out.append(shape_div(x, b))
            b = shape_div(b, product(x))
        return tuple(out)
    if a % b == 0:
        return a // b
    if b % a == 0:
        return 1
    raise ValueError(f"shape_div: {a} and {b} are not divisible either way")


def ceil_div(a, b):
    return -(-a // b)


def _congruent(a, b):
    if is_tuple(a) != is_tuple(b):
        return False
    if is_tuple(a):
        return len(a) == len(b) and all(_congruent(x, y) for x, y in zip(a, b))
    return True


def idx2crd(idx, shape, stride=None):
    """Linear index -> natural coordinate (colexicographic)."""
    if stride is None:
        stride = prefix_product(shape)
    if is_tuple(shape):
        return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
    return (idx // stride) % shape


def crd2idx(crd, shape, stride):
    """Coordinate (int or congruent tuple) -> offset. An int coordinate for a tuple
    shape is unpacked colexicographically; the last mode is not wrapped (CuTe)."""
    if is_tuple(crd):
        if not is_tuple(shape) or len(crd) != len(shape):
            raise ValueError(f"coordinate {crd} does not match shape {shape}")
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if crd is None:
        crd = 0
    if is_tuple(shape):
        out = 0
        for s, d in zip(shape[:-1], stride[:-1]):
            n = product(s)
            out += crd2idx(crd % n, s, d)
            crd //= n
        return out + crd2idx(crd, shape[-1], stride[-1])
    return crd * stride


# --------------------------------------------------------------------------- Layout


class Layout:
    """(shape, stride) layout. Strides may be strings (opaque C expressions) for the
    outermost, run-time sized modes; those layouts support `c_expr` and evaluation
    with an `env`, but not the algebra (which needs static integers)."""

    __slots__ = ("shape", "stride")

    def __init__(self, shape, stride=None):
        if stride is None:
            stride = prefix_product(shape)
        if not _congruent(shape, stride):
            raise ValueError(f"shape {shape} and stride {stride} are not congruent")
        self.shape, self.stride = shape, stride

    # -- structure
    def __len__(self):
        return len(self.shape) if is_tuple(self.shape) else 1

    def __getitem__(self, i):
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])
        if i != 0:
            raise IndexError(i)
        return self

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __eq__(self, other):
        return isinstance(other, Layout) and self.shape == other.shape and self.stride == other.stride

    def __hash__(self):
        return hash((self.shape, self.stride))

    def __repr__(self):
        return f"{_fmt(self.shape)}:{_fmt(self.stride)}"

    def size(self):
        return product(self.shape)

    def cosize(self):
        """1 + the largest offset (non-negative static strides)."""
        n = self.size()
        return self(n - 1) + 1 if n else 0

    def rank(self):
        return len(self)

    def depth(self):
        def d(t):
            return 1 + max((d(x) for x in t), default=0) if is_tuple(t) else 0

        return d(self.shape)

    # -- function
    def __call__(self, *crd, env=None):
        if len(crd) == 1:
            crd = crd[0]
        stride = _bind(self.stride, env) if env else self.stride
        return crd2idx(crd, self.shape, stride)

    def leaves(self):
        """Flat ((shape, stride), ...) of the leaf modes, colexicographic order."""
        return tuple(zip(flatten(self.shape), flatten(self.stride)))

    def image(self):
        return sorted({self(i) for i in range(self.size())})

    def is_injective(self):
        return len(self.image()) == self.size()

    def is_bijective(self):
        return self.image() == list(range(self.size()))

    # -- C emission
    def c_expr(self, crd, leaf_coords=None):
        """C expression for this layout at coordinate `crd`: one C expression per
        top-level mode (or one for a rank-1 layout). Hierarchical modes are unpacked
        colexicographically from that mode's expression with `/` and `%` by constants.
        `leaf_coords` maps a flat leaf index to a function(expr) -> expr that rewrites
        that leaf's coordinate (used for swizzles)."""
        crd = (crd,) if isinstance(crd, str) else tuple(crd)
        modes = list(self) if len(crd) > 1 else [self]
        if len(modes) != len(crd):
            raise ValueError(f"{len(crd)} coordinate expressions for a rank-{len(modes)} layout")
        terms, leaf = [], 0
        for mode, c in zip(modes, crd):
            lv = mode.leaves()
            div = 1
            for i, (s, d) in enumerate(lv):
                last = i == len(lv) - 1
                e = c if div == 1 else f"({c}) / {div}"
                if not last:
                    e = f"({e}) % {s}" if div != 1 else f"({c}) % {s}"
                if leaf_coords and leaf in leaf_coords:
                    e = leaf_coords[leaf](e)
                if s != 1 and d != 0:
                    terms.append(e if d == 1 else f"({e}) * {d}" if is_int(d) else f"({e}) * ({d})")
                div *= s
                leaf += 1
        return " + ".join(terms) if terms else "0"


def _fmt(t):
    if is_tuple(t):
        return "(" + ",".join(_fmt(x) for x in t) + ")"
    return str(t)


def _bind(t, env):
    if is_tuple(t):
        return tuple(_bind(x, env) for x in t)
    return env[t] if isinstance(t, str) else t


def make_layout(*layouts):
    """Concatenate layouts as the modes of a new layout."""
    if len(layouts) == 1 and not isinstance(layouts[0], Layout):
        layouts = tuple(layouts[0])
    return Layout(tuple(L.shape for L in layouts), tuple(L.stride for L in layouts))


def coalesce(layout, profile=None):
    """Flatten and merge modes (s0:d0)(s1:d1) with d1 == s0 * d0; drop size-1 modes.
    With a profile (tuple), coalesce each top-level mode separately (by-mode)."""
    if profile is not None and is_tuple(profile):
        return make_layout(*(coalesce(layout[i], profile[i] if i < len(profile) else None) for i in range(len(layout))))
    out_s, out_d = [1], [0]
    for s, d in layout.leaves():
        if s == 1:
            continue
        if out_s[-1] == 1:
            out_s[-1], out_d[-1] = s, d
        elif out_s[-1] * out_d[-1] == d:
            out_s[-1] *= s
        else:
            out_s.append(s)
            out_d.append(d)
    if len(out_s) == 1:
        return Layout(out_s[0], out_d[0])
    return Layout(tuple(out_s), tuple(out_d))


def filter_zeros(layout):
    """Replace stride-0 modes by size 1, then coalesce."""
    return coalesce(Layout(tuple(1 if d == 0 else s for s, d in layout.leaves()), tuple(0 if d == 0 else d for s, d in layout.leaves())))


def composition(a, b):
    """R = A o B, R(c) = A(B(c)) on the domain of B. `b` may be a Layout, an int
    (shorthand for b:1) or a tuple of tilers applied mode-by-mode."""
    if b is None:
        return a
    if is_int(b):
        return composition(a, Layout(b))
    if is_tuple(b):
        if len(a) < len(b):
            raise ValueError("tiler has more modes than the layout")
        return make_layout(*(composition(a[i], b[i]) for i in range(len(b))), *(a[i] for i in range(len(b), len(a))))
    if is_tuple(b.shape):
        return make_layout(*(composition(a, bi) for bi in b))
    if b.stride == 0:
        return Layout(b.shape, 0)
    res_s, res_d = [], []
    rest_s, rest_d = b.shape, b.stride
    fa = coalesce(a)
    lv = fa.leaves()
    for s, d in lv[:-1]:
        if not (s % rest_d == 0 or rest_d % s == 0):
            raise ValueError(f"composition: stride {rest_d} does not divide shape {s} (inadmissible)")
        ns = min(max(1, s // rest_d), rest_s)
        if ns != 1:
            res_s.append(ns)
            res_d.append(rest_d * d)
        rest_s //= ns
        rest_d = ceil_div(rest_d, s)
    if rest_s != 1 or not res_s:
        res_s.append(rest_s)
        res_d.append(rest_d * lv[-1][1])
    if len(res_s) == 1:
        return Layout(res_s[0], res_d[0])
    return Layout(tuple(res_s), tuple(res_d))


def complement(layout, cotarget=1):
    """The layout that fills the holes of `layout` up to `cotarget`: the concatenation
    (layout, complement) is a bijection onto [0, cosize) when layout is injective."""
    if is_int(layout):
        layout = Layout(layout)
    res_s, res_d = [], []
    cur = 1
    for d, s in sorted((d, s) for s, d in layout.leaves()):
        if d == 0 or s == 1:
            continue
        if cur > s * d:
            raise ValueError(f"complement: {layout} is not injective")
        res_s.append(d // cur)
        res_d.append(cur)
        cur = s * d
    res_s.append(ceil_div(cotarget, cur))
    res_d.append(cur)
    return coalesce(Layout(tuple(res_s), tuple(res_d)))


def right_inverse(layout):
    """R with layout(R(i)) == i for i < size(R) (the largest such contiguous range)."""
    res_s, res_d = [], []
    cur = 1
    fs, fd = flatten(layout.shape), flatten(layout.stride)
    for d, s, r in sorted(zip(fd, fs, prefix_product(fs))):
        if s == 1:
            continue
        if cur != d:
            break
        res_s.append(s)
        res_d.append(r)
        cur = s * d
    if not res_s:
        return Layout(1, 0)
    return coalesce(Layout(tuple(res_s), tuple(res_d)))


def left_inverse(layout):
    """L with L(layout(i)) == i for every i in the domain (layout injective)."""
    return right_inverse(make_layout(layout, complement(layout)))


def logical_divide(a, tiler):
    """Tile `a` by `tiler`: result mode 0 = the tile, mode 1 = the tile index ("rest").
    A tuple tiler divides mode-by-mode."""
    if tiler is None:
        return a
    if is_int(tiler):
        tiler = Layout(tiler)
    if is_tuple(tiler):
        return make_layout(*(logical_divide(a[i], tiler[i]) for i in range(len(tiler))), *(a[i] for i in range(len(tiler), len(a))))
    return composition(a, make_layout(tiler, complement(tiler, a.size())))


def zipped_divide(a, tiler):
    """((tile modes...), (rest modes...)) for a mode-wise tiler."""
    if not is_tuple(tiler):
        return logical_divide(a, tiler)
    d = logical_divide(a, tiler)
    tiles = make_layout(*(d[i][0] for i in range(len(tiler))))
    rests = make_layout(*(d[i][1] for i in range(len(tiler))), *(d[i] for i in range(len(tiler), len(a))))
    return make_layout(tiles, rests)


def tiled_divide(a, tiler):
    z = zipped_divide(a, tiler)
    return make_layout(z[0], *z[1])


def logical_product(a, b):
    """(A, complement(A, size(A) * cosize(B)) o B): B's pattern of copies of A."""
    if is_int(b):
        b = Layout(b)
    if is_tuple(b):
        return make_layout(*(logical_product(a[i], b[i]) for i in range(len(b))), *(a[i] for i in range(len(b), len(a))))
    return make_layout(a, composition(complement(a, a.size() * b.cosize()), b))


def blocked_product(a, b):
    """Rank-wise product: mode i = (a_i, b_i) (A tiles are contiguous blocks)."""
    r = max(len(a), len(b))
    a, b = _append(a, r), _append(b, r)
    p = logical_product(a, b)
    return make_layout(*(make_layout(p[0][i], p[1][i]) for i in range(r)))


def raked_product(a, b):
    """Rank-wise product with B innermost: mode i = (b_i, a_i) (A elements interleaved)."""
    r = max(len(a), len(b))
    a, b = _append(a, r), _append(b, r)
    p = logical_product(a, b)
    return make_layout(*(make_layout(p[1][i], p[0][i]) for i in range(r)))


def _append(layout, r):
    modes = list(layout) if is_tuple(layout.shape) else [layout]
    while len(modes) < r:
        modes.append(Layout(1, 0))
    return make_layout(*modes)


def make_ordered_layout(shape, order):
    """Compact layout of `shape` whose strides increase in the rank order `order`
    (order[i] = rank of mode i; smallest rank gets stride 1)."""
    shape, order = tuple(shape), tuple(order)
    stride = [0] * len(shape)
    cur = 1
    for i in sorted(range(len(shape)), key=lambda i: order[i]):
        stride[i] = cur
        cur *= product(shape[i])
    return Layout(shape, tuple(stride))


def regroup(layout, groups):
    """New layout whose mode g is made of the flat leaves `groups[g]` (leaf indices of
    `layout`), in that order. Used to bind storage leaves to logical modes."""
    lv = layout.leaves()
    used = sorted(i for g in groups for i in g)
    if used != list(range(len(lv))):
        raise ValueError(f"regroup must use every leaf exactly once (got {groups} for {len(lv)} leaves)")
    modes = []
    for g in groups:
        s = tuple(lv[i][0] for i in g)
        d = tuple(lv[i][1] for i in g)
        modes.append(Layout(s[0], d[0]) if len(g) == 1 else Layout(s, d))
    return make_layout(*modes)


@dataclass(frozen=True)
class Swizzle:
    """CuTe Swizzle<B, M, S>: XOR the B bits at [M+S, M+S+B) into [M, M+B).
    An involution (and so a bijection) on the integers."""

    bits: int
    base: int
    shift: int

    def __call__(self, off):
        if self.bits == 0:
            return off
        msk = ((1 << self.bits) - 1) << (self.base + self.shift)
        return off ^ ((off & msk) >> self.shift)

    def c_expr(self, e):
        if self.bits == 0:
            return e
        msk = ((1 << self.bits) - 1) << (self.base + self.shift)
        return f"(({e}) ^ ((({e}) & {msk}) >> {self.shift}))"


@dataclass(frozen=True)
class ComposedLayout:
    """outer(offset + inner(crd)); outer is a Swizzle or a Layout."""

    outer: object
    offset: int
    inner: Layout

    def __call__(self, *crd):
        return self.outer(self.offset + self.inner(*crd))

    def size(self):
        return self.inner.size()


# --------------------------------------------------------------------------- CPU packed records

VNNI_DEPTH = 4  # vpdpbusd: 4 consecutive u8 x s8 products per 32-bit lane


def lanes_atom(lanes, depth=VNNI_DEPTH):
    """rows -> lanes interleave: (row, j) -> slot, j fastest (one lane = `depth` values of one row)."""
    return Layout((lanes, depth), (depth, 1))


def planes(epb):
    """Bit planes inside a byte: element = plane + epb * slot."""
    return Layout(epb, 1)


@dataclass(frozen=True)
class Field:
    """A metadata field of a record: `count` values of `size` bytes per (row, period)."""

    name: str
    ctype: str  # f16 | f32 | u8 | i16
    count: int
    layout: Layout  # (row, idx[, period]) -> byte offset in the record
    per_kg: bool = False  # lives inside each K-group chunk (correction sums)


@dataclass(frozen=True)
class RecordLayout:
    """A packed weight record: R rows x kblock K values.

    code: (row, k) -> code element index (bits wide, little-endian packed); the element
          index counts from the start of the record in `bits` units.
    fields: metadata (scales, sums), byte offsets.
    rec_bytes: record size including padding; rg_pad: extra bytes per row-group stride.
    Records are stored row-group-major: record (gr, p) at (gr * nrec_k + p) * rec_bytes
    + gr * rg_pad (the outer layout, see `outer`).
    """

    params: dict
    bits: int
    lanes: int  # rows per vector (lane width)
    rows: int  # R = rows per record
    kblock: int
    code: Layout
    fields: tuple
    code_base: int  # first code byte
    kg_bytes: int  # bytes per K-group chunk (codes of every row group + per-kg fields)
    lg_code_bytes: int  # code bytes per (lane group, K-group)
    rec_bytes: int
    rg_pad: int
    swizzle: Optional[Swizzle] = None
    vec_bytes: int = 64
    doc: str = ""

    @property
    def epb(self):
        return 8 // self.bits

    def field(self, name):
        return next((f for f in self.fields if f.name == name), None)

    def outer(self):
        """(record index along K, row group) -> record byte offset; run-time sized."""
        return Layout((1, 1), (self.rec_bytes, f"nrec_k * {self.rec_bytes} + {self.rg_pad}"))

    def element(self, row, k, gr=0):
        """Code element index of (row, k) inside the record of row group `gr` (with swizzle)."""
        e = self.code(row, k)
        if self.swizzle is None:
            return e
        byte, sub = divmod(e, self.epb)
        return self.swizzle_byte(byte, gr) * self.epb + sub

    def swizzle_byte(self, byte, gr):
        """XOR-swizzle the vector index of a code byte inside its (lane group, K-group)
        chunk by the low bits of the row-group index (Swizzle<B, log2(VB), S> applied to
        the virtual offset chunk_off + (gr << S))."""
        if self.swizzle is None:
            return byte
        rel = byte - self.code_base
        kg, rem = divmod(rel, self.kg_bytes)
        if rem >= self.lg_code_bytes * self.params["rgroup"]:
            return byte  # per-kg fields are not swizzled
        lg, off = divmod(rem, self.lg_code_bytes)
        sw = self.swizzle
        virt = off + (gr << (sw.base + sw.shift))
        off2 = sw(virt) & ((1 << (sw.base + sw.shift)) - 1)
        return self.code_base + kg * self.kg_bytes + lg * self.lg_code_bytes + off2

    def describe(self):
        return f"code {self.code}  fields " + ", ".join(f"{f.name}{f.layout}" for f in self.fields) + f"  rec {self.rec_bytes} B"


def _meta_fields(recipe_meta, lanes_total, periods):
    """Lay metadata fields out field-major: field f of period q at a running offset,
    rows contiguous (stride = element size), entries `idx` strided by the row count.
    A 4-tuple (name, ctype, count, il) interleaves `il` consecutive entries per row
    (row, idx) -> (idx // il) * il * R + il * row + idx % il  (in elements)."""
    fields, off = [], 0
    for q in range(periods):
        for name, ctype, count, *il in recipe_meta:
            size = {"f16": 2, "f32": 4, "u8": 1, "i16": 2}[ctype]
            if q == 0:
                fields.append([name, ctype, count, off, il[0] if il else 1])
            off += size * lanes_total * count
    per_bytes = off // periods if periods else 0
    out = []
    for name, ctype, count, base, il in fields:
        size = {"f16": 2, "f32": 4, "u8": 1, "i16": 2}[ctype]
        if il > 1:
            shape = (lanes_total, (il, count // il), periods)
            stride = (size * il, (size, size * il * lanes_total), per_bytes)
        else:
            shape = (lanes_total, count, periods)
            stride = (size, size * lanes_total, per_bytes)
        out.append((name, ctype, count, base, Layout(shape, stride)))
    return out, off


def record_layout(
    bits,
    meta,
    *,
    lanes=16,
    plane="kstep",
    kblock=32,
    period=32,
    rgroup=1,
    place="head",
    corr_fields=(),
    align=0,
    rg_pad=0,
    swizzle=0,
    depth=VNNI_DEPTH,
    doc="",
):
    """Compose a packed record layout from primitives.

    bits      code width (8, 4, 2, 1)
    meta      ((name, ctype, count), ...) metadata per row per scale period
    lanes     rows per vector (lane width L); one vector = L * depth slots
    plane     bit-plane binding for sub-byte codes (kstep | khalf | rows | atom | none)
    kblock    K values per record (multiple of 32 and of `period`)
    rgroup    vector groups (lane groups) per record: interleaves row groups into one stream
    place     metadata before (head) or after (tail) the codes
    corr_fields ((name, ctype), ...) per-row values stored inside each 32-value K-group
    align     round the record up to a multiple of `align` bytes (0: packed)
    rg_pad    extra bytes added to the row-group stride (breaks 4 KiB aliasing)
    swizzle   B: XOR the vector index inside a (lane group, K-group) chunk with B row-group bits
    """
    if kblock % 32 or kblock % period and period % kblock:
        raise ValueError(f"kblock {kblock} must be a multiple of 32 and of the scale period {period}")
    if period > kblock:
        raise ValueError(f"kblock {kblock} is smaller than the scale period {period}")
    epb = 8 // bits
    if bits == 8:
        plane = "none"
    if plane in ("none",) and bits != 8 or plane == "atom" and bits != 1:
        raise ValueError(f"plane={plane} is not valid for {bits}-bit codes")
    vb = lanes * depth  # bytes per vector (8-bit slots) or bits per vector (atom planes)
    steps = 32 // depth  # K steps per 32-value K-group
    kgs = kblock // 32
    # rows per chunk (one vector group) and code bytes per (vector group, K-group)
    if plane == "rows":
        rows_vg, nvec = lanes * epb, steps
        lg_code = nvec * vb
    elif plane == "atom":
        rows_vg, nvec = lanes, steps
        lg_code = nvec * vb // 8
    elif plane in ("kstep", "khalf"):
        rows_vg = lanes
        if steps % epb:
            raise ValueError(f"{steps} K steps do not split into {epb} planes")
        nvec = steps // epb
        lg_code = nvec * vb
    else:  # 8-bit
        rows_vg, nvec = lanes, steps
        lg_code = nvec * vb
    rows = rows_vg * rgroup
    corr_bytes = sum({"f16": 2, "f32": 4, "u8": 1, "i16": 2}[t] for _, t in corr_fields) * rows
    kg_bytes = rgroup * lg_code + corr_bytes
    meta_l, meta_bytes = _meta_fields(meta, rows, kblock // period)
    code_bytes = kgs * kg_bytes
    code_base = meta_bytes if place == "head" else 0
    meta_base = 0 if place == "head" else code_bytes
    rec = meta_bytes + code_bytes
    if align:
        rec = ceil_div(rec, align) * align

    # -- code layout, composed from primitives (all strides in element units)
    atom = lanes_atom(lanes, depth)  # (row, j) -> slot
    if plane == "atom":  # 1-bit: slots are bits, 8 per byte
        vecs = logical_product(atom, Layout(nvec))  # ((r, j), step) -> bit
        chunk = logical_product(vecs, Layout(rgroup))  # + vector group
        # leaves: r, j, step, vg
        flat = coalesce_keep(make_layout(chunk, Layout(kgs, kg_bytes * 8)))
        leaves = flat.leaves()  # r j step vg kg
        base = code_base * 8
        code = regroup(Layout(tuple(s for s, _ in leaves), tuple(d for _, d in leaves)), [[0, 3], [1, 2, 4]])
    else:
        vecs = logical_product(atom, Layout(nvec))  # ((r, j), vec) -> byte (slot)
        elems = logical_product(planes(epb), vecs) if epb > 1 else vecs  # (plane, ((r, j), vec)) -> element
        chunk = logical_product(elems, Layout(rgroup))
        flat = coalesce_keep(make_layout(chunk, Layout(kgs, kg_bytes * epb)))
        lv = flat.leaves()
        base = code_base * epb
        if epb > 1:  # leaves: plane, r, j, vec, vg, kg
            P, R, J, V, G, KG = range(6)
            if plane == "kstep":
                groups = [[R, G], [J, P, V, KG]]
            elif plane == "khalf":
                groups = [[R, G], [J, V, P, KG]]
            else:  # rows: the planes carry consecutive row groups
                groups = [[R, P, G], [J, V, KG]]
        else:  # leaves: r, j, vec, vg, kg
            R, J, V, G, KG = range(5)
            groups = [[R, G], [J, V, KG]]
        code = regroup(Layout(tuple(s for s, _ in lv), tuple(d for _, d in lv)), groups)
    code = _offset(code, base)

    fields = []
    for name, ctype, count, fbase, lay in meta_l:
        fields.append(Field(name, ctype, count, _offset(lay, meta_base + fbase)))
    off = rgroup * lg_code
    for name, ctype in corr_fields:
        size = {"f16": 2, "f32": 4, "u8": 1, "i16": 2}[ctype]
        lay = Layout((rows, kgs), (size, kg_bytes))
        fields.append(Field(name, ctype, 1, _offset(lay, code_base + off), per_kg=True))
        off += size * rows
    sw = None
    if swizzle:
        if (1 << swizzle) > nvec:
            raise ValueError(f"swizzle {swizzle} needs >= {1 << swizzle} vectors per chunk (have {nvec})")
        vbytes = vb // 8 if plane == "atom" else vb
        sw = Swizzle(swizzle, vbytes.bit_length() - 1, 16)
    params = dict(
        bits=bits,
        lanes=lanes,
        plane=plane,
        kblock=kblock,
        period=period,
        rgroup=rgroup,
        place=place,
        align=align,
        rg_pad=rg_pad,
        swizzle=swizzle,
        meta=tuple(meta),
        corr=tuple(corr_fields),
    )
    return RecordLayout(
        params,
        bits,
        lanes,
        rows,
        kblock,
        code,
        tuple(fields),
        code_base,
        kg_bytes,
        lg_code,
        rec,
        rg_pad,
        sw,
        vb // 8 if plane == "atom" else vb,
        doc,
    )


class _Offset(Layout):
    """A layout plus a constant offset (c_expr / evaluation include it)."""

    __slots__ = ("off",)

    def __init__(self, shape, stride, off):
        super().__init__(shape, stride)
        self.off = off

    def __call__(self, *crd, env=None):
        return self.off + super().__call__(*crd, env=env)

    def __getitem__(self, i):
        return Layout.__getitem__(self, i)

    def c_expr(self, crd, leaf_coords=None):
        e = super().c_expr(crd, leaf_coords)
        return e if not self.off else f"{self.off} + {e}"

    def __repr__(self):
        return (f"{self.off}+" if self.off else "") + super().__repr__()

    def __eq__(self, other):
        return isinstance(other, _Offset) and super().__eq__(other) and self.off == other.off

    __hash__ = Layout.__hash__


def _offset(layout, off):
    return _Offset(layout.shape, layout.stride, off)


def coalesce_keep(layout):
    """Flatten without merging (keeps every primitive leaf addressable for regroup)."""
    lv = layout.leaves()
    return Layout(tuple(s for s, _ in lv), tuple(d for _, d in lv))


# --------------------------------------------------------------------------- recipe -> metadata


def recipe_meta(recipe, scales="unpacked"):
    """Metadata fields per row per scale period for a generic.Recipe."""
    if recipe.two_level:
        if scales == "packed":
            return (("d", "f16", 1), ("dmin", "f16", 1), ("raw", "u8", 12))
        if scales == "fold":
            return (("dsc", "f32", 8), ("dmn", "f32", 8))
        return (("d", "f16", 1), ("dmin", "f16", 1), ("sc", "u8", 8), ("mn", "u8", 8))
    return (("d", "f16", 1),)


def lut_record_layout(bits, period, rows=32, doc="l32: T-MAC LUT indices"):
    """The l32 layout: 4-bit table indices (one per g = 4 / bits codes), 32 rows in
    int16 lanes; byte (pair * 32 + row) holds chunk 2*pair (low nibble) and 2*pair + 1
    (high nibble). Expressed in `bits`-wide code elements."""
    g = 4 // bits  # codes per index
    chunks = 32 // g
    idx_bytes = rows * chunks // 2
    kgs = period // 32
    hdr = 2 * rows
    epb = 8 // bits
    # leaves (code-element units): j (code in index), nib (index in byte), row, pair, kg
    leaves = ((g, 1), (2, g), (rows, 2 * g), (chunks // 2, rows * 2 * g), (kgs, idx_bytes * epb))
    flat = Layout(tuple(s for s, _ in leaves), tuple(d for _, d in leaves))
    code = _offset(regroup(flat, [[2], [0, 1, 3, 4]]), hdr * epb)
    fields = (Field("d", "f16", 1, Layout((rows, 1, 1), (2, 2 * rows, hdr))),)
    params = dict(
        bits=bits,
        lanes=rows,
        plane="lut",
        kblock=period,
        period=period,
        rgroup=1,
        place="head",
        align=0,
        rg_pad=0,
        swizzle=0,
        meta=(("d", "f16", 1),),
        corr=(),
    )
    return RecordLayout(params, bits, rows, rows, period, code, fields, hdr, idx_bytes, idx_bytes, hdr + kgs * idx_bytes, 0, None, 64, doc)


def preset(name, recipe, target="avx512_vnni", **kw):
    """The fixed layouts of kurn <= v0.2 as compositions (see module doc)."""
    lanes = 8 if target == "avx2_vnni" else 16
    corr = (("wsum", "i16"),) if kw.get("correction") == "weight" else ()
    meta = recipe_meta(recipe, kw.get("scales", "unpacked"))
    if name == "vnni16":
        return record_layout(
            8,
            meta,
            lanes=16,
            kblock=32,
            period=32,
            place="tail",
            align=64 if kw.get("align") == 64 else 0,
            doc="vnni16: 16 rows into lanes, codes then d[16]",
        )
    if name in ("i16", "i8"):
        plane = "atom" if recipe.bits == 1 else ("kstep" if recipe.bits < 8 else "none")
        return record_layout(
            recipe.bits,
            meta,
            lanes=lanes,
            plane=plane,
            kblock=recipe.period,
            period=recipe.period,
            place="head",
            corr_fields=corr,
            doc=f"{name}: {lanes} rows into lanes, metadata first",
        )
    if name == "l32":
        return lut_record_layout(recipe.bits, recipe.period)
    raise KeyError(name)
