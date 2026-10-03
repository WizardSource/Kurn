"""Workstream `q4fix` (weak 4-bit end to end): the `ilv` schedule key of the 4-bit nibble
lowering (kurn.generic.lower_nibble).

ilv = I > 1 stores the records of I consecutive row groups next to each other for every K
step, so a GEMV pass over I row groups (rows = I) reads one sequential stream instead of I
streams one row-group length apart. Pure re-ordering of the same records: results are
identical to ilv = 1. The GEMV and the verify kernels of one packed buffer must use the same
ilv (kurn/integration/llama.cpp/gen_ggml_sources.py passes it on).
"""

from .. import generic, hooks


def _ilv_values(op, f, t):
    if t in generic.TARGET_VEC and f in generic.RECIPES and generic.RECIPES[f].bits == 4 and op in ("gemv", "verify"):
        return (1, 2, 4)
    return (1,)


hooks.new_key("ilv", _ilv_values, 1)
# keep the enumerated product (spec.legal_configs) as it was: ilv only re-orders records
hooks.ENUM_KEYS.add("ilv")
hooks.EXTRA_INVALID.append(
    (lambda c: c.get("ilv", 1) != 1 and not (c["layout"] in ("i16", "i8") and generic.nibble_path(generic.RECIPES[c["weights"]], c)),
     "ilv applies to the 4-bit nibble lowering only (layout=i16/i8, unpack=mask16/perm/pair or MX scales)")
)

_A5 = {"target": "avx512_vnni", "layout": "i16"}
hooks.GOLDEN.update({
    "q4_0_gemv_avx512_i16_pair_rows4_ilv4": {"op": "gemv", "weights": "q4_0", **_A5, "unpack": "pair", "rows": 4, "ilv": 4},
})  # fmt: skip
