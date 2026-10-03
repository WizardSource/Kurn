"""Extension points for modules in `kurn.ext`.

The core registries (formats.FORMATS, generic.RECIPES, kernels.KERNELS) are plain
dicts that extension modules may add to directly. Everything that spec.py derives
from (op, weights, target) goes through the tables below instead, so a new layout,
algorithm key or format never needs an edit to spec.py or kernels.py:

    EXTRA_VALUES[key]   functions (op, weights, target) -> tuple of extra legal values
                        for an existing schedule key (e.g. a new `layout`)
    NEW_KEYS[key]       (legal_fn, default, changes_codegen) for a new schedule key
    EXTRA_INVALID       (predicate(config), message) pairs, like spec.INVALID_COMBOS
    RESOLVE_HOOKS       functions(config) -> None, run at the end of spec.resolve to
                        replace `auto` values for the extension's layouts
    LOWERINGS[layout]   (target, config) -> C source, for layouts of existing kernels
    TEST_BLOCKS[fmt]    (rng, nblocks, extreme) -> bytes, random blocks for the tests
    GOLDEN[name]        config dict for a golden codegen file (tests/golden/<name>.c)
"""

EXTRA_VALUES = {}
NEW_KEYS = {}
EXTRA_INVALID = []
RESOLVE_HOOKS = []
LOWERINGS = {}
TEST_BLOCKS = {}
GOLDEN = {}


def extra_values(key, fn):
    EXTRA_VALUES.setdefault(key, []).append(fn)


def new_key(key, legal_fn, default, changes_codegen=True):
    if key in NEW_KEYS:
        raise ValueError(f"schedule key {key!r} registered twice")
    NEW_KEYS[key] = (legal_fn, default, changes_codegen)


# --- layout ---
# A layout whose keys would explode the full product in spec.legal_configs (kurn.ext.layout's
# `composed`: lanes x plane x kblock x rgroup x ... x algorithm keys) registers an enumerator
# instead: ENUMERATORS[layout](op, weights, target) -> iterable of override dicts (a covering
# set); legal_configs resolves those and skips `layout` in its product.
ENUMERATORS = {}
# NEW_KEYS that only an enumerated layout uses; legal_configs pins them to their defaults
# (other extensions' keys still vary in the product).
ENUM_KEYS = set()
# AUTO_VALUES[key] = the value of a NEW_KEYS key that means `auto` (e.g. lanes=0); a search
# space derived from the registry leaves it out, like the string `auto`.
AUTO_VALUES = {}

# --- attn ---
# OPS[op]       module path of an op that lives outside the GEMV registry (its own spec keys,
#               generator, harness and tuner), e.g. "attn" -> "kurn.attention"
# COMMANDS[cmd] callable(argv) -> exit code, dispatched by `kurn <cmd> ...` (see cli.py)
OPS = {}
COMMANDS = {}

# --- gpu ---
# TARGET_BACKENDS[target] callable(cmd, argv) -> exit code: a target with its own spec keys,
#               generator and harness (e.g. "cuda" -> kurn.gpu). `kurn check|gen|build|verify|tune
#               SPEC` route there when the spec (or a target=... override) names that target.
TARGET_BACKENDS = {}
