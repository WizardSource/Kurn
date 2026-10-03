"""Compile every legal configuration of the fourbit tuner specs into KURN_CACHE_DIR, outside
the timing lock, so `kurn tune` under benchlock.sh only measures.
    nice -n 19 python prebuild_tune.py specs/*.kurn
"""

import itertools
import sys
from concurrent.futures import ThreadPoolExecutor

from kurn.spec import SpecError, load, resolve
from kurn.toolchain import build


def configs(path):
    spec, space = load(path)
    keys = list(space)
    seen = {}
    for combo in itertools.product(*(space[k] for k in keys)):
        try:
            c = resolve(spec, dict(zip(keys, combo)))
        except SpecError:
            continue
        seen[repr(sorted((k, v) for k, v in c.items() if k != "threads"))] = c
    return list(seen.values())


def main(paths):
    cs = [c for p in paths for c in configs(p)]
    print(f"{len(cs)} distinct builds")
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(build, cs)):
            if i % 20 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
