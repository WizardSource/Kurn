#!/usr/bin/env python3
"""Pairwise greedy agreement from results/greedy.txt (written by `measure.sh greedy-MODEL`):
for every model and pair of implementations, the first differing generated token per prompt.

  greedy_pairs.py results/greedy.txt
"""

import itertools
import re
import sys


def main():
    runs = {}
    for line in open(sys.argv[1]):
        m = re.match(r"(\S+) \[(.*)\] (\S+) first_diff\S* tokens:(.*)", line)
        if m:
            model, prompt, impl, toks = m.groups()
            runs.setdefault(model, {}).setdefault(impl, {})[prompt] = toks.split()
    for model, impls in runs.items():
        names = list(impls)
        prompts = list(impls[names[0]])
        print(f"{model}: first differing token per prompt ({len(prompts)} prompts x 64 tokens; 64 = identical)")
        for a, b in itertools.combinations(names, 2):
            d = []
            for p in prompts:
                x, y = impls[a].get(p, []), impls[b].get(p, [])
                d.append(next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))))
            print(f"  {a:9s} vs {b:9s} {' '.join(f'{v:2d}' for v in d)}   identical {sum(v == 64 for v in d)}/{len(d)}")


if __name__ == "__main__":
    main()
