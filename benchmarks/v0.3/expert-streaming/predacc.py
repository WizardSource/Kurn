#!/usr/bin/env python3
"""Offline next-layer prediction accuracy (GGML_KXS_MEASURE=1: no speculative reads), decode only."""
import json, sys
sys.path.insert(0, '/home/ubuntu/kw')
from xsrun import run

out = open('/home/ubuntu/kw/results/predacc.jsonl', 'a')
for model in sys.argv[1:]:
    tot = {}
    for i in range(1, 6):
        r = run(model, 'xs', f'/home/ubuntu/kw/prompts/p{i}.txt', n_gen=128, cache_mb=8000, env={'GGML_KXS_MEASURE': 1})
        out.write(json.dumps(r) + '\n')
        out.flush()
        d = r['xs_decode']
        w = d['pred_n']
        for k in ['recall_ratio8', 'recall_ratio12', 'recall_ratio16', 'recall_raw8', 'recall_raw16', 'recall_prevtok', 'recall_prevlayer']:
            tot[k] = tot.get(k, 0) + d[k] * w
        tot['n'] = tot.get('n', 0) + w
        print(model, i, {k: round(d[k], 3) for k in d if k.startswith('recall')}, flush=True)
    print(model, 'ALL', {k: round(v / tot['n'], 4) for k, v in tot.items() if k != 'n'}, 'n', tot['n'], flush=True)
