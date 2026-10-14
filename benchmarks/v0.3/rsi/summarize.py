import json, statistics, collections, sys
rows=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'results.jsonl')]
ref={}
for r in rows:
    if r['config']=='none': ref.setdefault((r['workload'],r['prompt']), r['text'])
by=collections.defaultdict(list)
for r in rows: by[(r['workload'],r['config'])].append(r)
order=['none','draft_selector','ngram','ngram_selector','union_naive','arms_v1_eager','arms_v1','arms']
for wl in ['refine','think','eight']:
    base=None
    for cfg in order:
        rs=by.get((wl,cfg))
        if not rs: continue
        if wl=='eight':
            reps=collections.defaultdict(list)
            for r in rs: reps[r['rep']].append(r)
            # aggregate tok/s per rep = total tokens / total predicted time
            agg=[sum(r['n'] for r in v)/sum(r['n']/r['tok_s'] for r in v) for v in reps.values()]
        else:
            agg=[r['tok_s'] for r in rs]
        med=statistics.median(agg)
        if cfg=='none': base=med
        tot=[ (r['prompt_ms']+r['predicted_ms'])/1000 for r in rs if 'prompt_ms' in r]
        ident=sum(r['text']==ref.get((wl,r['prompt'])) for r in rs)
        print(f"{wl:7s} {cfg:15s} tok/s {' '.join(f'{x:6.1f}' for x in agg):30s} median {med:6.1f}  x{med/base:4.2f}  identical {ident}/{len(rs)}  end-to-end s {statistics.median(tot) if tot else float('nan'):.2f}")
