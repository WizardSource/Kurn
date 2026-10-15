#!/usr/bin/env python3
"""Interleaved, paired experiments: every repetition runs all configurations on the same prompt, in a
rotated order, each in a fresh process with the guest page cache dropped. Summaries pair each
configuration with the baseline per repetition (ratio of means, 95% t-interval of the per-pair ratios)."""
import json, math, os, statistics, sys, time
sys.path.insert(0, os.path.expanduser('~/kw'))
from xsrun import run

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18,
       14: 2.14, 19: 2.09}

PROMPTS = [os.path.expanduser(f'~/kw/prompts/p{i}.txt') for i in range(1, 6)]
GEMMA_BASE, QWEN_BASE, MARGIN = 1814, 916, 250


def cache_for(model, mem_mb):
    base = GEMMA_BASE if model == 'gemma4' else QWEN_BASE
    return max(1, mem_mb - base - MARGIN)


def experiment(name, configs, reps=5, prompts=PROMPTS, n_gen=128, out_dir=os.path.expanduser('~/kw/results')):
    """configs: list of (label, kwargs for run())"""
    path = f'{out_dir}/{name}.jsonl'
    f = open(path, 'a')
    for r in range(reps):
        prompt = prompts[r % len(prompts)]
        order = configs[r % len(configs):] + configs[:r % len(configs)]
        if r % 2:
            order = order[::-1]
        for label, kw in order:
            kw = dict(kw)
            kw.setdefault('n_gen', n_gen)
            t0 = time.time()
            rec = run(prompt_file=prompt, **kw)
            rec['label'] = label
            rec['rep'] = r
            rec['exp'] = name
            f.write(json.dumps(rec) + '\n')
            f.flush()
            print(f"[{name}] rep {r} {label:28s} rc={rec['rc']} decode={rec.get('decode_tps', 0):7.3f} tok/s "
                  f"ttft={rec.get('ttft_s', 0):7.2f}s hash={rec.get('logits_hash', '-')} "
                  f"hit={rec.get('xs_decode', {}).get('hit_rate', 0) if isinstance(rec.get('xs_decode'), dict) else '-'} "
                  f"peak={rec['cg_peak_mib']:.0f}MiB ({time.time() - t0:.0f}s)", flush=True)
    return path


def load(path):
    return [json.loads(l) for l in open(path)]


def ci(vals):
    n = len(vals)
    m = statistics.mean(vals)
    if n < 2:
        return m, float('nan')
    return m, T95.get(n - 1, 2.0) * statistics.stdev(vals) / math.sqrt(n)


def paired(recs, base, other, metric='decode_tps'):
    b = {r['rep']: r for r in recs if r['label'] == base and r['rc'] == 0}
    o = {r['rep']: r for r in recs if r['label'] == other and r['rc'] == 0}
    reps = sorted(set(b) & set(o))
    ratios = [o[k][metric] / b[k][metric] for k in reps if b[k][metric] > 0]
    m, h = ci(ratios)
    return m, h, len(ratios)


def summary(recs, base=None, metric='decode_tps'):
    labels = []
    for r in recs:
        if r['label'] not in labels:
            labels.append(r['label'])
    rows = []
    for lab in labels:
        rs = [r for r in recs if r['label'] == lab and r['rc'] == 0]
        if not rs:
            rows.append((lab, None))
            continue
        row = {'n': len(rs)}
        for k in ['decode_tps', 'ttft_s', 'cg_peak_mib']:
            row[k] = ci([r[k] for r in rs])
        xd = [r['xs_decode'] for r in rs if isinstance(r.get('xs_decode'), dict) and r['xs_decode'].get('open')]
        if xd:
            toks = sum(r['n_gen'] - 1 for r in rs)
            row['hit_rate'] = sum(x['hits'] + x['hits_inflight'] for x in xd) / max(1, sum(x['uses'] for x in xd))
            row['mb_per_tok'] = sum(x['bytes_demand'] + x['bytes_spec'] for x in xd) / toks / 1e6
            row['spec_mb_per_tok'] = sum(x['bytes_spec'] for x in xd) / toks / 1e6
        else:
            # mmap / ram: disk bytes of the whole run from the cgroup (load + prefill + decode)
            row['cg_gb'] = statistics.mean(r['cg_rbytes'] for r in rs) / 1e9
        if base and lab != base:
            row['vs_base'] = paired(recs, base, lab, metric)
        rows.append((lab, row))
    return rows


def print_summary(path, base=None, metric='decode_tps'):
    recs = load(path)
    print(f'== {os.path.basename(path)}  (base: {base})')
    for lab, row in summary(recs, base, metric):
        if row is None:
            print(f'  {lab:30s} FAILED')
            continue
        s = f"  {lab:30s} n={row['n']} decode {row['decode_tps'][0]:7.3f}±{row['decode_tps'][1]:.3f} tok/s  ttft {row['ttft_s'][0]:6.2f}s  peak {row['cg_peak_mib'][0]:6.0f} MiB"
        if 'hit_rate' in row:
            s += f"  hit {row['hit_rate']:.3f}  {row['mb_per_tok']:.0f} MB/tok (spec {row['spec_mb_per_tok']:.0f})"
        if 'cg_gb' in row:
            s += f"  disk {row['cg_gb']:.1f} GB/run"
        if 'vs_base' in row:
            m, h, n = row['vs_base']
            s += f"  | x{m:.3f} ±{h:.3f} (n={n})"
        print(s)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print_summary(p)
