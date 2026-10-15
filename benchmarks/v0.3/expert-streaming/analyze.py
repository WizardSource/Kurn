#!/usr/bin/env python3
"""Tables for the write-up: per model and budget, every config's decode / TTFT / bytes / hit rate,
paired ratios between steps, and the bitwise gate against the all-in-RAM kurn run (same prompt)."""
import glob, json, os, sys
sys.path.insert(0, os.path.expanduser('~/kw'))
from xsexp import load, summary, paired, ci

R = os.path.expanduser('~/kw/results')


def ram_hashes(model):
    return {r['prompt']: r['logits_hash'] for r in load(f'{R}/ram_{model}.jsonl') if r['label'] == 'kurn_ram' and r['rc'] == 0}


def gate(model, recs, ref):
    ok = bad = 0
    for r in recs:
        if r['mode'] != 'xs' or r['rc'] != 0:
            continue
        if ref.get(r['prompt']) == r['logits_hash']:
            ok += 1
        else:
            bad += 1
    return ok, bad


def main(models):
    for m in models:
        if not os.path.exists(f'{R}/ram_{m}.jsonl'):
            continue
        ref = ram_hashes(m)
        print(f'\n### {m}')
        rr = summary(load(f'{R}/ram_{m}.jsonl'))
        for lab, row in rr:
            print(f"| {lab} | uncapped | {row['decode_tps'][0]:.2f} ± {row['decode_tps'][1]:.2f} | {row['ttft_s'][0]:.2f} | – | – |")
        tot_ok = tot_bad = 0
        for mem in (2048, 4096, 8192):
            p = f'{R}/decode_{m}_{mem}.jsonl'
            if not os.path.exists(p):
                continue
            recs = load(p)
            ok, bad = gate(m, recs, ref)
            tot_ok += ok
            tot_bad += bad
            for lab, row in summary(recs, 'mmap'):
                if row is None:
                    print(f'| {lab} | {mem} | FAILED |')
                    continue
                mb = f"{row['mb_per_tok']:.0f}" if 'mb_per_tok' in row else f"({row['cg_gb']:.1f} GB/run)"
                hit = f"{row['hit_rate']:.3f}" if 'hit_rate' in row else '–'
                vs = row.get('vs_base')
                vs = f"{vs[0]:.2f}x ± {vs[1]:.2f}" if vs else ''
                print(f"| {lab} | {mem} | {row['decode_tps'][0]:.2f} ± {row['decode_tps'][1]:.2f} | {row['ttft_s'][0]:.2f} | {mb} | {hit} | {vs} |")
            for a, b in (('s2_cache', 's3_overlap'), ('s3_overlap', 's4_prefetch'), ('s2_cache', 's4_prefetch')):
                mm, h, n = paired(recs, a, b)
                print(f'  paired {mem}: {b} / {a} = {mm:.3f} ± {h:.3f} (n={n})')
            # TTFT paired
            mm, h, n = paired(recs, 'mmap', 's4_prefetch', 'ttft_s')
            print(f'  paired TTFT {mem}: s4 / mmap = {mm:.3f} ± {h:.3f}')
        print(f'  bitwise gate vs kurn_ram (same prompt): {tot_ok} identical, {tot_bad} different')


if __name__ == '__main__':
    main(sys.argv[1:] or ['gemma4', 'qwen3'])
