#!/usr/bin/env python3
"""One fresh-process run of llama-kurn-xs-bench inside its own cgroup (memory.max = RAM budget,
optional io.max read cap), after dropping the guest page cache. Prints/returns one JSON record:
the bench output plus cgroup memory.peak and disk bytes read (io.stat rbytes)."""
import json, os, subprocess, sys, time, uuid

ROOT = os.path.expanduser('~/kw')
BIN = os.environ.get('XS_BIN', f'{ROOT}/llama-kurn/build/bin/llama-kurn-xs-bench')
CG = '/sys/fs/cgroup/xs'
DEV = '254:32'  # vdc

MODELS = {
    'gemma4': (f'{ROOT}/models/gemma4-26b-a4b-q4_0.gguf', f'{ROOT}/xs/gemma4.kxs'),
    'qwen3': (f'{ROOT}/models/qwen3-30b-a3b-q4_0-pure.gguf', f'{ROOT}/xs/qwen3.kxs'),
}


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def drop_caches():
    sh("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null")


def run(model, mode, prompt_file, n_gen=128, mem_mb=None, cache_mb=None, env=None, ubatch=512, ctx=2048, rbps=None,
        threads=8, extra=None, timeout=3600):
    """mode: ram (all in RAM, kurn) | mmap (stock ggml, no repack, mmap) | xs (expert streaming)"""
    gguf, kxs = MODELS[model]
    e = dict(os.environ)
    for k in list(e):
        if k.startswith('GGML_KXS') or k.startswith('LLAMA_EXPERT') or k.startswith('GGML_KURN'):
            del e[k]
    args = [BIN, '-m', gguf, '-f', prompt_file, '--chat', '-n', str(n_gen), '-t', str(threads), '-c', str(ctx),
            '-ub', str(ubatch), '--kv', 'q8_0']
    if mode == 'mmap':
        e['GGML_KURN'] = '0'
        args.append('--no-repack')
    elif mode == 'xs':
        e['LLAMA_EXPERT_STREAM'] = kxs
        e['LLAMA_EXPERT_CACHE_MB'] = str(cache_mb)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    if extra:
        args += extra
    drop_caches()
    cg = f'{CG}/r{uuid.uuid4().hex[:8]}'
    sh(f'sudo mkdir {cg}')
    try:
        if mem_mb:
            sh(f'echo {int(mem_mb) << 20} | sudo tee {cg}/memory.max >/dev/null')
        if rbps:
            sh(f'echo "{DEV} rbps={int(rbps)}" | sudo tee {cg}/io.max >/dev/null')
        t0 = time.time()
        p = subprocess.run(['bash', '-c', f'echo $$ | sudo tee {cg}/cgroup.procs >/dev/null; exec "$@"', '_'] + args,
                           env=e, capture_output=True, text=True, errors='replace', timeout=timeout)
        wall = time.time() - t0
        peak = int(open(f'{cg}/memory.peak').read())
        rbytes = 0
        for line in open(f'{cg}/io.stat'):
            if line.startswith(DEV):
                rbytes = int(dict(kv.split('=') for kv in line.split()[1:])['rbytes'])
        events = open(f'{cg}/memory.events').read()
    finally:
        subprocess.run(f'sudo rmdir {cg}', shell=True)
    rec = {'model': model, 'mode': mode, 'mem_mb': mem_mb, 'cache_mb': cache_mb, 'env': env or {}, 'ubatch': ubatch,
           'rbps': rbps, 'prompt': os.path.basename(prompt_file), 'wall_s': wall, 'cg_peak_mib': peak / 2**20,
           'cg_rbytes': rbytes, 'oom_kill': 'oom_kill 0' not in events, 'rc': p.returncode}
    if p.returncode == 0:
        try:
            rec.update(json.loads(p.stdout.strip().splitlines()[-1]))
        except Exception as ex:  # noqa
            rec['parse_error'] = str(ex)
    else:
        rec['stderr_tail'] = p.stderr[-2000:]
    return rec


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('mode')
    ap.add_argument('prompt')
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--mem', type=int)
    ap.add_argument('--cache', type=int)
    ap.add_argument('--ub', type=int, default=512)
    ap.add_argument('--rbps', type=float)
    ap.add_argument('--env', nargs='*', default=[])
    a = ap.parse_args()
    r = run(a.model, a.mode, a.prompt, a.n, a.mem, a.cache, dict(x.split('=', 1) for x in a.env), a.ub, rbps=a.rbps)
    print(json.dumps(r))
