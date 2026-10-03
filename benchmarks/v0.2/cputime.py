#!/usr/bin/env python3
"""Run a command; print its stdout, then to stderr:
'CPU_S <user+sys> WALL_S <wall> RSS_MB <peak> ANON_MB <peak anon> FILE_MB <peak file-backed>'
(memory sampled from /proc/<pid>/status every 0.1 s)."""
import resource
import subprocess
import sys
import threading
import time

t = time.time()
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
peak = {"VmRSS": 0, "RssAnon": 0, "RssFile": 0}


def sample():
    while p.poll() is None:
        try:
            for line in open(f"/proc/{p.pid}/status"):
                k = line.split(":")[0]
                if k in peak:
                    peak[k] = max(peak[k], int(line.split()[1]))
        except OSError:
            pass
        time.sleep(0.1)


th = threading.Thread(target=sample, daemon=True)
th.start()
out = p.communicate()[0]
th.join(0.5)
r = resource.getrusage(resource.RUSAGE_CHILDREN)
sys.stdout.write(out)
sys.stderr.write(f"CPU_S {r.ru_utime + r.ru_stime:.2f} WALL_S {time.time() - t:.2f} RSS_MB {peak['VmRSS'] / 1024:.0f} "
                 f"ANON_MB {peak['RssAnon'] / 1024:.0f} FILE_MB {peak['RssFile'] / 1024:.0f}\n")
