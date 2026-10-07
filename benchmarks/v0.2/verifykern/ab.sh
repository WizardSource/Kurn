#!/usr/bin/env bash
# ab.sh TYPE "M list" ROUNDS "ENV_A" "ENV_B" ... : alternate opbench runs of each env config (use "-" for none),
# print median (over rounds) of the per-run median ms, and of the per-run min, per config and M.
cd $HOME/work
T=$1; MS=$2; R=$3; shift 3
SH="4096:4096,4096:1024,4096:1024,4096:4096,4096:12288,4096:12288,12288:4096"
tmp=$(mktemp)
for r in $(seq $R); do
  for M in $MS; do
    i=0
    for E in "$@"; do
      i=$((i+1))
      ENVV=""; [ "$E" != "-" ] && ENVV="$E"
      line=$(env $ENVV Kurn-gpu/benchmarks/v0.2/benchlock.sh bin/opbench ${BUFT:-KURN} $T $M 8 9 8 $SH 2>/dev/null | grep "ms (min")
      echo "$i $M $line" >> $tmp
    done
  done
done
python3 - "$tmp" "$@" <<'EOF'
import sys,re,statistics as st,collections
f=sys.argv[1]; names=sys.argv[2:]
d=collections.defaultdict(list)
for l in open(f):
    m=re.match(r'(\d+) (\d+) .*: ([\d.]+) ms \(min ([\d.]+)',l)
    if m: d[(int(m[1]),int(m[2]))].append((float(m[3]),float(m[4])))
Ms=sorted({k[1] for k in d})
for i,n in enumerate(names,1):
    print(f"{n:40s}", "  ".join(f"M={M}: {st.median(x[0] for x in d[(i,M)]):6.2f}/{st.median(x[1] for x in d[(i,M)]):6.2f}" for M in Ms))
EOF
rm -f $tmp
