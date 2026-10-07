import sys,re,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    p=l.split()
    if len(p)<10 or not p[0].isdigit(): continue
    cols=[x for x in l.split('|')]
    try:
        b=p[1]; pl=int(cols[3]); stg=float(cols[8])
    except Exception: continue
    d[(b,pl)].append(stg)
npl=sorted({k[1] for k in d}); builds=list(dict.fromkeys(k[0] for k in d))
print('| build | ' + ' | '.join(str(x) for x in npl)+' |')
for b in builds:
    print(f'| {b} | ' + ' | '.join(f"{st.median(d[(b,x)]):.1f}" for x in npl)+' |', ' reps', len(d[(b,npl[0])]))
