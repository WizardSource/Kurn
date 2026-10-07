import sys,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    p=l.split()
    if len(p)==6: d[(p[1],p[2],int(p[3]))].append(float(p[4]))
for model in ('Qwen3-8B-Q8_0','Qwen3-8B-Q4_K_M'):
    Ms=sorted({k[2] for k in d if k[0]==model})
    print(f'{model}: ms per verify step (median of rounds); M =', Ms)
    for b in ('stock','before','after','ik'):
        if (model,b,1) not in d: continue
        v=[st.median(d[(model,b,m)]) for m in Ms]
        print(f'  {b:7s}', ' '.join(f'{x:6.1f}' for x in v), '| x M=1:', ' '.join(f'{x/v[0]:.2f}' for x in v))
