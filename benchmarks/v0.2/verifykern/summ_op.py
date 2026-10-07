import re,sys,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'(\d+) (\S+) \S+ (\S+) M=(\d+) .*: ([\d.]+) ms',l)
    if m: d[(m[3],m[2],int(m[4]))].append(float(m[5]))
Ms=[1,2,3,4,5,6,7,8,16]
for t in ('q8_0','q4_K','q6_K'):
    print(f'| {t} ms | '+' | '.join(f'M={M}' for M in Ms)+' |')
    print('|---|'+'---|'*len(Ms))
    for c,name in (('stockamx','stock (ggml AMX buffer)'),('before','before'),('after','**after**')):
        if (t,c,1) in d: print(f'| {name} | '+' | '.join(f"{st.median(d[(t,c,M)]):.1f}" if d[(t,c,M)] else '-' for M in Ms)+' |')
    print()
