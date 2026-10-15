#!/bin/bash
# after pipeline.sh: reference rows, KV analysis, GSM8K (strictly one heavy job at a time)
set -x
. /tmp/venv/bin/activate
B=/tmp/llama-kurn/build/bin; P=/tmp/proto; G=/tmp/gguf
cd $G
S=$P/calib-1.7b.json.pt
python $P/oaquant.py --f16 qwen3-1.7b-f16.gguf --base qwen3-1.7b-Q4_K_M.gguf --stats $S --out oa16nb-q4_k_m.gguf --k 16 --no-bias > /dev/null 2>&1
python $P/oaquant.py --f16 qwen3-1.7b-f16.gguf --base qwen3-1.7b-IQ2_XS-im.gguf --stats $S --out oa16nb-iq2_xs.gguf --k 16 --no-bias > /dev/null 2>&1
python $P/oaquant.py --f16 qwen3-1.7b-f16.gguf --base qwen3-1.7b-IQ2_XS-im.gguf --stats $S --out im-iq2_xs.gguf --no-side > /dev/null 2>&1
python $P/oaquant.py --f16 qwen3-1.7b-f16.gguf --base qwen3-1.7b-Q2_0.gguf --stats $S --out oa64nb-q2_0.gguf --k 64 --no-bias > /dev/null 2>&1
$P/ppl_matrix.sh 17b.kld $P/ppl-1.7b-extra.tsv OA16nb-Q4_K_M=oa16nb-q4_k_m.gguf OA16nb-IQ2_XS=oa16nb-iq2_xs.gguf im-IQ2_XS=im-iq2_xs.gguf OA64nb-Q2_0=oa64nb-q2_0.gguf
S=$P/calib-llama3b.json.pt
python $P/oaquant.py --f16 llama3b-f16.gguf --base llama3b-Q4_0.gguf --stats $S --out llama3b-oa16nb-q4_0.gguf --k 16 --no-bias > /dev/null 2>&1
python $P/oaquant.py --f16 llama3b-f16.gguf --base llama3b-Q4_K_M.gguf --stats $S --out llama3b-oa16nb-q4_k_m.gguf --k 16 --no-bias > /dev/null 2>&1
python $P/oaquant.py --f16 llama3b-f16.gguf --base llama3b-Q2_0.gguf --stats $S --out llama3b-oa64nb-q2_0.gguf --k 64 --no-bias > /dev/null 2>&1
CHUNKS=32 $P/ppl_matrix.sh llama3b.kld $P/ppl-llama3b-extra.tsv OA16nb-Q4_0=llama3b-oa16nb-q4_0.gguf OA16nb-Q4_K_M=llama3b-oa16nb-q4_k_m.gguf OA64nb-Q2_0=llama3b-oa64nb-q2_0.gguf
[ -f qwen3-1.7b-Q6_K.gguf ] || $B/llama-quantize qwen3-1.7b-f16.gguf qwen3-1.7b-Q6_K.gguf Q6_K 8 > /dev/null 2>&1
$P/ppl_matrix.sh 17b.kld $P/ppl-1.7b-extra.tsv Q6_K=qwen3-1.7b-Q6_K.gguf
GGML_KURN_FA_MODE=exact $P/ppl_matrix.sh 17b.kld $P/ppl-1.7b-extra-exact.tsv Q8_0-exact=qwen3-1.7b-Q8_0.gguf R8-exact=r8-1.7b.gguf
for t in llama3b qwen3-8b; do
  [ -f $t-Q6_K.gguf ] || $B/llama-quantize $t-f16.gguf $t-Q6_K.gguf Q6_K 8 > /dev/null 2>&1
  CHUNKS=32 $P/ppl_matrix.sh $t.kld $P/ppl-$t-extra.tsv Q6_K=$t-Q6_K.gguf
done
python $P/qstream.py /tmp/models/Qwen3-1.7B --mode kv --calib-text /tmp/calib.txt --calib 4 --eval 8 --out $P/kv-1.7b.json > /tmp/kv17.log 2>&1
python $P/qstream.py /tmp/models/Llama-3.2-3B-Instruct --mode kv --calib-text /tmp/calib.txt --calib 4 --eval 8 --out $P/kv-llama3b.json > /tmp/kvl.log 2>&1
for t in qwen3-1.7b llama3b qwen3-8b; do
  n=100; [ $t = qwen3-8b ] && n=60
  pre=$t; [ $t = qwen3-1.7b ] && pre=""
  for f in $t-Q8_0.gguf $t-Q4_K_M-im.gguf ${pre:+$pre-}oa16nb-q4_k_m.gguf $t-IQ2_XS-im.gguf ${pre:+$pre-}oa16nb-iq2_xs.gguf; do
    python $P/gsm8k.py $G/$f $n >> $P/gsm8k.tsv 2>/dev/null
  done
done
python $P/gsm8k.py $G/r8-1.7b.gguf 100 >> $P/gsm8k.tsv 2>/dev/null
python $P/gsm8k.py $G/llama3b-r8.gguf 100 >> $P/gsm8k.tsv 2>/dev/null
python $P/gsm8k.py $G/qwen3-8b-r8.gguf 60 >> $P/gsm8k.tsv 2>/dev/null
echo QUEUE2 DONE >> $P/gsm8k.tsv
