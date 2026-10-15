#!/bin/bash
# pipeline.sh TAG HF_DIR   -> /tmp/gguf/TAG-*.gguf baselines + OA variants, calibration stats, F16 KL base, ppl matrix
set -x
TAG=$1; HF=$2
B=/tmp/llama-kurn/build/bin; P=/tmp/proto; G=/tmp/gguf
. /tmp/venv/bin/activate
cd $G
F=$TAG-f16.gguf
[ -f $F ] || python /tmp/llama-kurn/convert_hf_to_gguf.py $HF --outtype f16 --outfile $F
for t in Q8_0 Q4_0 Q4_K_M Q2_K Q2_0; do [ -f $TAG-$t.gguf ] || $B/llama-quantize $F $TAG-$t.gguf $t 8 > /dev/null 2>&1; done
[ -f $TAG.imatrix.gguf ] || $B/llama-imatrix -m $TAG-Q8_0.gguf -f /tmp/calib.txt -c 512 --chunks 64 -t 8 -o $TAG.imatrix.gguf > /tmp/imatrix-$TAG.log 2>&1
for t in Q4_0 Q4_K_M Q2_K IQ2_XS IQ2_S; do
  # Q4_0: keep every linear weight in Q4_0 (imatrix mode would pick Q4_1 for some ffn_down, which runs in ggml's
  # unguarded AMX buffer on this VM)
  ex=; [ $t = Q4_0 ] && ex="--tensor-type ffn_down=q4_0"
  [ -f $TAG-$t-im.gguf ] || $B/llama-quantize $ex --imatrix $TAG.imatrix.gguf $F $TAG-$t-im.gguf $t 8 > /dev/null 2>&1
done
S=$P/calib-$TAG.json.pt
[ -f $S ] || python $P/qstream.py $HF --mode stats --text /tmp/calib.txt --calib 32 --out $P/calib-$TAG.json > /tmp/calib-$TAG.log 2>&1
oa() { [ -f $3 ] || python $P/oaquant.py --f16 $F --base $1 --stats $S --out $3 $2 > $3.log 2>&1; }
# one at a time: oaquant quantizes with every core (OA_THREADS)
# OAk = k side channels, no bias; OAkb = with the mean-fold bias (ablation)
oa $TAG-Q4_0.gguf "--no-side" $TAG-im-q4_0.gguf
oa $TAG-Q4_0.gguf "--k 16 --no-bias" $TAG-oa16nb-q4_0.gguf
oa $TAG-Q4_K_M.gguf "--k 16 --no-bias" $TAG-oa16nb-q4_k_m.gguf
oa $TAG-Q4_K_M.gguf "--k 16" $TAG-oa16-q4_k_m.gguf
oa $TAG-IQ2_XS-im.gguf "--no-side" $TAG-im-iq2_xs.gguf
oa $TAG-IQ2_XS-im.gguf "--k 16 --no-bias" $TAG-oa16nb-iq2_xs.gguf
oa $TAG-IQ2_XS-im.gguf "--k 16" $TAG-oa16-iq2_xs.gguf
oa $TAG-Q2_0.gguf "--k 16 --no-bias" $TAG-oa16nb-q2_0.gguf
oa $TAG-Q2_0.gguf "--k 64 --no-bias" $TAG-oa64nb-q2_0.gguf
oa $TAG-Q2_0.gguf "--no-side" $TAG-im-q2_0.gguf
[ -f $TAG-r8.gguf ] || python $P/r8quant.py --f16 $F --base $TAG-Q8_0.gguf --out $TAG-r8.gguf --clip absmax > /dev/null 2>&1
[ -f $TAG-r8mse.gguf ] || python $P/r8quant.py --f16 $F --base $TAG-Q8_0.gguf --out $TAG-r8mse.gguf --clip mse > /dev/null 2>&1
[ -f $TAG.kld ] || GGML_KURN=0 $B/llama-perplexity -m $F -f /tmp/wiki.txt -c 512 --chunks 64 -t 8 --kl-divergence-base $TAG.kld > /tmp/kldbase-$TAG.log 2>&1
CHUNKS=32 $P/ppl_matrix.sh $TAG.kld $P/ppl-$TAG.tsv Q8_0=$TAG-Q8_0.gguf R8=$TAG-r8.gguf Q4_0=$TAG-Q4_0.gguf \
  Q4_0-im=$TAG-Q4_0-im.gguf im-Q4_0=$TAG-im-q4_0.gguf OA16nb-Q4_0=$TAG-oa16nb-q4_0.gguf Q4_K_M=$TAG-Q4_K_M.gguf \
  Q4_K_M-im=$TAG-Q4_K_M-im.gguf OA16nb-Q4_K_M=$TAG-oa16nb-q4_k_m.gguf OA16-Q4_K_M=$TAG-oa16-q4_k_m.gguf \
  IQ2_XS-im=$TAG-IQ2_XS-im.gguf im-IQ2_XS=$TAG-im-iq2_xs.gguf OA16nb-IQ2_XS=$TAG-oa16nb-iq2_xs.gguf \
  OA16-IQ2_XS=$TAG-oa16-iq2_xs.gguf IQ2_S-im=$TAG-IQ2_S-im.gguf Q2_K-im=$TAG-Q2_K-im.gguf im-Q2_0=$TAG-im-q2_0.gguf \
  OA16nb-Q2_0=$TAG-oa16nb-q2_0.gguf OA64nb-Q2_0=$TAG-oa64nb-q2_0.gguf R8mse=$TAG-r8mse.gguf
echo DONE >> $P/ppl-$TAG.tsv
