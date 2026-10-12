#!/bin/bash
# Quant lab (runs on the Mac, Metal). usage: scripts/qlab.sh <name> <base-type> [llama-quantize extra args...]
# Quantizes BF16 -> models/q/<name>.gguf (with imatrix) and scores KLD/PPL vs BF16 on wikitext-2 (40x512).
# Appends a TSV row to results/quant.tsv.   Use name "eval:<path>" as first arg to only evaluate an existing gguf.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MB=third_party/llama.cpp-master/build-mac/bin
name=$1; type=$2; shift 2
if [[ "$name" == eval:* ]]; then f=${name#eval:}; name=$(basename "$f" .gguf); else
  f=models/q/$name.gguf
  if [ ! -f "$f" ]; then
    $MB/llama-quantize --imatrix data/imatrix-wiki.dat "$@" models/LFM2.5-2.6B-BF16.gguf "$f" "$type" > "data/out/$name.quant.log" 2>&1 || { echo "quantize failed: $name"; tail -3 data/out/$name.quant.log; exit 1; }
  fi
fi
out=$($MB/llama-perplexity -m "$f" -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base data/bf16-wiki40.kld --kl-divergence 2>&1 | tee "data/out/$name.eval.log")
ppl=$(echo "$out" | sed "s/\x1b\[[0-9;]*m//g" | grep -a "^Mean PPL(Q)  " | head -1 | awk '{print $4}')
kld=$(echo "$out" | grep -a "Mean    KLD" | head -1 | awk '{print $3}')
top=$(echo "$out" | grep -a "^Same top p:" | head -1 | awk '{print $4}')
size=$(stat -f%z "$f")
printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$size" "$ppl" "$kld" "$top" | tee -a results/quant.tsv
