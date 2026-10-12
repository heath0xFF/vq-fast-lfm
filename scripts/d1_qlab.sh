#!/bin/bash
# d1 quant lab (Mac/Metal). usage: scripts/d1_qlab.sh <name> <type> [imat|noimat] [extra llama-quantize args...]
# quantizes d1-3B-F16 -> models/d1q/<name>.gguf, serves it on :8097 (Metal), scores vs data/d1/ref_f16_seed1.json
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; MB=third_party/llama.cpp-master/build-mac/bin
name=$1; type=$2; im=${3:-imat}; shift 3 2>/dev/null; mkdir -p models/d1q
f=models/d1q/$name.gguf
if [ ! -f "$f" ]; then
  IM=""; [ "$im" = imat ] && IM="--imatrix data/d1/imatrix-d1.dat"
  $MB/llama-quantize $IM "$@" models/d1/d1-3B-F16.gguf "$f" "$type" > data/d1/$name.quant.log 2>&1 || { echo "quantize failed: $name"; tail -3 data/d1/$name.quant.log; exit 1; }
fi
pkill -x llama-server 2>/dev/null; sleep 1
$MB/llama-server -m "$f" -ngl 99 -fa on --port 8097 > /tmp/d1_q_$name.log 2>&1 &
for i in $(seq 1 90); do sleep 2; curl -s localhost:8097/health | grep -q ok && break; done
line=$(python3 scripts/d1_fidelity.py --seed 1 --ref data/d1/ref_f16_seed1.json --label "$name")
pkill -x llama-server; sleep 1
echo "$line  | $(stat -f%z "$f" | awk '{printf "%.2f GB", $1/1e9}')" | tee -a results/d1_quant.txt
