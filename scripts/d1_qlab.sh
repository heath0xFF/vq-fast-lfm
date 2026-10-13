#!/bin/bash
# Quantize and score d1 on Mac/Metal without stopping other model servers.
# usage: d1_qlab.sh <name> <type> [imat|noimat] [extra llama-quantize args...]
# IMATRIX can select a rebuilt matrix; use a new name for each quantization recipe.
set -eo pipefail
[ "$#" -ge 2 ] || { echo "usage: d1_qlab.sh name type [imat|noimat] [quantize-args...]" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MB=third_party/llama.cpp-master/build-mac/bin
name=$1; type=$2; shift 2
im=${1:-imat}; [ "$#" -eq 0 ] || shift
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "Invalid quantization name" >&2; exit 1; }
[[ "$im" = imat || "$im" = noimat ]] || { echo "Expected imat or noimat" >&2; exit 1; }
url=http://127.0.0.1:8097/health
if curl -s --max-time 2 -o /dev/null "$url"; then
  echo "Port 8097 already serves HTTP; refusing to use or stop it" >&2
  exit 1
fi
pid=; tmp=
cleanup() {
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  # A stopped or failed quantizer must not leave a reusable model file.
  [ -z "$tmp" ] || rm -f "$tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p models/d1q data/d1 results
f=models/d1q/$name.gguf
if [ ! -f "$f" ]; then
  IM=()
  if [ "$im" = imat ]; then
    IM=(--imatrix "${IMATRIX:-data/d1/imatrix-d1.dat}")
    if [ -z "${IMATRIX:-}" ]; then
      echo "Warning: the historical imatrix includes evaluation tickets; see wiki/experiments/D01-d1-format-fidelity.md" >&2
    fi
  fi
  tmp=$(mktemp "$f.XXXXXX")
  "$MB/llama-quantize" "${IM[@]}" "$@" models/d1/d1-3B-F16.gguf "$tmp" "$type" > "data/d1/$name.quant.log" 2>&1 || {
    echo "Quantize failed: $name" >&2; tail -3 "data/d1/$name.quant.log" >&2; exit 1;
  }
  mv "$tmp" "$f"; tmp=
fi
log=$(mktemp "${TMPDIR:-/tmp}/d1-quant.XXXXXX")
echo "Server log: $log"
"$MB/llama-server" -m "$f" -ngl 99 -fa on --host 127.0.0.1 --port 8097 > "$log" 2>&1 &
pid=$!
ready=0
for ((i=0; i<90; i++)); do
  kill -0 "$pid" 2>/dev/null || break
  if curl -fsS --max-time 2 "$url" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    if kill -0 "$pid" 2>/dev/null; then ready=1; fi
    break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "SERVER FAILED: see $log" >&2
  tail -10 "$log" >&2
  exit 1
fi
line=$(python3 scripts/d1_fidelity.py --seed 1 --ref data/d1/ref_f16_seed1.json --label "$name")
size=$(wc -c < "$f" | awk '{printf "%.2f GB", $1/1e9}')
echo "$line  | $size" | tee -a results/d1_quant.txt
