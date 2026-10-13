#!/bin/bash
# On the VQ: run clients against a dedicated d1 server, then stop only that server.
# usage: d1_run.sh "<label>" "<taskset cpus|none>" "<server extra args>" "<client cmd1>" ["<client cmd2>" ...]
set -eo pipefail
[ "$#" -ge 4 ] || { echo "usage: d1_run.sh label cpus server-args client-command [...]" >&2; exit 1; }
source "${LFM_ENV:-$HOME/lfm-fast/lenv-master.sh}"
label=$1; cpus=$2; sargs=$3; shift 3
[[ "$label" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "Invalid experiment label" >&2; exit 1; }
MODEL=${MODEL:-$HOME/models/d1/d1-3B-Q4_K_M.gguf}
MMPROJ=${MMPROJ-$HOME/models/d1/mmproj-d1-3B-Q8_0.gguf}
pre=(); [ "$cpus" = none ] || pre=(taskset -c "$cpus")
projector=(); [ -z "$MMPROJ" ] || projector=(--mmproj "$MMPROJ")
read -r -a server_args <<< "$sargs"
url=http://127.0.0.1:8098/health
if curl -s --max-time 2 -o /dev/null "$url"; then
  echo "Port 8098 already serves HTTP; refusing to use or stop it" >&2
  exit 1
fi
mkdir -p "$LFM/results"
out=$(mktemp -d "$LFM/results/$label.XXXXXX")
{
  printf 'SERVER: '
  printf '%q ' "${pre[@]}" "$L/bin/llama-server" -m "$MODEL" "${projector[@]}" "${server_args[@]}" --host 127.0.0.1 --port 8098
  printf '\n'
  printf 'CLIENT: %s\n' "$@"
} > "$out/cmd.txt"
echo "##### $label  [cpus=$cpus; $sargs]  results: $out"
"${pre[@]}" "$L/bin/llama-server" -m "$MODEL" "${projector[@]}" "${server_args[@]}" --host 127.0.0.1 --port 8098 > "$out/log.txt" 2>&1 &
pid=$!
cleanup() {
  # The child PID keeps cleanup separate from other model workloads.
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
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
  echo "SERVER FAILED: see $out/log.txt" >&2
  tail -10 "$out/log.txt" >&2
  exit 1
fi
for c in "$@"; do
  echo "  \$ $c" | tee -a "$out/client.txt"
  (cd "$LFM" && eval "$c") 2>&1 | tee -a "$out/client.txt"
done
