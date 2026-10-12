#!/bin/bash
# On the VQ: start llama-server (d1) with given args, run client commands, stop server.
# usage: d1_run.sh "<label>" "<taskset cpus|none>" "<server extra args>" "<client cmd1>" ["<client cmd2>" ...]
source ~/lfm-fast/lenv-master.sh; D=$HOME/models/d1
label=$1; cpus=$2; sargs=$3; shift 3
MODEL=${MODEL:-$D/d1-3B-Q4_K_M.gguf}; MMPROJ=${MMPROJ-$D/mmproj-d1-3B-Q8_0.gguf}
pre=""; [ "$cpus" != none ] && pre="taskset -c $cpus"
pkill -x llama-server 2>/dev/null; sleep 1
$pre $L/bin/llama-server -m $MODEL ${MMPROJ:+--mmproj $MMPROJ} --port 8098 $sargs > /tmp/d1_server_$label.log 2>&1 &
for i in $(seq 1 90); do sleep 2; curl -s localhost:8098/health | grep -q ok && break; done
echo "##### $label  [cpus=$cpus; $sargs]"
if ! curl -s localhost:8098/health | grep -q ok; then echo "  SERVER FAILED: $(LC_ALL=C grep -a -E 'what\(\)|error|failed' /tmp/d1_server_$label.log | LC_ALL=C sed 's/\x1b\[[0-9;]*m//g' | tail -1 | cut -c1-150)"; pkill -x llama-server; exit 0; fi
for c in "$@"; do echo "  \$ $c"; (cd ~/lfm-fast && eval "$c" 2>&1 | sed 's/^/    /'); done
pkill -x llama-server; sleep 2
