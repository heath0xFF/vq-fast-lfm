#!/bin/bash
# On-device experiment runner (run on the VQ).
# usage: exp.sh <name> "<ENV=VAL ENV2=VAL2>" <geniex-bench args...>
# Writes ~/lfm-fast/results/<name>/{result.json,log.txt,cmd.txt} and prints the [ok] line.
source ~/lfm-fast/env.sh
name=$1; envs=$2; shift 2
out=$LFM/results/$name; rm -rf "$out"; mkdir -p "$out"
echo "ENV: $envs" > "$out/cmd.txt"; echo "ARGS: $*" >> "$out/cmd.txt"
# shellcheck disable=SC2086
env $envs $BENCH "$@" --output-json "$out/result.json" > "$out/log.txt" 2>&1
rc=$?
printf '%-28s ' "$name"
grep -a -E "\[ok  \]|\[FAIL|\[fail|ERROR" "$out/log.txt" | sed 's/\x1b\[[0-9;]*m//g' | tail -2 | cut -c1-220
[ $rc -ne 0 ] && echo "  (exit $rc)"
exit 0
