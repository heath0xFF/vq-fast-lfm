#!/bin/bash
# On the VQ: speed bench of GGUFs on HTP0 with master build, pinned to prime core 2.
# usage: qbench.sh <label>=<file> ...    (prints pp512 / tg128 medians)
source ~/lfm-fast/lenv-master.sh
for kv in "$@"; do
  label=${kv%%=*}; f=${kv#*=}
  out=$(taskset -c 2 $L/bin/llama-bench -m "$f" --device HTP0 -ngl 99 -fa 1 -p 512 -n 128 -r 3 -t 1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -a -E '^\|.*(pp|tg)[0-9]+' | awk -F'|' '{gsub(/ /,"",$(NF-2)); printf "%s=%s ", $(NF-2), $(NF-1)}')
  printf '%-22s %s\n' "$label" "${out:-FAILED}"
done
