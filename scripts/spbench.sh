#!/bin/bash
# On the VQ: plain vs DSpark decode speed grid. usage: spbench.sh <target.gguf> <draft.gguf> <nmax list> [prompts...]
source ~/lfm-fast/lenv-master.sh || exit 1
cd "$LFM" || exit 1
T=$1; D=$2; NMAXS=$3; shift 3; PROMPTS=${@:-"chat code math summarize json"}
CORE=${CORE:-2}; NGEN=${NGEN:-200}; PMIN=${PMIN:-0}; SKIP_PLAIN=${SKIP_PLAIN:-0}
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
for p in $PROMPTS; do
  if [ "$SKIP_PLAIN" = 1 ]; then plain=-; else plain=$(taskset -c $CORE $L/bin/llama-completion -m $T --device HTP0 -ngl 99 -fa on -no-cnv -f prompts/chat/$p.txt -n $NGEN --temp 0 --top-k 1 -t 1 2>&1 | strip | grep -a "eval time" | grep -a -v "prompt eval" | grep -a -o "[0-9.]* tokens per second" | head -1 | awk '{print $1}'); fi
  line="$p plain=${plain}"
  for n in $NMAXS; do
    out=$(taskset -c $CORE $L/bin/llama-speculative-simple -m $T -md $D --spec-type draft-dspark --spec-draft-n-max $n --spec-draft-p-min $PMIN --device HTP0 -ngl 99 -devd HTP0 -ngld 99 -fa on -f prompts/chat/$p.txt -n $NGEN --temp 0 --top-k 1 -t 1 2>&1 | strip)
    sp=$(echo "$out" | grep -a "decoded" | awk '{print $(NF-1)}'); nd=$(echo "$out" | grep -a "n_drafted" | awk '{print $NF}'); na=$(echo "$out" | grep -a "n_accept " | awk '{print $NF}')
    line="$line | n$n/p$PMIN=${sp:-FAIL} (acc ${na}/${nd})"
  done
  echo "$line"
done
