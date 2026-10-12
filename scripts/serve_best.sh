#!/bin/bash
# On the VQ: best-known LFM2.5-2.6B server (OpenAI-compatible) on the Hexagon NPU.
#   plain:   ./serve_best.sh
#   dspark:  SPEC=1 ./serve_best.sh          (helps on code/math/summarize/JSON; hurts on free-form chat -- see wiki)
# Env: PORT (8080), CORE (2 = prime A78 core), CTX (8192), MODEL, DRAFT, NMAX (9)
source ~/lfm-fast/lenv-master.sh
PORT=${PORT:-8080}; CORE=${CORE:-2}; CTX=${CTX:-8192}; NMAX=${NMAX:-9}
MODEL=${MODEL:-$M_QAD}; DRAFT=${DRAFT:-$D_Q8}
ARGS=(-m "$MODEL" --device HTP0 -ngl 99 -fa on -t 1 -c "$CTX" --host 0.0.0.0 --port "$PORT" --jinja --temp 0.1 --top-k 50)
if [ "${SPEC:-0}" = 1 ]; then ARGS+=(-md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max "$NMAX" -devd HTP0 -ngld 99); fi
exec taskset -c "$CORE" "$L/bin/llama-server" "${ARGS[@]}"
