#!/bin/bash
# On the VQ: serve a d1 decision model (POST /v1/systemone) on the Hexagon NPU.
#   omni (default): ./serve_d1.sh           d1-omni-600M Q8_0 (+ image & audio), ~110 ms / 3-question decision
#   3b:   MODEL=3b ./serve_d1.sh            d1-3B Q4_K_M (+ image), ~530 ms, better calibrated -- see wiki D00-D02
# Env: PORT (8098), CORE (2 = prime A78 core)
source ~/lfm-fast/lenv-master.sh
PORT=${PORT:-8098}; CORE=${CORE:-2}; M=$HOME/models
case "${MODEL:-omni}" in
  omni) ARGS=(-m $M/d1o/d1-omni-600M-Q8_0.gguf --mmproj $M/d1o/mmproj-d1-omni-600M-Q8_0.gguf -b 4096 -ub 4096) ;;  # card: a question is read in one batch
  3b)   ARGS=(-m $M/d1/d1-3B-Q4_K_M.gguf --mmproj $M/d1/mmproj-d1-3B-Q8_0.gguf) ;;                                   # Q5_K_M/Q6_K/Q8_0 don't map on the HTP with the mmproj (D01)
  *)    echo "MODEL must be omni or 3b"; exit 1 ;;
esac
exec taskset -c "$CORE" "$L/bin/llama-server" "${ARGS[@]}" --device HTP0 -ngl 99 -fa on -t 1 --host 0.0.0.0 --port "$PORT"
