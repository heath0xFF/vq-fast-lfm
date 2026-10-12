# Methodology

## Benchmark tool
Primary: **`geniex-bench` v0.8.0** (Qualcomm; llama.cpp @ `9425611`, QAIRT 2.45), installed at
`~/lfm-fast/bench/geniex-bench-linux-arm64-v0.8.0` on the VQ.
It reports median TTFT / prefill t/s / decode t/s over N measured reps (+1 warmup), writes JSON.
For our own llama.cpp builds we use `llama-bench` (pp/tg) and `llama-completion`/`llama-cli` for real prompts.

## Two benchmark modes
1. **Synthetic** (`-p 512 -n 64`): random token ids → clean prefill/decode throughput. Good for kernel/placement comparisons.
   *Not valid for speculative decoding* (random text can't be drafted).
2. **Real prompts** (`--prompt-file prompts/chat/<name>.txt`): five LFM-chat-templated prompts
   (`code`, `math`, `chat`, `summarize`, `json`). Needed for speculative decoding/acceptance.
   - timing mode feeds the file **raw** (no chat template, BOS is added automatically) → `prompts/chat/` has the template baked in *without* BOS.
   - `--accuracy` mode applies the chat template itself → use `prompts/raw/` there.

## Harness
- `scripts/env.sh` → copied to `~/lfm-fast/env.sh` on the VQ (LD_LIBRARY_PATH, Adreno libs, `OCL_ICD_FILENAMES`, libcdsprpc symlink, model paths).
- `scripts/exp.sh <name> "<ENV=VAL ...>" <geniex-bench args>` runs an experiment and stores `result.json/log.txt/cmd.txt` in `~/lfm-fast/results/<name>/`.
- `scripts/pull_results.sh` rsyncs into `results/raw/`; `scripts/summarize.py [filter]` prints a markdown table (medians).
- `scripts/prof_agg.py` aggregates `GGML_HEXAGON_PROFILE=1` logs per op.

## Conventions
- Experiment IDs `Exx-short-name`, one wiki page each in `wiki/experiments/`.
- Always state: model file, device, env vars, args, n reps. Report median (sd) of ≥3 reps.
- Machine must be otherwise idle (check `uptime` load) – nothing else on the VQ was running during tests unless stated.
- Thermals/clock: CPU governor `schedutil`, HTP power mode `burst` (geniex-bench default) unless the experiment varies it.
