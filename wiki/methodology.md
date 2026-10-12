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

## d1 decision models (D-series)
- Serving: `llama-server … --device HTP0 -ngl 99 -fa on -t 1` pinned to cpu2 (`scripts/serve_d1.sh`); client calls `POST /v1/systemone`.
- Latency/throughput: `scripts/d1bench.py {ref|lat|img}` – `lat` sends N *distinct* seeded sensor states (cache-miss workload; identical repeated states hit the server's prompt cache), reports p50/p95 and decisions/s; `--conc K` / `--split` test concurrency. `d1_audio_test.py` for the audio path. `scripts/d1_run.sh` starts a server with given args, runs client commands, stops it.
- Fidelity: `scripts/d1_fidelity.py` sends 80 seeded decisions (40 sensor, 40 ticket; 240 typed answers). `--save` stores a reference (F16 on llama.cpp/Metal, which reproduces the transformers-fp32 numbers from the hackathon repo to 3e-4); `--ref` reports noul mean/max |ΔP| + threshold flips at 0.5, choice total-variation + top-1, score error; `--acc` scores against rule-based labels (my heuristics, not human ground truth). Quant lab: `scripts/d1_qlab.sh` (imatrix from `data/d1/calib.txt`, 720 `systemone`-formatted prompts from different seeds than the eval set).
- Gotcha: the server's `confidence` field is not the reference's `category_confidence` (that one equals the probability of the chosen category).
