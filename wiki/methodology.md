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
- `python3 scripts/prof_agg.py <log.txt>` reports batch time and per-operation means, grouped by the token count in the first model norm. It reads batch headers before their operations and rejects incomplete or mixed-device logs. Counts that cannot be read are marked `unknown`, not decode. Profile timing includes measurement overhead; use separate unprofiled runs for latency claims.

## Conventions
- Experiment IDs `Exx-short-name`, one wiki page each in `wiki/experiments/`.
- Always state: model file, device, env vars, args, n reps. Report median (sd) of ≥3 reps.
- Machine must be otherwise idle (check `uptime` load) – nothing else on the VQ was running during tests unless stated.
- Thermals/clock: CPU governor `schedutil`, HTP power mode `burst` (geniex-bench default) unless the experiment varies it.

## d1 decision models (D-series)
- Serving: `llama-server … --device HTP0 -ngl 99 -fa on -t 1` pinned to cpu2 (`scripts/serve_d1.sh`); client calls `POST /v1/systemone`.
- Latency/throughput: `scripts/d1bench.py {ref|lat|img}` – `lat` sends N *distinct* seeded sensor states (cache-miss workload; identical repeated states hit the server's prompt cache), reports p50/p95 and decisions/s; `--conc K` / `--split` test concurrency. `d1_audio_test.py` for the audio path. `scripts/d1_run.sh` starts a dedicated server with given args, runs client commands, and stops only its child PID. It refuses an occupied HTTP port and returns failure on startup or client errors. Each run stores `cmd.txt`, `log.txt`, and `client.txt` in a new `~/lfm-fast/results/<label>.<suffix>/` directory; `pull_results.sh` copies all three.
- Fidelity: `scripts/d1_fidelity.py` sends 80 seeded decisions (40 sensor, 40 ticket; 240 typed answers). `--save` stores a reference (F16 on llama.cpp/Metal, which reproduces the transformers-fp32 numbers from the hackathon repo to 3e-4); `--ref` reports noul mean/max |ΔP| + threshold flips at 0.5, choice total-variation + top-1, score error; `--acc` scores against rule-based labels (my heuristics, not human ground truth). Quant lab: `scripts/d1_qlab.sh`. The historical imatrix uses 720 prompts from `data/d1/calib.txt`, but different seeds did **not** prevent repeated tickets: 32/40 evaluation ticket rows occur in calibration. `scripts/d1_calibration.py` checks this offline or writes a filtered copy; see [reproduce.md](reproduce.md#6-d1-models). The existing matrix and scores remain historical until the matrix and weights are rebuilt.
- Gotcha: the server's `confidence` field is not the reference's `category_confidence` (that one equals the probability of the chosen category).
