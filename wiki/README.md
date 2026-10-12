# vq-fast-lfm — LFM2.5-2.6B as fast as possible on the Arduino Ventuno Q

**Focus update:** work is now on Liquid AI's **d1-3B decision model** (see `experiments/D*.md`); the LFM2.5-2.6B chat results below stay valid and carry over (same backbone, same NPU kernels).

**Original goal:** maximize inference speed of Liquid AI's LFM2.5-2.6B (Q4_0 GGUF) on the Ventuno Q
(Qualcomm Dragonwing IQ-8275 / QCS8275) using every tool available: GenieX, llama.cpp
(CPU / Adreno OpenCL / Hexagon HTP), Qualcomm AI Hub / QAIRT / Genie, speculative decoding, etc.

**Metrics we track** (all single-stream, batch 1, greedy unless noted):
- **decode t/s** – token generation speed (what users feel as "typing speed")
- **prefill t/s** – prompt processing speed
- **TTFT** – time to first token for a given prompt length
- quality guard: speculative decoding is lossless; any re-quantization gets a perplexity / output check.

## Status board (updated as we go)

| # | Path | Prefill t/s | Decode t/s | Notes |
|---|------|------------:|-----------:|-------|
| E00 | geniex-bench llama.cpp **CPU** (8 thr, default) | 58.6 | 12.4 | baseline |
| E00 | geniex-bench llama.cpp **GPU** (Adreno OpenCL) | 39.6 | 5.3 | slower than CPU |
| E00 | geniex-bench llama.cpp **NPU** (Hexagon HTP) | 678 | 19.7 | pp512/tg64 random tokens, unpinned |
| E00 | geniex-bench llama.cpp **hybrid** | 108 | 9.8 | worse than pure NPU |
| E05/E07 | **own llama.cpp master build, NPU, host thread pinned to cpu2** | **691** | **22.5** (21.0 in "slow mode", E13) | best plain config |
| E09 | same, **QAD Q4_0** weights | 690 | 22.4 | identical speed, better quality |
| E10 | + **DSpark** speculative (lossless), n-max 9 | – | math 25.3, code 24.3, json 20.6, chat 16.2 (plain ≈ 20.8); echo-style prompt 42.3 | task dependent; small-batch HTP cost is the limiter (the 42.3 case is the model copying its input verbatim) |
| E14 | + **n-gram** speculation (`ngram-mod`, match 4) | – | edit-text 26.3 vs 22.4 plain (+17 %), edit-code +5 %, otherwise = plain | free: no draft model, never slower in tests |

**Best known recipe today:** `scripts/serve_best.sh` (plain, QAD Q4_0) or `SPEC=1 scripts/serve_best.sh` (DSpark) on the VQ.

_See [findings.md](findings.md) for the curated conclusions and [experiments/](experiments/) for the raw log of each test._

## Wiki index
- [hardware.md](hardware.md) – what's in the box (SoC, cores, memory, NPU, software stack)
- [methodology.md](methodology.md) – how we benchmark, prompts, harness scripts
- [setup-notes.md](setup-notes.md) – environment gotchas & how-to (libcdsprpc, OpenCL ICD, Docker/Netskope CA, builds)
- [findings.md](findings.md) – curated key findings
- [ideas.md](ideas.md) – backlog of avenues, with status
- [experiments/](experiments/) – one page per experiment (E00, E01, … for LFM2.5-2.6B chat; **D00, D01, … for the d1-3B decision model**)

## Repo layout
```
scripts/      harness (exp.sh runs on VQ), build_lcpp.sh (Docker cross-build), summarize.py, prof_agg.py
prompts/      raw/ (user text) and chat/ (LFM chat-templated) benchmark prompts
results/raw/  JSON + logs pulled from the VQ
third_party/  llama.cpp checkout (gitignored) used for our own builds
wiki/         this wiki
```
