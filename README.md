# vq-fast-lfm

Making **Liquid AI's LFM models** fast on the **Arduino Ventuno Q** (Qualcomm Dragonwing IQ-8275, Hexagon v75 NPU) using llama.cpp (CPU / Adreno OpenCL / Hexagon HTP), GenieX, speculative decoding, quantization experiments and kernel-level profiling.
Two tracks, one board:

**1. LFM2.5-2.6B chat model (E-series).** NPU decode 19.7 → **22.5 tok/s** (own llama.cpp master build + host thread pinned to a big core), prefill 691 tok/s.
DSpark speculative decoding gives +6–22 % on code/math but loses on free-form chat (limited by small-batch cost on the HTP); n-gram speculation is free (+5–17 % on edit-style tasks).

**2. d1 decision models (D-series).** Typed questions in, probabilities out, one forward pass, no generated tokens, served via llama.cpp's `POST /v1/systemone` on the NPU:

| model | 3-question decision | image + question | audio (11 s) | notes |
|---|---:|---:|---:|---|
| d1-omni-600M (Q8_0) | ~110 ms (≈9/s) | ~120 ms | ~1.5 s | near-lossless Q8_0; clearly less calibrated than the 3B |
| d1-3B (Q4_K_M) | ~530 ms (≈1.9/s) | ~1.2 s | – | published Q4_K_M shifts probabilities (mean 0.04, up to 0.18; 10/80 alert decisions flip across 0.5 vs F16) |

Everything is documented in **[wiki/](wiki/README.md)** – start with [`wiki/findings.md`](wiki/findings.md); every experiment has a page in [`wiki/experiments/`](wiki/experiments/). How to rebuild: [`wiki/reproduce.md`](wiki/reproduce.md).
Run on the VQ: `scripts/serve_best.sh` (chat model) and `scripts/serve_d1.sh` (`MODEL=omni|3b`).

```
wiki/       findings, hardware, methodology, setup notes, ideas backlog, per-experiment pages
scripts/    benchmark harness (runs on the VQ), Docker cross-build, quant labs, profilers, d1 clients
prompts/    benchmark prompts (raw + LFM chat-templated)
results/    raw JSON/logs from the VQ, quantization tables (LFM KLD, d1 probability error)
data/       imatrices, d1 reference answers + eval/calibration sets, small eval logs (large reference data not committed)
tools/      membw.c – CPU memory-bandwidth probe
```
