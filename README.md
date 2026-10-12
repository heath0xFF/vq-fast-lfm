# vq-fast-lfm

Making **Liquid AI LFM2.5-2.6B** run as fast as possible on the **Arduino Ventuno Q** (Qualcomm Dragonwing IQ-8275, Hexagon v75 NPU) using GenieX, llama.cpp (CPU / Adreno OpenCL / Hexagon HTP), speculative decoding (DSpark), quantization experiments, and kernel-level profiling.

**Headline (as of the last wiki update):** NPU decode 19.7 → **22.5 tok/s** (own llama.cpp master build + host thread pinned to a big core), prefill 691 tok/s;
DSpark speculative decoding gives +6–22 % on code/math but loses on free-form chat – limited by small-batch cost on the HTP; n-gram speculation is free (+5–17 % on edit-style tasks).

Everything is documented in **[wiki/](wiki/README.md)** – start with [`wiki/findings.md`](wiki/findings.md); every experiment has a page in [`wiki/experiments/`](wiki/experiments/). How to rebuild: [`wiki/reproduce.md`](wiki/reproduce.md).

```
wiki/       findings, hardware, methodology, setup notes, ideas backlog, per-experiment pages
scripts/    benchmark harness (runs on the VQ), Docker cross-build, quant lab, profilers
prompts/    benchmark prompts (raw + LFM chat-templated)
results/    raw JSON/logs from the VQ, quantization KLD table
data/       imatrix + small eval logs (large reference data not committed)
tools/      membw.c – CPU memory-bandwidth probe
```
