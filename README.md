# vq-fast-lfm

Making **Liquid AI's LFM models** fast on the **Arduino Ventuno Q** (Qualcomm Dragonwing IQ-8275, Hexagon v75 NPU) using llama.cpp (CPU / Adreno OpenCL / Hexagon HTP), GenieX, speculative decoding, quantization experiments and kernel-level profiling.
Two tracks, one board:

**1. LFM2.5-2.6B chat model (E-series).** NPU decode 19.7 → **22.5 tok/s** (own llama.cpp master build + host thread pinned to a big core), prefill 691 tok/s.
DSpark gains depend on workload and output length. [E15](wiki/experiments/E15-dspark-server-recheck.md) found slower long-code generation and token differences from plain. [E16](wiki/experiments/E16-dspark-cost-and-prefix.md) reproduced two token flips with the target alone, profiled the cycle cost, and gained about 7–9% on short code/math by moving draft sampling to CPU. Q8 remained faster than the tested Q4/F16 drafts. **Keep plain as the default.** N-gram speculation gave +5–17 % on the tested edit tasks.

**2. d1 decision models (D-series).** Typed questions in, probabilities out, one forward pass, no generated tokens, served via llama.cpp's `POST /v1/systemone` on the NPU:

| model | 3-question decision | image + question | audio (11 s) | notes |
|---|---:|---:|---:|---|
| d1-omni-600M (Q8_0) | ~110 ms (≈9/s) | ~120 ms | ~1.5 s | near-lossless Q8_0; clearly less calibrated than the 3B |
| d1-3B (Q4_K_M) | ~530 ms (≈1.9/s) | ~1.2 s | – | published Q4_K_M shifts probabilities (mean 0.04, up to 0.18; 10/80 alert decisions flip across 0.5 vs F16) |

**Jetson comparison — [J00](wiki/experiments/J00-orin-nano-comparison.md):** on an Orin Nano 8 GB in its current **15 W mode**, plain LFM decode is slower than VQ (19.4 versus 22.3 tok/s), but DSpark reaches 46.4 code / 44.5 math / 30.8 chat tok/s on short prefixes. Complete-answer gains vary and outputs differ. Tuned Jetson d1-3B Q8 takes **413 ms** per text decision; Omni F16 takes **74 ms text / 65 ms image / 618 ms audio**. The report separates matched weights from tuned formats, and records the CUDA audio workaround and quality limits. No Super-mode result is claimed.

Everything is documented in **[wiki/](wiki/README.md)** – start with [`wiki/findings.md`](wiki/findings.md); every experiment has a page in [`wiki/experiments/`](wiki/experiments/). How to rebuild: [`wiki/reproduce.md`](wiki/reproduce.md).
Run on the VQ: `scripts/serve_best.sh` (chat model) and `scripts/serve_d1.sh` (`MODEL=omni|3b`).
Offline checks: `python3 -m unittest discover -s tests -v` (no device or models needed).

**Quality caveat:** the historical d1 imatrix includes some evaluation tickets. Existing scores are preserved, but need a clean rerun. See [D01](wiki/experiments/D01-d1-format-fidelity.md) and the [calibration rebuild steps](wiki/reproduce.md#6-d1-models).

```
wiki/       findings, hardware, methodology, setup notes, ideas backlog, per-experiment pages
scripts/    benchmark harnesses (VQ and Jetson), Docker cross-build, quant labs, profilers, d1 clients
prompts/    benchmark prompts (raw + LFM chat-templated)
results/    raw JSON/logs from both boards, quantization tables (LFM KLD, d1 probability error)
data/       imatrices, d1 reference answers + eval/calibration sets, small eval logs (large reference data not committed)
tools/      membw.c – CPU memory-bandwidth probe
```
