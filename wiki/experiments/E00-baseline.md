# E00 – Baseline: geniex-bench llama.cpp, all compute units

**Date:** 2026-10-12 · **Tool:** geniex-bench v0.8.0 (llama.cpp 9425611) · **Model:** user's `LFM2.5-2.6B-Q4_0.gguf` (md5 1363ae03…)
**Args:** `--plugin llama_cpp --device {cpu|gpu|npu|hybrid} -p 512 -n 64 -r 3 --warmup 1` (random token prefill, greedy)
**Env:** see [setup-notes](../setup-notes.md) (libcdsprpc symlink, Adreno libs, `OCL_ICD_FILENAMES`). Defaults: ctx 4096, HTP power mode `burst`, 8 CPU threads.

| device | prefill t/s | decode t/s (sd) | TTFT (pp512) |
|---|---:|---:|---:|
| cpu (8 thr) | 58.6 | 12.38 (±0.11) | 8.7 s |
| gpu (Adreno 623, OpenCL) | 39.6 | 5.28 | 12.9 s |
| **npu (Hexagon v75 HTP)** | **678.1** | **19.68 (±0.06)** | **0.76 s** |
| hybrid (HTP + CPU scheduler) | 108.3 | 9.79 | 4.7 s |

## Observations
- **The NPU wins everything**: 11.6× prefill and 1.6× decode vs CPU.
- GPU OpenCL path is *slower than the CPU* here (LFM2's short-conv/other ops likely fall back; Q4_0 images not tuned for Adreno 623).
- `hybrid` is worse than pure `npu` for this model (opposite of the geniex docs' generic advice). Its numbers look like CPU+GPU – see
  [setup-notes](../setup-notes.md): when HTP isn't loadable, hybrid silently degrades.  *(Re-test hybrid with HTP loaded – TODO E0x.)*
- Raw JSON: `results/raw/E00-baseline-*/result.json`.
