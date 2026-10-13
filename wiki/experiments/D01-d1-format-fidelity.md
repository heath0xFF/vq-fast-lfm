# D01 – d1-3B: which weight format? (fidelity vs. what the NPU can load)

d1 outputs **calibrated probabilities**, so unlike chat quality, quantization error shows up directly as shifted probabilities and flipped thresholds.

## Method
`scripts/d1_fidelity.py`: 80 seeded decisions (40 sensor states: noul/choice/score; 40 support tickets: noul/choice/score) = 240 typed answers.
Reference = **F16 GGUF on llama.cpp/Metal** (which matches the transformers-fp32 numbers from the hackathon repo to 3e-4, see D00).
imatrix for the quantized variants: `data/d1/imatrix-d1.dat`, computed from 720 `systemone`-formatted prompts (different seeds than the eval set). `scripts/d1_qlab.sh`, table in `results/d1_quant.txt`.
Metrics: noul mean/max |ΔP| and number of decisions whose P crosses 0.5 vs the reference; choice total-variation distance and top-1 agreement; score expected-level error / range.

**Offline review correction:** different seeds did not give separate datasets. Of the 40 evaluation ticket rows, 32 occur verbatim in calibration (23 distinct tickets). No evaluation sensor state occurs verbatim. These imatrix results may therefore overstate quality on new inputs. `scripts/d1_calibration.py` removes the 153 matching calibration prompts and keeps 567; see [the rebuild steps](../reproduce.md#6-d1-models). The table, saved references, calibration file, and imatrix below remain historical and unchanged. A clean matrix and new weights have not yet been built.

| format | size | noul mean\|ΔP\| (max) | flips @0.5 (of 80) | choice TV | choice top-1 (of 80) | score err |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 | 2.87 GB | 0.0027 (0.011) | 1 | 0.0044 | 79 | 0.0015 |
| **Q6_K** | 2.22 GB | 0.0055 (0.028) | 0 | 0.0066 | 79 | 0.0065 |
| **Q5_K_M** | 1.94 GB | 0.0113 (0.048) | 2 | 0.0155 | 79 | 0.0097 |
| Q5_0 | 1.90 GB | 0.0114 (0.056) | 2 | 0.0326 | 76 | 0.0123 |
| Q4_K_M + imatrix | 1.67 GB | 0.0248 (0.083) | 4 | 0.0276 | 76 | 0.0191 |
| IQ4_NL + imatrix | 1.59 GB | 0.0211 (0.081) | 5 | **0.0559** | **72** | 0.0196 |
| Q4_0 + imatrix | 1.60 GB | 0.0308 (0.162) | 4 | 0.0286 | 77 | 0.0245 |
| Q4_1 + imatrix | 1.75 GB | 0.0346 (0.122) | 7 | 0.0224 | 79 | 0.0167 |
| **published Q4_K_M** (no imatrix) | 1.67 GB | 0.0394 (0.175) | **10** | 0.0262 | 77 | 0.0148 |

- The published `d1-3B-Q4_K_M` flips **10/80** alert decisions vs F16; an in-distribution imatrix halves that, Q5_K_M cuts it to 2, Q6_K to 0.
- IQ4_NL (best for the chat model, [E08](E08-quant-lab.md)) is the *worst* for d1's choice questions.

## What the HTP can actually load (llama.cpp master, `--device HTP0`)
| format | pp64 / pp128 / pp512 t/s (llama-bench) | server |
|---|---|---|
| Q4_K_M (1.67 GB) | 446 / 599 / 688 | OK with defaults, with mmproj Q8_0 (image path works) |
| **Q5_K_M (1.94 GB)** | 443 / 606 / 689 (**same prefill speed**) | only text-only with small buffers (`-c 1536 -np 3 -ub 128 -b 512`, ~590–600 ms/decision); with the 583 MB mmproj or default buffers: `fastrpc_mmap failed` |
| Q6_K (2.22 GB) | – | `fastrpc_mmap failed` even in llama-bench |
| Q8_0 (2.87 GB) | – | `fastrpc_mmap failed` |
→ the HTP can map roughly ≤ ~2 GiB of weights plus buffers per session in practice; K-quant prefill costs the same as Q4_K_M (prefill is HMX-bound; dequant type barely matters).
(The Q5_K_M used for the speed test was requantized from Q8_0 on the board, so its probabilities differ slightly from the F16-derived file scored above.)

## Recommendation (so far)
- Image + text: Q4_K_M, **but self-quantize with the imatrix** (halves threshold flips) or accept ±0.1 probability error.
- Text only: **Q5_K_M** – 3.5× lower noul error than the published Q4_K_M at the same prefill speed (~10 % slower per request in the one server test, mostly due to the tiny `-ub 128`; to be re-measured).
- Open: quantizing the mmproj (583 MB Q8_0) lower would free HTP mapping budget for Q5_K_M + images.
