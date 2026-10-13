# D02 – d1-omni-600M (the smaller d1) on the Ventuno Q

`LiquidAI/d1-omni-600M-GGUF`: same "decision model" idea as d1-3B (typed questions → probabilities, one forward pass, no tokens) but ~0.6B params and **text + JSON + image + audio (≤ 30 s)**. llama.cpp type `lfm2-d1-omni`.
Files: F16 764 MB, Q8_0 407 MB (+ BF16); mmproj F16 441 MB / Q8_0 263 MB (no Q4). Model card serves it with `-b 4096 -ub 4096` (a question's whole prompt must fit one batch).
Same tests as D00/D01: `scripts/d1bench.py` (latency), `scripts/d1_fidelity.py` (fidelity vs F16 reference + rule-based label accuracy), HTP0, `-t 1` pinned to cpu2.

## Speed (HTP0, `-b 1024 -ub 1024`)
| workload | omni-600M Q8_0 | omni-600M F16 | d1-3B Q4_K_M ([D00](D00-d1-baseline.md)) | omni Q8_0, CPU only (4 big cores) |
|---|---:|---:|---:|---:|
| 3-question sensor decision (~224 tok) | **111 ms** (p95 119) | 121 ms | 534 ms | 702 ms |
| throughput, 1 client | 8.9 /s | 8.2 /s | 1.87 /s | 1.2 /s |
| throughput, 4 concurrent | 9.1 /s | 8.5 /s | 1.87 /s | – |
| 256×256 image + 1 question | **119 ms** | 140 ms | 1198 ms | 1875 ms |
| 11 s audio clip + 2 questions (375 tok) | **~1.5 s** (`-ub 4096`) | – | n/a (no audio) | – |
- **One NPU pass per request** (profile: tokens=241 / 217 / 241 per request, 100–107 ms of op time each), i.e. all questions are batched in one pass, unlike d1-3B's 3–4 passes. This is total operation time, not evidence of a 100 ms fixed dequantization cost. Measure the cost against state length and question count before choosing a kernel change.
- Within that pass: MUL_MAT_NX 22 ms, MUL_MAT 19 ms, FLASH_ATTN_EXT 14 ms (full bidirectional attention over the batch), rest many small ops.
- Concurrency doesn't raise throughput (server evaluates one decision batch at a time).
- Vision encoder on the NPU too (one unsupported op, `UPSCALE`, falls back to CPU).

## Fidelity of the deployed formats (reference = omni F16 on llama.cpp/Metal, 80 decisions)
| config | noul mean\|ΔP\| (max) | flips @0.5 | choice TV | top-1 |
|---|---:|---:|---:|---:|
| F16 on HTP vs F16 on Metal | 0.0013 (0.014) | 0 | 0.0014 | 80/80 |
| **Q8_0 on HTP** | 0.0082 (0.060) | **0** | 0.0063 | 79/80 |
→ the NPU is numerically faithful, and **Q8_0 is effectively lossless** at 407 MB (only +8 % faster than F16 at 764 MB).

## Is the small model good enough? (rule-based labels on the synthetic set — my own labels, weak)
Labels: alert = temp ≥ 55 °C or "sudden … g spike"; category = thermal/motion/normal accordingly; refund/team from the ticket wording; ambiguous cases excluded (`d1_fidelity.py --acc`).
| model | noul acc (67 labeled) | noul Brier ↓ | choice acc (80) |
|---|---:|---:|---:|
| d1-3B F16 | 60 | 0.082 | 55 |
| d1-3B Q5_K_M (imatrix) | 60 | 0.085 | 56 |
| d1-3B Q4_K_M (published, on HTP) | 60 | 0.093 | 52 |
| **d1-omni-600M F16 / Q8_0 (HTP)** | **56** | **0.152 / 0.151** | 52 / 52 |
- The omni model is clearly less well calibrated: Brier ~1.7× worse, and on the 80-decision set it **disagrees with d1-3B F16 on 16/80 alert decisions** (mean |ΔP| 0.23, category top-1 agreement 56/80).
- Concretely, on the hackathon's sensor examples: state `{"temp_c": 71.5, "temp_c_1min_ago": 58.0, "motion": "none"}` → alert **0.0004** (3B: 0.70); `{37.0 °C, "sudden 2.1 g spike"}` → alert 0.997 but category "thermal" (3B: 0.60, "motion"). Looks like the 600M model under-reads numeric changes. (The image example is fine: p_red 0.9999.)
- Caveat: no human ground truth; the labels are my heuristics. But both models face identical labels, and the gap is consistent.

## Take-aways
- Speed: ~5× faster than the 3B for text, ~10× for images, 8.9 decisions/s, plus audio at ~1.5 s — all on the NPU, with Q8_0 (use that, not F16).
- Quality: not a drop-in replacement for the 3B on numeric/sensor-style reasoning. A confidence-only router is not supported by these results: in the saved Omni F16 answers, 31 of 67 rule-labeled binary questions have at least 99% confidence, but four are wrong under those rules. The labels are weak, but the model can be confidently wrong. Validate routing on real labeled tasks and test whether both models can stay loaded.
