# E10 – DSpark speculative decoding, per-task grid (HTP, pinned core 2)

> **Correction (added later):** the `summarize` prompt does *not* produce a summary – the greedy model copies the input paragraph back verbatim (and then loops). The +104 % row is therefore a best-case "verbatim echo" number, not representative of summarization. Real gains seen so far: code/math/edit-style +6..22 %, JSON ~0, free-form chat negative.

Target = geniex `LFM2.5-2.6B-Q4_0.gguf`, draft = `DSpark-Q8_0` (349 MB), both on HTP0, `-fa on`, greedy, 200 new tokens, `-t 1`, master build (8bb3cf0).
Plain = `llama-completion` same prompt. `scripts/spbench.sh`. The target verifies draft tokens, but this experiment did not establish token-by-token parity. The later [E15 QAD/server check](E15-dspark-server-recheck.md) found repeatable differences from plain greedy output on the HTP; do not treat these historical timings as proof of lossless execution.

| prompt | plain t/s | n-max 3 | n-max 4 | n-max 6 | n-max 9 | best vs plain |
|---|---:|---:|---:|---:|---:|---:|
| chat (free-form reasoning prose) | 20.83 | 19.1 | 17.1 | 16.7 | 16.2 | **−8 %** (n3) |
| code (python + tests) | 20.95 | 17.9 | 22.8 | 23.9 | 24.3 | +16 % |
| math (step-by-step word problem) | 20.84 | 22.5 | 23.0 | 24.2 | 25.3 | +22 % |
| "summarize" (⚠ see correction: model echoes the input paragraph verbatim) | 20.71 | 27.7 | 32.1 | 38.2 | 42.3 | +104 % (echo case only) |
| json (structured generation) | 20.86 | 19.5 | 20.0 | 20.6 | 20.6 | −1 % |

Acceptance (tokens emitted per verify cycle at n-max 9) measured on the Mac/Metal with the same sidecar, to confirm the HTP integration is correct:
BF16 target: chat 2.6, code 3.5, math 4.8, summarize 7.7, json 4.5 · Q4_0 target: 2.8 / 4.0 / 4.3 / 7.4 / 4.5 · QAD target: 2.9 / 3.0 / 4.7 / 7.7 / 4.8
→ matches Liquid's claim (mean 4.4); HTP numbers agree, so **acceptance is task-dependent, not an HTP bug**.

## Cycle anatomy on the HTP (n-max 6, JSON prompt, master, `GGML_HEXAGON_PROFILE=1`)
| part | NPU time |
|---|---:|
| target verify, M=7 tokens (535 ops, HMX path) | **102.9 ms** |
| draft: encoder/fc batch | 1.6 ms |
| draft: 5 attn layers (Q8_0, M=9) + Markov chain over 128k logits (`TOP_K` 8 ms, `CONCAT`s) | **37.9 ms** |
| **sum / wall per cycle** | 142 ms / ~155 ms |

Pinning the host thread to a prime core removed the earlier mysterious ~70 ms/cycle (228 → 155 ms).
`--spec-draft-p-min` (confidence truncation) does not help: chat 16.6/18.3/15.7, code 23.7/20.7/22.3 for p-min 0.3/0.5/0.7.

Other failures seen: draft on CPU (`-devd none`) crashes (shares target embedding in HTP buffer); F16 draft hits `fastrpc_mmap failed` (HTP mapping budget). **Later correction:** [E16](E16-dspark-cost-and-prefix.md) loaded F16 at context 2048 / batch 128, but it was slower than Q8. Moving only draft sampling to CPU also worked and improved short-prefix speed.

## What would make it pay everywhere
Verify cost is the problem (103 ms for M≥5 vs 45 ms for M=1). With cheaper verification (target ≈ 55–60 ms) and the same acceptance,
cycle ≈ 95 ms → 3–4.5 tokens / 95 ms = 32–47 t/s typical, 80 t/s on copy-heavy text. See [E06](E03-E07-own-llamacpp-dspark-verify-curve.md), [E11](E11-hvx-thread-scaling.md).
