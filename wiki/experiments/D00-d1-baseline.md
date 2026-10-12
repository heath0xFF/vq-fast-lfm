# D00 – d1-3B on the Ventuno Q: baseline (llama.cpp master, Hexagon NPU)

**d1-3B** (`LiquidAI/d1-3B`) is a *decision model*: given a `state` (text/JSON/images) and named typed questions (`noul` yes/no, `choice`, `score`) it returns calibrated probabilities from **one forward pass, zero generated tokens**.
Base = LFM2.5-VL-3B: the language backbone is the same size/architecture as LFM2.5-2.6B (F16 GGUF 5,403,160,160 B vs 5,403,158,528 B) + a SigLIP2 NaFlex 400M vision tower in a separate `mmproj` GGUF. GGUF metadata: `lfm2.decision.type = lfm2-d1`, a dedicated `systemone` chat template.
→ **Latency of a decision = prefill time**, not decode speed. Everything we learned about NPU prefill (HMX, ~690 tok/s at large batch, ~100 ms fixed cost per pass) applies; decode work does not.

## Serving path
- llama.cpp master (8bb3cf0, our existing NPU build) already has the endpoint: `POST /v1/systemone` (`tools/server/server-decision.cpp`). Run:
  `llama-server -m d1-3B-Q4_K_M.gguf --mmproj mmproj-d1-3B-Q8_0.gguf --device HTP0 -ngl 99 -fa on -t 1` (pinned to cpu2).
- GenieX v0.4.0/0.8.0 return 404 for `/v1/systemone` (per the hackathon repo's notes), so llama.cpp is the way.
- Files: `d1-3B-Q4_K_M.gguf` 1.67 GB, `d1-3B-Q8_0.gguf` 2.87 GB, F16/BF16 5.4 GB, `mmproj-d1-3B-{Q8_0 583 MB, F16 854 MB, BF16 856 MB}` (HF `LiquidAI/d1-3B-GGUF`; there is no Q4_0).

## Reference (prior work, `~/code/liquidai-network-crash-hackathon`, IQ-9075, transformers fp32 on CPU)
3-question sensor state ≈ **4.3–4.6 s** (125–133 input tokens), image + 1 noul ≈ **7.6 s**, batch of 3 states 13.7 s, peak RSS 19.4 GB. Probabilities from that run are our fidelity reference (`scripts/d1bench.py ref`).

## Ours (Ventuno Q, HTP0, Q4_K_M, `-t 1` pinned cpu2) — `scripts/d1bench.py`
| workload | latency | notes |
|---|---:|---|
| 3 questions (noul+choice+score), sensor JSON, ~224 input tokens, distinct states | **534 ms** (p95 536) | **≈ 8.5× faster** than the CPU reference |
| image 256×256 + 1 noul (91 input tokens) | **1198 ms** | ≈ 6.4× faster; p_red 0.9856 vs reference 0.9865 |
| throughput, 1/3/4 concurrent clients | 1.87 decisions/s each | server serializes the passes; concurrency doesn't help |
Server knobs: `-t 4` 536 ms, `-t 8` unpinned 608, `-ub 256` 534, `-np 3 -ub 1024` 607 → no gain.

## Where the 534 ms goes (GGML_HEXAGON_PROFILE, one request)
Per request the NPU runs **3–4 separate passes**, each ≥ 111 ms even for tiny batches (`tokens=22: 126 ms`, `62: 141`, `5: 111`, `50: 127`; sum ≈ 504 ms):
1. parent task: shared state prefix (50 tokens) — split exactly at `decision.pos_first` so children can branch the KV/conv state,
2. parent suffix (22 tokens) — finished *before* children start,
3. children (the other two questions, 36+31 tokens) batched together.
Real work is only ~140 new tokens (one pass would take ~215 ms at pp128), but every pass pays the ~100 ms HMX weight-dequant fixed cost ([E11](E11-hvx-thread-scaling.md)). Fewer passes and/or a cheaper fixed cost are the levers.

## Fidelity caveat (Q4_K_M vs fp32 reference) — details in [D01](D01-d1-format-fidelity.md)
| state | p_alert (ours, HTP) | (fp32 ref) | Δ | p(category) | (ref) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.6228 | 0.6983 | −0.076 | 0.743 | 0.752 |
| 1 | 0.0639 | 0.0686 | −0.005 | 0.627 | 0.732 |
| 2 | **0.4978** | **0.5998** | −0.102 | 0.621 | 0.636 |
State 2 flips across the 0.5 alert threshold. The same deviation appears on Metal (0.5017), so it is quantization, not the NPU: **F16 through llama.cpp reproduces the transformers fp32 reference to 3e-4** (p_alert 0.6980/0.0689/0.5999, p(category) 0.752/0.730/0.635, image 0.9865), i.e. the llama.cpp d1 implementation is correct.
(The server's `confidence` field is a different quantity from the reference's `category_confidence`; the reference equals the probability of the chosen category.)
