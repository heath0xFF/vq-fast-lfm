# Key findings (curated)

_Updated after E00–E14 and D00–D02. Device: Ventuno Q (QCS8275, Hexagon v75). Sections 1–7: LFM2.5-2.6B chat model; sections 8–9: d1 decision models._

## 1. Which engine? → the Hexagon NPU, by a mile  (E00)
| | prefill t/s (pp512) | decode t/s |
|---|---:|---:|
| CPU 8 thr | 58.6 | 12.4 |
| GPU (Adreno OpenCL) | 39.6 | 5.3 |
| hybrid | 108 | 9.8 |
| **NPU (llama.cpp ggml-hexagon via GenieX)** | **678** | **19.7** |
Whole graph (conv, attention, norms, LM head) runs on the HTP. CPU/GPU cannot help: they share the same DDR and the CPU tops out at ~28 GB/s vs the NPU's ~45 GB/s (E12).

## 2. Plain decode is at the memory roofline (E02, E11, E12)
~1.59 GB streamed per token at 42–45 GB/s ≈ 35 ms + ~6 ms of small ops ≈ 41–44 ms/token ⇒ **ceiling ≈ 23–24 t/s**. We measure 22.5 (good mode).
Kernel tuning of the plain decode path can't buy more than a few %. Faster than this ⇒ read fewer bytes per *emitted* token ⇒ speculative decoding.

## 3. Free wins already banked
- **Pin the host thread to a prime A78 core** (`taskset -c 2 … -t 1`): 19.3–20.9 → 21.7–22.5 t/s (E05). The CPU only feeds the HTP op queue; A55s/migration add latency.
- **llama.cpp master (Oct 2026)** vs geniex-pinned 9425611: +3 % (CPY/CONCAT/SSM_CONV rewrites). We build it ourselves (Docker cross-toolchain, E03/E07).
- **QAD Q4_0 weights** (`LFM2.5-2.6B-QAD-Q4_0.gguf`): same bytes and speed as plain Q4_0 (22.4 t/s) but Liquid's quantization-aware-distilled → better quality. Use it as the default.
- Prefill 690 t/s (pp512) is HMX-compute-bound; nothing cheap left.

## 4. Quantization can't buy speed here (E08/E09)
Only Q4_0 streams at the DDR roofline. Q3_K_M (−14 % bytes) is no faster; Q4_K_M −15 % slower; IQ4_NL −12 % slower (better fidelity, KLD 0.133 vs 0.225). Sub-4-bit also doubles KLD.

## 5. Speculative decoding with Liquid's DSpark drafter (E04, E10) – works, lossless, but the HTP's small-batch cost caps the gain
- GenieX's plugin can't run `draft-dspark`; our own llama.cpp build can.
- Per-task decode, plain 20.8 → DSpark n-max 9: math 25.3 (+22 %), code 24.3 (+16 %), json 20.6 (±0), free-form chat 16.2 (**−22 %**, n-max 3: 19.1), and 42.3 (+104 %) on the "summarize" prompt – **but that prompt makes the model echo its input verbatim, so it is a best-case echo number, not typical summarization** (corrected).
- Reason: a verify pass of M ≥ 5 tokens costs ~100 ms on the HTP (HMX path, dequant-bound) vs 45 ms for M = 1; M = 3–4 uses a slow HVX kernel (83–99 ms). The drafter adds ~38 ms/cycle (Q8_0 matmuls + Markov head over 128k logits).
- If verification were ~55 ms: projected 32–47 t/s typical, ~80 t/s on copy-heavy text.

## 5b. N-gram speculation: free, modest (E14)
`--spec-type ngram-mod` needs no draft model and never drafts when nothing matches, so it cost nothing in any test (code/json/chat-like prompts ran at plain speed). With a short match length (`--spec-ngram-mod-n-match 4 --spec-ngram-mod-n-min 2`) it gives **+17 % on a typo-fix edit and +5 % on a code edit** (all drafted tokens accepted). Default match length 24 drafts nothing on those tasks. It also gives big numbers whenever the output repeats earlier text (loops, regenerated identical requests: up to 10× on an exact repeat) – real but not a general speed-up.

## 6. Dead ends / blocked
- **AI Hub / Genie / QAIRT**: `qai-hub-models` 0.64 has no LFM2 (short-conv hybrid) model and GenieX's QAIRT registry has no LFM entry; no AI Hub token available here. Even if ported, W4 decode is DDR-bound like Q4_0 → little to gain for decode.
- GPU OpenCL path (5.3 t/s) and hybrid (9.8) are slower than CPU; CPU draft model for HTP target crashes; F16 draft exceeds HTP mapping budget.
- **Second NPU**: Qualcomm docs say "dual Hexagon" for the IQ-8275 EVK, but the Ventuno Q exposes only `/dev/fastrpc-cdsp` (+ `gdsp0`, a general-purpose DSP, not an HTP); `HTP1` appears in ggml-hex only as a static fallback and `--device HTP1` is rejected. It also wouldn't help decode: DDR bandwidth is shared (NPU at ~88 % of the 51 GB/s theoretical peak).
- `GGML_HEXAGON_*` power mode, OPPOLL, HOSTBUF: no measurable effect. A 21.0/22.5 t/s bimodality (E13) exists independent of all of them.

## 7. Where the remaining big upside is
1. **Cheaper small-batch verification on the HTP** (HVX multi-column Q4_0 kernel for M=2–8 and/or faster Q4_0→fp16 HMX dequant) – unlocks DSpark/n-gram speculation: 1.4–2× typical, up to ~4× on copy-heavy tasks.
2. Cheaper drafting (TOP_K/CONCAT over 128k logits, Q4_0 draft weights).
3. Task-adaptive speculation (fall back to plain when acceptance is low; n-gram for copy-heavy).

## 8. d1-3B (decision model) on the Ventuno Q  (D00, D01)
- d1 = LFM2.5-VL-3B-based "decision" model: typed questions in, calibrated probabilities out, **one forward pass, zero output tokens** → latency = prefill, so the NPU prefill path (HMX, ~690 tok/s) is what matters.
- llama.cpp master already serves it (`POST /v1/systemone`); GenieX 0.4/0.8 don't. **534 ms per 3-question decision (text), 1.2 s with an image**, vs 4.3–4.6 s / 7.6 s for the previous transformers-fp32-on-CPU path (IQ-9075) ≈ 8.5× / 6.4×. The vision encoder runs on the NPU too (CPU-only would be 5.5–8.9 s).
- Each request = 3–4 separate NPU passes (shared 50-token state prefix, parent suffix, then the other questions batched) and **every pass pays the ~100 ms HMX fixed cost** → ~70 % of the latency is fixed cost, not work. A server scheduling change (one pass for parent-suffix + children) or a faster Q4→fp16 dequant kernel are the next levers (est. 534 → ~300 ms).
- **Quantization matters here**: the published Q4_K_M shifts probabilities by up to 0.10 and flips 10/80 alert decisions across 0.5; Q5_K_M flips 2, Q6_K 0, at identical NPU prefill speed – but Q6_K/Q8_0 don't map on the HTP and Q5_K_M doesn't fit together with the vision projector.

## 9. d1-omni-600M (D02)
- 111 ms per 3-question decision (8.9/s), 119 ms with an image, ~1.5 s for an 11 s audio clip, all on the NPU with Q8_0 (near-lossless: 0 threshold flips vs F16; F16 on the HTP matches Metal). It evaluates a request in **one** NPU pass, so the 100 ms fixed cost is paid once.
- But it is a much weaker decision-maker than d1-3B on our synthetic labeled set (noul Brier 0.15 vs 0.08–0.09; disagrees with the 3B on 16/80 alert decisions) — speed and quality are a real tradeoff, so validate on your own task before swapping.
