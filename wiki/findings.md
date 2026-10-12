# Key findings (curated)

_Updated after E00–E13. Device: Ventuno Q (QCS8275, Hexagon v75). Model: LFM2.5-2.6B Q4_0._

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
- Per-task decode, plain 20.8 → DSpark n-max 9: **summarize 42.3 (+104 %)**, math 25.3 (+22 %), code 24.3 (+16 %), json 20.6 (±0), free-form chat 16.2 (**−22 %**, n-max 3: 19.1).
- Reason: a verify pass of M ≥ 5 tokens costs ~100 ms on the HTP (HMX path, dequant-bound) vs 45 ms for M = 1; M = 3–4 uses a slow HVX kernel (83–99 ms). The drafter adds ~38 ms/cycle (Q8_0 matmuls + Markov head over 128k logits).
- If verification were ~55 ms: projected 32–47 t/s typical, ~80 t/s on copy-heavy text.

## 6. Dead ends / blocked
- **AI Hub / Genie / QAIRT**: `qai-hub-models` 0.64 has no LFM2 (short-conv hybrid) model and GenieX's QAIRT registry has no LFM entry; no AI Hub token available here. Even if ported, W4 decode is DDR-bound like Q4_0 → little to gain for decode.
- GPU OpenCL path (5.3 t/s) and hybrid (9.8) are slower than CPU; CPU draft model for HTP target crashes; F16 draft exceeds HTP mapping budget.
- `GGML_HEXAGON_*` power mode, OPPOLL, HOSTBUF: no measurable effect. A 21.0/22.5 t/s bimodality (E13) exists independent of all of them.

## 7. Where the remaining big upside is
1. **Cheaper small-batch verification on the HTP** (HVX multi-column Q4_0 kernel for M=2–8 and/or faster Q4_0→fp16 HMX dequant) – unlocks DSpark/n-gram speculation: 1.4–2× typical, up to ~4× on copy-heavy tasks.
2. Cheaper drafting (TOP_K/CONCAT over 128k logits, Q4_0 draft weights).
3. Task-adaptive speculation (fall back to plain when acceptance is low; n-gram for copy-heavy).
