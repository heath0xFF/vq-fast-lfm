# Ideas backlog

Legend: ✅ done · 🔄 in progress · ⏳ queued · ❌ rejected/blocked · ❓ unknown

Next work: [offline preparation and board test plan](next-tests.md). No new speed gains have been measured during the offline work.

| # | Idea | Expected gain | Status |
|---|---|---|---|
| 1 | Baseline all compute units | – | ✅ E00 |
| 2 | **DSpark speculative decoding** (Liquid's official drafter, block 9) | +16–22% on code/math in E10; loses on free-form chat | 🔄 measured; cheaper small-batch verification is still needed |
| 3 | n-gram / prompt-lookup speculation (`ngram-mod`, `ngram-map-k`) | +5–17% on tested edit tasks; larger gains on repeats are not general | ✅ E14 (`ngram-mod`); other variants not established |
| 4 | HTP env knobs: `NHVX`, `NHMX`, `OPBATCH`, `OPQUEUE`, `OPFUSION`, `OPPOLL`, `MM_SELECT`, `FA_SELECT`, `HOSTBUF`, `DMA64`, power mode | few % | ⏳ |
| 5 | CPU-side overhead trim: pin driver thread to prime core | +5–10 % | ✅ E05 (cpu2 → 21.7 t/s) |
| 6 | System tuning: CPU governor/`powerprofilesctl`, DDR/LLCC/bus devfreq clocks (needs root?) | unknown | ⏳ |
| 7 | Requantize LM head Q6_K→Q4_0/Q8_0 (4.9 ms/token on NPU) | ≤ 3 % speed, KLD 0.176→0.198 | ❌ E08 not worth it |
| 8 | Vocab pruning (shrink 128k LM head) | ≤ 5 %, changes model behaviour | ⏳ (low priority) |
| 9 | QAD Q4_0 variant vs plain Q4_0 (quality at same speed) | quality | ✅ E09 same speed → default |
| 10 | Qualcomm AI Hub / QAIRT / Genie compile of LFM2.5-2.6B (W4A16 context binaries) | prefill maybe; decode is BW-bound | ❌ blocked: `qai-hub-models` 0.64.0 has no LFM/Liquid model (list: Qwen3, Llama, Granite, Gemma4, Phi, Mistral…), no AI Hub token on this machine; would need custom porting of the short-conv arch + AIMET |
| 11 | Re-test `hybrid` with HTP actually loaded | – | ⏳ |
| 12 | Adreno OpenCL tuning (`GGML_OPENCL_*` env), Q4_0 | GPU path is 5.3 t/s – unlikely to beat NPU | ⏳ low |
| 13 | Upstream llama.cpp master Hexagon backend (newer kernels) vs geniex-pinned 9425611 | few % | ✅ E07 (+3 %: tg128 22.5 pinned) |
| 14 | Concurrent streams (n_seq_max>1) aggregate throughput | high aggregate | ⏳ |
| 15 | Custom HTP kernel work (we have the Hexagon SDK via toolchain image) | last resort | ❓ |
| 16 | **Lower-bit weights**: Q3_K_*, Q2_K, IQ4_NL, Q4_K_M | – | ❌ E08/E09: no format beats Q4_0 speed on HTP; sub-4-bit KLD doubles |
| 17 | **Cheaper small-batch verify on HTP**: wider-register HVX Q4_0 GEMM for M=2..8 and/or `vlut16`-based faster Q4_0→fp16 HMX dequant (E11) | unlock 1.4–2× typical, ~4× copy-heavy via DSpark | ⏳ proposed next (big effort; SDK+toolchain ready) |
| 18 | Task-adaptive speculation (plain fallback when acceptance low; n-gram for copy-heavy) | removes the −22 % chat loss | ⏳ |
| 19 | Cheaper DSpark draft: Q4_0 draft weights, faster TOP_K/CONCAT on 128k logits | −10..20 ms/cycle | ⏳ |
| 20 | Understand the 21.0/22.5 t/s bimodality (E13) | +7 % in slow mode | ❓ no root |

## d1 decision models (added later)
| # | Idea | Expected gain | Status |
|---|---|---|---|
| 21 | Serve d1 on the NPU via llama.cpp `/v1/systemone` | – | ✅ D00: 534 ms (3B), D02: 111 ms (omni) |
| 22 | Better d1-3B weights: imatrix Q4_K_M / Q5_K_M vs published Q4_K_M | historical gains in D01; clean evaluation needed | 🔄 calibration overlaps evaluation tickets; rebuild matrix and weights. Q5_K_M doesn't fit with the vision projector |
| 23 | Quantize the mmproj lower so Q5_K_M + images fits the HTP mapping budget | accuracy for image requests | ⏳ |
| 24 | Server change: evaluate d1-3B's parent-suffix together with the children in one pass (3–4 passes → 2) | 534 → ~300 ms is a hypothesis | ⏳ first latency target; inspect pinned source, preserve attention/conv state, then test on board |
| 25 | Faster Q4→fp16 dequant kernel (same work as idea 17) | possible gain on 3B; no measured Omni gain | ⏳ profile the actual deployed format first; Omni's total pass time is not a fixed dequantization cost |
| 26 | omni-then-3B router | unknown until routing quality and model residency are tested | ❓ confidence alone is unsafe: saved Omni F16 answers include four wrong binary answers at ≥99% confidence under the weak labels |
| 27 | Validate both d1 models on the real hackathon task with human labels (my synthetic labels are weak) | trust | ⏳ needs data |
| 28 | Cross-request batching of decisions (server evaluates one decision batch at a time; throughput flat at 1–4 clients) | omni: higher decisions/s | ⏳ server change |
