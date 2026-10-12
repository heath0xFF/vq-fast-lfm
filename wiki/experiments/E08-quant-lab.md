# E08 – Quantization lab (fidelity side)

**Goal:** decode is at the DDR roofline (E02), so fewer bytes/token = faster. Which lower-bit formats (all supported by the HTP backend: Q4_0, Q4_1, Q8_0, MXFP4, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) keep acceptable quality?

**Setup (Mac, Metal):** llama.cpp master built natively (`build-mac`); source = `LiquidAI/LFM2.5-2.6B-GGUF` **BF16** (5.4 GB).
imatrix from wikitext-2 *train* (200×512 tokens). Score = `llama-perplexity --kl-divergence` vs the BF16 logits on wikitext-2 *test* (40×512 tokens): **mean KLD** (lower better) and **same-top-token %** (higher better).
Script: `scripts/qlab.sh`; table: `results/quant.tsv`; logs `data/out/*.eval.log`.

> ⚠ Raw PPL on wikitext is **not** meaningful for this chat model (several quants score "better" PPL than BF16 = 50.1). Use KLD / top-p agreement.
> ⚠ `QAD` (quantization-aware distillation) is a *different set of weights* trained for Q4_0 (it even gets lower wikitext PPL than BF16), so its KLD vs the BF16 base is not a fidelity measure — Liquid's card says it was optimized for Q4_0 only and says nothing about other formats.

| variant | bytes | bpw-ish | mean KLD ↓ | same-top-p ↑ | note |
|---|---:|---:|---:|---:|---|
| published Q4_0 (PTQ) | 1.594 GB | 4.5 | 0.225 | 77.9 % | |
| my Q4_0 + imatrix | 1.598 GB | 4.5 | 0.176 | 80.4 % | imatrix helps |
| my Q4_0 + imatrix, head Q4_0 | 1.530 GB | | 0.198 | 79.1 % | head Q6_K→Q4_0 costs quality |
| my Q4_0 + imatrix, head Q8_0 | 1.662 GB | | 0.175 | 80.4 % | |
| **my IQ4_NL + imatrix** | 1.594 GB | 4.5 | **0.133** | **82.7 %** | same size as Q4_0, markedly closer to BF16 (HTP supports IQ4_NL) |
| published Q4_K_M | 1.674 GB | 4.9 | 0.157 | 81.0 % | |
| my Q3_K_M + imatrix | 1.367 GB | 3.9 | 0.308 | 74.7 % | −14 % bytes |
| my Q3_K_S + imatrix | 1.271 GB | 3.6 | 0.434 | 70.9 % | −20 % bytes |
| my Q2_K + imatrix | 1.095 GB | 3.2 | 1.017 | 57.8 % | broken |
| published **QAD** Q4_0 | 1.594 GB | 4.5 | (0.140, not comparable) | – | Liquid's Q4_0-specific QAD checkpoint → best known Q4_0 quality |

Take-aways so far:
1. At constant 4.5 bpw, **IQ4_NL+imatrix** and the **QAD Q4_0** are the quality picks; plain published Q4_0 is the weakest.
2. Going below 4 bits roughly doubles KLD (Q3_K_M) or worse; a 2.6B model has little redundancy. Speed side: see [E09](E09-quant-speed.md).
