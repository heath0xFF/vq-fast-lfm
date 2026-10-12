# E09 – Quantization format vs HTP speed

Master build (8bb3cf0), `taskset -c 2`, `-t 1`, `llama-bench -p 512 -n 128 -r 3`, fidelity numbers from [E08](E08-quant-lab.md).
(Note the 21.0 / 22.5 tg bimodality described in [E13](E13-run-to-run-bimodality.md): compare only within a session / bracket with a reference.)

| format | size | pp512 t/s | tg128 t/s | KLD vs BF16 | verdict |
|---|---:|---:|---:|---:|---|
| **Q4_0** (published) | 1.59 GB | 691–693 | 20.7–22.5 | 0.225 | HTP sweet spot |
| **Q4_0 QAD** (Liquid) | 1.59 GB | 690 | 22.4 | n/a (different weights, best Q4_0 quality) | **use this** – identical speed, better quality |
| IQ4_NL + imatrix | 1.59 GB | 690 | 18.8 | 0.133 | −12 % speed (LUT dequant) for better fidelity |
| Q4_K_M (published) | 1.67 GB | 685–689 | 17.8 | 0.157 | slower (K-quant kernel + Q6_K tensors) |
| Q3_K_M + imatrix | 1.37 GB | 682 | 21.9 | 0.308 | 14 % fewer bytes but **no faster** (Q3_K dequant is compute-bound), 2× worse KLD |
| Q8_0 | 2.87 GB | – | – | – | failed to run on HTP (mapping budget) / would be ≤ 11 t/s anyway |
| Q3_K_S / Q2_K | | not benchmarked | | 0.43 / 1.02 | quality unacceptable |

**Conclusion:** on this HTP backend the byte reduction of K-quants does not translate into speed (their HVX kernels are compute-bound,
see [E11](E11-hvx-thread-scaling.md)); only plain Q4_0 streams at the DDR roofline. **Q4_0 (QAD) is the right format.**
The bigger lever is the LM head (Q6_K, 215 MB = 4.9 ms/token): Q4_0 head saves ~1.5 ms but costs fidelity (KLD 0.176→0.198) – not worth it.
