# E11 – Is the HTP kernel HVX-compute or DDR bound? (`GGML_HEXAGON_NHVX` sweep)

master, pinned, `llama-bench -p 1,8,128 -n 0 -r 2` – prompt-batch time in ms (4 = default):

| NHVX threads | pp1 (decode-like) | pp8 (HMX path) | pp128 |
|---:|---:|---:|---:|
| 4 | 47.5 | 99.6 | 205 |
| 3 | 50.2 | 114 | 223 |
| 2 | 68 | 144 | 268 |
| 1 | 128 | 238 | 415 |

Fit `t = a/n + b`: decode (pp1) a ≈ 107 ms, b ≈ 20 ms → at 4 threads HVX work ≈ 27 ms vs ~35 ms DDR floor: **decode is simultaneously close to the HVX limit and the DDR limit**.
pp8: a ≈ 185 ms (Q4_0→fp16 tile dequant, perfectly parallel), b ≈ 54 ms (DDR floor 35 ms + other ops + sync). So the HMX path is **dequant-bound**: a 1.3–2× faster dequant would cut ~15–45 ms from every M≥5 pass.
Code reading (`htp/hmx-mm-kernels-tiled.h::dequantize_tiled_weight_to_fp16_task_q4_0`): ~100 vector ops per 1024-weight tile (and/lsr/vsub/vshuff/vunpack/convert/vmpy/convert/store); a `vlut16` formulation could roughly halve that.
Small-M HVX kernel (`hvx-mm-kernels-tiled.h::tiled_vec_dot_q4_0_32x2`): ~45 vector ops per (32×32 tile, column) of which only 8 are `vrmpy` → 22 ms per extra column; wider register blocking (2×4) and a Q8_1-style activation-sum trick to drop the `vsub 8` could give ≈ 30–40 %.

# E12 – CPU memory bandwidth (for calibration; `tools/membw.c`, static aarch64)
Streaming read: 1 prime core 22 GB/s, 2 prime cores **27.7 GB/s (CPU ceiling)**, 4 cores 25.6, 8 cores 12 (A55s hurt); A55 cluster alone 5.7 GB/s.
The NPU's 44–45 GB/s (matmul kernels in decode) is far above anything the CPU can pull → LPDDR peak is ≥ 45 GB/s (Qualcomm docs for the IQ-8275: 4×16 LPDDR5-3200 → ~51 GB/s theoretical, so the NPU is at ~88 % of peak); **the NPU is the only engine that can saturate DDR**.
(Explains why CPU decode is 12.4 t/s ≈ 1.6 GB × 12.4 = 20 GB/s and why GPU/CPU cannot help the NPU: they share the same DDR.)
