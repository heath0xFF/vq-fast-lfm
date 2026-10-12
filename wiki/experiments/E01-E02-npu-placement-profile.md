# E01/E02 – How the graph lands on the NPU + per-op profile

**Args:** `GGML_HEXAGON_VERBOSE=1` (E01) / `GGML_HEXAGON_PROFILE=1` (E02), `--device npu -p 64 -n 32 -r 1 --warmup 0`.

## Placement (E01)
- `offloaded 31/31 layers` to HTP0: model buffer on HTP0 = 1525.9 MiB, on CPU = 205 MiB (the input token-embedding table, Q6_K; used only for `GET_ROWS` lookup).
- The tied LM head is a *duplicate* of the Q6_K embedding living on HTP0 → the whole decode graph is on the NPU, including SSM_CONV (short conv), SWIGLU, RMS_NORM fusion, flash-attn.
- KV cache: 8 attention layers only, f16, 64 MiB @ 4k. Recurrent state 0.34 MiB. HTP compute buffer 516 MiB @ ubatch 1024.
- HTP: `hvx 4, hmx 1, vtcm 8 MB`, one session on domain cdsp:3. Op batching: 1280 ops / batch, `n-ops 403` per decode token.

## Decode per-op profile (E02, avg of 31 decode tokens; profile mode adds some overhead: 17.1 t/s)
Total NPU batch time ≈ **42.6 ms / token**; wall time in this run 58 ms/token.

| op | µs / token | calls | µs / call | effective BW |
|---|---:|---:|---:|---:|
| MUL_MAT_NX (fused ffn gate+up, Q4_0) | 17 359 | 30 | 554 | ≈ 45 GB/s |
| MUL_MAT+ADD (ffn_down, Q4_0) | 8 882 | 30 | 296 | ≈ 42 GB/s |
| MUL_MAT token_embd (**LM head, Q6_K 2048×128000**) | 4 921 | 1 | 4921 | ≈ 44 GB/s |
| MUL_MAT shortconv.in_proj | 3 601 | 22 | 164 | ≈ 43 GB/s |
| MUL_MAT+ADD shortconv.out_proj | 1 256 | 21 | 60 | |
| CONCAT (conv state) | 1 517 | 22 | 69 | |
| SSM_CONV | 837 | 22 | 38 | |
| FLASH_ATTN_EXT | 814 | 8 | 102 | |
| everything else | < 2 000 | | | |

**Key insight: decode on the NPU already streams weights at ≈ 42–45 GB/s – the LPDDR roofline.**
Matmuls are ~85 % of the NPU time and all run at memory-bandwidth speed. Plain kernel tuning can't buy much; we need to
(a) read fewer bytes per generated token (speculative decoding amortizes the weight read over several tokens), and
(b) trim the ~8–10 ms/token of non-NPU overhead (wall 50.8 ms vs ~42 ms NPU time).
Prefill (pp64 shown, profile) is compute-bound on HMX: MUL_MAT_NX 1267 µs/call.
