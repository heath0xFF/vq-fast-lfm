# E03–E07 – Own llama.cpp build, DSpark attempt, host pinning, small-batch "verify" curve

All on the VQ, model = geniex-cached `LFM2.5-2.6B-Q4_0.gguf` (newer revision, md5 0f7e563b…) unless noted, `--device HTP0 -ngl 99 -fa 1`.

## E03 – Own cross-build of llama.cpp @ `9425611` (same commit as geniex-bench v0.8.0)
Built with `scripts/build_lcpp.sh` (Docker `snapdragon-toolchain/arm64-linux:v0.7`, preset `arm64-linux-snapdragon-release`; HTP skels v73/v75/v79/v81 are built too).
`llama-bench -m <user's Q4_0> -p 512 -n 64 -r 3`: **pp512 680.7 ± 2.2, tg64 19.33 ± 0.13** → parity with geniex-bench (678 / 19.7). Toolchain validated.
Run env: `LD_LIBRARY_PATH=lcpp/lib:rpclib ADSP_LIBRARY_PATH=lcpp/lib` (see `~/lfm-fast/lenv.sh`).

## E04 – DSpark speculative decoding (Liquid's official drafter) — **currently a LOSS**
`llama-speculative-simple --spec-type draft-dspark -md LFM2.5-2.6B-DSpark-Q8_0.gguf`, chat prompt, greedy, 128 tokens:

| config | decode t/s | accepted / drafted | note |
|---|---:|---:|---|
| plain (llama-completion, `-t 4`) | **22.09** | – | same prompt, same build |
| DSpark Q8_0, n-max 9 | 9.5 | 88/369 (23.8 %) ≈ 3.1 tok/cycle | |
| DSpark Q8_0, n-max 4 | 12.2 | 83/186 (44.6 %) ≈ 2.8 tok/cycle | |
| DSpark draft on CPU (`-devd none`) | – | crash in `llama_context` ctor (draft shares target embeddings that live in HTP buffers) | |
| DSpark F16 draft on HTP | – | `ggml-hex: fastrpc_mmap failed` (HTP buffer-mapping budget exceeded: target 1.5 GiB + 664 MB draft + compute buffers) | |

GenieX's own plugin cannot run it either (`setup_speculative` only loads a draft for `draft-mtp|eagle3|simple`).

**Per-cycle NPU time (n-max 4, `GGML_HEXAGON_PROFILE=1 … -v`)**: 8.1 + 11.5 + **113.4 (verify, 491 ops, M=5)** + 1.7 + **20.5 (draft fwd)** + 3×0.9 ms ≈ 158 ms NPU, vs 228 ms wall/cycle (~70 ms unexplained host/rollback overhead).
A plain token costs ~45 ms, so a cycle must yield >5 tokens to break even. It yields ~3.

## E06 – Why: cost of a small batch ("verification") on the HTP (llama-bench `-p N -n 0`)
| M tokens | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms / pass (9425611, MM_SELECT=2 default) | 49 | 60 | 88 | 104 | 101 | 101 | 105 | 108 | 112 | 122 | 135 | 215 |
| ms / pass (MM_SELECT=1, HVX only) | 46 | 60 | 87 | 103 | – | 145 | 188 | 274 | 359 | – | – | – |
| ms / pass (master 8bb3cf0, default) | 45 | 58 | 83 | 99 | 96 | 97 | 99 | 104 | 106 | 115 | – | – |

- `HTP_MM_HMX_MIN_NROWS = 4`: M ≤ 4 uses the HVX Q4_0×Q8_0 kernel (processes columns in pairs, **≈ 22 ms per extra column** → compute-bound at ~120 GMAC/s, ~16 % of vrmpy peak);
  M ≥ 5 uses HMX, which first **dequantizes all 1.5 GB of Q4_0 weights to fp16 tiles on HVX — a ~100 ms fixed cost** (ffn gate+up 1453 µs/call ≈ 17 GB/s vs 554 µs ≈ 45 GB/s in decode).
- So on this stack verifying 1+k tokens costs ≥ 2× a plain token; speculative decoding needs avg accepted length ≫ 2.5 *and* a very cheap drafter to pay off.
- Possible fixes (kernel work in `ggml/src/ggml-hexagon/htp/{matmul-ops.c,hvx-mm-kernels-tiled.h}`): wider HVX register blocking (2x4/2x8 columns, amortize nibble-unpack), a faster Q4_0→fp16 dequant for HMX, or caching.
  Back-of-envelope: even a 2× faster HVX kernel only gets M=4 to ~60 ms. → deferred; see [ideas](../ideas.md).

## E05 – Host-thread placement matters (tg128, `llama-bench -p 0 -n 128 -r 3`, 9425611)
| setup | tg128 t/s |
|---|---:|
| default (no pin) `-t 8` | 20.6 ± 0.0 (run-to-run 19.3–20.9) |
| `taskset -c 0-3 -t 4` | 20.30 |
| `taskset -c 4-7 -t 4` (A55) | 19.14 |
| `taskset -c 2-3 -t 4` | 21.91 |
| `taskset -c 2 -t 1` / `-c 3 -t 1` (prime A78 core) | **21.7** (very stable, sd 0.03) |
| `taskset -c 2-3 -t 1/2` | 20.3–20.4 (scheduler bouncing between the 2 cores hurts) |
| thread count `-t 1,2,4,6,8` unpinned | 19.8, 19.5, 20.4, 20.9, 20.6 (no effect) |
| `GGML_HEXAGON_OPPOLL` 0/1/100/1000/10000 | 21.77 / 21.74 / 21.89 / 21.99 / 21.86 (noise) |

Takeaway: the CPU only drives the HTP op queue; **pin that single thread to one big core** → +5–10 % and low variance.
> Caveat (added later): these comparisons predate the discovery of the 21.0/22.5 bimodality ([E13](E13-run-to-run-bimodality.md)); in later interleaved tests cpu0/1/2/3 were indistinguishable. cpu2-3 = Kryo *Prime* and cpu0-1 = *Gold* are the same Cortex-A78C design (2.36 vs 2.11 GHz); A55 (cpu4-7) = Silver. Pinned-vs-unpinned looks real, big-core-vs-big-core is not established, and A55-vs-A78C should be re-tested with repeated interleaved launches.

## E07 – llama.cpp master (8bb3cf0, 2026-10-12; 400 commits / ~25 Hexagon commits newer than 9425611)
Build: `SRC=llama.cpp-master scripts/build_lcpp.sh`. Pinned to cpu2, `-t 1`: **pp512 691.0, tg128 22.50 ± 0.03**, pp1 22.4, pp2 34.7 — a few % better than 9425611
(CPY/CONCAT/SSM_CONV optimizations). Small-batch verify curve unchanged (table above).
