# Hardware & software inventory (Ventuno Q, `ssh vq`)

Observed 2026-10-12 via SSH.

| Item | Value |
|---|---|
| Board | Arduino VENTUNO Q (DT model: "Qualcomm Technologies, Inc. Monaco Monza addons") |
| SoC | Qualcomm Dragonwing **IQ-8275 (QCS8275)** – AI Hub chipset id `qualcomm-qcs8275` |
| CPU | 8 cores, Qualcomm Kryo Gen 6: **cpu2-3 = Prime** (Cortex-A78C, 2.36 GHz), **cpu0-1 = Gold** (Cortex-A78C, 2.11 GHz), **cpu4-7 = Silver** (Cortex-A55, 1.96 GHz). Verified on device (`/proc/cpuinfo` parts 0xd4b/0xd05, sysfs max freqs, kernel `cpu_capacity` 915/915/1024/1024/446×4) and matches Qualcomm docs (System/Kernel/runtime-performance-tuning, IQ-8275 tab: capacity-dmips-mhz A78C 1946 vs A55 1024; Prime+Gold share an L3, Silver has its own; Gold and Prime pairs are separate cpufreq policies 0 and 2) |
| CPU ISA | fp, asimd, asimdhp (fp16), asimddp (dotprod), aes, sha; **no i8mm / bf16 / SVE** → build llama.cpp with `-march=armv8.2a+fp16+dotprod` |
| GPU | Adreno 623 (OpenCL 3.0, devfreq 479–877 MHz, `simple_ondemand`) |
| NPU | Hexagon **v75** HTP: 4 HVX threads, 1 HMX, 8 MB VTCM (reported by ggml-hex). `npu_count = 1` |
| RAM | ≈14.9 GiB visible (board is presumably 16 GB; not verified), no swap. Qualcomm docs for the IQ-8275 SoC: 4×16-bit LPDDR5-3200 MHz (6400 MT/s) → ~51.2 GB/s theoretical peak (board's actual DRAM config not verified) |
| Storage | eMMC 55 GB root (≈15 GB free at start) |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-1080-qcom, aarch64 |
| CPU governor | `schedutil` (power-profiles-daemon: `balanced`) |
| sudo | needs password (not available non-interactively) |
| Toolchain on device | none (no gcc/cmake); docker is available to user `arduino` |
| Network | internet reachable (GitHub, HuggingFace), behind Netskope TLS interception on the Mac side |

## Pre-existing software on the VQ (found, not installed by us)
- `geniex` CLI v0.7.0 (`~/.local/bin/geniex`, launcher sets `LD_LIBRARY_PATH`), QAIRT 2.45, llama.cpp hash `4ff829e`
- Adreno/OpenCL/dmabufheap user-prefix libs in `~/.local/share/geniex-deps/root` (no sudo needed)
- fastrpc: `/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1`, `/usr/lib/dsp/{adsp,cdsp,gdsp0}`
- Docker images: `llamacpp-npu-runner:0.11.2` (Arduino App Lab), `python-apps-base:0.11.0`
- LiteRT demos with QNN HTP libs in `~/litert-demos/native/npu_lib`
- Three different Q4_0 GGUFs of the model (see below)

## Model files
| File | md5 | size | notes |
|---|---|---|---|
| `~/models/lfm25-2.6b-q4_0/LFM2.5-2.6B-Q4_0.gguf` (user's) | 1363ae03… | 1,593,894,720 | 33 KV pairs, older revision (ctx 128000) |
| `~/.cache/geniex/models/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q4_0.gguf` | 0f7e563b… | 1,593,894,912 | 36 KV pairs, newer revision (ctx 131072, base_model metadata) |
| `…/LFM2.5-2.6B-QAD-Q4_0.gguf` | 319278866… | 1,593,894,944 | QAD (quantization-aware distillation) variant, ships `qad/tokenizer.json` |

### Model architecture (from GGUF metadata)
LFM2.5-2.6B: 30 blocks = **22 gated short-conv blocks (l_cache 3) + 8 attention blocks** (GQA 32 heads / 8 KV heads),
d_model 2048, FFN 10752, vocab 128000, ctx 128k. 2.70 B params. Tensor types: 166× Q4_0, 1× **Q6_K**
(tied token-embedding / LM head, 205 MiB), 99× F32 (norms/conv weights).
Only 8 layers keep a KV cache (64 MiB at 4k ctx) → context cost is tiny; the 22 conv blocks keep a small recurrent state.
