# J00 — Jetson Orin Nano versus Ventuno Q

Measured 2026-10-13. **Jetson Orin Nano 8 GB, current 15 W mode**, versus the VQ Hexagon NPU. These are tested configurations, not a claim that every possible optimization has been exhausted.

## Main result

- **Plain LFM generation favors the VQ:** about 22.3 versus 19.4 tokens/s.
- **DSpark helps much more on CUDA**, but still changes greedy output and can make a complete chat answer slower.
- **The Jetson is faster for both d1 models**, especially images and audio. It can also use higher-precision weights without losing speed.
- Omni audio initially crashed. A small, recorded CUDA virtual-memory workaround fixed the tested case.

## 1. Matched weights and inputs

Both boards use identical model/projector files, checked by SHA-256. Each result is the median of three independent server-launch summaries. Models run one at a time. No model-load time is included.

| Workload | VQ NPU | Jetson CUDA, 15 W | Result |
|---|---:|---:|---|
| LFM2.5-2.6B QAD Q4_0, plain decode | **22.3 tok/s** | 19.4 tok/s | VQ ~15% faster |
| LFM2.5-2.6B QAD Q4_0, prefill, 512 tokens | 692 tok/s | **722 tok/s** | Jetson ~4% faster |
| d1-3B Q4_K_M, three-question text decision | 597 ms | **418 ms** | Jetson 1.43× |
| d1-3B Q4_K_M + Q8 projector, image | 1,200 ms | **390 ms** | Jetson 3.08× |
| Omni Q8_0, three-question text decision | 110 ms | **85 ms** | Jetson 1.30× |
| Omni Q8_0 + Q8 projector, image | 119 ms | **75 ms** | Jetson 1.59× |
| Omni Q8_0 + Q8 projector, 11-second audio clip | 1,481 ms | **643 ms** | Jetson 2.30× |

Decode uses the code/math/chat prompts and 128 output tokens. Prefill is a separate `llama-bench` random-token test with three repetitions in one process, not three server launches. All other latency rows are HTTP request p50, measured locally on each board.

Text decision p95: VQ/Jetson **603/435 ms** for d1-3B and **119/92 ms** for Omni. The text tests use 100 measured requests per launch; images use 10 and audio uses five. Image/audio p95 estimates therefore have few samples.

The older d1-3B result was ~534 ms. The fresh comparison uses explicit context, slots, batches, and the new saved-request harness; do not silently replace this 597 ms baseline with that historical result. Against the historical 534 ms, the tuned Jetson's 413 ms is about 1.29× faster, not 1.45×.

## 2. Best tested Jetson decision configurations

The fastest useful format is not the same on both accelerators. The VQ retains Q4_K_M for multimodal d1-3B and Q8_0 for Omni; its fresh results are in the matched table above. Its historical ~534 ms d1-3B text result remains the lower reference, but was not reproduced under this protocol.

| Jetson configuration | Text p50 / p95 | Text requests/s, one client | Image p50 | Audio p50 |
|---|---:|---:|---:|---:|
| d1-3B **Q8_0**, **F16 projector** | **413 / 428 ms** | **2.40** | **375 ms** | unsupported by model |
| Omni **F16**, **F16 projector** | **74 / 84 ms** | **13.22** | **65 ms** | **618 ms** |

These rows change precision; they are not the matched-weight rows above. Each has three launches, with the same request counts as the matched tests.

On the saved 80-decision fidelity set, compared with each model's existing F16/Metal reference:

| Jetson target format | Mean binary probability difference | Maximum | Flips at 0.5 | Choice agreement |
|---|---:|---:|---:|---:|
| d1-3B Q4_K_M | 0.0399 | 0.197 | 10/80 | 77/80 |
| **d1-3B Q8_0** | **0.0037** | **0.021** | **1/80** | 78/80 |
| Omni Q8_0 | 0.0145 | 0.116 | 0/80 | 79/80 |
| **Omni F16** | **0.0026** | **0.030** | **0/80** | **80/80** |

Thus, d1-3B Q8 improves fidelity substantially at almost the same latency. Omni F16 is both faster and closer to its reference than Q8. These are fidelity checks, not human-labeled accuracy scores. Omni remains the weaker model on the earlier synthetic sensor/ticket task; faster inference does not remove that quality gap.

### Concurrency is a tradeoff, not a free gain

Single screening launch per row, 100 measured text requests, four concurrent clients:

| Tuned model | Server slots | Requests/s | p50 | p95 |
|---|---:|---:|---:|---:|
| d1-3B Q8 | 4 | 2.44 | 825 ms | 4,509 ms |
| d1-3B Q8 | 12 | 3.04 | 1,301 ms | 1,456 ms |
| Omni F16 | 4 | 14.11 | 275 ms | 351 ms |
| Omni F16 | 12 | 13.96 | 270 ms | 403 ms |

Use four slots and one in-flight request when low latency matters. Twelve slots can improve d1-3B throughput, but need a larger context and produce much longer request latency. These are closed-loop client tests; latency includes HTTP/server wait, not time waiting in the client's executor queue.

Concurrency also changed some probabilities. Against the serial result, Omni had two binary threshold flips with four slots and one with twelve, over 105 requests including warmups. d1-3B had no threshold flips in either case. Do not assume batching is numerically identical.

## 3. LFM and DSpark

Three interleaved plain/DSpark launch pairs, one warmup and three measured requests per prompt per launch. QAD Q4_0 target, Q8_0 draft, n-max 9:

| Jetson mode | Code, 128 tokens | Math, 128 tokens | Chat, 128 tokens |
|---|---:|---:|---:|
| Plain | 19.38 tok/s | 19.39 tok/s | 19.43 tok/s |
| **DSpark** | **46.37 tok/s** | **44.46 tok/s** | **30.79 tok/s** |

The short code/math token IDs match plain CUDA decoding with n-max 9. Chat first differs at token **8**, consistently. n-max 3 also changes math at token 75. These results do **not** establish lossless greedy execution.

Selected fast short-prefix settings on each board:

| Workload | VQ | Jetson |
|---|---:|---:|
| Code, Q8/n9 DSpark | ~33.5 tok/s, CPU sampling | 46.37 tok/s, GPU sampling |
| Math, Q8/n9 DSpark | ~32.6 tok/s, CPU sampling | 44.46 tok/s, GPU sampling |
| Chat | 22.26 tok/s, plain | 30.79 tok/s, Q8/n9 DSpark |

The VQ speculative figures are historical E16 short-prefix results, not fresh J00 full-answer comparisons. The plain and Jetson figures are repeated J00 results. The one-launch CUDA n3 chat screen reached 31.06 tok/s, but n9 was selected for the repeated mixed-workload test. On CUDA, GPU draft sampling was faster than CPU sampling.

### Complete answers change the result

One diagnostic request per prompt/mode, **not repeated latency estimates**. All entries below reached EOS and closed their reasoning block:

| Prompt | VQ plain | Jetson plain | Jetson DSpark |
|---|---:|---:|---:|
| Code | 83.4 s / 1,781 tokens | 136.1 s / 2,620 tokens | **44.0 s / 1,795 tokens** |
| Math | 125.9 s / 2,644 tokens | 106.1 s / 2,037 tokens | **73.7 s / 3,695 tokens** |
| Chat | **32.6 s / 708 tokens** | 38.1 s / 734 tokens | 45.1 s / 1,216 tokens |

The initial speculative math run hit its 3,072-token cap inside reasoning. It is retained as an incomplete-answer sample, not counted as a completed-task speedup. The math values above use an additional Jetson plain/speculative pair at context 8192 and output cap 6144; the others use context 4096 and cap 3072. The extended plain math output matched the shorter-context plain output.

The outputs and their lengths differ. These are observed application latencies, not controlled equal-output speed ratios. **DSpark is useful here for some tasks, but it is still not a universal faster mode.**

### Small output-quality check

- All three generated `merge_intervals` functions passed ten reviewed contract cases without modifying their input.
- VQ plain and Jetson DSpark passed their three generated unit tests. Jetson plain generated one incorrect expected value in its `test_unsorted`; that test failed.
- All math outputs solved the main equation, but reported 11:33 AM rather than the exact 11:33:20 AM.
- All three hardware-explanation answers contained unsupported general claims about NPU/GPU memory and efficiency. Do not use that model output as engineering evidence.

This checks a few saved outputs. It is not a broad quality benchmark or production acceptance test.

## 4. What tuning helped

- Full CUDA offload, flash attention on, four host threads. The thread-count gain over one thread was small; six did not improve it further.
- QAD Q4_0 was faster than ordinary Q4_K_M for LFM. Native decode screening gave ~19.6 versus ~17.2 tokens/s.
- DSpark Q8/n9 with **GPU** draft sampling was the best tested code/math draft setting. CPU sampling gave ~44.2/42.3/29.2 code/math/chat tokens/s versus ~46.3/44.4/30.8 with GPU sampling.
- DSpark F16 was slower (~36.0/34.1/23.6), as was Q4_K_M (~42.4/40.8/26.2). n-max 3 reduced code/math speed; chat stayed near 31 tokens/s.
- `GGML_CUDA_GRAPH_OPT=1` did not improve the plain screen. It is off in the final runs.
- For d1, one slot was slower; eight slots gave no useful single-request improvement over four. Flash attention off did not help d1-3B.
- Omni Q4_0 did not beat Q8, while F16 did. F16 projectors also improved the tested image paths modestly.

## 5. Audio crash and local workaround

Stock CUDA execution aborted on the first Omni audio request at:

```text
cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0)
CUDA error: out of memory
```

The failure was at a **32 GiB virtual-address reservation**, after the physical allocation call had succeeded. It matches [upstream issue #29142](https://github.com/ggml-org/llama.cpp/issues/29142), not proof that the 8 GB board lacked enough physical memory for this request.

[`patches/jetson-vmm-pool.patch`](../../patches/jetson-vmm-pool.patch) reduces that per-pool cap to 8 GiB for this build. The shared constant also controls the free operation and growth check. The original binaries remain separate on the board.

After rebuilding, audio completed repeatedly. All 35 original Omni text responses were exactly unchanged by the patch. Final text, image, audio, and higher-precision configurations were tested with the patch.

**Limit:** this is a local workaround, not an upstream fix. A workload needing a scratch pool larger than 8 GiB would hit the cap; do not apply it blindly to larger GPUs.

## 6. Hardware and method limits

- Jetson: Orin Nano developer kit, 8 GB RAM, no swap, Ubuntu 24.04.4 / L4T R39.2.0, driver 595.78, CUDA toolkit 13.2.86, GCC 13, native SM87 build.
- Current profile: **15 W**, GPU clock table ends at **624.75 MHz**, CPU maximum **1.5104 GHz**. No clock locking, power-profile change, Super/MAXN enablement, firmware change, or reboot was performed. These are **not Orin Nano Super results**.
- Jetson source: `8bb3cf0d216a5ae48360179f2931daf284be4d01`, plus the VMM patch. The existing VQ engine's source commit is still unproven; its binaries are hashed. This compares deployed hardware/software configurations, not hardware alone.
- d1: context 8192, four slots, flash attention; batch/microbatch 1024 for 3B and 4096 for Omni. VQ: one host thread pinned to CPU 2. Jetson: four host threads, no affinity. The API bodies and model hashes match in the matched-format tests.
- Text inputs are seeded, distinct sensor states. Image/audio tests repeat the same fixture after warmup. No cold-start, camera pipeline, or variable-audio workload claim.
- Same weights do not give identical probabilities across backends. On 100 measured matched-format sensor requests, d1-3B differed by mean/max 0.0130/0.0470, with one binary flip and five choice differences. Omni differed by 0.0165/0.1794, with three binary flips and seven choice differences.
- All servers were localhost-only and stopped after testing. Existing VQ GenieX and unrelated Jetson services were left untouched. No persistent inference service was installed.

Selected final-run telemetry maxima, including load/warmup and the tested workloads:

| Jetson configuration | System RAM used | Board input power, VDD_IN | Junction temperature |
|---|---:|---:|---:|
| d1-3B Q8 + F16 projector | 5,576 MB | 12.12 W | 58.75 °C |
| Omni F16 + F16 projector | 3,717 MB | 10.92 W | 57.41 °C |
| LFM plain | 3,944 MB | 11.79 W | 58.84 °C |
| LFM + DSpark Q8 | 5,027 MB | 10.96 W | 58.41 °C |

These are one-second samples, not peak-capture measurements, model-only VRAM, or energy per request. No equivalent VQ power trace was recorded, so this does not establish a power-efficiency ranking.

## 7. Reproduce

The approved build dependencies were installed before testing:

```bash
sudo apt-get install --no-install-recommends \
  build-essential cmake ninja-build cuda-nvcc-13-2 \
  cuda-cudart-dev-13-2 cuda-driver-dev-13-2 cuda-cccl-13-2 libcublas-dev-13-2
```

Use a separate workspace and the pinned source. Apply the patch only to the Jetson build:

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout 8bb3cf0d216a5ae48360179f2931daf284be4d01
git -C llama.cpp apply /path/to/vq-fast-lfm/patches/jetson-vmm-pool.patch
cmake -S llama.cpp -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87-real \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DGGML_CUDA_NCCL=OFF
cmake --build build -j 4 --target llama-server llama-bench llama-quantize
```

Use the official model files from `LiquidAI/LFM2.5-2.6B-GGUF`, `LiquidAI/LFM2.5-2.6B-DSpark-GGUF`, `LiquidAI/d1-3B-GGUF`, and `LiquidAI/d1-omni-600M-GGUF`. Verify them against the saved SHA-256 manifests.

Base server flags:

```text
--device CUDA0 -ngl 99 -fa on -t 4 --host 127.0.0.1 --port 8098
```

Add:

| Model/mode | Weights and additional flags |
|---|---|
| LFM plain | QAD Q4_0; `-c 4096 -np 1 -b 1024 -ub 1024 --spec-type none` |
| LFM DSpark experiment | Same target; `-c 4096 -np 1 -b 1024 -ub 1024 --spec-type draft-dspark -md <Q8-draft> -devd CUDA0 -ngld 99 --spec-draft-n-max 9` |
| d1-3B | Q8_0 target + F16 mmproj; `-c 8192 -np 4 -b 1024 -ub 1024` |
| Omni | F16 target + F16 mmproj; `-c 8192 -np 4 -b 4096 -ub 4096` |

Run `decisionbench.py text --label <name> --output <new-file.json> -n 100 --warm 5` for decisions; use `image` or `audio --audio jfk.wav` for the other modes. Keep `d1bench.py` beside it. Use `specbench.py` and the checked-in chat prompts for generation.

The existing safe runner supports `LFM_ENV=/path/to/env.sh`; that file must define `LFM` as the workspace and `L` as the build directory containing `bin/llama-server`. Set `MODEL` and `MMPROJ` explicitly. The default VQ environment path is unchanged.

## Evidence

- [Derived comparison and reviewed code checks](../../results/raw/J00-comparison.json).
- [Jetson raw results, commands, logs, telemetry, source patch, and hashes](../../results/raw/J00-jetson/).
- [Fresh VQ requests and results](../../results/raw/J00-vq-matched/), with server logs in sibling `J00-vq-*` directories.
- `scripts/decisionbench.py`, `tests/test_decisionbench.py`; offline suite: `python3 -m unittest discover -s tests -v`.

Next useful work: confirm supported higher-power firmware/modes with the owner; test distinct real tasks and longer contexts; investigate batch-dependent probabilities before safety-critical use. No Super-mode, TensorRT-engine, or broad production-quality result is claimed here.
