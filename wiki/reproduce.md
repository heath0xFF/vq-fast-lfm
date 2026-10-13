# Reproducing the setup

Nothing in `third_party/`, `models/` or the BF16 reference logits is committed; everything is re-creatable. **No llama.cpp source patches were made** – we use upstream as-is (only the docs' `CMakeUserPresets.json` is copied in).

## 1. Source trees (Mac)
```bash
mkdir -p third_party && cd third_party
# (a) commit that geniex-bench v0.8.0 ships (parity baseline)
git init llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp
git fetch --depth 1 origin 94256114c229674ef96e76eb2dea596e65b43818 && git checkout FETCH_HEAD
cp docs/backend/snapdragon/CMakeUserPresets.json . ; cd ..
# (b) master used for the best numbers (2026-10-12)
git -C llama.cpp fetch --depth 400 origin master
git -C llama.cpp worktree add -f ../llama.cpp-master 8bb3cf0   # FETCH_HEAD at the time
cp llama.cpp/CMakeUserPresets.json llama.cpp-master/
```

## 2. Docker toolchain (Mac, Apple Silicon)
```bash
brew install colima docker
colima start --arch aarch64 --vm-type vz --vz-rosetta --cpu 14 --memory 32 --disk 100
# Behind Netskope TLS interception: import the corporate CA into the VM first (see setup-notes.md)
docker pull --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.7
SRC=llama.cpp-master scripts/build_lcpp.sh build-snapdragon     # -> third_party/llama.cpp-master/pkg-build-snapdragon
```
Deploy only when the board is reachable and available for testing:
- Copy the **contents** of `third_party/llama.cpp-master/pkg-build-snapdragon/` to `~/lfm-fast/lcpp-master/`. The server must be at `~/lfm-fast/lcpp-master/bin/llama-server`.
- Copy both `scripts/env.sh` and `scripts/lenv-master.sh` to `~/lfm-fast/`. Keep a copy of any existing board-specific environment file before replacing it.
- `lenv-master.sh` sets `L`, `LFM`, model paths, and the host/DSP library paths. It puts the master build's libraries before GenieX libraries. `env.sh` creates the unversioned RPC library links.
- The DSpark default is `~/models/LFM2.5-2.6B-DSpark-Q8_0.gguf`. Set `D_Q8` or the serving script's `DRAFT` if the file is elsewhere.

The checked-in environment file has local tests, but has not yet been tested on the board. See [setup-notes.md](setup-notes.md) for library requirements.

## 3. Models
| what | where |
|---|---|
| Q4_0 / QAD-Q4_0 / Q4_K_M / BF16 | `https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/<file>` (`LFM2.5-2.6B-QAD-Q4_0.gguf` is the recommended one) |
| DSpark drafter (Q8_0 used) | `https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/resolve/main/LFM2.5-2.6B-DSpark-Q8_0.gguf` |
| wikitext-2 raw | `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip` → `data/wikitext-2-raw/` |
| imatrix (committed) | `data/imatrix-wiki.dat` (200×512 tokens of wikitext-2 train, from BF16) |

## 4. Quant lab (Mac, Metal)
```bash
# Run from the repository root.
cmake -S third_party/llama.cpp-master -B third_party/llama.cpp-master/build-mac -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build third_party/llama.cpp-master/build-mac -j10 \
  --target llama-server llama-quantize llama-imatrix llama-perplexity llama-speculative-simple
# reference logits (2.6 GB, not committed):
third_party/llama.cpp-master/build-mac/bin/llama-perplexity -m models/LFM2.5-2.6B-BF16.gguf -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base data/bf16-wiki40.kld
scripts/qlab.sh my-IQ4_NL IQ4_NL        # quantize + score, appends to results/quant.tsv
```

## 5. Benchmarks on the VQ
`scripts/qbench.sh`, `scripts/spbench.sh`, `scripts/exp.sh`, `scripts/serve_best.sh` (copy to `~/lfm-fast/` on the board). Pull results back with `scripts/pull_results.sh`; table with `scripts/summarize.py`.

## 6. d1 models
```bash
# on the VQ (HF is reachable directly and fast there)
mkdir -p ~/models/d1 ~/models/d1o
curl -fL -o ~/models/d1/d1-3B-Q4_K_M.gguf          https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/d1-3B-Q4_K_M.gguf
curl -fL -o ~/models/d1/mmproj-d1-3B-Q8_0.gguf     https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/mmproj-d1-3B-Q8_0.gguf
for f in d1-omni-600M-Q8_0.gguf mmproj-d1-omni-600M-Q8_0.gguf; do curl -fL -o ~/models/d1o/$f https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/$f; done
MODEL=omni ~/lfm-fast/serve_d1.sh        # or MODEL=3b
```
Fidelity work (Mac, Metal): download `d1-3B-F16.gguf` + `mmproj-d1-3B-F16.gguf` (and the omni F16 pair) to `models/d1*/`, build `llama-server`/`llama-quantize`/`llama-imatrix` in `third_party/llama.cpp-master/build-mac`, then
`python3 scripts/d1_fidelity.py --save <new-reference.json>` against a running F16 server. Keep the committed seed-1 references unchanged.

**The historical imatrix overlaps the evaluation set.** The original inputs and results remain committed for reproduction, not as a clean held-out evaluation. To prepare new calibration inputs, from the repository root:
```bash
# Offline: removes 153 overlapping prompts and keeps 567. Refuses to overwrite files.
python3 scripts/d1_calibration.py data/d1/calib.txt --output data/d1/calib-heldout.txt
python3 scripts/d1_calibration.py data/d1/calib-heldout.txt

# Requires the local Metal build and F16 weights. Not part of the offline tests.
third_party/llama.cpp-master/build-mac/bin/llama-imatrix \
  -m models/d1/d1-3B-F16.gguf -f data/d1/calib-heldout.txt \
  -o data/d1/imatrix-d1-heldout.dat -c 512 -ngl 99
IMATRIX=data/d1/imatrix-d1-heldout.dat bash scripts/d1_qlab.sh Q4_K_M-heldout Q4_K_M imat
```
Use a new quantization name when changing the matrix or options: the lab reuses existing model files. Rebuild the matrix **and** weights before claiming new quality results. Filtering removes exact duplicate states; it does not replace evaluation on human-labeled, real tasks.
> Note: Colima/Docker were uninstalled after the llama.cpp cross-build. Rebuilding or patching llama.cpp for the NPU means redoing §2 (reinstall colima+docker, import the Netskope CA, pull the toolchain image). The prebuilt NPU binaries on the VQ (`~/lfm-fast/lcpp-master`) are what the d1 experiments use.

## 7. Offline checks
```bash
python3 -m unittest discover -s tests -v
python3 scripts/prof_agg.py results/raw/E02-npu-profile/log.txt
```
These checks use saved logs and mock processes. They do not open network connections, load models, or stop existing servers. The calibration checker returns a nonzero status on the original `data/d1/calib.txt`; that overlap is an expected historical finding.
