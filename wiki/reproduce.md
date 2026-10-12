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
Deploy: `tar czf` the `pkg-build-snapdragon` dir, copy to the VQ as `~/lfm-fast/lcpp-master/` (then `~/lfm-fast/lenv-master.sh` sets `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`; see `scripts/` and `wiki/setup-notes.md`; the unversioned `libcdsprpc.so` symlink dir `~/lfm-fast/rpclib` is created by `scripts/env.sh`).

## 3. Models
| what | where |
|---|---|
| Q4_0 / QAD-Q4_0 / Q4_K_M / BF16 | `https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/<file>` (`LFM2.5-2.6B-QAD-Q4_0.gguf` is the recommended one) |
| DSpark drafter (Q8_0 used) | `https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/resolve/main/LFM2.5-2.6B-DSpark-Q8_0.gguf` |
| wikitext-2 raw | `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip` → `data/wikitext-2-raw/` |
| imatrix (committed) | `data/imatrix-wiki.dat` (200×512 tokens of wikitext-2 train, from BF16) |

## 4. Quant lab (Mac, Metal)
```bash
cd third_party/llama.cpp-master && cmake -B build-mac -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON && cmake --build build-mac -j10 --target llama-quantize llama-imatrix llama-perplexity llama-speculative-simple
# reference logits (2.6 GB, not committed):
build-mac/bin/llama-perplexity -m models/LFM2.5-2.6B-BF16.gguf -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base data/bf16-wiki40.kld
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
`scripts/d1_fidelity.py --save data/d1/ref_f16_seed1.json` against a running F16 server and `scripts/d1_qlab.sh <name> <type> imat` per format (imatrix `data/d1/imatrix-d1.dat` is committed).
> Note: Colima/Docker were uninstalled after the llama.cpp cross-build. Rebuilding or patching llama.cpp for the NPU means redoing §2 (reinstall colima+docker, import the Netskope CA, pull the toolchain image). The prebuilt NPU binaries on the VQ (`~/lfm-fast/lcpp-master`) are what the d1 experiments use.
