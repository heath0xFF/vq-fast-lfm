# Setup notes & gotchas

## 1. geniex-bench on the VQ without sudo
`geniex-bench` plugin load fails with `libOpenCL.so.1: cannot open shared object file` and NPU fails with
`Device 'HTP0' not found` unless:
- Adreno/OpenCL libs from geniex's user prefix are on `LD_LIBRARY_PATH`: `~/.local/share/geniex-deps/root/usr/lib/aarch64-linux-gnu`
- `OCL_ICD_FILENAMES=libOpenCL_adreno.so.1` (no `/etc/OpenCL/vendors`).
- An **unversioned** `libcdsprpc.so` is on `LD_LIBRARY_PATH` (ggml-hex `dlopen("libcdsprpc.so")`; system only has `.so.1`). We symlink into `~/lfm-fast/rpclib/`.
All encoded in `scripts/env.sh`.

A silent failure mode: if the HTP can't load, `--device hybrid` falls back (we saw 9.8 t/s = CPU+GPU speed) *without erroring*.

## 2. GenieX plugin does not support DSpark (as of v0.8.0 / main)
`sdk/plugins/llama_cpp/src/llm.cpp::setup_speculative` only treats `draft-mtp | draft-eagle3 | draft-simple`
as "needs draft model". `--spec-type draft-dspark` therefore logs `failed to initialize speculative context` and falls
back to plain decoding. The underlying llama.cpp (`9425611`) *does* contain DSpark. → we build llama.cpp ourselves.

## 3. Cross-building llama.cpp with the Hexagon backend
- Needs the Hexagon SDK even for the host library (htp_iface IDL), so use the official toolchain image
  `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7` (x86_64 image; ~7 GB).
- On the Mac (Apple Silicon): `brew install colima docker`, `colima start --arch aarch64 --vm-type vz --vz-rosetta --cpu 14 --memory 32 --disk 100`
  → amd64 images run under Rosetta at near-native speed.
- **Netskope TLS interception**: Colima VM doesn't trust the corporate CA → `docker pull` fails with
  `x509: certificate signed by unknown authority`. Fix: export certs from the macOS keychain
  (`security find-certificate -a -p -c certadmin|goskope|Netskope`) and install them into the VM:
  `colima ssh -- sudo tee /usr/local/share/ca-certificates/netskope.crt; colima ssh -- sudo update-ca-certificates; colima ssh -- sudo systemctl restart docker`.
- Build script: `scripts/build_lcpp.sh <build-dir>` (uses preset `arm64-linux-snapdragon-release`).
- Deploy: scp `pkg-<build>/` to the VQ; run with `LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib`.

## 4. d1 on the HTP: memory-mapping limits
- Weights are mapped lazily into the DSP (`ggml-hex: fastrpc_mmap failed`, "HTP0 buffer mapping failed … error 0x1"). In practice a single model session handles ≲ ~2 GiB of weights plus buffers: d1-3B **Q4_K_M (1.67 GB) + mmproj Q8_0 (583 MB)** works with defaults; **Q5_K_M (1.94 GB)** needs text-only and small buffers (`-c 1536 -np 3 -ub 128 -b 512`); **Q6_K (2.2 GB) and Q8_0 (2.9 GB)** never map (LFM2.5-2.6B Q8_0 also failed to run on the HTP).
- d1-omni-600M: use the card's `-b 4096 -ub 4096` only when you need audio / long states; `-b/-ub 1024` is enough for ~240-token text and 89-token image requests and keeps the compute buffer small.
- The vision encoder runs on the HTP except one `UPSCALE` op (falls back to CPU, harmless). `--no-mmproj-offload` makes image requests ~5–7× slower (1.2 s → 5.5–8.9 s on d1-3B).
- Logs: server output has ANSI colors and some non-UTF8 bytes → use `LC_ALL=C sed 's/\x1b\[[0-9;]*m//g'` on macOS.
