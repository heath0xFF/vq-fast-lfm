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
