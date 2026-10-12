#!/bin/bash
# Cross-build llama.cpp (Hexagon backend) for Snapdragon Linux using the official toolchain image.
# usage: scripts/build_lcpp.sh [build-dir-name] [extra cmake args...]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=${SRC:-llama.cpp}   # dir under third_party/
BD=${1:-build-snapdragon}; shift || true
docker run --rm --platform linux/amd64 -u "$(id -u):$(id -g)" \
  --volume "$ROOT/third_party/$SRC:/workspace" -w /workspace \
  ghcr.io/snapdragon-toolchain/arm64-linux:v0.7 bash -c "
    set -e
    cmake --preset arm64-linux-snapdragon-release -B $BD $* &&
    cmake --build $BD -j 14 &&
    rm -rf pkg-$BD && cmake --install $BD --prefix pkg-$BD
  "
