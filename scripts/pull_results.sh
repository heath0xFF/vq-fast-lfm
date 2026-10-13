#!/bin/bash
# Pull benchmark artifacts; an explicit SSH destination avoids relying on a local alias.
# usage: pull_results.sh [ssh-alias-or-user@host]
set -e
cd "$(dirname "$0")/.."
mkdir -p results/raw
rsync -a -e 'ssh -o BatchMode=yes -o ForwardAgent=no -o ConnectTimeout=8' \
  --include='*/' --include='*.json' --include='cmd.txt' --include='log.txt' \
  --include='client.txt' --include='hashes.txt' --include='*.cmd.txt' \
  --include='*.log.txt' --include='*.out.txt' --exclude='*' \
  "${1:-vq}:lfm-fast/results/" results/raw/
