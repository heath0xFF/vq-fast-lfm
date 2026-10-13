#!/bin/bash
# Pull all experiment results from the VQ into results/raw/<name>/
cd "$(dirname "$0")/.."
mkdir -p results/raw
rsync -a --include='*/' --include='result.json' --include='cmd.txt' --include='log.txt' --include='client.txt' --exclude='*' vq:lfm-fast/results/ results/raw/
