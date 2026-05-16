#!/usr/bin/env bash
# run_gpu3.sh — n_assets=20  (GPU 3)
# 20-asset universe: full ETF basket.
# Longest job (~4-6h). FD HJB disabled. GRPO is the key novel method here.
set -euo pipefail
mkdir -p logs results/gpu3

export CUDA_VISIBLE_DEVICES=3

echo "============================================================"
echo "  GPU 3  |  n_assets=20  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    20 \
    --seeds       1,2,3,4,5 \
    --results-dir results/gpu3 \
    --include-grpo \
    --device      cuda \
    --resume \
    2>&1 | tee logs/gpu3.log

echo ""
echo "GPU 3 finished at $(date)"
echo "Results in: results/gpu3/"
