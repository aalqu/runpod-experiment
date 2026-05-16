#!/usr/bin/env bash
# run_gpu4.sh — n_assets=100  (GPU 4)
# 100-asset synthetic universe (factor-model calibrated to real market statistics:
# mu ~6% excess, sigma ~20%, 3-factor correlation ~0.30).
# FD HJB disabled at this dimensionality. NN and GRPO only.
# Longest job (~6-8h due to large policy networks at n=100).
set -euo pipefail
mkdir -p logs results/gpu4

export CUDA_VISIBLE_DEVICES=4

echo "============================================================"
echo "  GPU 4  |  n_assets=100  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    100 \
    --seeds       1,2,3,4,5 \
    --results-dir results/gpu4 \
    --include-grpo \
    --device      cuda \
    --resume \
    2>&1 | tee logs/gpu4.log

echo ""
echo "GPU 4 finished at $(date)"
echo "Results in: results/gpu4/"
