#!/usr/bin/env bash
# run_gpu2.sh — n_assets=10  (GPU 2)
# 10-asset universe: IVV, QQQ, IWM, VEA, VNQ, TLT, IEF, LQD, GLD, XLK.
# Longer job (~3-4h). FD HJB skipped at n=10 (curse of dimensionality).
# NN and GRPO are the primary comparison at this dimensionality.
set -euo pipefail
mkdir -p logs results/gpu2

export CUDA_VISIBLE_DEVICES=2

echo "============================================================"
echo "  GPU 2  |  n_assets=10  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    10 \
    --seeds       1,2,3,4,5 \
    --results-dir results/gpu2 \
    --include-grpo \
    --device      cuda \
    --resume \
    2>&1 | tee logs/gpu2.log

echo ""
echo "GPU 2 finished at $(date)"
echo "Results in: results/gpu2/"
