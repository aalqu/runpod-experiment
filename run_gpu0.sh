#!/usr/bin/env bash
# run_gpu0.sh — n_assets=1  (GPU 0)
# 1-asset universe: IVV only.
# Fastest job (~1-2h). All architectures + ES-GRPO.
# FD HJB is solved analytically (1D) — gold-standard reference.
set -euo pipefail
mkdir -p logs results/gpu0

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "  GPU 0  |  n_assets=1  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    1 \
    --seeds       1,2,3,4,5 \
    --results-dir results/gpu0 \
    --include-grpo \
    --device      cuda \
    --resume \
    2>&1 | tee logs/gpu0.log

echo ""
echo "GPU 0 finished at $(date)"
echo "Results in: results/gpu0/"
