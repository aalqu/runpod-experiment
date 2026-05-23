#!/usr/bin/env bash
# setup.sh — run once on a fresh RunPod instance before launching any GPU scripts.
# Usage:  bash setup.sh
set -euo pipefail

echo "============================================"
echo "  RunPod Environment Setup"
echo "============================================"

# 1. Python deps (torch is assumed pre-installed by the RunPod image)
echo "[1/3] Installing Python dependencies..."
pip install --quiet numpy>=1.24 scipy>=1.10 pandas>=2.0 matplotlib>=3.7

# 2. Verify GPU visibility
echo "[2/3] GPU check..."
python3 - <<'PYEOF'
import torch
n = torch.cuda.device_count()
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  GPU count      : {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    gb = props.total_memory / 1e9
    print(f"  GPU {i}          : {props.name}  ({gb:.1f} GB)")
if n < 2:
    print(f"  WARNING: found {n} GPU(s). Scripts gpu0-gpu1 each need their own GPU.")
    print(f"           Run only the scripts matching available GPU indices.")
PYEOF

# 3. Verify data file
echo "[3/3] Data file check..."
python3 - <<'PYEOF'
import numpy as np, pathlib
npz = pathlib.Path("real_etf_data.npz")
if not npz.exists():
    print("  ERROR: real_etf_data.npz not found in the current directory!")
    raise SystemExit(1)
d = np.load(npz, allow_pickle=True)
print(f"  real_etf_data.npz OK — keys: {list(d.keys())[:8]} ...")
PYEOF

echo ""
echo "Setup complete. You can now launch the GPU scripts:"
echo "  nohup bash run_gpu0.sh > logs/gpu0.log 2>&1 &   # n=1"
echo "  nohup bash run_gpu1.sh > logs/gpu1.log 2>&1 &   # n=5"
echo ""
echo "Tail any log with:  tail -f logs/gpu1.log"
echo "Merge results when done:  python3 merge_results.py"
