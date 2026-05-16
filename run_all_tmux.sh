#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

echo "Running setup before launching GPU jobs..."
bash setup.sh

SESSION="${TMUX_SESSION:-}"
if [[ -z "$SESSION" ]]; then
  SESSION="$(tmux display-message -p '#S' 2>/dev/null || true)"
fi

if [[ -z "$SESSION" ]]; then
  echo "This script should be run inside tmux."
  echo "Example: tmux new-session -s runpod-exp './run_all_tmux.sh'"
  exit 1
fi

GPU_COUNT="$(python3 - <<'PYEOF'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(1)
PYEOF
)"

if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No CUDA GPUs detected."
  exit 1
fi

MAX_JOB=4
LAST_JOB=$((GPU_COUNT - 1))
if [[ "$LAST_JOB" -gt "$MAX_JOB" ]]; then
  LAST_JOB="$MAX_JOB"
fi

tmux rename-window -t "$SESSION:0" monitor

for gpu in $(seq 0 "$LAST_JOB"); do
  script="run_gpu${gpu}.sh"
  if [[ -f "$script" ]]; then
    tmux new-window -t "$SESSION" -n "gpu${gpu}" \
      "cd '$PWD' && bash '$script'; exec bash"
    echo "Started $script in tmux window gpu${gpu}"
  fi
done

tmux select-window -t "$SESSION:gpu0"

echo ""
echo "All available GPU jobs have been launched."
echo "Use Ctrl-b then n/p to switch tmux windows."
echo "Detach with Ctrl-b then d."
