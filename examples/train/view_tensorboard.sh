#!/bin/bash
# View training curves (train/val loss + per-position acceptance) in TensorBoard.
#
# The training script logs to ./train_logs when LOGGER=tensorboard (default).
# Run this ON the training box:
#   bash examples/train/view_tensorboard.sh
# Then from your laptop, tunnel and open the browser:
#   ssh -N -L 6006:localhost:6006 <user>@<gpu-box>
#   # browser -> http://localhost:6006

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-./train_logs}"
PORT="${PORT:-6006}"

python3 -c "import tensorboard" 2>/dev/null || python3 -m pip install -q tensorboard

echo "Serving TensorBoard for $LOG_DIR on :$PORT  (Ctrl+C to stop)"
exec tensorboard --logdir "$LOG_DIR" --port "$PORT" --bind_all
