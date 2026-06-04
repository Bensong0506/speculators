#!/bin/bash
# Detached long-run launcher for Qwen3.5-9B multimodal DFlash on ALLaVA LAION.
# Defaults to the two local LAION json files: caption + instruct = 937,340 rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dflash_qwen35_9b_allava_full_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

export MODEL="${MODEL:-/home/models/Qwen3.5-9B}"
export FINETUNE_FROM="${FINETUNE_FROM:-/home/models/Qwen3.5-9B-DFlash-spec}"
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
export ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"

export MAX_SAMPLES="${MAX_SAMPLES:-937340}"
export EPOCHS="${EPOCHS:-1000}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export FORCE_EAGER="${FORCE_EAGER:-0}"
export DFLASH_COMPILE="${DFLASH_COMPILE:-1}"

export LOGGER="${LOGGER:-wandb}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.155.156.175:38080}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators}"
export RUN_NAME

# The nohup redirect is the single source of the full log for this launcher.
export LOG_TO_FILE=0

echo "Starting detached training:"
echo "  run_name: $RUN_NAME"
echo "  max_samples: $MAX_SAMPLES"
echo "  epochs: $EPOCHS"
echo "  checkpoint_freq: $CHECKPOINT_FREQ"
echo "  block_size: $BLOCK_SIZE"
echo "  num_spec: $((BLOCK_SIZE - 1))"
echo "  max_anchors: $MAX_ANCHORS"
echo "  force_eager_training: $FORCE_EAGER"
echo "  dflash_compile_training: $DFLASH_COMPILE"
echo "  logger: $LOGGER"
echo "  wandb: ${WANDB_BASE_URL:-unset}"
echo "  log: $NOHUP_LOG_PATH"

nohup bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh \
    > "$NOHUP_LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

echo "Started PID $PID"
echo "PID file: $PID_PATH"
echo "Follow log:"
echo "  tail -f $NOHUP_LOG_PATH"
