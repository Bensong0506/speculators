#!/bin/bash
# Detached long-run launcher for Qwen3.5-9B multimodal DFlash on ALLaVA LAION.
# Defaults to a 10k-sample ALLaVA warm-start run from a converted DFlash model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dflash_qwen35_9b_allava_10k_continue_dflash_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

export MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
export FINETUNE_FROM="${FINETUNE_FROM:-/data/wenxuan/Qwen3.5-9B-DFlash-spec}"
export CONVERTED_DFLASH_OUT="${CONVERTED_DFLASH_OUT:-}"
export AUTO_CONVERT_DFLASH="${AUTO_CONVERT_DFLASH:-0}"
export REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3.5_9b_mm_10k}"
export SAVE_PATH="${SAVE_PATH:-./output/dflash_qwen3.5_9b_mm_10k_continue_dflash/${RUN_NAME}/checkpoints}"
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/wenxuan/ALLaVA-4V}"
export ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"

export MAX_SAMPLES="${MAX_SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-1000}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5}"
export LR="${LR:-1e-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE-}"
export DRAFT_ARCH="${DRAFT_ARCH:-qwen3}"
export MASK_TOKEN_ID="${MASK_TOKEN_ID:-248070}"
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 8 15 22 29}"
export NO_RESUME_FROM_CHECKPOINT="${NO_RESUME_FROM_CHECKPOINT:-1}"
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
echo "  output_dir: $OUTPUT_DIR"
echo "  save_path: $SAVE_PATH"
echo "  max_samples: $MAX_SAMPLES"
echo "  epochs: $EPOCHS"
echo "  checkpoint_freq: $CHECKPOINT_FREQ"
echo "  lr: $LR"
echo "  finetune_from: $FINETUNE_FROM"
echo "  converted_dflash_out: ${CONVERTED_DFLASH_OUT:-unset}"
echo "  auto_convert_dflash: $AUTO_CONVERT_DFLASH"
echo "  require_pretrained_weights: $REQUIRE_PRETRAINED_WEIGHTS"
echo "  block_size: $BLOCK_SIZE"
echo "  num_spec: $((BLOCK_SIZE - 1))"
echo "  max_anchors: $MAX_ANCHORS"
echo "  draft_vocab_size: ${DRAFT_VOCAB_SIZE:-full}"
echo "  draft_arch: $DRAFT_ARCH"
echo "  mask_token_id: $MASK_TOKEN_ID"
echo "  target_layer_ids: $TARGET_LAYER_IDS"
echo "  no_resume_from_checkpoint: $NO_RESUME_FROM_CHECKPOINT"
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
