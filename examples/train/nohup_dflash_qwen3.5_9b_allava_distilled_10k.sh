#!/bin/bash
# Detached DFlash continue-training run on Qwen-distilled 10k ALLaVA data.
#
# First create the distilled data:
#   bash examples/train/distill_allava_qwen35_10k.sh
#
# Then run:
#   bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dflash_qwen35_9b_allava_distilled_10k_continue_dflash_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

export MODEL="${MODEL:-/home/wenxuan/Qwen3.5-9B}"
export FINETUNE_FROM="${FINETUNE_FROM:-/home/wenxuan/Qwen3.5-9B-DFlash-spec}"
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
export DISTILLED_ALLAVA_JSONL="${DISTILLED_ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"

if [ ! -s "$DISTILLED_ALLAVA_JSONL" ]; then
    echo "[fatal] Distilled ALLaVA jsonl not found: $DISTILLED_ALLAVA_JSONL"
    echo "        Run: bash examples/train/distill_allava_qwen35_10k.sh"
    exit 1
fi

export USE_ALLAVA=0
export USE_MMSTAR=0
export DATASET="$DISTILLED_ALLAVA_JSONL"
export MEDIA_ROOT="$ALLAVA_IMAGE_ROOT"
export CONVERTED_DFLASH_OUT="${CONVERTED_DFLASH_OUT:-}"
export AUTO_CONVERT_DFLASH="${AUTO_CONVERT_DFLASH:-0}"
export REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3.5_9b_mm_distilled_10k}"
export SAVE_PATH="${SAVE_PATH:-./output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/${RUN_NAME}/checkpoints}"

export MAX_SAMPLES="${MAX_SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-20}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export LR_FT="${LR_FT:-1e-5}"
export LR="${LR:-$LR_FT}"

# --- Diagnostic: LR=0 control run ------------------------------------------
# With LR=0 the optimizer makes no weight update, so the model is frozen and the
# teacher-forced val MUST stay flat across epochs (initial_val == epoch0 == ...,
# within tiny packing noise of ~0.005). If val still dips/moves under
# CONTROL_LR0=1, something mutates the model OUTSIDE optimizer.step() -- a real
# bug. If val stays flat, the observed dip-then-recover is genuine warm-start
# optimization dynamics (cold Adam + LR warmup), not a bug.
#   CONTROL_LR0=1 EPOCHS=2 MAX_SAMPLES=500 \
#     bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
if [ "${CONTROL_LR0:-0}" = "1" ]; then
    export LR_FT=0
    export LR=0
    echo "  [CONTROL_LR0] LR forced to 0 -- teacher-forced val must stay flat; any drift = bug"
fi

# Param/optimizer precision. bf16 (default) trains params AND AdamW moments in
# bf16; at LR~1e-5 updates smaller than the bf16 ULP round to zero, so small
# updates are lost -- which hits warm-start finetuning hardest. Set
# HIDDEN_STATES_DTYPE=float32 to train params + optimizer state in fp32 (fixes
# the precision loss; uses more GPU memory). Controls both model and data dtype.
export HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-bfloat16}"
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
export VALIDATE_INITIAL="${VALIDATE_INITIAL:-1}"

export LOGGER="${LOGGER:-wandb}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.155.156.175:38080}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators}"
export RUN_NAME

# The nohup redirect is the single source of the full log for this launcher.
export LOG_TO_FILE=0

echo "Starting detached DFlash training on Qwen-distilled ALLaVA:"
echo "  run_name: $RUN_NAME"
echo "  dataset: $DATASET"
echo "  media_root: $MEDIA_ROOT"
echo "  output_dir: $OUTPUT_DIR"
echo "  save_path: $SAVE_PATH"
echo "  max_samples: $MAX_SAMPLES"
echo "  epochs: $EPOCHS"
echo "  checkpoint_freq: $CHECKPOINT_FREQ"
echo "  lr: $LR"
echo "  lr_ft: $LR_FT"
echo "  hidden_states_dtype: $HIDDEN_STATES_DTYPE"
echo "  finetune_from: $FINETUNE_FROM"
echo "  block_size: $BLOCK_SIZE"
echo "  num_spec: $((BLOCK_SIZE - 1))"
echo "  max_anchors: $MAX_ANCHORS"
echo "  draft_vocab_size: ${DRAFT_VOCAB_SIZE:-full}"
echo "  draft_arch: $DRAFT_ARCH"
echo "  mask_token_id: $MASK_TOKEN_ID"
echo "  target_layer_ids: $TARGET_LAYER_IDS"
echo "  dflash_compile_training: $DFLASH_COMPILE"
echo "  validate_initial: $VALIDATE_INITIAL"
echo "  logger: $LOGGER"
echo "  log: $NOHUP_LOG_PATH"

nohup bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh \
    > "$NOHUP_LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

echo "Started PID $PID"
echo "PID file: $PID_PATH"
echo "Follow log:"
echo "  tail -f $NOHUP_LOG_PATH"
