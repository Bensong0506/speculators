#!/bin/bash
# Detached 100k Domino run: continue-train a Domino causal-correction head on top
# of your BEST trained DFlash (warm-start), on the 100k Qwen-distilled ALLaVA.
# One-line start; wandb / paths / Domino params baked in (override via env).
#
# REQUIRED: FINETUNE_FROM = your best trained DFlash (same one your smoke used).
# USAGE:
#   FINETUNE_FROM=$PWD/output/<your-best-dflash-run>/checkpoints/checkpoint_best \
#     bash examples/train/nohup_dflash_domino_100k.sh
#
# Smoke first (50) to confirm val/full_acc != 0:  add  MAX_SAMPLES=50 EPOCHS=1 RUN_NAME=domino_smoke
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-domino_100k_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

# --- warm-start from your BEST DFlash (already ALLaVA-trained = the right base) ---
: "${FINETUNE_FROM:?set FINETUNE_FROM=/abs/path/to/your-best-dflash/checkpoints/checkpoint_best (same as smoke)}"
[ -d "$FINETUNE_FROM" ] || { echo "[fatal] FINETUNE_FROM not found: $FINETUNE_FROM"; exit 1; }
export FINETUNE_FROM
export MODEL="${MODEL:-/home/wenxuan/Qwen3.5-9B}"
export REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-1}"
export NO_RESUME_FROM_CHECKPOINT="${NO_RESUME_FROM_CHECKPOINT:-1}"   # fresh run, don't resume a prior ckpt
export OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_domino_100k}"
export SAVE_PATH="${SAVE_PATH:-$OUTPUT_DIR/${RUN_NAME}/checkpoints}"

# --- data: prebuilt 100k DISTILLED jsonl (NOT raw ALLaVA) ---
export USE_ALLAVA="${USE_ALLAVA:-0}"
export DATASET="${DATASET:-$REPO_ROOT/data/allava/allava_qwen35_distill_100k.jsonl}"
export MEDIA_ROOT="${MEDIA_ROOT:-/home/wenxuan/ALLaVA-4V}"   # --allowed-local-media-path (parent of image abs-paths)
export MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export EPOCHS="${EPOCHS:-2}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"               # val + save each epoch; checkpoint_best by val loss
export LR="${LR:-1e-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"

# --- DFlash arch: must match the base you warm-start from (best DFlash = bs8) ---
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export DRAFT_ARCH="${DRAFT_ARCH:-qwen3}"
export MASK_TOKEN_ID="${MASK_TOKEN_ID:-248070}"
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 8 15 22 29}"
export FORCE_EAGER="${FORCE_EAGER:-0}"
export DFLASH_COMPILE="${DFLASH_COMPILE:-1}"

# --- Domino head ---
export ENABLE_DOMINO="${ENABLE_DOMINO:-1}"
export DOMINO_LOSS_DECAY_GAMMA="${DOMINO_LOSS_DECAY_GAMMA:-4}"          # bs8 -> 4 (bs16 -> 7)
export DOMINO_LAMBDA_BASE_DECAY_RATIO="${DOMINO_LAMBDA_BASE_DECAY_RATIO:-0.3}"  # base already converged -> short anchor, more head training
export DOMINO_LAMBDA_BASE_START="${DOMINO_LAMBDA_BASE_START:-1.0}"
export DOMINO_PURE_DRAFT_PREFIX_LEN="${DOMINO_PURE_DRAFT_PREFIX_LEN:-1}"

# --- wandb (host same as all your launchers; auth = persisted login / WANDB_API_KEY on the box) ---
export LOGGER="${LOGGER:-wandb}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.155.156.175:38080}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators}"
export RUN_NAME
export LOG_TO_FILE=0   # nohup redirect is the single full log

echo "Starting detached Domino 100k:"
echo "  run_name:      $RUN_NAME"
echo "  finetune_from: $FINETUNE_FROM   (best DFlash = warm-start base)"
echo "  output_dir:    $OUTPUT_DIR"
echo "  dataset:       $DATASET"
echo "  media_root:    $MEDIA_ROOT"
echo "  max_samples:   $MAX_SAMPLES   epochs: $EPOCHS"
echo "  block_size:    $BLOCK_SIZE   num_spec: $((BLOCK_SIZE - 1))   gamma: $DOMINO_LOSS_DECAY_GAMMA"
echo "  domino:        enabled=$ENABLE_DOMINO  lambda_base=$DOMINO_LAMBDA_BASE_START->0 over ratio $DOMINO_LAMBDA_BASE_DECAY_RATIO"
echo "  logger:        $LOGGER  ($WANDB_BASE_URL / $WANDB_PROJECT)"
echo "  log:           $NOHUP_LOG_PATH"

nohup bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh \
    > "$NOHUP_LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"
echo "Started PID $PID  (tail -f $NOHUP_LOG_PATH)"
