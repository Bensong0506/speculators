#!/bin/bash
# Detached 10k ALLaVA run for a 3-layer DFlash pruned from native 5-layer DFlash.
#
# Default recipe:
#   source checkpoint: /data/wenxuan/Qwen3.5-9B-DFlash-spec
#   kept draft layers: 0 2 4  -> new layers 0 1 2
#   train samples:     10k
#
# Usage:
#   bash examples/train/nohup_dflash_qwen3.5_9b_allava_10k_prune3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
KEEP_LAYERS="${KEEP_LAYERS:-0 2 4}"
KEEP_TAG="$(printf '%s' "$KEEP_LAYERS" | tr -d ' ')"
RUN_NAME="${RUN_NAME:-dflash_qwen35_9b_allava_10k_prune3_${KEEP_TAG}_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

export MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
export PRUNE_SRC="${PRUNE_SRC:-/data/wenxuan/Qwen3.5-9B-DFlash-spec}"
export PRUNED_DFLASH_OUT="${PRUNED_DFLASH_OUT:-/data/wenxuan/Qwen3.5-9B-DFlash-spec-3layers-${KEEP_TAG}}"
export PRUNE_OVERWRITE="${PRUNE_OVERWRITE:-0}"

if [ ! -f "$PRUNED_DFLASH_OUT/config.json" ] || [ ! -f "$PRUNED_DFLASH_OUT/model.safetensors" ]; then
    echo "=== Creating 3-layer pruned DFlash warm-start ==="
    echo "  source:      $PRUNE_SRC"
    echo "  output:      $PRUNED_DFLASH_OUT"
    echo "  keep_layers: $KEEP_LAYERS"
    PRUNE_ARGS=(
        python3 scripts/prune_dflash_layers.py
        --src "$PRUNE_SRC"
        --out "$PRUNED_DFLASH_OUT"
        --keep-layers $KEEP_LAYERS
    )
    if [ "$PRUNE_OVERWRITE" = "1" ]; then
        PRUNE_ARGS+=(--overwrite)
    fi
    "${PRUNE_ARGS[@]}"
else
    echo "=== Reusing existing 3-layer pruned DFlash checkpoint ==="
    echo "  output:      $PRUNED_DFLASH_OUT"
    echo "  keep_layers: $KEEP_LAYERS"
fi

python3 - "$PRUNED_DFLASH_OUT/config.json" $KEEP_LAYERS <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1]))
expected_keep = [int(x) for x in sys.argv[2:]]
if cfg.get("speculators_model_type") != "dflash":
    raise SystemExit(
        "[fatal] pruned checkpoint is not speculators-format DFlash; "
        "this run refuses to train from scratch."
    )
nested = cfg.get("transformer_layer_config") or cfg
num_layers = nested.get("num_hidden_layers") or cfg.get("num_hidden_layers")
if int(num_layers) != 3:
    raise SystemExit(f"[fatal] expected 3 draft layers, got {num_layers}")
actual_keep = cfg.get("pruned_keep_layers")
if actual_keep is not None and actual_keep != expected_keep:
    raise SystemExit(
        f"[fatal] reused pruned checkpoint keep layers {actual_keep} "
        f"do not match requested {expected_keep}. Set PRUNED_DFLASH_OUT or "
        "PRUNE_OVERWRITE=1."
    )
print("Pruned checkpoint sanity OK")
print(f"  speculators_model_type: {cfg.get('speculators_model_type')}")
print(f"  num_hidden_layers:      {num_layers}")
print(f"  pruned_keep_layers:     {actual_keep}")
PY

export FINETUNE_FROM="$PRUNED_DFLASH_OUT"
export CONVERTED_DFLASH_OUT="${CONVERTED_DFLASH_OUT:-}"
export AUTO_CONVERT_DFLASH="${AUTO_CONVERT_DFLASH:-0}"
export REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3.5_9b_mm_10k_prune3}"
export SAVE_PATH="${SAVE_PATH:-./output/dflash_qwen3.5_9b_mm_10k_prune3_continue_dflash/${RUN_NAME}/checkpoints}"
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/wenxuan/ALLaVA-4V}"
export ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"

export MAX_SAMPLES="${MAX_SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-20}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export LR_FT="${LR_FT:-1e-5}"
export LR="${LR:-$LR_FT}"
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

echo "Starting detached 3-layer DFlash 10k training:"
echo "  run_name: $RUN_NAME"
echo "  prune_src: $PRUNE_SRC"
echo "  pruned_dflash_out: $PRUNED_DFLASH_OUT"
echo "  keep_layers: $KEEP_LAYERS"
echo "  output_dir: $OUTPUT_DIR"
echo "  save_path: $SAVE_PATH"
echo "  max_samples: $MAX_SAMPLES"
echo "  epochs: $EPOCHS"
echo "  checkpoint_freq: $CHECKPOINT_FREQ"
echo "  lr: $LR"
echo "  lr_ft: $LR_FT"
echo "  finetune_from: $FINETUNE_FROM"
echo "  block_size: $BLOCK_SIZE"
echo "  num_spec: $((BLOCK_SIZE - 1))"
echo "  max_anchors: $MAX_ANCHORS"
echo "  draft_vocab_size: ${DRAFT_VOCAB_SIZE:-full}"
echo "  draft_arch: $DRAFT_ARCH"
echo "  target_layer_ids: $TARGET_LAYER_IDS"
echo "  dflash_compile_training: $DFLASH_COMPILE"
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
