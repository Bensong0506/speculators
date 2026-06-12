#!/bin/bash
# Train an MTP speculator against an already running vLLM hidden-state server.
#
# This is the modular version of examples/train/mtp_online_pipeline.sh for
# schedulers and clusters where data preparation and vLLM serving happen in
# separate jobs.
#
# Required inputs:
#   MODEL       verifier model ID or local path
#   DATA_PATH   preprocessed dataset directory from scripts/prepare_data.py
#
# Example:
#   MODEL=Qwen/Qwen3.5-9B \
#   DATA_PATH=./output/mtp \
#   SAVE_PATH=./output/mtp/checkpoints \
#   VLLM_ENDPOINT=http://localhost:8000/v1 \
#     bash examples/train/mtp_train_online.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-./output/mtp}"
SAVE_PATH="${SAVE_PATH:-$DATA_PATH/checkpoints}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-32}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"
STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"
FROM_PRETRAINED="${FROM_PRETRAINED:-}"
ON_GENERATE="${ON_GENERATE:-delete}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}"
SAVE_BEST="${SAVE_BEST:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"

# Optional post-training stitch.
STITCH_MTP="${STITCH_MTP:-0}"
STITCHED_DIR="${STITCHED_DIR:-$DATA_PATH/stitched}"
CHECKPOINT_TO_STITCH="${CHECKPOINT_TO_STITCH:-$SAVE_PATH/checkpoint_best}"

read -r -a TARGET_LAYER_ARGS <<< "$TARGET_LAYER_IDS"

train_args=(
    scripts/train.py
    --verifier-name-or-path "$MODEL"
    --data-path "$DATA_PATH"
    --vllm-endpoint "$VLLM_ENDPOINT"
    --save-path "$SAVE_PATH"
    --speculator-type mtp
    --num-speculative-steps "$NUM_SPECULATIVE_STEPS"
    --target-layer-ids "${TARGET_LAYER_ARGS[@]}"
    --step-weight-beta "$STEP_WEIGHT_BETA"
    --epochs "$EPOCHS"
    --lr "$LR"
    --total-seq-len "$SEQ_LENGTH"
    --on-missing generate
    --on-generate "$ON_GENERATE"
)

if [[ -n "$FROM_PRETRAINED" ]]; then
    train_args+=(--from-pretrained "$FROM_PRETRAINED")
fi
if [[ -n "$HIDDEN_STATES_PATH" ]]; then
    train_args+=(--hidden-states-path "$HIDDEN_STATES_PATH")
fi
if [[ "$SAVE_BEST" == "1" ]]; then
    train_args+=(--save-best)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    train_args+=(--trust-remote-code)
fi

echo "=== Training MTP ==="
if [[ "$NUM_TRAIN_GPUS" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
        --standalone \
        --nproc_per_node "$NUM_TRAIN_GPUS" \
        "${train_args[@]}"
else
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python3 "${train_args[@]}"
fi

if [[ "$STITCH_MTP" == "1" ]]; then
    echo "=== Stitching MTP weights ==="
    if [[ ! -e "$CHECKPOINT_TO_STITCH" ]]; then
        echo "No checkpoint found at $CHECKPOINT_TO_STITCH"
        echo "Set CHECKPOINT_TO_STITCH to a checkpoint directory,"
        echo "or keep SAVE_BEST=1."
        exit 1
    fi

    python3 scripts/stitch_mtp.py \
        "$CHECKPOINT_TO_STITCH" \
        "$MODEL" \
        --output-path "$STITCHED_DIR"
    echo "Done. Stitched checkpoint saved to $STITCHED_DIR/"
else
    echo "Done. MTP checkpoints saved to $SAVE_PATH/"
fi
