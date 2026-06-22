#!/bin/bash
# Detached MTP training on Qwen-distilled ALLaVA (multimodal).
#
# MTP trains the verifier's NATIVE multi-token-prediction head: speculators
# extracts the verifier's mtp.* weights and fine-tunes them (there is NO separate
# draft and NO FINETUNE_FROM). The verifier (MODEL) MUST contain native MTP
# weights -- Qwen3.5-9B does (that's what vLLM's `qwen3_5_mtp` serves). Reuses the
# exact multimodal online pipeline as DFlash, just with --speculator-type mtp.
#
#   bash examples/train/nohup_mtp_qwen3.5_9b_allava_distilled.sh
#
# SMOKE TEST FIRST (one short run validates the cross-fork MTP integration):
#   MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=1 \
#     bash examples/train/nohup_mtp_qwen3.5_9b_allava_distilled.sh
#   tail -f run_logs/mtp_*.nohup.log
# The most likely failure point is the gen<->model hidden-states width: MTP wants
# ONLY the last hidden state (width = 1 x hidden_size). If you see a width
# mismatch, that's the layer-alignment to fix (TARGET_LAYER_IDS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-mtp_qwen35_9b_allava_distilled_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

export MODEL="${MODEL:-/home/wenxuan/Qwen3.5-9B}"   # verifier; must contain native mtp.* weights
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
export DISTILLED_ALLAVA_JSONL="${DISTILLED_ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"
if [ ! -s "$DISTILLED_ALLAVA_JSONL" ]; then
    echo "[fatal] Distilled ALLaVA jsonl not found: $DISTILLED_ALLAVA_JSONL"
    exit 1
fi
if [ "${ALLOW_NON_9B_MODEL:-0}" != "1" ] && [[ ! "$MODEL" =~ (9[Bb]|9b) ]]; then
    echo "[fatal] This is the 9B MTP launcher, but MODEL=$MODEL"
    echo "        Use MODEL=/home/wenxuan/Qwen3.5-9B or set ALLOW_NON_9B_MODEL=1 intentionally."
    echo "        For 122B, use examples/train/nohup_mtp_122b_allava_distilled.sh."
    exit 1
fi
if [ "${ALLOW_NON_9B_DATA:-0}" != "1" ] && [[ "$DISTILLED_ALLAVA_JSONL" =~ 122[Bb] ]]; then
    echo "[fatal] This is the 9B MTP launcher, but DISTILLED_ALLAVA_JSONL=$DISTILLED_ALLAVA_JSONL"
    echo "        Use the 9B-distilled ALLaVA jsonl, or set ALLOW_NON_9B_DATA=1 intentionally."
    exit 1
fi
ROOT_TAG="$(python3 - "$ALLAVA_IMAGE_ROOT" <<'PY'
import hashlib
import sys

print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])
PY
)"
LOCAL_DISTILLED_ALLAVA_JSONL="${LOCAL_DISTILLED_ALLAVA_JSONL:-${DISTILLED_ALLAVA_JSONL%.jsonl}.local_${ROOT_TAG}.jsonl}"
if [ "${REWRITE_DISTILLED_IMAGE_PATHS:-1}" = "1" ]; then
    if [ ! -s "$LOCAL_DISTILLED_ALLAVA_JSONL" ] \
        || [ "$DISTILLED_ALLAVA_JSONL" -nt "$LOCAL_DISTILLED_ALLAVA_JSONL" ]; then
        echo "Rewriting distilled image paths for local image root:"
        echo "  source:     $DISTILLED_ALLAVA_JSONL"
        echo "  image_root: $ALLAVA_IMAGE_ROOT"
        echo "  output:     $LOCAL_DISTILLED_ALLAVA_JSONL"
        python3 scripts/rewrite_jsonl_image_paths.py \
            --in-jsonl "$DISTILLED_ALLAVA_JSONL" \
            --out-jsonl "$LOCAL_DISTILLED_ALLAVA_JSONL" \
            --image-root "$ALLAVA_IMAGE_ROOT"
    fi
    export DISTILLED_ALLAVA_JSONL="$LOCAL_DISTILLED_ALLAVA_JSONL"
fi

# --- MTP-specific ---
export SPECULATOR_TYPE=mtp
export NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"
export STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"
export MTP_SELF_FORCING_P="${MTP_SELF_FORCING_P:-0.0}"
export MTP_VAL_SELF_FORCING_P="${MTP_VAL_SELF_FORCING_P:-$MTP_SELF_FORCING_P}"
export FINETUNE_FROM=""        # MTP extracts the head from MODEL itself (no warm-start dir)
export DRAFT_VOCAB_SIZE=""     # full vocab

# --- data (same contract as the DFlash distilled launcher) ---
export USE_ALLAVA=0
export USE_MMSTAR=0
export DATASET="$DISTILLED_ALLAVA_JSONL"
export MEDIA_ROOT="$ALLAVA_IMAGE_ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/mtp_qwen3.5_9b_mm_distilled}"
export SAVE_PATH="${SAVE_PATH:-./output/mtp_qwen3.5_9b_mm_distilled/${RUN_NAME}/checkpoints}"

# Keep the 9B MTP smoke path isolated from the base multimodal launcher's DP=4
# default. DP=4 is useful for fast data generation, but it opens neighboring
# vLLM ports and can collide with stale 122B/old smoke servers.
export VLLM_GPUS="${VLLM_GPUS:-0}"
export VLLM_TP="${VLLM_TP:-1}"
export VLLM_DP="${VLLM_DP:-1}"
export GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.85}"
export TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# --- training knobs ---
export MAX_SAMPLES="${MAX_SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-20}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export LR="${LR:-3e-5}"        # MTP head is a warm-start; keep LR conservative (tune as needed)
export HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-bfloat16}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"
export NO_RESUME_FROM_CHECKPOINT="${NO_RESUME_FROM_CHECKPOINT:-1}"
export VALIDATE_INITIAL="${VALIDATE_INITIAL:-1}"
export VLLM_PORT="${VLLM_PORT:-18009}"  # avoid stale/default :8000 servers during smoke

export LOGGER="${LOGGER:-wandb}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.155.156.175:38080}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators}"
export RUN_NAME
export LOG_TO_FILE=0
export CLEAN_STALE_PROCS="${CLEAN_STALE_PROCS:-1}"

if [ "$CLEAN_STALE_PROCS" = "1" ]; then
    echo "Cleaning stale vLLM/train processes before launching 9B MTP run..."
    pkill -f '[v]llm.*serve' || true
    pkill -f '[t]orchrun.*scripts/train.py' || true
    pkill -f '[s]cripts/train.py' || true
    sleep 2
fi

echo "Starting detached MTP training on Qwen-distilled ALLaVA:"
echo "  run_name:    $RUN_NAME"
echo "  verifier:    $MODEL   (must contain native mtp.* weights)"
echo "  dataset:     $DATASET"
echo "  save_path:   $SAVE_PATH"
echo "  vLLM:        TP=$VLLM_TP DP=$VLLM_DP on GPUs [$VLLM_GPUS]  (mem_util=$GEN_GPU_MEM_UTIL)"
echo "  trainer:     $NUM_TRAIN_GPUS GPUs [$TRAIN_GPUS]"
echo "  spec_steps:  $NUM_SPECULATIVE_STEPS   step_weight_beta=$STEP_WEIGHT_BETA"
echo "  self_force:  train=$MTP_SELF_FORCING_P   val=$MTP_VAL_SELF_FORCING_P"
echo "  epochs:      $EPOCHS   lr=$LR   max_samples=$MAX_SAMPLES"
echo "  validate_initial: $VALIDATE_INITIAL"
echo "  vllm_port:   $VLLM_PORT"
echo "  cleanup:     $CLEAN_STALE_PROCS"
echo "  log:         $NOHUP_LOG_PATH"

nohup bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh \
    > "$NOHUP_LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"
echo "Started PID $PID  (pid file: $PID_PATH)"
echo "Follow log:"
echo "  tail -f $NOHUP_LOG_PATH"
