#!/bin/bash
# Detached DFlash continue-training on the ~122B verifier (Qwen3.5-122B-A10B, MoE),
# warm-started from a downloaded z-lab DFlash draft.
#
# Mirrors the 122B MTP launcher's GPU layout (verifier TP=4 on GPUs 0-3, DFlash
# trainer on GPUs 4-7) and the 9B DFlash *winning* recipe (CE + fp32 + LR 3e-5).
#
# block_size / aux target-layer-ids / draft_arch / mask are read from
# $FINETUNE_FROM/config.json by the base launcher, so the verifier exposes exactly
# the layers the draft expects -- you do NOT set those here. A raw z-lab checkpoint
# is auto-converted to speculators format (AUTO_CONVERT_DFLASH=1, uses $MODEL to
# extract verifier_norm) so the weights actually LOAD (true warm-start).
#
# DFlash trains a SEPARATE small draft, so fp32 is fine here (unlike MTP, where fp32
# breaks the verifier lm_head -> MTP stays bf16). heads=32 -> VLLM_TP in {4,8} (not 6).
#
# PREREQ: 122B distilled data (examples/train/distill_allava_122b.sh) + the
#         downloaded z-lab 122B DFlash draft dir.
#
# USAGE
#   MODEL=/data/wenxuan/Qwen3.5-122B-A10B \
#   FINETUNE_FROM=/data/wenxuan/Qwen3.5-122B-DFlash \
#   ALLAVA_IMAGE_ROOT=/home/wenxuan/ALLaVA-4V \
#   bash examples/train/nohup_dflash_122b_allava_distilled.sh
#
# SMOKE FIRST (validates 122B TP + warm-start weight load + DFlash wiring):
#   MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=0 MODEL=... FINETUNE_FROM=... \
#     bash examples/train/nohup_dflash_122b_allava_distilled.sh
#   tail -f run_logs/dflash_122b_*.nohup.log
#
# If TP=4 verifier OOMs: raise GEN_GPU_MEM_UTIL (->0.92), lower SEQ_LENGTH, or use
#   VLLM_TP=8 VLLM_GPUS=0,1,2,3,4,5,6,7 GEN_GPU_MEM_UTIL=0.55 TRAIN_GPUS=6,7.
# If the fp32 trainer OOMs: HIDDEN_STATES_DTYPE=bfloat16 (weaker recipe, less memory).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dflash_122b_allava_distilled_${STAMP}}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-$REPO_ROOT/run_logs}"
NOHUP_LOG_PATH="${NOHUP_LOG_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.nohup.log}"
PID_PATH="${PID_PATH:-$NOHUP_LOG_DIR/${RUN_NAME}.pid}"
mkdir -p "$NOHUP_LOG_DIR"

# ---- verifier (122B) + warm-start draft ----
export MODEL="${MODEL:-/data/wenxuan/Qwen3.5-122B-A10B}"
[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
export FINETUNE_FROM="${FINETUNE_FROM:-/data/wenxuan/Qwen3.5-122B-DFlash}"
if [ ! -f "$FINETUNE_FROM/config.json" ]; then
    echo "[fatal] FINETUNE_FROM/config.json not found: $FINETUNE_FROM"
    echo "        Point it at your downloaded z-lab 122B DFlash draft dir."
    echo "        (raw z-lab is OK -- AUTO_CONVERT_DFLASH=1 converts it here)."
    exit 1
fi
export AUTO_CONVERT_DFLASH="${AUTO_CONVERT_DFLASH:-1}"            # raw z-lab -> speculators so weights load
export REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-1}"  # true warm-start; fail if weights don't load

# ---- data: auto-pick the newest 122B distilled jsonl, MAX_SAMPLES from its rows ----
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/wenxuan/ALLaVA-4V}"
export DISTILLED_ALLAVA_JSONL="${DISTILLED_ALLAVA_JSONL:-$(ls -t "$REPO_ROOT"/data/allava/allava_122b_distill_*k.jsonl 2>/dev/null | head -1)}"
if [ ! -s "${DISTILLED_ALLAVA_JSONL:-}" ]; then
    echo "[fatal] no 122B distilled jsonl found (data/allava/allava_122b_distill_*k.jsonl)."
    echo "        Make it with: bash examples/train/distill_allava_122b.sh"
    exit 1
fi
export MAX_SAMPLES="${MAX_SAMPLES:-$(wc -l < "$DISTILLED_ALLAVA_JSONL")}"
export USE_ALLAVA=0
export USE_MMSTAR=0
export DATASET="$DISTILLED_ALLAVA_JSONL"
export MEDIA_ROOT="$ALLAVA_IMAGE_ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_122b_mm_distilled}"
export SAVE_PATH="${SAVE_PATH:-./output/dflash_122b_mm_distilled/${RUN_NAME}/checkpoints}"

# ---- GPU layout: verifier TP=4 (GPUs 0-3) + DFlash trainer (GPUs 4-7) ----
export VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
export VLLM_TP="${VLLM_TP:-4}"
export VLLM_DP="${VLLM_DP:-1}"
export GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.90}"
export TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# ---- DFlash winning recipe: CE + fp32 + LR 3e-5.
# block_size / target_layer_ids / draft_arch / mask are read from FINETUNE_FROM by
# the base launcher (do NOT set them here -- they must match the warm-start draft).
export SPECULATOR_TYPE=dflash
export LOSS_FN="${LOSS_FN:-ce}"
export HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-float32}"
export LR_FT="${LR_FT:-3e-5}"
export LR="${LR:-$LR_FT}"
export EPOCHS="${EPOCHS:-20}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"
export NO_RESUME_FROM_CHECKPOINT="${NO_RESUME_FROM_CHECKPOINT:-1}"
export DFLASH_COMPILE="${DFLASH_COMPILE:-1}"
export VALIDATE_INITIAL="${VALIDATE_INITIAL:-1}"

export LOGGER="${LOGGER:-wandb}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.155.156.175:38080}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators}"
export RUN_NAME
export LOG_TO_FILE=0

echo "Starting detached 122B DFlash training (warm-start):"
echo "  run_name:    $RUN_NAME"
echo "  verifier:    $MODEL"
echo "  warm-start:  $FINETUNE_FROM  (auto_convert=$AUTO_CONVERT_DFLASH, require_weights=$REQUIRE_PRETRAINED_WEIGHTS)"
echo "  vLLM:        TP=$VLLM_TP DP=$VLLM_DP on GPUs [$VLLM_GPUS]  (mem_util=$GEN_GPU_MEM_UTIL)"
echo "  trainer:     $NUM_TRAIN_GPUS GPUs [$TRAIN_GPUS]"
echo "  dataset:     $DATASET  (samples=$MAX_SAMPLES)"
echo "  recipe:      loss=$LOSS_FN dtype=$HIDDEN_STATES_DTYPE lr=$LR epochs=$EPOCHS"
echo "  (block_size / target_layer_ids / draft_arch read from FINETUNE_FROM)"
echo "  save_path:   $SAVE_PATH"
echo "  log:         $NOHUP_LOG_PATH"

nohup bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh \
    > "$NOHUP_LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"
echo "Started PID $PID  (pid file: $PID_PATH)"
echo "Follow log:  tail -f $NOHUP_LOG_PATH"
