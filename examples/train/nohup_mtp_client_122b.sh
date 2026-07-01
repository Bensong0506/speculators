#!/bin/bash
# STEP 2 (CLIENT) — Detached MTP training on the client's SFT'd 122B.
#
# Extracts and fine-tunes the post-SFT verifier's NATIVE mtp.* head on the
# client-domain self-distilled data from STEP 1. No separate draft, no
# FINETUNE_FROM — the head lives inside CLIENT_MODEL's own weights.
#
# Thin wrapper over the validated examples/train/nohup_mtp_122b_allava_distilled.sh
# (verifier vLLM TP=4 on GPUs 0-3 + MTP trainer on GPUs 4-7; bf16 — fp32 breaks the
# MTP lm_head). Only swaps in the client paths.
#
# USAGE
#   CLIENT_MODEL=/data/wenxuan/Qwen3.5-122B-A10B-sft \
#   CLIENT_DISTILL_JSONL=/data/wenxuan/speculators/data/client/client_122b_distill_10k.jsonl \
#   CLIENT_IMAGE_ROOT=/data/client/images \
#   bash examples/train/nohup_mtp_client_122b.sh
#
# SMOKE TEST FIRST (validates TP layout + MTP extraction on the SFT'd model):
#   MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=0 \
#     CLIENT_MODEL=... CLIENT_DISTILL_JSONL=... CLIENT_IMAGE_ROOT=... \
#     bash examples/train/nohup_mtp_client_122b.sh
#   tail -f run_logs/mtp_client_122b_*.nohup.log
#
# Best-known 122B MTP recipe (from the ALLaVA runs): NUM_SPECULATIVE_STEPS=3,
# STEP_WEIGHT_BETA=0.6, LR=3e-5, bf16. Tune EPOCHS to data size (10 for ~10k).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- client knobs ----
CLIENT_MODEL="${CLIENT_MODEL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B}"
CLIENT_DISTILL_JSONL="${CLIENT_DISTILL_JSONL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/client_122b_distill_multimodal_8137.jsonl}"
CLIENT_IMAGE_ROOT="${CLIENT_IMAGE_ROOT:-/mnt/tidal-alsh01}"

if [ ! -d "$CLIENT_MODEL" ]; then
    echo "[fatal] CLIENT_MODEL not found: $CLIENT_MODEL  (must contain native mtp.* weights)"
    exit 1
fi
if [ ! -s "$CLIENT_DISTILL_JSONL" ]; then
    echo "[fatal] CLIENT_DISTILL_JSONL not found/empty: $CLIENT_DISTILL_JSONL"
    echo "        Run STEP 1 (examples/train/distill_client_122b.sh) first."
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_NAME="${RUN_NAME:-mtp_client_122b_${STAMP}}"

# Map client knobs onto the base 122B MTP launcher's contract.
export MODEL="$CLIENT_MODEL"
export DISTILLED_ALLAVA_JSONL="$CLIENT_DISTILL_JSONL"
export ALLAVA_IMAGE_ROOT="$CLIENT_IMAGE_ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/mtp_client_122b}"
export SAVE_PATH="${SAVE_PATH:-$OUTPUT_DIR/${RUN_NAME}/checkpoints}"

# Client RAG prompts are LONG (system ~5k + retrieved notes up to ~56k chars ->
# ~9k-15k tokens typical). The 9B/ALLaVA default SEQ_LENGTH=4096 would truncate
# the context the answer was generated from. Bump it; note longer seq raises
# trainer memory (full-vocab logits scale seq x steps), so if STEP 2 OOMs lower
# this or NUM_SPECULATIVE_STEPS first.
export SEQ_LENGTH="${SEQ_LENGTH:-16384}"
export PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-16384}"

echo "=== STEP 2: MTP training on post-SFT 122B (client domain) ==="
echo "  run_name:      $RUN_NAME"
echo "  client_model:  $MODEL"
echo "  distill_jsonl: $DISTILLED_ALLAVA_JSONL"
echo "  save_path:     $SAVE_PATH"
echo "  (verifier TP=${VLLM_TP:-4} GPUs ${VLLM_GPUS:-0,1,2,3} + trainer GPUs ${TRAIN_GPUS:-4,5,6,7})"

exec bash "$SCRIPT_DIR/nohup_mtp_122b_allava_distilled.sh"
