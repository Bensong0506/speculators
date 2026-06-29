#!/bin/bash
# STEP 3 (CLIENT) — Test the trained 122B MTP head against the client's stock one.
#
# Two arms, SAME method (qwen3_5_mtp), SAME spec count, SAME client-domain val tail
# (last 10% of the STEP-1 distill jsonl); only the MTP-head weights differ:
#   - original_mtp : serve CLIENT_MODEL raw          -> the post-SFT model's NATIVE head
#   - trained_mtp  : stitch MTP_CKPT into CLIENT_MODEL -> our STEP-2 fine-tuned head
#
# So this measures exactly what the re-train bought on top of the client's own SFT.
# stitch_mtp.py already handles the 122B sharded / MoE checkpoint (index.json +
# inverse expert-fuse), so no special handling is needed beyond setting TP.
#
# Run from the mtp-training env (stitch_mtp.py imports speculators.convert.mtp).
#
# USAGE
#   CLIENT_MODEL=/data/wenxuan/Qwen3.5-122B-A10B-sft \
#   MTP_CKPT=output/mtp_client_122b/<run>/checkpoints/checkpoint_best \
#   CLIENT_DISTILL_JSONL=data/client/client_122b_distill_10k.jsonl \
#   CLIENT_IMAGE_ROOT=/data/client/images \
#   bash examples/evaluate/test_mtp_client_122b.sh
#
# OUTPUT
#   output/mtp_orig_vs_trained/<stamp>/mtp_orig_vs_trained_summary.md (+ per-arm json)
#   Trust first-pos / mean-accept; tok/s is noisy. Real tokens/step L = mean-accept + 1.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- client knobs ----
CLIENT_MODEL="${CLIENT_MODEL:-/data/wenxuan/Qwen3.5-122B-A10B-sft}"
CLIENT_DISTILL_JSONL="${CLIENT_DISTILL_JSONL:-$REPO_ROOT/data/client/client_122b_distill_10k.jsonl}"
CLIENT_IMAGE_ROOT="${CLIENT_IMAGE_ROOT:-/data/client/images}"

if [ -z "${MTP_CKPT:-}" ]; then
    echo "[fatal] set MTP_CKPT=output/mtp_client_122b/<run>/checkpoints/checkpoint_best"
    exit 1
fi

# Map client knobs onto the base orig-vs-trained eval. 122B needs TP (heads=32 -> {4,8}).
export MODEL="$CLIENT_MODEL"
export ALLAVA_JSONL="$CLIENT_DISTILL_JSONL"      # its tail-10% is the val slice
export ALLAVA_IMAGE_ROOT="$CLIENT_IMAGE_ROOT"
export MTP_CKPT
export TP="${TP:-8}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
export INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b-sft-mtp-orig-vs-trained}"

echo "=== STEP 3: client stock-SFT MTP vs trained MTP (122B) ==="
echo "  client_model:  $MODEL"
echo "  trained ckpt:  $MTP_CKPT"
echo "  val jsonl:     $ALLAVA_JSONL  (tail 10%)"
echo "  serve:         TP=$TP on GPUs [$GPUS]   spec=$INFER_NUM_SPEC   prompts=$NUM_PROMPTS"

exec bash "$SCRIPT_DIR/test_mtp_allava_orig_vs_trained.sh"
