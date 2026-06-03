#!/bin/bash
# Quick test: serve Qwen3.5-9B + OUR trained DFlash draft on a GPU, then curl it.
#
#   bash examples/serve/test_trained_dflash_gpu.sh
#
# It prints a ready-to-paste curl command, then starts vLLM in the foreground.
# Run the curl in ANOTHER terminal once you see "Application startup complete".
#
# Override any path via env, e.g.:
#   DRAFT=/some/other/checkpoint PORT=8100 bash examples/serve/test_trained_dflash_gpu.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/models/Qwen3.5-9B}"
DRAFT="${DRAFT:-/home/wenxuan/speculators/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"          # we trained block_size=8 -> 7
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8100}"
export CUDA_VISIBLE_DEVICES

echo "== checkpoint files =="
ls -l "$DRAFT" || { echo "[fatal] trained draft not found: $DRAFT"; exit 1; }
echo

echo "== once the server is UP, run this in another terminal =="
echo "curl http://localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍北京\"}],\"max_tokens\":128}'"
echo

echo "== starting vLLM (target=$MODEL_PATH  +  our dflash draft=$DRAFT) =="
exec vllm serve "$MODEL_PATH" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768 \
  --port "$PORT"
# To test with IMAGES, append these to the line above and curl with an image_url
# whose path is under the allowed dir:
#   --allowed-local-media-path /home/wenxuan/multimodel_test \
#   --limit-mm-per-prompt '{"image":40}' \
#   --trust-remote-code
