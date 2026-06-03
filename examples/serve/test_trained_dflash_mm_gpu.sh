#!/bin/bash
# Test OUR trained DFlash draft with a MULTIMODAL (image) request on GPU.
#
# First neutralize the 0.22.0 M-RoPE guard (one time):
#   bash examples/serve/patch_vllm_mrope_guard.sh
# Then:
#   bash examples/serve/test_trained_dflash_mm_gpu.sh
# It prints an image curl (using the first image under MM_MEDIA_DIR), then serves.
#
# vLLM 0.22.0's spec-decode propose() already has an `if self.uses_mrope` branch,
# so once the init guard is gone this often "just works" for images. If it errors
# at mrope position init instead, send that traceback — there's a second patch
# (T=H=W=arange fallback in _init_mrope_positions).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/models/Qwen3.5-9B}"
DRAFT="${DRAFT:-/home/wenxuan/speculators/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"           # trained block_size=8 -> 7
MM_MEDIA_DIR="${MM_MEDIA_DIR:-/home/wenxuan/mmstar/images}"   # must contain images
MAX_IMAGES="${MAX_IMAGES:-40}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8100}"
SERVED="${SERVED:-qwen3.5-9b}"
export CUDA_VISIBLE_DEVICES

ls -d "$DRAFT" > /dev/null || { echo "[fatal] trained draft not found: $DRAFT"; exit 1; }

# warn if the M-RoPE guard is still active
GUARD=$(python3 -c 'import vllm,os;p=os.path.join(os.path.dirname(vllm.__file__),"v1","spec_decode","llm_base_proposer.py");print(p)')
if grep -q "self._raise_if_mrope()" "$GUARD" 2>/dev/null && ! grep -q "patched-out _raise_if_mrope" "$GUARD" 2>/dev/null; then
    echo "[warn] M-RoPE guard still active in $GUARD"
    echo "       run:  bash examples/serve/patch_vllm_mrope_guard.sh   first"
fi

IMG=$(ls "$MM_MEDIA_DIR"/* 2>/dev/null | head -1 || true)
echo
echo "== once the server is UP, run this IMAGE request in another terminal =="
if [ -n "$IMG" ]; then
    echo "curl http://localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${SERVED}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file://${IMG}\"}},{\"type\":\"text\",\"text\":\"描述这张图\"}]}],\"max_tokens\":128}'"
else
    echo "[warn] no image under $MM_MEDIA_DIR — put a .jpg there (it must be under --allowed-local-media-path)"
fi
echo

echo "== starting vLLM (multimodal, target + our dflash draft) =="
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
  --compilation-config "{\"cudagraph_mode\":\"NONE\"}" \
  --attention-backend flash_attn \
  --allowed-local-media-path "$MM_MEDIA_DIR" \
  --limit-mm-per-prompt "{\"image\":$MAX_IMAGES}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-batched-tokens 32768 \
  --no-enable-chunked-prefill \
  --port "$PORT"
