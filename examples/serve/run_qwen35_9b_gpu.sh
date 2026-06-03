#!/bin/bash
# Qwen3.5-9B single-case launcher — GPU / CUDA version.
#
# Mirrors the proven Ascend launcher (run_qwen35_9b_onecase.sh) but for NVIDIA
# GPUs, and points the DFlash draft at OUR trained speculator checkpoint.
#
# Modes:
#   RUN_MODE=baseline bash examples/serve/run_qwen35_9b_gpu.sh
#   RUN_MODE=mtp    MTP_SPEC=3   bash examples/serve/run_qwen35_9b_gpu.sh
#   RUN_MODE=dflash DFLASH_SPEC=5 bash examples/serve/run_qwen35_9b_gpu.sh
#
# vLLM BUILD: DFlash multimodal needs a vLLM with the DFlash drafter + the
# M-RoPE input-position handling. The z-lab/Qwen3.5-9B-DFlash card installs
# nightly:
#   uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
# If your vLLM ignores --attention-backend, use: export VLLM_ATTENTION_BACKEND=FLASH_ATTN

set -euo pipefail
cd "$(dirname "$0")/../.."

# ===== model + device config (EDIT THESE) =====
export MODEL_PATH="${MODEL_PATH:-/home/models/Qwen3.5-9B}"
# DFlash draft = OUR trained speculator checkpoint (swap in a published one,
# e.g. z-lab/Qwen3.5-9B-DFlash, by overriding DFLASH_DRAFT_PATH).
export DFLASH_DRAFT_PATH="${DFLASH_DRAFT_PATH:-$(pwd)/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP="${TP:-1}"
export VLLM_PORT="${VLLM_PORT:-8100}"
export MM_MEDIA_DIR="${MM_MEDIA_DIR:-/home/wenxuan/multimodel_test}"  # allowed image dir

# ===== run mode =====
RUN_MODE="${RUN_MODE:-baseline}"     # baseline | mtp | dflash

# ===== spec params =====
MTP_SPEC="${MTP_SPEC:-3}"
DFLASH_SPEC="${DFLASH_SPEC:-5}"      # tokens drafted/step; must be <= trained block_size (8)

# ===== inference params =====
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
MAX_IMAGES="${MAX_IMAGES:-40}"
ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"   # z-lab card requires flash_attn on GPU
DFLASH_GRAPH_MODE="${DFLASH_GRAPH_MODE:-NONE}"   # DFlash: cudagraph off by default

case "$RUN_MODE" in
  baseline|mtp|dflash) ;;
  *) echo "[fatal] RUN_MODE must be baseline / mtp / dflash, got: $RUN_MODE"; exit 1 ;;
esac

echo "================================"
echo "vLLM serve Qwen3.5-9B single case (GPU)"
echo "  mode:          $RUN_MODE"
echo "  target:        $MODEL_PATH"
echo "  draft:         $DFLASH_DRAFT_PATH"
echo "  served-as:     $SERVED_MODEL_NAME"
echo "  port:          $VLLM_PORT"
echo "  TP:            $TP"
echo "  devices:       $CUDA_VISIBLE_DEVICES"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  max_num_seqs:  $MAX_NUM_SEQS"
echo "  attn backend:  $ATTN_BACKEND"
echo "  dflash spec:   $DFLASH_SPEC"
echo "  dflash graph:  $DFLASH_GRAPH_MODE"
echo "================================"

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null | head -20 || true

[ -d "$MODEL_PATH" ] || { echo "[fatal] MODEL_PATH not found: $MODEL_PATH"; exit 1; }
if [ "$RUN_MODE" = "dflash" ]; then
  [ -d "$DFLASH_DRAFT_PATH" ] || { echo "[fatal] DFLASH_DRAFT_PATH not found: $DFLASH_DRAFT_PATH"; exit 1; }
fi

ARGS=(
  vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --allowed-local-media-path "$MM_MEDIA_DIR"
  --limit-mm-per-prompt "{\"image\":$MAX_IMAGES}"
  --trust-remote-code
  --dtype bfloat16
  --attention-backend "$ATTN_BACKEND"
  --no-enable-chunked-prefill
  --generation-config vllm
  --host 0.0.0.0
  --port "$VLLM_PORT"
)
# Note: dropped the Ascend launcher's `--block-size 128` (NPU-specific tuning;
# CUDA attention backends use smaller KV block sizes — let vLLM default it).

if [ "$RUN_MODE" = "mtp" ]; then
  ARGS+=(
    --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP_SPEC,\"enforce_eager\":true}"
  )
fi

if [ "$RUN_MODE" = "dflash" ]; then
  ARGS+=(
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT_PATH\",\"num_speculative_tokens\":$DFLASH_SPEC}"
    --compilation-config "{\"cudagraph_mode\":\"$DFLASH_GRAPH_MODE\"}"
  )
fi

echo
echo "[cmd]"
printf ' %q' "${ARGS[@]}"
echo
echo

exec "${ARGS[@]}"
