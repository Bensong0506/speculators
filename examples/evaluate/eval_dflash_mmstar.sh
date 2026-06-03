#!/bin/bash
# Auto-evaluate a trained DFlash multimodal speculator.
#
# Serves the base verifier + the trained speculator in vLLM, sends a handful of
# multimodal prompts (taken from your MMStar jsonl), then parses the vLLM log for
# speculative-decoding acceptance (per-position rate + mean accepted length).
#
# Run it AFTER training finishes (GPUs free):
#   bash examples/evaluate/eval_dflash_mmstar.sh
#
# NOTE: this needs a vLLM build that can RUN a DFlash speculator (DFlash inference
# landed via vLLM PR #38300). If your vLLM doesn't have it, the server will fail to
# start with an "unknown method dflash" style error — the script reports that.

set -euo pipefail

# ===================== CONFIG — EDIT THESE ============================
MODEL="/home/models/Qwen3.5-9B"
SPECULATOR="$(pwd)/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best"
DATA_JSONL="$(pwd)/data/mmstar/mmstar.jsonl"   # prompts (image+text) come from here
MEDIA_ROOT="/home/wenxuan/mmstar/images"        # vLLM may read images under this dir

GPUS="0"                 # GPUs for the eval server (1 card holds a 9B fine)
TP=1                     # tensor-parallel size; bump (and add GPUs) if it OOMs
NUM_SPEC_TOKENS=7        # tokens drafted per step; must be <= trained block_size (8)
MAX_MODEL_LEN=8192       # keep small — Qwen3-Next defaults to ~256K and will OOM
GPU_MEM_UTIL=0.85
PORT=8001

NUM_PROMPTS=64           # how many prompts to send
MAX_TOKENS=128           # output tokens per prompt (longer = more spec-decode signal)
SERVER_LOG="$(pwd)/output/eval_vllm.log"
PARSE_LOGS="examples/evaluate/eval-guidellm/scripts/parse_logs.py"
# =====================================================================

# vLLM speculative-decoding config: run the trained checkpoint as a dflash drafter.
SPEC_CONFIG="{\"model\": \"${SPECULATOR}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}, \"method\": \"dflash\", \"max_model_len\": ${MAX_MODEL_LEN}}"

echo "=== Serving verifier + speculator ==="
echo "    base:       ${MODEL}"
echo "    speculator: ${SPECULATOR}"
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
    --seed 42 \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --trust-remote-code \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --speculative-config "$SPEC_CONFIG" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping eval server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server to be ready (log: $SERVER_LOG)..."
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died during startup. Last 40 log lines:"
        tail -n 40 "$SERVER_LOG"
        echo
        if grep -qiE "M-?RoPE|does not support multimodal" "$SERVER_LOG"; then
            echo "DIAGNOSIS: your vLLM does NOT support speculative decoding for"
            echo "multimodal (M-RoPE) verifiers like Qwen-VL yet. This is a vLLM"
            echo "limitation (vLLM issue #42005), NOT a problem with your trained"
            echo "speculator. Judge quality with the training-time val acceptance"
            echo "metrics (val/full_acc, val/position_k_acc); the end-to-end serving"
            echo "speedup eval needs a vLLM build that adds M-RoPE to the spec path."
        elif grep -qiE "unknown.*dflash|method.*dflash" "$SERVER_LOG"; then
            echo "DIAGNOSIS: your vLLM lacks DFlash inference support (vLLM PR #38300)."
        fi
        exit 1
    fi
    sleep 5
done

echo "=== Sending ${NUM_PROMPTS} multimodal prompts (max_tokens=${MAX_TOKENS}) ==="
python3 examples/evaluate/send_mm_requests.py \
    --endpoint "http://localhost:${PORT}/v1" \
    --data-jsonl "$DATA_JSONL" \
    --num "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS"

sleep 3   # let vLLM flush its SpecDecoding metrics into the log

echo "=== Speculative-decoding acceptance analysis ==="
if ! python3 "$PARSE_LOGS" "$SERVER_LOG"; then
    echo "(parse_logs found no 'SpecDecoding metrics' lines.)"
    echo "Raw spec-related log lines:"
    grep -iE "spec|accept|drafted" "$SERVER_LOG" | tail -n 20 || true
    echo "If empty, spec decoding may not have run — try more/longer requests,"
    echo "or check $SERVER_LOG for errors."
fi

echo
echo "Done. Full server log: $SERVER_LOG"
