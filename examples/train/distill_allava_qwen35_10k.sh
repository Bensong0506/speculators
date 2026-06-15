#!/bin/bash
# Distill 10k ALLaVA samples with Qwen3.5-9B into training-ready conversations jsonl.
#
# This starts a plain verifier vLLM server, removes the original ALLaVA/GT
# assistant answers, generates new Qwen answers, and writes:
#
#   data/allava/allava_qwen35_distill_10k.jsonl
#
# The output can be used directly as DATASET for DFlash training.
#
# Usage:
#   bash examples/train/distill_allava_qwen35_10k.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/home/wenxuan/Qwen3.5-9B}"
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"
OUT_JSONL="${OUT_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"

MAX_SAMPLES="${MAX_SAMPLES:-10000}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-}"
SKIP_SAMPLES="${SKIP_SAMPLES:-0}"
STRIDE="${STRIDE:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
RESUME="${RESUME:-1}"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-allava-distill}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Client-side in-flight requests. Default to the server batch size so the vLLM
# batch is actually saturated; before this flag the client was sequential (=1).
CONCURRENCY="${CONCURRENCY:-$MAX_NUM_SEQS}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
DTYPE="${DTYPE:-bfloat16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
START_SERVER="${START_SERVER:-1}"
ENDPOINT="${ENDPOINT:-http://localhost:${PORT}/v1}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/run_logs}"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/distill_allava_qwen35_10k_${STAMP}_vllm.log}"
mkdir -p "$LOG_DIR" "$(dirname "$OUT_JSONL")"

SERVER_PID=""
cleanup_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server pid=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

wait_for_server() {
    echo "Waiting for verifier server on :$PORT (log: $SERVER_LOG)"
    for _ in $(seq 1 180); do
        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM server died during startup. Last 100 lines:"
            tail -n 100 "$SERVER_LOG" || true
            return 1
        fi
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "Verifier server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for verifier server. Last 100 lines:"
    tail -n 100 "$SERVER_LOG" || true
    return 1
}

start_server() {
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server."
        echo "       Set START_SERVER=0 to reuse it, or choose PORT=...."
        return 1
    fi

    local args=(
        vllm serve "$MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --seed 42
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEMORY_UTIL"
        --trust-remote-code
        --dtype "$DTYPE"
        --allowed-local-media-path "$ALLAVA_IMAGE_ROOT"
        --limit-mm-per-prompt '{"image":1}'
        --generation-config vllm
        --host 0.0.0.0
        --port "$PORT"
    )
    if [ "$ENFORCE_EAGER" = "1" ]; then
        args+=(--enforce-eager)
    fi
    if [ -n "$ATTENTION_BACKEND" ]; then
        args+=(--attention-backend "$ATTENTION_BACKEND")
    fi

    echo "=== Starting Qwen verifier for ALLaVA distillation ==="
    echo "  model:        $MODEL"
    echo "  image_root:   $ALLAVA_IMAGE_ROOT"
    echo "  out_jsonl:    $OUT_JSONL"
    echo "  max_samples:  $MAX_SAMPLES"
    echo "  total_samples:${TOTAL_SAMPLES:-unset}"
    echo "  shard:        $SHARD_INDEX/$NUM_SHARDS"
    echo "  port:         $PORT"
    echo "  gpus:         $GPUS"
    echo "  max_num_seqs: $MAX_NUM_SEQS   client_concurrency: $CONCURRENCY"
    echo "  log:          $SERVER_LOG"
    printf '[cmd]'
    printf ' %q' env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}"
    echo

    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

if [ "$START_SERVER" = "1" ]; then
    start_server
else
    echo "START_SERVER=0 -> reusing endpoint $ENDPOINT"
fi

INPUT_ARGS=()
for src in $ALLAVA_INPUTS; do
    INPUT_ARGS+=(--in "$src")
done

DISTILL_ARGS=(
    python3 scripts/distill_allava_with_qwen.py
    --endpoint "$ENDPOINT"
    "${INPUT_ARGS[@]}"
    --image-root "$ALLAVA_IMAGE_ROOT"
    --out-jsonl "$OUT_JSONL"
    --max-samples "$MAX_SAMPLES"
    --skip-samples "$SKIP_SAMPLES"
    --stride "$STRIDE"
    --num-shards "$NUM_SHARDS"
    --shard-index "$SHARD_INDEX"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --request-timeout "$REQUEST_TIMEOUT"
    --concurrency "$CONCURRENCY"
)
if [ "$RESUME" = "1" ]; then
    DISTILL_ARGS+=(--resume)
fi
if [ -n "$TOTAL_SAMPLES" ]; then
    DISTILL_ARGS+=(--total-samples "$TOTAL_SAMPLES")
fi

echo "=== Distilling ALLaVA prompts with Qwen ==="
printf '[cmd]'
printf ' %q' "${DISTILL_ARGS[@]}"
echo
"${DISTILL_ARGS[@]}"

echo
echo "Distilled data ready:"
echo "  $OUT_JSONL"
echo
echo "Use it for training with:"
echo "  USE_ALLAVA=0 DATASET=$OUT_JSONL MEDIA_ROOT=$ALLAVA_IMAGE_ROOT bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh"
