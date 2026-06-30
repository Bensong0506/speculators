#!/bin/bash
# STEP 1 (CLIENT) — Data distillation for MTP on the client's SFT'd 122B.
#
# Serves the FULL-PRECISION post-SFT verifier (qwen3.5-vl-122B, NOT the
# msModelSlim-quantized dir) and regenerates the assistant answers with it, so the
# MTP head learns THIS post-SFT model's own continuations (on-policy alignment).
#
# Client data = `messages` (OpenAI role/content) + `images` (abs paths), so this
# drives scripts/distill_client_messages.py (multi-image aware), not the ALLaVA
# single-image distiller.
#
# DEFAULT MODE = TEXT-ONLY: the task (小红书 "问一问" RAG search) is text-dominated;
# text-only bootstraps the whole MTP pipeline without the 20-images/sample +
# long-context cost. Set MODE=multimodal for the full image path (phase 2).
#
# USAGE
#   CLIENT_MODEL=/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B \
#   CLIENT_TRAIN_JSONL=/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/train.jsonl \
#   MAX_SAMPLES=8137 \
#   bash examples/train/distill_client_122b.sh
#   # -> data/client/client_122b_distill_<N>.jsonl  (feed to STEP 2)
#
#   # multimodal (needs the image root visible + per-prompt image cap):
#   MODE=multimodal IMAGE_MEDIA_ROOT=/mnt/tidal-alsh01 LIMIT_IMAGES=20 \
#     CLIENT_MODEL=... CLIENT_TRAIN_JSONL=... bash examples/train/distill_client_122b.sh
#
# Two unconnected machines? split with SKIP_SAMPLES + per-machine OUT_JSONL, cat after.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- client knobs ----
CLIENT_MODEL="${CLIENT_MODEL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B}"
CLIENT_TRAIN_JSONL="${CLIENT_TRAIN_JSONL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/train.jsonl}"
MODE="${MODE:-text}"                       # text | multimodal
MAX_SAMPLES="${MAX_SAMPLES:-8137}"
SKIP_SAMPLES="${SKIP_SAMPLES:-0}"

# SAFETY: this is the client's machine. We NEVER touch the original train.jsonl.
# Our files land in the SAME folder as train.jsonl (the /huawei dir, per request);
# we operate on a READ-ONLY COPY of the source, never the original. The copy and
# the outputs have distinct names, so they never collide with train.jsonl.
HUAWEI_ROOT="${HUAWEI_ROOT:-$(cd "$(dirname "$CLIENT_TRAIN_JSONL")" && pwd)}"
WORK_DIR="${WORK_DIR:-$HUAWEI_ROOT}"
SOURCE_COPY="${SOURCE_COPY:-$WORK_DIR/train_source_copy.jsonl}"
OUT_JSONL="${OUT_JSONL:-$WORK_DIR/client_122b_distill_${MODE}_${MAX_SAMPLES}.jsonl}"

[ -d "$CLIENT_MODEL" ] || { echo "[fatal] CLIENT_MODEL not found: $CLIENT_MODEL"; exit 1; }
[ -s "$CLIENT_TRAIN_JSONL" ] || { echo "[fatal] CLIENT_TRAIN_JSONL not found: $CLIENT_TRAIN_JSONL"; exit 1; }
mkdir -p "$WORK_DIR" "$(dirname "$OUT_JSONL")"

# --- self-detach: survive SSH drops (the box loses connection ~every 5 min).
#     Re-exec under nohup so serve+distill keep running; DETACH=0 stays foreground.
NOHUP_LOG="${NOHUP_LOG:-$WORK_DIR/distill_client_122b_${MODE}_${STAMP}.nohup.log}"
if [ "${DETACH:-1}" = "1" ] && [ -z "${_DETACHED:-}" ]; then
    echo "Detaching distillation (survives disconnect). Follow with:"
    echo "  tail -f $NOHUP_LOG"
    _DETACHED=1 nohup bash "$0" "$@" > "$NOHUP_LOG" 2>&1 &
    echo "  PID $!   (stop: kill $!)"
    echo "$!" > "${NOHUP_LOG%.log}.pid"
    exit 0
fi

# Guard: refuse if any output path resolves to the original source.
SRC_RP="$(readlink -f "$CLIENT_TRAIN_JSONL")"
for p in "$SOURCE_COPY" "$OUT_JSONL"; do
    mkdir -p "$(dirname "$p")"
    if [ "$(readlink -f "$(dirname "$p")")/$(basename "$p")" = "$SRC_RP" ]; then
        echo "[fatal] output path '$p' would hit the original source $SRC_RP — refusing"; exit 1
    fi
done

# Make/refresh the read-only working copy (original stays untouched). cp -n keeps an
# existing copy (rerun-safe); we verify it matches the source by size before reuse.
if [ ! -f "$SOURCE_COPY" ] || [ "$(stat -c%s "$SOURCE_COPY" 2>/dev/null)" != "$(stat -c%s "$CLIENT_TRAIN_JSONL" 2>/dev/null)" ]; then
    echo "Copying source -> read-only working copy (original NOT modified):"
    echo "  src:  $CLIENT_TRAIN_JSONL"
    echo "  copy: $SOURCE_COPY"
    cp -f "$CLIENT_TRAIN_JSONL" "$SOURCE_COPY"
    chmod 0444 "$SOURCE_COPY"   # read-only copy; never the original
fi
# From here on we read ONLY the copy.
CLIENT_TRAIN_JSONL="$SOURCE_COPY"

# ---- serving knobs (8x H800, full-precision 122B, LONG context) ----
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
PORT="${PORT:-8100}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-vl-122b-sft}"
# client prompts are long (system ~5k + retrieved notes up to ~56k chars) -> big ctx
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
IMAGE_MEDIA_ROOT="${IMAGE_MEDIA_ROOT:-/mnt/tidal-alsh01}"   # images live here (abs paths)
LIMIT_IMAGES="${LIMIT_IMAGES:-20}"
START_SERVER="${START_SERVER:-1}"
ENDPOINT="${ENDPOINT:-http://localhost:${PORT}/v1}"

# ---- distill knobs ----
MAX_TOKENS="${MAX_TOKENS:-2048}"           # answers ~1k chars; headroom for the long tail
TEMPERATURE="${TEMPERATURE:-0}"            # greedy: learn the verifier's argmax continuations
CONCURRENCY="${CONCURRENCY:-$MAX_NUM_SEQS}"
RESUME="${RESUME:-1}"

LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/distill_client_122b_${STAMP}_vllm.log}"
mkdir -p "$LOG_DIR"

if [ "$MODE" = "multimodal" ]; then MODE_FLAG="--multimodal"; else MODE_FLAG="--text-only"; fi

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }; }
trap cleanup EXIT

wait_for_server() {
    echo "Waiting for verifier on :$PORT (log: $SERVER_LOG)"
    for _ in $(seq 1 240); do
        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server died during startup. Last 100 lines:"; tail -n 100 "$SERVER_LOG" || true; return 1
        fi
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo "Server ready."; return 0; }
        sleep 5
    done
    echo "ERROR: timed out. Last 100 lines:"; tail -n 100 "$SERVER_LOG" || true; return 1
}

if [ "$START_SERVER" = "1" ]; then
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo "[fatal] port $PORT busy (START_SERVER=0 to reuse)"; exit 1; }
    args=(
        vllm serve "$CLIENT_MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --seed 42
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEMORY_UTIL"
        --trust-remote-code
        --dtype "$DTYPE"
        --generation-config vllm
        --host 0.0.0.0 --port "$PORT"
    )
    [ "$ENFORCE_EAGER" = "1" ] && args+=(--enforce-eager)
    if [ "$MODE" = "multimodal" ]; then
        args+=(--allowed-local-media-path "$IMAGE_MEDIA_ROOT" --limit-mm-per-prompt "{\"image\":$LIMIT_IMAGES}")
    fi
    echo "=== Serving post-SFT 122B for client distillation ($MODE) ==="
    echo "  model: $CLIENT_MODEL"
    echo "  serve: TP=$TP GPUs[$GPUS] max_len=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS"
    [ "$MODE" = multimodal ] && echo "  media: $IMAGE_MEDIA_ROOT  limit_images=$LIMIT_IMAGES"
    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_server
else
    echo "START_SERVER=0 -> reusing $ENDPOINT"
fi

DISTILL=(
    python3 scripts/distill_client_messages.py
    --endpoint "$ENDPOINT"
    --in "$CLIENT_TRAIN_JSONL"
    --out-jsonl "$OUT_JSONL"
    --max-samples "$MAX_SAMPLES"
    --skip-samples "$SKIP_SAMPLES"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --concurrency "$CONCURRENCY"
    "$MODE_FLAG"
)
[ "$RESUME" = "1" ] && DISTILL+=(--resume)

echo "=== Distilling client prompts ($MODE) ==="
printf '[cmd]'; printf ' %q' "${DISTILL[@]}"; echo
"${DISTILL[@]}"

echo
echo "Distilled data ready: $OUT_JSONL"
echo "Next (STEP 2):"
echo "  CLIENT_MODEL=$CLIENT_MODEL CLIENT_DISTILL_JSONL=$OUT_JSONL \\"
echo "    bash examples/train/nohup_mtp_client_122b.sh"
