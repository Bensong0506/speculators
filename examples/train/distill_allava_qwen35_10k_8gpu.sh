#!/bin/bash
# 8-GPU ALLaVA distillation launcher.
#
# It starts one vLLM verifier server per GPU, shards the first MAX_SAMPLES
# ALLaVA prompts across those servers, then merges shard jsonl files into the
# same training-ready file used by the single-GPU launcher.
#
# Usage:
#   bash examples/train/distill_allava_qwen35_10k_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/home/wenxuan/Qwen3.5-9B}"
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"

MAX_SAMPLES="${MAX_SAMPLES:-10000}"
FINAL_JSONL="${FINAL_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"
SHARD_ROOT="${SHARD_ROOT:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k_shards}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARRAY[@]}}"
BASE_PORT="${BASE_PORT:-8100}"

MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
RESUME="${RESUME:-1}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
DTYPE="${DTYPE:-bfloat16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

if [ "$NUM_SHARDS" -le 0 ]; then
    echo "[fatal] NUM_SHARDS must be positive"
    exit 1
fi
if [ "${#GPU_ARRAY[@]}" -lt "$NUM_SHARDS" ]; then
    echo "[fatal] GPU_LIST has ${#GPU_ARRAY[@]} entries but NUM_SHARDS=$NUM_SHARDS"
    exit 1
fi

mkdir -p "$SHARD_ROOT" "$(dirname "$FINAL_JSONL")"

echo "=== 8-GPU ALLaVA Qwen distillation ==="
echo "  model:        $MODEL"
echo "  image_root:   $ALLAVA_IMAGE_ROOT"
echo "  final_jsonl:  $FINAL_JSONL"
echo "  shard_root:   $SHARD_ROOT"
echo "  max_samples:  $MAX_SAMPLES"
echo "  num_shards:   $NUM_SHARDS"
echo "  gpu_list:     $GPU_LIST"
echo "  base_port:    $BASE_PORT"

pids=()
driver_logs=()

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_ARRAY[$shard]}"
    port=$((BASE_PORT + shard))
    shard_jsonl="$SHARD_ROOT/shard_${shard}_of_${NUM_SHARDS}.jsonl"
    driver_log="$SHARD_ROOT/shard_${shard}_driver.log"
    server_log="$SHARD_ROOT/shard_${shard}_vllm.log"
    driver_logs+=("$driver_log")

    echo
    echo "Launching shard $shard/$NUM_SHARDS on GPU $gpu port $port"
    (
        MODEL="$MODEL" \
        ALLAVA_IMAGE_ROOT="$ALLAVA_IMAGE_ROOT" \
        ALLAVA_INPUTS="$ALLAVA_INPUTS" \
        OUT_JSONL="$shard_jsonl" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        TOTAL_SAMPLES="$MAX_SAMPLES" \
        NUM_SHARDS="$NUM_SHARDS" \
        SHARD_INDEX="$shard" \
        GPUS="$gpu" \
        TP="$TP" \
        PORT="$port" \
        SERVED_MODEL_NAME="qwen3.5-9b-allava-distill-shard-${shard}" \
        MAX_TOKENS="$MAX_TOKENS" \
        TEMPERATURE="$TEMPERATURE" \
        TOP_P="$TOP_P" \
        REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
        RESUME="$RESUME" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
        ATTENTION_BACKEND="$ATTENTION_BACKEND" \
        DTYPE="$DTYPE" \
        ENFORCE_EAGER="$ENFORCE_EAGER" \
        SERVER_LOG="$server_log" \
        bash examples/train/distill_allava_qwen35_10k.sh
    ) > "$driver_log" 2>&1 &
    pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    log="${driver_logs[$idx]}"
    if wait "$pid"; then
        echo "Shard $idx finished OK (log: $log)"
    else
        echo "Shard $idx FAILED (log: $log)"
        tail -n 80 "$log" || true
        status=1
    fi
done

if [ "$status" != "0" ]; then
    echo "[fatal] one or more distillation shards failed; not merging."
    exit "$status"
fi

echo
echo "=== Merging shard jsonl files ==="
python3 - "$FINAL_JSONL" "$SHARD_ROOT" "$NUM_SHARDS" "$MAX_SAMPLES" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
num_shards = int(sys.argv[3])
expected = int(sys.argv[4])

shards = []
for idx in range(num_shards):
    path = root / f"shard_{idx}_of_{num_shards}.jsonl"
    if not path.exists():
        raise SystemExit(f"[fatal] missing shard file: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shards.append(lines)
    print(f"  shard {idx}: {len(lines)} rows")

written = 0
with out.open("w", encoding="utf-8") as handle:
    max_len = max((len(lines) for lines in shards), default=0)
    for row_idx in range(max_len):
        for lines in shards:
            if row_idx >= len(lines):
                continue
            handle.write(lines[row_idx] + "\n")
            written += 1
            if written >= expected:
                break
        if written >= expected:
            break

print(f"merged rows: {written} -> {out}")
if written < expected:
    print(f"WARNING: expected {expected} rows but merged only {written}")
PY

echo
echo "Distilled data ready:"
echo "  $FINAL_JSONL"
echo
echo "Use it for training with:"
echo "  DISTILLED_ALLAVA_JSONL=$FINAL_JSONL bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh"
