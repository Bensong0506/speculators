#!/bin/bash
# Four-way benchmark on the ALLaVA validation tail.
#
# This mirrors the training split used by ArrowDataset:
#   train = first 90% of the preprocessed/order-preserving jsonl
#   val   = last 10%
#
# Usage:
#   DRAFT=/path/to/trained/checkpoint \
#   ALLAVA_JSONL=/home/wenxuan/speculators/data/allava/allava_10000.jsonl \
#   INFER_NUM_SPEC=7 \
#   bash examples/evaluate/test_dflash_allava_val_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_ROOT="${DEFAULT_ROOT:-/home/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /data/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/data/wenxuan"
fi

DEFAULT_ALLAVA_ROOT="${ALLAVA_ROOT:-$DEFAULT_ROOT/ALLaVA-4V}"
if [ ! -d "$DEFAULT_ALLAVA_ROOT" ] && [ -d /data/wenxuan/ALLaVA-4V ]; then
    DEFAULT_ALLAVA_ROOT="/data/wenxuan/ALLaVA-4V"
fi

MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"
BASELINE_DRAFT="${BASELINE_DRAFT:-$DEFAULT_ROOT/Qwen3.5-9B-DFlash}"
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-$DEFAULT_ALLAVA_ROOT}"
ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"
ALLAVA_MAX_SAMPLES="${ALLAVA_MAX_SAMPLES:-10000}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_${ALLAVA_MAX_SAMPLES}.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_TAG="${VAL_TAG:-tail10pct}"
ALLAVA_VAL_JSONL="${ALLAVA_VAL_JSONL:-$REPO_ROOT/data/allava/allava_${ALLAVA_MAX_SAMPLES}_val_${VAL_TAG}.jsonl}"
BUILD_ALLAVA_JSONL="${BUILD_ALLAVA_JSONL:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/allava_val_weight_tests}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
mkdir -p "$RUN_DIR"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
MTP_METHOD="${MTP_METHOD:-qwen3_5_mtp}"
MTP_SPEC="${MTP_SPEC:-$INFER_NUM_SPEC}"
ORIGINAL_DFLASH_SPEC="${ORIGINAL_DFLASH_SPEC:-${BASELINE_SPEC:-$INFER_NUM_SPEC}}"
TRAINED_DFLASH_SPEC="${TRAINED_DFLASH_SPEC:-${DFLASH_SPEC:-$INFER_NUM_SPEC}}"
MAX_CONFIG_SPEC="$(python3 - "$MTP_SPEC" "$ORIGINAL_DFLASH_SPEC" "$TRAINED_DFLASH_SPEC" <<'PY'
import sys

print(max(int(x) for x in sys.argv[1:]))
PY
)"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_CONFIG_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-allava-val-weight-test}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

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

if [ -z "${DRAFT:-}" ]; then
    DRAFT="$(python3 - "$REPO_ROOT/output/mmstar_trained_dflash_best_vs_baselines" <<'PY' || true
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = []
if root.exists():
    for path in root.rglob("best_checkpoint.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        checkpoint = obj.get("checkpoint")
        if checkpoint and Path(checkpoint).exists():
            items.append((path.stat().st_mtime, checkpoint))
if items:
    print(max(items)[1])
PY
)"
fi

if [ -z "${DRAFT:-}" ]; then
    echo "[fatal] DRAFT is not set and no prior best_checkpoint.json was found."
    echo "        Set DRAFT=/path/to/trained/checkpoint."
    exit 1
fi

[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$BASELINE_DRAFT" ] || { echo "[fatal] BASELINE_DRAFT not found: $BASELINE_DRAFT"; exit 1; }
[ -d "$DRAFT" ] || { echo "[fatal] DRAFT not found: $DRAFT"; exit 1; }
[ -d "$ALLAVA_IMAGE_ROOT" ] || { echo "[fatal] ALLAVA_IMAGE_ROOT not found: $ALLAVA_IMAGE_ROOT"; exit 1; }

build_allava_jsonl() {
    local in_args=()
    local src
    for src in $ALLAVA_INPUTS; do
        in_args+=(--in "$src")
    done
    python3 scripts/llava_to_jsonl.py \
        "${in_args[@]}" \
        --image-root "$ALLAVA_IMAGE_ROOT" \
        --out-jsonl "$ALLAVA_JSONL" \
        --max-samples "$ALLAVA_MAX_SAMPLES"
}

prepare_val_jsonl() {
    python3 - "$ALLAVA_JSONL" "$ALLAVA_VAL_JSONL" "$VAL_RATIO" <<'PY'
import math
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ratio = float(sys.argv[3])
if not 0 < ratio < 1:
    raise SystemExit(f"[fatal] VAL_RATIO must be in (0, 1), got {ratio}")
if not src.exists():
    raise SystemExit(f"[fatal] ALLAVA_JSONL not found: {src}")

lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"[fatal] ALLAVA_JSONL is empty: {src}")

split_idx = int(len(lines) * (1.0 - ratio))
val_lines = lines[split_idx:]
if not val_lines:
    raise SystemExit(
        f"[fatal] validation split is empty: rows={len(lines)} ratio={ratio}"
    )

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
print(
    f"[info] wrote ALLaVA val jsonl: {dst} "
    f"rows={len(val_lines)} source_rows={len(lines)} start_index={split_idx}"
)
PY
}

if [ ! -s "$ALLAVA_JSONL" ]; then
    if [ "$BUILD_ALLAVA_JSONL" = "1" ]; then
        echo "=== Building ALLaVA conversations jsonl ==="
        build_allava_jsonl
    else
        echo "[fatal] ALLAVA_JSONL missing: $ALLAVA_JSONL"
        echo "        Point ALLAVA_JSONL at your distilled training jsonl, or set BUILD_ALLAVA_JSONL=1."
        exit 1
    fi
fi

echo "=== Preparing ALLaVA validation tail ==="
prepare_val_jsonl

wait_for_server() {
    local log="$1"
    local mode="$2"
    echo "Waiting for $mode server on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*(mtp|dflash)|unsupported.*(mtp|dflash)|unrecognized.*(mtp|dflash)" "$log"; then
                echo "DIAGNOSIS: this vLLM build may not support the requested speculative method."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: vLLM speculative scheduling budget is too small."
                echo "           Lower MAX_NUM_SEQS/spec, or raise MAX_NUM_BATCHED_TOKENS."
            fi
            return 1
        fi
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "$mode server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for $mode server. Last 100 log lines:"
    tail -n 100 "$log" || true
    return 1
}

start_server() {
    local mode="$1"
    local log="$2"
    local spec_config="${3:-}"

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting $mode."
        echo "       Stop the old vLLM process or choose a different PORT."
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
        --allowed-local-media-path "$ALLAVA_IMAGE_ROOT"
        --limit-mm-per-prompt '{"image":1}'
        --generation-config vllm
        --host 0.0.0.0
        --port "$PORT"
    )

    if [ "$ENFORCE_EAGER" = "1" ]; then
        args+=(--enforce-eager)
    fi
    if [ -n "$DTYPE" ]; then
        args+=(--dtype "$DTYPE")
    fi
    if [ -n "$ATTENTION_BACKEND" ]; then
        args+=(--attention-backend "$ATTENTION_BACKEND")
    fi
    if [ "$DISABLE_CHUNKED_PREFILL" = "1" ]; then
        args+=(--no-enable-chunked-prefill)
    fi
    if [ -n "$spec_config" ]; then
        args+=(--speculative-config "$spec_config")
    fi

    echo
    echo "=== Starting $mode server ==="
    echo "  model:             $MODEL"
    echo "  spec_config:       ${spec_config:-none}"
    echo "  max_model_len:     $MAX_MODEL_LEN"
    echo "  max batched toks:  $MAX_NUM_BATCHED_TOKENS"
    echo "  max seqs:          $MAX_NUM_SEQS"
    echo "  port:              $PORT"
    echo "  devices:           $GPUS"
    echo "  media_root:        $ALLAVA_IMAGE_ROOT"
    echo "  log:               $log"
    printf '[cmd]'
    printf ' %q' env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}"
    echo

    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$log" "$mode"
}

run_client() {
    local mode="$1"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${PORT}/v1" \
        --data-jsonl "$ALLAVA_VAL_JSONL" \
        --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
        --summary-json "$RUN_DIR/${mode}_summary.json" \
        --num "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS"
}

parse_acceptance() {
    local mode="$1"
    local log="$2"
    local out="$RUN_DIR/${mode}_acceptance_from_log.txt"

    if ! python3 examples/evaluate/eval-guidellm/scripts/parse_logs.py "$log" \
        > "$out" 2>&1; then
        echo "(No parseable 'SpecDecoding metrics:' lines found for $mode.)" >> "$out"
        grep -iE "spec|accept|draft|mtp|dflash" "$log" | tail -n 60 >> "$out" || true
    fi
}

run_one() {
    local mode="$1"
    local spec_config="${2:-}"
    local log="$RUN_DIR/${mode}_vllm.log"

    start_server "$mode" "$log" "$spec_config"
    run_client "$mode"
    cleanup_server
    sleep 5
    parse_acceptance "$mode" "$log"
}

echo "=== ALLaVA val four-way benchmark ==="
echo "  model:               $MODEL"
echo "  original_dflash:     $BASELINE_DRAFT"
echo "  trained_dflash:      $DRAFT"
echo "  mtp_method:          $MTP_METHOD"
echo "  mtp_spec:            $MTP_SPEC"
echo "  original spec:       $ORIGINAL_DFLASH_SPEC"
echo "  trained spec:        $TRAINED_DFLASH_SPEC"
echo "  allava_jsonl:        $ALLAVA_JSONL"
echo "  allava_val_jsonl:    $ALLAVA_VAL_JSONL"
echo "  media_root:          $ALLAVA_IMAGE_ROOT"
echo "  num_prompts:         $NUM_PROMPTS"
echo "  output:              $RUN_DIR"

BASELINE_SPEC_CONFIG=""
MTP_SPEC_CONFIG="{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$MTP_SPEC,\"enforce_eager\":true}"
ORIGINAL_DFLASH_SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$BASELINE_DRAFT\",\"num_speculative_tokens\":$ORIGINAL_DFLASH_SPEC}"
TRAINED_DFLASH_SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$TRAINED_DFLASH_SPEC}"

run_one baseline "$BASELINE_SPEC_CONFIG"
run_one mtp "$MTP_SPEC_CONFIG"
run_one dflash_original "$ORIGINAL_DFLASH_SPEC_CONFIG"
run_one trained_dflash "$TRAINED_DFLASH_SPEC_CONFIG"

python3 - \
    "$RUN_DIR" \
    "$RUN_DIR/allava_val_four_way_summary.jsonl" \
    "$RUN_DIR/allava_val_four_way_summary.csv" \
    "$RUN_DIR/allava_val_summary.md" <<'PY' | tee "$RUN_DIR/allava_val_summary.stdout.txt"
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_jsonl = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
out_md = Path(sys.argv[4])

methods = ["baseline", "mtp", "dflash_original", "trained_dflash"]
rows = []
for method in methods:
    summary_path = run_dir / f"{method}_summary.json"
    summary = json.loads(summary_path.read_text())
    rows.append(
        {
            "method": method,
            "tok_s": summary.get("output_tok_per_sec"),
            "mean_accept_per_draft": summary.get("spec_mean_accepted_tokens_per_draft"),
            "token_accept": summary.get("spec_token_acceptance_rate"),
            "first_pos_accept": summary.get("spec_first_position_acceptance_rate"),
            "draft_tokens": summary.get("spec_draft_tokens_total"),
            "accepted_tokens": summary.get("spec_accepted_tokens_total"),
            "completed": summary.get("completed"),
            "requested": summary.get("num_requested"),
            "summary_path": str(summary_path),
        }
    )

with out_jsonl.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

fields = [
    "method",
    "tok_s",
    "mean_accept_per_draft",
    "token_accept",
    "first_pos_accept",
    "draft_tokens",
    "accepted_tokens",
    "completed",
    "requested",
    "summary_path",
]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def ratio(a, b):
    return a / b if a is not None and b else None

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

ranked = sorted(
    [row for row in rows if row["tok_s"] is not None],
    key=lambda row: row["tok_s"],
    reverse=True,
)
by_method = {row["method"]: row for row in rows}
trained = by_method["trained_dflash"]
original = by_method["dflash_original"]
mtp = by_method["mtp"]
baseline = by_method["baseline"]

md = [
    "# ALLaVA Val Four-Way Benchmark",
    "",
    "| rank | method | tok/s | mean accept/draft | token accept | first-pos accept | completed |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for idx, row in enumerate(ranked, 1):
    md.append(
        "| {idx} | {method} | {tok_s} | {mean_accept} | {token_accept} | "
        "{first_accept} | {completed}/{requested} |".format(
            idx=idx,
            method=row["method"],
            tok_s=fmt(row.get("tok_s")),
            mean_accept=fmt(row.get("mean_accept_per_draft")),
            token_accept=fmt(row.get("token_accept")),
            first_accept=fmt(row.get("first_pos_accept")),
            completed=fmt(row.get("completed")),
            requested=fmt(row.get("requested")),
        )
    )

md.extend(
    [
        "",
        "## Key Ratios",
        "",
        f"- trained/original DFlash tok/s: `{fmt(ratio(trained.get('tok_s'), original.get('tok_s')))}`",
        f"- trained/MTP tok/s: `{fmt(ratio(trained.get('tok_s'), mtp.get('tok_s')))}`",
        f"- trained/baseline tok/s: `{fmt(ratio(trained.get('tok_s'), baseline.get('tok_s')))}`",
        f"- original DFlash/baseline tok/s: `{fmt(ratio(original.get('tok_s'), baseline.get('tok_s')))}`",
        f"- MTP/baseline tok/s: `{fmt(ratio(mtp.get('tok_s'), baseline.get('tok_s')))}`",
        "",
        "## Verdict",
        "",
    ]
)
if ranked and ranked[0]["method"] == "trained_dflash":
    md.append("Trained DFlash is the fastest config on this ALLaVA val slice.")
elif ratio(trained.get("mean_accept_per_draft"), original.get("mean_accept_per_draft")) and ratio(trained.get("mean_accept_per_draft"), original.get("mean_accept_per_draft")) > 1:
    md.append("Trained DFlash improves acceptance over original DFlash but is not the fastest config.")
else:
    md.append("Trained DFlash does not beat original DFlash on acceptance or tok/s here.")

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print()
print(f"summary_jsonl={out_jsonl}")
print(f"summary_csv={out_csv}")
print(f"summary_md={out_md}")
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
