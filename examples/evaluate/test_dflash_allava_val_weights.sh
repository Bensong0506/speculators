#!/bin/bash
# Benchmark original vs trained DFlash on the ALLaVA validation tail.
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
BASELINE_PORT="${BASELINE_PORT:-8100}"
DFLASH_PORT="${DFLASH_PORT:-8101}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
BASELINE_SPEC="${BASELINE_SPEC:-$INFER_NUM_SPEC}"
DFLASH_SPEC="${DFLASH_SPEC:-$INFER_NUM_SPEC}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-allava-val-weight-test}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

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

echo "=== ALLaVA val DFlash benchmark ==="
echo "  model:             $MODEL"
echo "  original_dflash:   $BASELINE_DRAFT"
echo "  trained_dflash:    $DRAFT"
echo "  allava_jsonl:      $ALLAVA_JSONL"
echo "  allava_val_jsonl:  $ALLAVA_VAL_JSONL"
echo "  media_root:        $ALLAVA_IMAGE_ROOT"
echo "  infer_num_spec:    $INFER_NUM_SPEC"
echo "  num_prompts:       $NUM_PROMPTS"
echo "  output:            $RUN_DIR"

env \
    MODEL="$MODEL" \
    BASELINE_DRAFT="$BASELINE_DRAFT" \
    DRAFT="$DRAFT" \
    MMSTAR_SRC="$ALLAVA_JSONL" \
    MMSTAR_JSONL="$ALLAVA_VAL_JSONL" \
    MMSTAR_IMAGE_DIR="$REPO_ROOT/data/allava/images" \
    MEDIA_ROOT="$ALLAVA_IMAGE_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_DIR="$RUN_DIR" \
    GPUS="$GPUS" \
    TP="$TP" \
    BASELINE_PORT="$BASELINE_PORT" \
    DFLASH_PORT="$DFLASH_PORT" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    MAX_TOKENS="$MAX_TOKENS" \
    INFER_NUM_SPEC="$INFER_NUM_SPEC" \
    BASELINE_SPEC="$BASELINE_SPEC" \
    DFLASH_SPEC="$DFLASH_SPEC" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    ENFORCE_EAGER="$ENFORCE_EAGER" \
    ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    bash examples/evaluate/test_dflash_mmstar_weights.sh \
    | tee "$RUN_DIR/allava_val_stdout.log"

python3 - "$RUN_DIR/baseline_summary.json" "$RUN_DIR/dflash_summary.json" "$RUN_DIR/allava_val_summary.md" <<'PY' | tee "$RUN_DIR/allava_val_summary.stdout.txt"
import json
import sys
from pathlib import Path

native = json.loads(Path(sys.argv[1]).read_text())
trained = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])

def ratio(a, b):
    return a / b if a is not None and b else None

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

rows = [
    ("original_dflash", native),
    ("trained_dflash", trained),
]
native_tps = native.get("output_tok_per_sec")
trained_tps = trained.get("output_tok_per_sec")
native_mean = native.get("spec_mean_accepted_tokens_per_draft")
trained_mean = trained.get("spec_mean_accepted_tokens_per_draft")
native_token = native.get("spec_token_acceptance_rate")
trained_token = trained.get("spec_token_acceptance_rate")
native_first = native.get("spec_first_position_acceptance_rate")
trained_first = trained.get("spec_first_position_acceptance_rate")

md = [
    "# ALLaVA Val Original vs Trained DFlash",
    "",
    "| method | tok/s | mean accept/draft | token accept | first-pos accept | completed |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, row in rows:
    md.append(
        "| {name} | {tok_s} | {mean_accept} | {token_accept} | {first_accept} | "
        "{completed}/{requested} |".format(
            name=name,
            tok_s=fmt(row.get("output_tok_per_sec")),
            mean_accept=fmt(row.get("spec_mean_accepted_tokens_per_draft")),
            token_accept=fmt(row.get("spec_token_acceptance_rate")),
            first_accept=fmt(row.get("spec_first_position_acceptance_rate")),
            completed=fmt(row.get("completed")),
            requested=fmt(row.get("num_requested")),
        )
    )

md.extend(
    [
        "",
        "## Ratios",
        "",
        f"- trained/original tok/s: `{fmt(ratio(trained_tps, native_tps))}`",
        f"- trained/original mean accept: `{fmt(ratio(trained_mean, native_mean))}`",
        f"- trained/original token accept: `{fmt(ratio(trained_token, native_token))}`",
        f"- trained/original first-pos accept: `{fmt(ratio(trained_first, native_first))}`",
        "",
        "## Verdict",
        "",
    ]
)
if ratio(trained_tps, native_tps) and ratio(trained_tps, native_tps) > 1:
    md.append("Trained DFlash is faster than original DFlash on this ALLaVA val slice.")
elif ratio(trained_mean, native_mean) and ratio(trained_mean, native_mean) > 1:
    md.append("Trained DFlash improves acceptance but not end-to-end tok/s on this ALLaVA val slice.")
else:
    md.append("Trained DFlash does not improve acceptance or tok/s on this ALLaVA val slice.")

out.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
