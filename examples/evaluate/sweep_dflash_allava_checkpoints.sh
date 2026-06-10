#!/bin/bash
# Sweep DFlash checkpoints on the ALLaVA validation tail (in-domain selection).
#
# This is the ALLaVA analog of sweep_dflash_mmstar_checkpoints.sh. It answers the
# project's actual question: "which continued-training checkpoint beats the
# original/native DFlash draft IN-DOMAIN?" -- measured by REAL speculative
# acceptance on the same val tail the trainer used (last VAL_RATIO of the
# distilled training jsonl).
#
# It runs the native DFlash draft ONCE as the baseline, then every discovered
# checkpoint as the dflash draft against the identical val prompts, and writes a
# ranked table sorted by mean accepted tokens / draft. The per-checkpoint config
# is byte-identical to test_dflash_allava_val_weights.sh's trained_dflash config,
# so the best row here == that script's trained_dflash number.
#
# Usage:
#   CHECKPOINT_FIND_ROOT=output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/<CE_RUN>/checkpoints \
#   ALLAVA_JSONL="$(pwd)/data/allava/allava_qwen35_distill_10k.jsonl" \
#   INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/sweep_dflash_allava_checkpoints.sh
#
# Notes:
#   - Point CHECKPOINT_FIND_ROOT at ONE run's checkpoints dir (not all of
#     output/), so the old kl_div checkpoints are not swept = wasted GPU.
#   - ALLAVA_JSONL must be the SAME jsonl used for training, so the last
#     VAL_RATIO tail matches the trainer's val split (same prompts).

set -uo pipefail

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

ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_TAG="${VAL_TAG:-tail10pct}"
ALLAVA_VAL_JSONL="${ALLAVA_VAL_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k_val_${VAL_TAG}.jsonl}"

CHECKPOINT_FIND_ROOT="${CHECKPOINT_FIND_ROOT:-$REPO_ROOT/output}"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/output/allava_checkpoint_sweeps/$STAMP}"
RESULTS_JSONL="$SWEEP_ROOT/results.jsonl"
RESULTS_CSV="$SWEEP_ROOT/results.csv"
RESULTS_MD="$SWEEP_ROOT/results.md"

INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * INFER_NUM_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-allava-sweep}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
INCLUDE_BEST="${INCLUDE_BEST:-0}"
DEDUP_REALPATH="${DEDUP_REALPATH:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

mkdir -p "$SWEEP_ROOT"

[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$BASELINE_DRAFT" ] || { echo "[fatal] BASELINE_DRAFT not found: $BASELINE_DRAFT"; exit 1; }
[ -d "$ALLAVA_IMAGE_ROOT" ] || { echo "[fatal] ALLAVA_IMAGE_ROOT not found: $ALLAVA_IMAGE_ROOT"; exit 1; }
[ -s "$ALLAVA_JSONL" ] || { echo "[fatal] ALLAVA_JSONL not found/empty: $ALLAVA_JSONL"; exit 1; }

SERVER_PID=""
cleanup_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

prepare_val_jsonl() {
    python3 - "$ALLAVA_JSONL" "$ALLAVA_VAL_JSONL" "$VAL_RATIO" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ratio = float(sys.argv[3])
if not 0 < ratio < 1:
    raise SystemExit(f"[fatal] VAL_RATIO must be in (0, 1), got {ratio}")
lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"[fatal] ALLAVA_JSONL is empty: {src}")
split_idx = int(len(lines) * (1.0 - ratio))
val_lines = lines[split_idx:]
if not val_lines:
    raise SystemExit(f"[fatal] validation split is empty: rows={len(lines)} ratio={ratio}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
print(f"[info] val jsonl: {dst} rows={len(val_lines)} source_rows={len(lines)} start_index={split_idx}")
PY
}

discover_checkpoints() {
    python3 - "$CHECKPOINT_FIND_ROOT" "$INCLUDE_BEST" "$DEDUP_REALPATH" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
include_best = sys.argv[2] == "1"
dedup_realpath = sys.argv[3] == "1"
if not root.exists():
    raise SystemExit(f"[fatal] CHECKPOINT_FIND_ROOT does not exist: {root}")

items = []
for path in root.rglob("*"):
    if not path.is_dir():
        continue
    name = path.name
    if name == "checkpoint_best" and not include_best:
        continue
    step_match = re.match(r"checkpoint[_-]?(\d+)$", name) or re.match(r"(\d+)$", name)
    if name != "checkpoint_best" and not step_match:
        continue
    if not (path / "config.json").exists() or not (path / "model.safetensors").exists():
        continue
    try:
        cfg = json.loads((path / "config.json").read_text())
    except Exception:
        continue
    if cfg.get("speculators_model_type") != "dflash":
        continue
    step = int(step_match.group(1)) if step_match else 10**18
    key = str(path.resolve()) if dedup_realpath else str(path)
    items.append(
        {
            "key": key,
            "parent": str(path.parent.parent),
            "step": step,
            "is_best": name == "checkpoint_best",
            "path": str(path),
        }
    )

if dedup_realpath:
    deduped = {}
    for item in items:
        current = deduped.get(item["key"])
        if current is None or (current["is_best"] and not item["is_best"]):
            deduped[item["key"]] = item
    items = list(deduped.values())

for item in sorted(items, key=lambda x: (x["parent"], x["step"], x["is_best"], x["path"])):
    print(item["path"])
PY
}

wait_for_server() {
    local log="$1"
    local mode="$2"
    echo "Waiting for $mode on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode server died during startup. Last 60 log lines:"
            tail -n 60 "$log" || true
            return 1
        fi
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "$mode server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for $mode. Last 60 log lines:"
    tail -n 60 "$log" || true
    return 1
}

start_server() {
    local mode="$1"
    local log="$2"
    local spec_config="${3:-}"

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before $mode."
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
    [ "$ENFORCE_EAGER" = "1" ] && args+=(--enforce-eager)
    [ -n "$DTYPE" ] && args+=(--dtype "$DTYPE")
    [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
    [ "$DISABLE_CHUNKED_PREFILL" = "1" ] && args+=(--no-enable-chunked-prefill)
    [ -n "$spec_config" ] && args+=(--speculative-config "$spec_config")

    echo "=== start $mode (spec_config=${spec_config:-none}) ==="
    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$log" "$mode"
}

run_client() {
    local tag="$1"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${PORT}/v1" \
        --data-jsonl "$ALLAVA_VAL_JSONL" \
        --out-jsonl "$SWEEP_ROOT/${tag}_responses.jsonl" \
        --summary-json "$SWEEP_ROOT/${tag}_summary.json" \
        --num "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS"
}

append_row() {
    local tag="$1"
    local label="$2"
    local draft="$3"
    local status="$4"
    python3 - "$tag" "$label" "$draft" "$status" "$SWEEP_ROOT/${tag}_summary.json" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

tag, label, draft, status = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
summ_path = Path(sys.argv[5])
out = Path(sys.argv[6])
row = {
    "tag": tag, "label": label, "draft": draft, "status": status,
    "tok_s": None, "mean_accept_per_draft": None, "token_accept": None,
    "first_pos_accept": None, "draft_tokens": None, "accepted_tokens": None,
    "completed": None, "requested": None,
}
if summ_path.exists():
    try:
        s = json.loads(summ_path.read_text())
        row.update({
            "tok_s": s.get("output_tok_per_sec"),
            "mean_accept_per_draft": s.get("spec_mean_accepted_tokens_per_draft"),
            "token_accept": s.get("spec_token_acceptance_rate"),
            "first_pos_accept": s.get("spec_first_position_acceptance_rate"),
            "draft_tokens": s.get("spec_draft_tokens_total"),
            "accepted_tokens": s.get("spec_accepted_tokens_total"),
            "completed": s.get("completed"),
            "requested": s.get("num_requested"),
        })
    except Exception as exc:
        row["error"] = str(exc)
with out.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[row] {label}: first_pos={row['first_pos_accept']} "
      f"mean={row['mean_accept_per_draft']} tok_s={row['tok_s']} status={status}")
PY
}

run_one() {
    local tag="$1"
    local label="$2"
    local draft="$3"
    local log="$SWEEP_ROOT/${tag}_vllm.log"
    local spec_config="{\"method\":\"dflash\",\"model\":\"$draft\",\"num_speculative_tokens\":$INFER_NUM_SPEC}"
    local status=0
    if start_server "$tag" "$log" "$spec_config"; then
        run_client "$tag" || status=$?
    else
        status=1
    fi
    cleanup_server
    sleep 4
    append_row "$tag" "$label" "$draft" "$status"
    return "$status"
}

echo "=== ALLaVA val checkpoint sweep ==="
echo "  model:            $MODEL"
echo "  native draft:     $BASELINE_DRAFT"
echo "  find root:        $CHECKPOINT_FIND_ROOT"
echo "  allava_jsonl:     $ALLAVA_JSONL"
echo "  val_ratio:        $VAL_RATIO"
echo "  infer_num_spec:   $INFER_NUM_SPEC"
echo "  num_prompts:      $NUM_PROMPTS"
echo "  gpus:             $GPUS"
echo "  output:           $SWEEP_ROOT"

echo "=== prepare val tail ==="
prepare_val_jsonl

CKPTS=()
while IFS= read -r line; do
    [ -n "$line" ] && CKPTS+=("$line")
done < <(discover_checkpoints)
echo "Discovered ${#CKPTS[@]} dflash checkpoint(s)."
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[fatal] no dflash checkpoints under $CHECKPOINT_FIND_ROOT"
    exit 1
fi

: > "$RESULTS_JSONL"

if ! run_one "native" "native_dflash" "$BASELINE_DRAFT"; then
    echo "WARN: native baseline run failed."
    [ "$CONTINUE_ON_FAIL" = "1" ] || { echo "[fatal] native baseline failed"; exit 1; }
fi

idx=0
for ck in "${CKPTS[@]}"; do
    idx=$((idx + 1))
    tag="ckpt_$(printf '%03d' "$idx")"
    echo "--- [$idx/${#CKPTS[@]}] $ck ---"
    if ! run_one "$tag" "$ck" "$ck"; then
        echo "WARN: $ck failed."
        [ "$CONTINUE_ON_FAIL" = "1" ] || exit 1
    fi
done

python3 - "$RESULTS_JSONL" "$RESULTS_CSV" "$RESULTS_MD" <<'PY' | tee "$SWEEP_ROOT/results.stdout.txt"
import csv
import json
import sys
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
csv_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])
native = next((r for r in rows if r["tag"] == "native"), None)
trained = [r for r in rows if r["tag"] != "native"]

def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None

for r in trained:
    r["mean_ratio_vs_native"] = ratio(
        r.get("mean_accept_per_draft"),
        native.get("mean_accept_per_draft") if native else None,
    )
    r["tok_s_ratio_vs_native"] = ratio(
        r.get("tok_s"), native.get("tok_s") if native else None
    )

ranked = sorted(
    [r for r in trained if r.get("mean_accept_per_draft") is not None],
    key=lambda r: r["mean_accept_per_draft"],
    reverse=True,
)

fields = [
    "label", "first_pos_accept", "mean_accept_per_draft", "token_accept",
    "tok_s", "mean_ratio_vs_native", "tok_s_ratio_vs_native", "completed",
    "requested", "status", "draft",
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    if native:
        writer.writerow(native)
    for r in ranked:
        writer.writerow(r)

def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def short(draft):
    p = Path(draft)
    return f"{p.name} ({p.parent.parent.name[-12:]})"

md = ["# ALLaVA Val Checkpoint Sweep", ""]
if native:
    md += [
        f"native DFlash @spec: first-pos `{fmt(native.get('first_pos_accept'))}`  "
        f"mean-accept `{fmt(native.get('mean_accept_per_draft'))}`  "
        f"tok/s `{fmt(native.get('tok_s'))}`",
        "",
    ]
md += [
    "| rank | checkpoint | first-pos | mean-accept | mean/native | tok/s | tok/s/native | beats native? |",
    "|---:|---|---:|---:|---:|---:|---:|:--:|",
]
for i, r in enumerate(ranked, 1):
    beats = "**yes**" if (r.get("mean_ratio_vs_native") or 0) > 1.0 else "no"
    md.append(
        "| {i} | {lbl} | {fp} | {ma} | {mr} | {ts} | {tr} | {b} |".format(
            i=i, lbl=short(r["draft"]),
            fp=fmt(r.get("first_pos_accept")), ma=fmt(r.get("mean_accept_per_draft")),
            mr=fmt(r.get("mean_ratio_vs_native")), ts=fmt(r.get("tok_s")),
            tr=fmt(r.get("tok_s_ratio_vs_native")), b=beats,
        )
    )
md.append("")
best_line = "BEST_CHECKPOINT=NONE"
if ranked:
    best = ranked[0]
    best_line = f"BEST_CHECKPOINT={best['draft']}"
    md += [
        "## Best by mean-accept",
        "",
        f"- checkpoint: `{best['draft']}`",
        f"- first-pos: `{fmt(best.get('first_pos_accept'))}`  "
        f"mean-accept: `{fmt(best.get('mean_accept_per_draft'))}`  "
        f"tok/s: `{fmt(best.get('tok_s'))}`",
        f"- vs native mean-accept ratio: `{fmt(best.get('mean_ratio_vs_native'))}` "
        f"(>1.0 = beats original in-domain)",
    ]
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print()
print(best_line)
PY

echo
echo "Artifacts: $SWEEP_ROOT"
echo "  results.md / results.csv / results.jsonl"
