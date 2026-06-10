#!/bin/bash
# Three-way MMStar comparison at a SINGLE spec depth, in one run:
#   mtp, original/native DFlash, trained DFlash -- all at INFER_NUM_SPEC.
#
# Unlike test_dflash_mmstar_weights.sh (2-way: original vs trained) this adds
# native MTP, so MTP / original DFlash / trained DFlash are measured back-to-back
# on the identical MMStar prompts and the same spec budget -- apples-to-apples,
# no cross-run drift. It mirrors test_dflash_allava_val_weights.sh but on MMStar
# and without the no-spec baseline (set WITH_BASELINE=1 to add it).
#
# Server args match test_dflash_mmstar_weights.sh so the original/trained numbers
# reproduce that 2-way run; MTP uses the qwen3_5_mtp config from the ALLaVA 4-way.
#
# Usage:
#   DRAFT=/path/to/checkpoint_best \
#   INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/test_dflash_mmstar_three_way.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_ROOT="${DEFAULT_ROOT:-/home/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /data/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/data/wenxuan"
fi
DEFAULT_MMSTAR_ROOT="${MMSTAR_ROOT:-$DEFAULT_ROOT/mmstar}"
if [ ! -s "$DEFAULT_MMSTAR_ROOT/mmstar_answers.json" ] && [ -s /data/wenxuan/mmstar/mmstar_answers.json ]; then
    DEFAULT_MMSTAR_ROOT="/data/wenxuan/mmstar"
fi

MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"
BASELINE_DRAFT="${BASELINE_DRAFT:-$DEFAULT_ROOT/Qwen3.5-9B-DFlash}"
MMSTAR_SRC="${MMSTAR_SRC:-$DEFAULT_MMSTAR_ROOT/mmstar_answers.json}"
MMSTAR_JSONL="${MMSTAR_JSONL:-$REPO_ROOT/data/mmstar/mmstar_eval.jsonl}"
MMSTAR_IMAGE_DIR="${MMSTAR_IMAGE_DIR:-$REPO_ROOT/data/mmstar/images}"
MEDIA_ROOT="${MEDIA_ROOT:-$DEFAULT_MMSTAR_ROOT/images}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_three_way_tests}"
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
ORIGINAL_DFLASH_SPEC="${ORIGINAL_DFLASH_SPEC:-$INFER_NUM_SPEC}"
TRAINED_DFLASH_SPEC="${TRAINED_DFLASH_SPEC:-$INFER_NUM_SPEC}"
WITH_BASELINE="${WITH_BASELINE:-0}"
MAX_CONFIG_SPEC="$(python3 - "$MTP_SPEC" "$ORIGINAL_DFLASH_SPEC" "$TRAINED_DFLASH_SPEC" <<'PY'
import sys
print(max(int(x) for x in sys.argv[1:]))
PY
)"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_CONFIG_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mmstar-three-way}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

if [ -z "${DRAFT:-}" ]; then
    echo "[fatal] DRAFT (trained checkpoint) is not set. Pass DRAFT=/path/to/checkpoint_best."
    exit 1
fi

[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$BASELINE_DRAFT" ] || { echo "[fatal] BASELINE_DRAFT not found: $BASELINE_DRAFT"; exit 1; }
[ -d "$DRAFT" ] || { echo "[fatal] DRAFT not found: $DRAFT"; exit 1; }
[ -d "$MEDIA_ROOT" ] || { echo "[fatal] MEDIA_ROOT (mmstar images) not found: $MEDIA_ROOT"; exit 1; }

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

echo "=== Preparing MMStar conversations jsonl ==="
if [ -s "$MMSTAR_JSONL" ]; then
    echo "reuse existing $MMSTAR_JSONL"
else
    python3 scripts/mmstar_to_jsonl.py \
        --mmstar "$MMSTAR_SRC" \
        --out-jsonl "$MMSTAR_JSONL" \
        --image-dir "$MMSTAR_IMAGE_DIR"
fi

wait_for_server() {
    local log="$1"
    local mode="$2"
    echo "Waiting for $mode server on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 80 log lines:"
            tail -n 80 "$log" || true
            if grep -qiE "unknown.*(mtp|dflash)|unsupported.*(mtp|dflash)|unrecognized.*(mtp|dflash)" "$log"; then
                echo "DIAGNOSIS: this vLLM build may not support the requested speculative method."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: vLLM speculative scheduling budget too small; lower MAX_NUM_SEQS/spec or raise MAX_NUM_BATCHED_TOKENS."
            fi
            return 1
        fi
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "$mode server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for $mode server. Last 80 log lines:"
    tail -n 80 "$log" || true
    return 1
}

start_server() {
    local mode="$1"
    local log="$2"
    local spec_config="${3:-}"

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting $mode."
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
        --allowed-local-media-path "$MEDIA_ROOT"
        --limit-mm-per-prompt '{"image":1}'
        --generation-config vllm
        --host 0.0.0.0
        --port "$PORT"
    )
    [ "$ENFORCE_EAGER" = "1" ] && args+=(--enforce-eager)
    [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
    [ -n "$spec_config" ] && args+=(--speculative-config "$spec_config")

    echo
    echo "=== Starting $mode server (spec_config=${spec_config:-none}) on :$PORT, devices=$GPUS ==="
    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$log" "$mode"
}

run_client() {
    local mode="$1"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${PORT}/v1" \
        --data-jsonl "$MMSTAR_JSONL" \
        --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
        --summary-json "$RUN_DIR/${mode}_summary.json" \
        --num "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS"
}

run_one() {
    local mode="$1"
    local spec_config="${2:-}"
    local log="$RUN_DIR/${mode}_vllm.log"
    if start_server "$mode" "$log" "$spec_config"; then
        run_client "$mode" || echo "WARN: client failed for $mode"
    else
        echo "WARN: server failed for $mode"
    fi
    cleanup_server
    sleep 5
}

echo "=== MMStar three-way benchmark ==="
echo "  model:            $MODEL"
echo "  mtp_method/spec:  $MTP_METHOD @ $MTP_SPEC"
echo "  original_dflash:  $BASELINE_DRAFT @ $ORIGINAL_DFLASH_SPEC"
echo "  trained_dflash:   $DRAFT @ $TRAINED_DFLASH_SPEC"
echo "  mmstar_jsonl:     $MMSTAR_JSONL"
echo "  media_root:       $MEDIA_ROOT"
echo "  num_prompts:      $NUM_PROMPTS"
echo "  with_baseline:    $WITH_BASELINE"
echo "  output:           $RUN_DIR"

MTP_SPEC_CONFIG="{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$MTP_SPEC,\"enforce_eager\":true}"
ORIGINAL_DFLASH_SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$BASELINE_DRAFT\",\"num_speculative_tokens\":$ORIGINAL_DFLASH_SPEC}"
TRAINED_DFLASH_SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$TRAINED_DFLASH_SPEC}"

if [ "$WITH_BASELINE" = "1" ]; then
    run_one baseline ""
fi
run_one mtp "$MTP_SPEC_CONFIG"
run_one dflash_original "$ORIGINAL_DFLASH_SPEC_CONFIG"
run_one trained_dflash "$TRAINED_DFLASH_SPEC_CONFIG"

echo
echo "=== Final three-way comparison ==="
python3 - "$RUN_DIR" "$WITH_BASELINE" "$RUN_DIR/mmstar_three_way_summary.md" <<'PY' | tee "$RUN_DIR/mmstar_three_way_summary.stdout.txt"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
with_baseline = sys.argv[2] == "1"
out_md = Path(sys.argv[3])

methods = (["baseline"] if with_baseline else []) + ["mtp", "dflash_original", "trained_dflash"]
rows = {}
for m in methods:
    p = run_dir / f"{m}_summary.json"
    if p.exists():
        try:
            s = json.loads(p.read_text())
        except Exception:
            s = {}
    else:
        s = {}
    rows[m] = {
        "method": m,
        "tok_s": s.get("output_tok_per_sec"),
        "mean_accept": s.get("spec_mean_accepted_tokens_per_draft"),
        "token_accept": s.get("spec_token_acceptance_rate"),
        "first_pos": s.get("spec_first_position_acceptance_rate"),
        "completed": s.get("completed"),
        "requested": s.get("num_requested"),
    }

def fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)

def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None

ranked = sorted(
    [r for r in rows.values() if r["tok_s"] is not None],
    key=lambda r: r["tok_s"], reverse=True,
)

md = [
    "# MMStar Three-Way Benchmark (single spec, one run)",
    "",
    "| rank | method | tok/s | mean accept/draft | token accept | first-pos accept | completed |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for i, r in enumerate(ranked, 1):
    md.append(
        f"| {i} | {r['method']} | {fmt(r['tok_s'])} | {fmt(r['mean_accept'])} | "
        f"{fmt(r['token_accept'])} | {fmt(r['first_pos'])} | {fmt(r['completed'])}/{fmt(r['requested'])} |"
    )

tr = rows.get("trained_dflash", {})
orig = rows.get("dflash_original", {})
mtp = rows.get("mtp", {})
md += [
    "",
    "## Key ratios",
    "",
    f"- trained / original DFlash : tok/s `{fmt(ratio(tr.get('tok_s'), orig.get('tok_s')))}`, "
    f"mean-accept `{fmt(ratio(tr.get('mean_accept'), orig.get('mean_accept')))}`, "
    f"first-pos `{fmt(ratio(tr.get('first_pos'), orig.get('first_pos')))}`",
    f"- trained / MTP            : tok/s `{fmt(ratio(tr.get('tok_s'), mtp.get('tok_s')))}`, "
    f"mean-accept `{fmt(ratio(tr.get('mean_accept'), mtp.get('mean_accept')))}`, "
    f"first-pos `{fmt(ratio(tr.get('first_pos'), mtp.get('first_pos')))}`",
    f"- original DFlash / MTP    : tok/s `{fmt(ratio(orig.get('tok_s'), mtp.get('tok_s')))}`, "
    f"mean-accept `{fmt(ratio(orig.get('mean_accept'), mtp.get('mean_accept')))}`",
    "",
    "## Verdict",
    "",
]
mr_to = ratio(tr.get("mean_accept"), orig.get("mean_accept"))
mr_tm = ratio(tr.get("mean_accept"), mtp.get("mean_accept"))
if mr_to is not None and mr_to > 1.0:
    md.append(f"- trained DFlash beats original DFlash on mean-accept ({fmt(mr_to)}x). [project bar met]")
elif mr_to is not None:
    md.append(f"- trained DFlash does NOT beat original DFlash on mean-accept ({fmt(mr_to)}x).")
if mr_tm is not None and mr_tm > 1.0:
    md.append(f"- trained DFlash beats MTP on mean-accept ({fmt(mr_tm)}x).")
elif mr_tm is not None:
    md.append(f"- MTP still stronger than trained DFlash on mean-accept ({fmt(mr_tm)}x).")
if ranked:
    md.append(f"- Fastest by tok/s: **{ranked[0]['method']}**. Order: " + " > ".join(r["method"] for r in ranked) + ".")

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
echo "  $RUN_DIR/mmstar_three_way_summary.md"
