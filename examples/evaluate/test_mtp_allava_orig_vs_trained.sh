#!/bin/bash
# ALLaVA: ORIGINAL MTP head vs your FINE-TUNED (trained) MTP head.
#
# Both arms run the SAME spec method (qwen3_5_mtp) at the SAME spec count on the
# SAME ALLaVA val tail; the ONLY difference is the MTP-head weights:
#   - original_mtp : serve the raw verifier ($MODEL)        -> verifier's native MTP head
#   - trained_mtp  : serve the STITCHED checkpoint           -> your finetuned MTP head
#
# "Stitching" = scripts/stitch_mtp.py merges the finetuned MTP-layer weights back
# into a full copy of the verifier checkpoint, producing a self-contained model dir
# that vLLM can serve. (vLLM can't mount a bare MTP speculator checkpoint; the head
# lives inside the verifier weights, so we replace those weights in place.)
#
# REQUIREMENTS
#   - Run on the box where the MTP-capable `speculators` is importable (the
#     mtp-training env), because stitch_mtp.py imports speculators.convert.mtp.
#   - A trained MTP checkpoint (speculators format), e.g. .../checkpoints/checkpoint_best.
#
# USAGE
#   MTP_CKPT=/home/wenxuan/speculators/output/<run>/checkpoints/checkpoint_best \
#   INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/test_mtp_allava_orig_vs_trained.sh
#
#   # IMPORTANT: point ALLAVA_JSONL at the SAME jsonl your MTP trained on, so the
#   # val tail (last 10%) matches the training val split. Defaults to the 10k
#   # distilled jsonl (the MTP train default). For the 100k run, set:
#   #   ALLAVA_JSONL=.../data/allava/allava_qwen35_distill_100k.jsonl
#
# OUTPUT
#   output/mtp_orig_vs_trained/<stamp>/mtp_orig_vs_trained_summary.md  (+ per-arm json)
#
# Notes
#   - Trust first-pos / mean-accept (read from vLLM /metrics by the client); tok/s
#     is noisy run-to-run (+/- several %).
#   - If trained_mtp comes out ~identical to original_mtp, the finetuned head may
#     not have loaded -- see the sanity check at the end and try TRAINED_MTP_METHOD=mtp.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- paths: verifier + ALLaVA (mirror the DFlash ALLaVA eval defaults) ----
DEFAULT_ROOT="${DEFAULT_ROOT:-/home/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /data/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/data/wenxuan"
fi
DEFAULT_ALLAVA_ROOT="${ALLAVA_ROOT:-$DEFAULT_ROOT/ALLaVA-4V}"
if [ ! -d "$DEFAULT_ALLAVA_ROOT" ] && [ -d /data/wenxuan/ALLaVA-4V ]; then
    DEFAULT_ALLAVA_ROOT="/data/wenxuan/ALLaVA-4V"
fi

MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-$DEFAULT_ALLAVA_ROOT}"

# ---- trained MTP checkpoint -> stitched model dir ----
MTP_CKPT="${MTP_CKPT:-}"
STITCHED_DIR="${STITCHED_DIR:-}"
FORCE_STITCH="${FORCE_STITCH:-0}"

# ---- ALLaVA val tail (must match the jsonl the MTP trained on) ----
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_TAG="${VAL_TAG:-tail10pct}"
ALLAVA_VAL_JSONL="${ALLAVA_VAL_JSONL:-$REPO_ROOT/data/allava/$(basename "${ALLAVA_JSONL%.jsonl}")_val_${VAL_TAG}.jsonl}"

# ---- run dir ----
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mtp_orig_vs_trained}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
mkdir -p "$RUN_DIR"

# ---- serving knobs (mirror test_dflash_allava_val_weights.sh) ----
GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mtp-orig-vs-trained}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

# Both arms use the same MTP method by default (apples-to-apples: only the weights
# differ). The native Qwen3.5 path reads the MTP head from the served checkpoint,
# so it picks up the stitched (finetuned) weights too. If the native path won't
# load the stitched head, switch the trained arm to the generic speculators MTP
# path: TRAINED_MTP_METHOD=mtp  (this is the tutorial's documented deploy method).
ORIG_MTP_METHOD="${ORIG_MTP_METHOD:-qwen3_5_mtp}"
TRAINED_MTP_METHOD="${TRAINED_MTP_METHOD:-qwen3_5_mtp}"

MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * INFER_NUM_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"

# ---- validation ----
if [ -z "$MTP_CKPT" ]; then
    echo "[fatal] set MTP_CKPT=/path/to/trained/mtp/checkpoint_best (speculators format)"
    exit 1
fi
[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$MTP_CKPT" ] || { echo "[fatal] MTP_CKPT not found: $MTP_CKPT"; exit 1; }
[ -d "$ALLAVA_IMAGE_ROOT" ] || { echo "[fatal] ALLAVA_IMAGE_ROOT not found: $ALLAVA_IMAGE_ROOT"; exit 1; }
[ -s "$ALLAVA_JSONL" ] || { echo "[fatal] ALLAVA_JSONL missing/empty: $ALLAVA_JSONL (point it at the jsonl your MTP trained on)"; exit 1; }

if [ -z "$STITCHED_DIR" ]; then
    STITCHED_DIR="$REPO_ROOT/output/mtp_stitched/$(basename "$(dirname "$(dirname "$MTP_CKPT")")")_$(basename "$MTP_CKPT")"
fi

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

# ---- step 1: stitch finetuned MTP head into a full verifier checkpoint ----
stitch_is_ready() {
    [ -f "$STITCHED_DIR/config.json" ] && \
    { [ -f "$STITCHED_DIR/model.safetensors" ] || [ -f "$STITCHED_DIR/model.safetensors.index.json" ]; }
}

echo "=== Step 1: stitch finetuned MTP head -> servable checkpoint ==="
if [ "$FORCE_STITCH" != "1" ] && stitch_is_ready; then
    echo "  reuse existing stitched dir: $STITCHED_DIR (set FORCE_STITCH=1 to rebuild)"
else
    echo "  stitching (this writes a full ~verifier-size copy; needs the MTP speculators env)"
    echo "  ckpt:     $MTP_CKPT"
    echo "  verifier: $MODEL"
    echo "  out:      $STITCHED_DIR"
    rm -rf "$STITCHED_DIR"
    if ! python3 scripts/stitch_mtp.py "$MTP_CKPT" "$MODEL" --output-path "$STITCHED_DIR"; then
        echo "[fatal] stitch_mtp.py failed. Run from the mtp-training env where"
        echo "        'import speculators.convert.mtp' works (pip install -e . --no-deps"
        echo "        or PYTHONPATH=\$PWD/src)."
        exit 1
    fi
fi
stitch_is_ready || { echo "[fatal] stitched dir looks incomplete: $STITCHED_DIR"; exit 1; }

# ---- step 2: prepare ALLaVA val tail (last VAL_RATIO of the training jsonl) ----
echo "=== Step 2: prepare ALLaVA val tail (last ${VAL_RATIO} of $(basename "$ALLAVA_JSONL")) ==="
python3 - "$ALLAVA_JSONL" "$ALLAVA_VAL_JSONL" "$VAL_RATIO" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ratio = float(sys.argv[3])
if not 0 < ratio < 1:
    raise SystemExit(f"[fatal] VAL_RATIO must be in (0, 1), got {ratio}")
lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not lines:
    raise SystemExit(f"[fatal] ALLAVA_JSONL is empty: {src}")
split_idx = int(len(lines) * (1.0 - ratio))
val_lines = lines[split_idx:]
if not val_lines:
    raise SystemExit(f"[fatal] validation split empty: rows={len(lines)} ratio={ratio}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
print(f"[info] val jsonl: {dst} rows={len(val_lines)} (of {len(lines)}, start_index={split_idx})")
PY

# ---- server helpers ----
wait_for_server() {
    local log="$1"; local mode="$2"
    echo "Waiting for $mode server on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*mtp|unsupported.*mtp|unrecognized.*mtp|no.*mtp" "$log"; then
                echo "DIAGNOSIS: this vLLM build may not support method='$( [ "$mode" = trained_mtp ] && echo "$TRAINED_MTP_METHOD" || echo "$ORIG_MTP_METHOD")'."
                echo "           For the trained arm, try TRAINED_MTP_METHOD=mtp (generic speculators MTP)."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: spec scheduling budget too small; lower MAX_NUM_SEQS/spec or raise MAX_NUM_BATCHED_TOKENS."
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
    local mode="$1"; local model_dir="$2"; local log="$3"; local spec_config="$4"
    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting $mode."
        return 1
    fi
    local args=(
        vllm serve "$model_dir"
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
        --speculative-config "$spec_config"
    )
    [ "$ENFORCE_EAGER" = "1" ] && args+=(--enforce-eager)
    [ -n "$DTYPE" ] && args+=(--dtype "$DTYPE")
    [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
    [ "$DISABLE_CHUNKED_PREFILL" = "1" ] && args+=(--no-enable-chunked-prefill)

    echo
    echo "=== Starting $mode server ==="
    echo "  model:        $model_dir"
    echo "  spec_config:  $spec_config"
    echo "  port:         $PORT   devices: $GPUS"
    echo "  log:          $log"
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
    local mode="$1"; local log="$2"
    local out="$RUN_DIR/${mode}_acceptance_from_log.txt"
    local parser="examples/evaluate/eval-guidellm/scripts/parse_logs.py"
    if [ -f "$parser" ] && python3 "$parser" "$log" > "$out" 2>&1; then
        :
    else
        echo "(No parseable 'SpecDecoding metrics:' lines for $mode.)" > "$out"
        grep -iE "spec|accept|draft|mtp" "$log" | tail -n 60 >> "$out" || true
    fi
}

run_one() {
    local mode="$1"; local model_dir="$2"; local spec_config="$3"
    local log="$RUN_DIR/${mode}_vllm.log"
    start_server "$mode" "$model_dir" "$log" "$spec_config" || {
        echo "WARN: $mode server failed; writing empty summary so the table still renders."
        echo '{}' > "$RUN_DIR/${mode}_summary.json"
        parse_acceptance "$mode" "$log"
        return 0
    }
    run_client "$mode"
    cleanup_server
    sleep 5
    parse_acceptance "$mode" "$log"
}

echo
echo "=== ALLaVA: original MTP vs trained MTP ==="
echo "  model:             $MODEL"
echo "  trained ckpt:      $MTP_CKPT"
echo "  stitched dir:      $STITCHED_DIR"
echo "  orig  method:      $ORIG_MTP_METHOD   (serves $MODEL)"
echo "  trained method:    $TRAINED_MTP_METHOD   (serves stitched dir)"
echo "  spec tokens:       $INFER_NUM_SPEC"
echo "  allava_val_jsonl:  $ALLAVA_VAL_JSONL"
echo "  num_prompts:       $NUM_PROMPTS"
echo "  output:            $RUN_DIR"

ORIG_SPEC="{\"method\":\"$ORIG_MTP_METHOD\",\"num_speculative_tokens\":$INFER_NUM_SPEC,\"enforce_eager\":true}"
TRAINED_SPEC="{\"method\":\"$TRAINED_MTP_METHOD\",\"num_speculative_tokens\":$INFER_NUM_SPEC,\"enforce_eager\":true}"

# SKIP_ORIGINAL=1 skips the native arm (it is alpha-independent) so a soup sweep
# only measures it once; the summary then reads original_mtp_summary.json as n/a.
if [ "${SKIP_ORIGINAL:-0}" = "1" ]; then
    echo "SKIP_ORIGINAL=1 -> skipping native (original_mtp) arm"
else
    run_one original_mtp "$MODEL"         "$ORIG_SPEC"
fi
run_one trained_mtp  "$STITCHED_DIR"  "$TRAINED_SPEC"

# ---- summary + sanity check ----
python3 - "$RUN_DIR" "$RUN_DIR/mtp_orig_vs_trained_summary.md" "$MTP_CKPT" "$ALLAVA_VAL_JSONL" <<'PY' | tee "$RUN_DIR/mtp_orig_vs_trained_summary.stdout.txt"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_md = Path(sys.argv[2])
ckpt = sys.argv[3]
val_jsonl = sys.argv[4]

def load(m):
    p = run_dir / f"{m}_summary.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

orig = load("original_mtp")
trained = load("trained_mtp")

def g(d, k):
    return d.get(k)

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v))

def ratio(a, b):
    return a / b if (isinstance(a, (int, float)) and isinstance(b, (int, float)) and b) else None

def delta(a, b):
    return a - b if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None

KEYS = [
    ("first-pos accept", "spec_first_position_acceptance_rate"),
    ("mean accept/draft", "spec_mean_accepted_tokens_per_draft"),
    ("token accept", "spec_token_acceptance_rate"),
    ("tok/s", "output_tok_per_sec"),
]

md = [
    "# ALLaVA — original MTP vs trained MTP",
    "",
    f"trained ckpt: `{ckpt}`  ",
    f"val jsonl: `{val_jsonl}`  ",
    f"requests: original {fmt(g(orig,'completed'))}/{fmt(g(orig,'num_requested'))}, "
    f"trained {fmt(g(trained,'completed'))}/{fmt(g(trained,'num_requested'))}",
    "",
    "| metric | original MTP | trained MTP | trained-original | trained/original |",
    "|---|---:|---:|---:|---:|",
]
for label, key in KEYS:
    a = g(orig, key)
    b = g(trained, key)
    md.append(
        f"| {label} | {fmt(a)} | {fmt(b)} | {fmt(delta(b, a))} | {fmt(ratio(b, a))} |"
    )

# per-position acceptance, if present
def by_pos(d):
    v = d.get("spec_accepted_tokens_by_position")
    return v if isinstance(v, list) else None

op, tp = by_pos(orig), by_pos(trained)
if op or tp:
    md += ["", "## accepted tokens by position", "",
           "| pos | original | trained |", "|---:|---:|---:|"]
    n = max(len(op or []), len(tp or []))
    for i in range(n):
        ov = op[i] if op and i < len(op) else None
        tv = tp[i] if tp and i < len(tp) else None
        md.append(f"| {i} | {fmt(ov)} | {fmt(tv)} |")

# verdict + sanity check
md += ["", "## verdict", ""]
fa, fb = g(orig, "spec_first_position_acceptance_rate"), g(trained, "spec_first_position_acceptance_rate")
ma, mb = g(orig, "spec_mean_accepted_tokens_per_draft"), g(trained, "spec_mean_accepted_tokens_per_draft")
identical = (
    isinstance(fa, float) and isinstance(fb, float) and abs(fa - fb) < 1e-4 and
    isinstance(ma, float) and isinstance(mb, float) and abs(ma - mb) < 1e-3
)
if identical:
    md += [
        "WARNING: trained == original almost exactly. The finetuned MTP head"
        " probably did NOT load. Check that the stitch overwrote the MTP weights,"
        " and try `TRAINED_MTP_METHOD=mtp` (generic speculators MTP path).",
    ]
elif isinstance(mb, float) and isinstance(ma, float):
    if mb > ma:
        md += [f"Trained MTP improves over original on this ALLaVA val slice "
               f"(mean-accept {fmt(mb)} vs {fmt(ma)}, first-pos {fmt(fb)} vs {fmt(fa)})."]
    else:
        md += [f"Trained MTP does NOT beat original here "
               f"(mean-accept {fmt(mb)} vs {fmt(ma)}, first-pos {fmt(fb)} vs {fmt(fa)}). "
               f"More data / pos-0 weighting may be needed."]
else:
    md += ["Incomplete metrics — see per-arm *_summary.json and *_vllm.log."]

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  summary:  $RUN_DIR/mtp_orig_vs_trained_summary.md"
echo "  per-arm:  $RUN_DIR/{original_mtp,trained_mtp}_summary.json"
echo "  stitched: $STITCHED_DIR"
