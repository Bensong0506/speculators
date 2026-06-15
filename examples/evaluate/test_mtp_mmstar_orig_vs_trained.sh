#!/bin/bash
# MMStar (OUT-OF-DOMAIN forgetting check): ORIGINAL MTP head vs FINE-TUNED MTP head.
#
# Sibling of test_mtp_allava_orig_vs_trained.sh. Same stitch + 2-arm serve + client,
# but on MMStar (which the MTP was NEVER trained on). This is a FORGETTING check:
# we want trained ~= original (no regression), NOT trained >> original.
#
#   - original_mtp : serve raw verifier ($MODEL)  -> native MTP head
#   - trained_mtp  : serve STITCHED checkpoint     -> your finetuned MTP head
#
# Reuses the SAME stitched dir as the ALLaVA run (same ckpt+verifier -> same default
# path), so if you already ran the ALLaVA eval it won't re-stitch.
#
# USAGE
#   MTP_CKPT=/data/wenxuan/speculators/output/<run>/checkpoints/checkpoint_best \
#   INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/test_mtp_mmstar_orig_vs_trained.sh
#
# Requires the mtp-training env (stitch imports speculators.convert.mtp).
# OUTPUT: output/mtp_mmstar_orig_vs_trained/<stamp>/mtp_mmstar_orig_vs_trained_summary.md

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- verifier ----
DEFAULT_ROOT="${DEFAULT_ROOT:-/data/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /home/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/home/wenxuan"
fi
MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"

# ---- MMStar data (OOD eval set; model never trained on it -> use the whole set) ----
MMSTAR_SRC="${MMSTAR_SRC:-$DEFAULT_ROOT/mmstar/mmstar_answers.json}"
MMSTAR_JSONL="${MMSTAR_JSONL:-$REPO_ROOT/data/mmstar/mmstar_eval.jsonl}"
MMSTAR_IMAGE_DIR="${MMSTAR_IMAGE_DIR:-$REPO_ROOT/data/mmstar/images}"
MEDIA_ROOT="${MEDIA_ROOT:-$DEFAULT_ROOT/mmstar/images}"

# ---- trained MTP checkpoint -> stitched model dir (MATCHES the ALLaVA script default) ----
MTP_CKPT="${MTP_CKPT:-}"
STITCHED_DIR="${STITCHED_DIR:-}"
FORCE_STITCH="${FORCE_STITCH:-0}"

# ---- run dir ----
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mtp_mmstar_orig_vs_trained}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
mkdir -p "$RUN_DIR"

# ---- serving knobs (same as the ALLaVA script) ----
GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mtp-mmstar-orig-vs-trained}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
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
[ -d "$MEDIA_ROOT" ] || { echo "[fatal] MEDIA_ROOT (MMStar images) not found: $MEDIA_ROOT"; exit 1; }

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

# ---- step 1: stitch (reuses ALLaVA run's stitched dir if present) ----
stitch_is_ready() {
    [ -f "$STITCHED_DIR/config.json" ] && \
    { [ -f "$STITCHED_DIR/model.safetensors" ] || [ -f "$STITCHED_DIR/model.safetensors.index.json" ]; }
}

echo "=== Step 1: stitch finetuned MTP head -> servable checkpoint ==="
if [ "$FORCE_STITCH" != "1" ] && stitch_is_ready; then
    echo "  reuse existing stitched dir: $STITCHED_DIR (set FORCE_STITCH=1 to rebuild)"
else
    echo "  stitching (writes a full ~verifier-size copy; needs the MTP speculators env)"
    echo "  ckpt:     $MTP_CKPT"
    echo "  verifier: $MODEL"
    echo "  out:      $STITCHED_DIR"
    rm -rf "$STITCHED_DIR"
    if ! python3 scripts/stitch_mtp.py "$MTP_CKPT" "$MODEL" --output-path "$STITCHED_DIR"; then
        echo "[fatal] stitch_mtp.py failed. Run from the mtp-training env where"
        echo "        'import speculators.convert.mtp' works."
        exit 1
    fi
fi
stitch_is_ready || { echo "[fatal] stitched dir looks incomplete: $STITCHED_DIR"; exit 1; }

# ---- step 2: build MMStar conversations jsonl (whole set; OOD, no train/val split) ----
echo "=== Step 2: prepare MMStar jsonl ==="
if [ -s "$MMSTAR_JSONL" ]; then
    echo "  reuse existing $MMSTAR_JSONL"
else
    [ -s "$MMSTAR_SRC" ] || { echo "[fatal] MMSTAR_SRC not found: $MMSTAR_SRC"; exit 1; }
    python3 scripts/mmstar_to_jsonl.py \
        --mmstar "$MMSTAR_SRC" \
        --out-jsonl "$MMSTAR_JSONL" \
        --image-dir "$MMSTAR_IMAGE_DIR"
fi
echo "  rows: $(wc -l < "$MMSTAR_JSONL")  (client samples first $NUM_PROMPTS)"

# ---- server helpers (identical to the ALLaVA script) ----
wait_for_server() {
    local log="$1"; local mode="$2"
    echo "Waiting for $mode server on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*mtp|unsupported.*mtp|unrecognized.*mtp|no.*mtp" "$log"; then
                echo "DIAGNOSIS: vLLM may not support method='$( [ "$mode" = trained_mtp ] && echo "$TRAINED_MTP_METHOD" || echo "$ORIG_MTP_METHOD")'."
                echo "           For the trained arm try TRAINED_MTP_METHOD=mtp."
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
        --allowed-local-media-path "$MEDIA_ROOT"
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
    echo "  port:         $PORT   devices: $GPUS   media: $MEDIA_ROOT"
    echo "  log:          $log"
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
echo "=== MMStar (OOD): original MTP vs trained MTP ==="
echo "  model:           $MODEL"
echo "  trained ckpt:    $MTP_CKPT"
echo "  stitched dir:    $STITCHED_DIR"
echo "  mmstar jsonl:    $MMSTAR_JSONL"
echo "  media root:      $MEDIA_ROOT"
echo "  spec tokens:     $INFER_NUM_SPEC   num_prompts: $NUM_PROMPTS"
echo "  output:          $RUN_DIR"

ORIG_SPEC="{\"method\":\"$ORIG_MTP_METHOD\",\"num_speculative_tokens\":$INFER_NUM_SPEC,\"enforce_eager\":true}"
TRAINED_SPEC="{\"method\":\"$TRAINED_MTP_METHOD\",\"num_speculative_tokens\":$INFER_NUM_SPEC,\"enforce_eager\":true}"

# SKIP_ORIGINAL=1 skips the native arm (alpha-independent) for soup sweeps.
if [ "${SKIP_ORIGINAL:-0}" = "1" ]; then
    echo "SKIP_ORIGINAL=1 -> skipping native (original_mtp) arm"
else
    run_one original_mtp "$MODEL"         "$ORIG_SPEC"
fi
run_one trained_mtp  "$STITCHED_DIR"  "$TRAINED_SPEC"

# ---- summary + OOD forgetting verdict ----
python3 - "$RUN_DIR" "$RUN_DIR/mtp_mmstar_orig_vs_trained_summary.md" "$MTP_CKPT" "$MMSTAR_JSONL" <<'PY' | tee "$RUN_DIR/mtp_mmstar_orig_vs_trained_summary.stdout.txt"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_md = Path(sys.argv[2])
ckpt = sys.argv[3]
data_jsonl = sys.argv[4]

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
    "# MMStar (OUT-OF-DOMAIN) — original MTP vs trained MTP",
    "",
    "Forgetting check: the MTP was trained on ALLaVA, NOT MMStar. We want trained "
    "**~= original** (no regression), not trained >> original.",
    "",
    f"trained ckpt: `{ckpt}`  ",
    f"eval set: `{data_jsonl}`  ",
    f"requests: original {fmt(g(orig,'completed'))}/{fmt(g(orig,'num_requested'))}, "
    f"trained {fmt(g(trained,'completed'))}/{fmt(g(trained,'num_requested'))}",
    "",
    "| metric | original MTP | trained MTP | trained-original | trained/original |",
    "|---|---:|---:|---:|---:|",
]
for label, key in KEYS:
    a = g(orig, key)
    b = g(trained, key)
    md.append(f"| {label} | {fmt(a)} | {fmt(b)} | {fmt(delta(b, a))} | {fmt(ratio(b, a))} |")

md += ["", "## verdict (forgetting check)", ""]
ma, mb = g(orig, "spec_mean_accepted_tokens_per_draft"), g(trained, "spec_mean_accepted_tokens_per_draft")
fa, fb = g(orig, "spec_first_position_acceptance_rate"), g(trained, "spec_first_position_acceptance_rate")
exact = (
    isinstance(fa, float) and isinstance(fb, float) and abs(fa - fb) < 1e-4 and
    isinstance(ma, float) and isinstance(mb, float) and abs(ma - mb) < 1e-3
)
r = ratio(mb, ma)
if exact:
    md += ["WARNING: trained == original EXACTLY. On OOD the drafts should still differ "
           "slightly; exact match suggests the finetuned head did not load. "
           "Try `TRAINED_MTP_METHOD=mtp`."]
elif r is None:
    md += ["Incomplete metrics — see per-arm *_summary.json and *_vllm.log."]
elif r >= 0.98:
    extra = " (and even improves OOD)" if r > 1.0 else ""
    md += [f"PASS — no forgetting{extra}: trained/original mean-accept = {fmt(r)} "
           f"(first-pos {fmt(fb)} vs {fmt(fa)}). Finetuning the MTP head did not hurt OOD."]
elif r >= 0.95:
    md += [f"BORDERLINE: trained/original mean-accept = {fmt(r)} (mild OOD regression). "
           f"Acceptable but watch it."]
else:
    md += [f"REGRESSION: trained/original mean-accept = {fmt(r)} (< 0.95). The finetune "
           f"hurt OOD — consider less data/epochs, lower LR, or mixing some general data."]

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  summary:  $RUN_DIR/mtp_mmstar_orig_vs_trained_summary.md"
echo "  per-arm:  $RUN_DIR/{original_mtp,trained_mtp}_summary.json"
