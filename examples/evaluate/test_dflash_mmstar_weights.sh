#!/bin/bash
# End-to-end MMStar comparison for Qwen3.5-9B DFlash checkpoints.
#
# This script answers a narrow question: "did my continued-training DFlash
# checkpoint improve over the native/downloaded DFlash checkpoint?" It runs two
# DFlash servers, native/raw baseline then trained candidate, against the same
# MMStar prompts and writes all outputs under RUN_DIR.
#
# Usage:
#   bash examples/evaluate/test_dflash_mmstar_weights.sh
#
# Common overrides:
#   BASELINE_DRAFT=/data/wenxuan/Qwen3.5-9B-DFlash \
#   DRAFT=/path/to/checkpoint_best \
#   INFER_NUM_SPEC=7 \
#   MMSTAR_SRC=/data/wenxuan/mmstar/mmstar_answers.json \
#   NUM_PROMPTS=128 \
#   bash examples/evaluate/test_dflash_mmstar_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
BASELINE_DRAFT="${BASELINE_DRAFT:-/data/wenxuan/Qwen3.5-9B-DFlash}"
MMSTAR_SRC="${MMSTAR_SRC:-/data/wenxuan/mmstar/mmstar_answers.json}"
MMSTAR_JSONL="${MMSTAR_JSONL:-$REPO_ROOT/data/mmstar/mmstar_eval.jsonl}"
MMSTAR_IMAGE_DIR="${MMSTAR_IMAGE_DIR:-$REPO_ROOT/data/mmstar/images}"
MEDIA_ROOT="${MEDIA_ROOT:-/data/wenxuan/mmstar/images}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_weight_tests}"
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
MAX_CONFIG_SPEC="$(python3 - "$BASELINE_SPEC" "$DFLASH_SPEC" <<'PY'
import sys
print(max(int(x) for x in sys.argv[1:]))
PY
)"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_CONFIG_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mmstar-weight-test}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

# If DRAFT is not explicitly set, try the most recent long-run checkpoint first,
# then fall back to the older default location.
if [ -z "${DRAFT:-}" ]; then
    DRAFT="$(find "$REPO_ROOT/output" -path '*/checkpoints/checkpoint_best' -type l -o -path '*/checkpoints/checkpoint_best' -type d 2>/dev/null | sort | tail -n 1 || true)"
    if [ -z "$DRAFT" ]; then
        DRAFT="$REPO_ROOT/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best"
    fi
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

checkpoint_json() {
    local label="$1"
    local draft_path="$2"
    local requested_spec="$3"
    python3 - "$MODEL" "$draft_path" "$label" "$requested_spec" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

model = sys.argv[1]
draft = Path(sys.argv[2])
label = sys.argv[3]
requested_spec = int(sys.argv[4])
if not draft.exists():
    raise SystemExit(f"[fatal] {label} DFlash draft does not exist: {draft}")

cfg_path = draft / "config.json"
weights_path = draft / "model.safetensors"
if not cfg_path.exists():
    raise SystemExit(f"[fatal] missing config.json under {label} DFlash draft: {draft}")
if not weights_path.exists():
    raise SystemExit(
        f"[fatal] missing model.safetensors under {label} DFlash draft: {draft}"
    )

cfg = json.loads(cfg_path.read_text())
speculators_type = cfg.get("speculators_model_type")
is_speculators_dflash = speculators_type == "dflash"
is_raw_dflash = bool(cfg.get("dflash_config")) and bool(cfg.get("block_size"))
if not is_speculators_dflash and not (label == "native" and is_raw_dflash):
    raise SystemExit(
        f"[fatal] {label} checkpoint is not an accepted DFlash model: {draft}\n"
        f"        speculators_model_type={speculators_type!r}, "
        f"has_dflash_config={bool(cfg.get('dflash_config'))}.\n"
        "        Native baseline may be raw/z-lab DFlash; trained candidate must "
        "be speculators-format DFlash."
    )

aux = cfg.get("aux_hidden_state_layer_ids") or cfg.get("dflash_config", {}).get(
    "target_layer_ids"
) or []
hidden = cfg.get("transformer_layer_config", {}).get("hidden_size") or cfg.get(
    "hidden_size"
)
block = int(cfg.get("block_size", 0))
max_spec = block - 1
verifier = (
    cfg.get("speculators_config", {})
    .get("verifier", {})
    .get("name_or_path")
)

with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
    keys = set(handle.keys())
    missing = [
        key
        for key in ("fc.weight", "hidden_norm.weight", "norm.weight")
        if key not in keys
    ]
    if missing:
        raise SystemExit(f"[fatal] missing expected trained tensor(s): {missing}")
    fc_shape = tuple(handle.get_tensor("fc.weight").shape)
    layer_keys = [key for key in keys if key.startswith("layers.")]

if hidden and aux and fc_shape != (hidden, hidden * len(aux)):
    raise SystemExit(
        "[fatal] fc.weight shape does not match aux hidden-state recipe: "
        f"fc_shape={fc_shape}, hidden={hidden}, aux={aux}"
    )
if not layer_keys:
    raise SystemExit("[fatal] no draft layer weights found under layers.*")
if block <= 0:
    raise SystemExit(f"[fatal] {label} has invalid block_size={block}: {draft}")
if requested_spec <= 0:
    raise SystemExit(f"[fatal] requested num_speculative_tokens must be positive: {requested_spec}")
if requested_spec > max_spec:
    raise SystemExit(
        f"[fatal] {label} config block_size={block} only supports "
        f"num_speculative_tokens <= {max_spec}, requested {requested_spec}."
    )

if verifier and verifier != model:
    print(f"[warn] {label} verifier path differs: {verifier} != {model}")

print(f"Checkpoint sanity OK ({label})")
print(f"  draft:      {draft}")
print(f"  format:     {'speculators' if is_speculators_dflash else 'raw'}")
print(f"  block_size: {block}")
print(f"  max_spec:   {max_spec}")
print(f"  infer_spec: {requested_spec}")
print(f"  aux_layers: {aux}")
print(f"  hidden:     {hidden}")
print(f"  fc.weight:  {fc_shape}")
print(f"  layers.*:   {len(layer_keys)} tensors")
print(f"NUM_SPEC_TOKENS={requested_spec}")
PY
}

echo "=== Checkpoint sanity ==="
BASELINE_INFO="$(checkpoint_json native "$BASELINE_DRAFT" "$BASELINE_SPEC")"
DFLASH_INFO="$(checkpoint_json trained "$DRAFT" "$DFLASH_SPEC")"
{
    printf '%s\n' "$BASELINE_INFO"
    echo
    printf '%s\n' "$DFLASH_INFO"
} | tee "$RUN_DIR/checkpoint_sanity.txt"

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
    local port="$1"
    local log="$2"
    local mode="$3"
    echo "Waiting for $mode server on :$port (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 80 log lines:"
            tail -n 80 "$log" || true
            if grep -qiE "unknown.*dflash|unsupported.*dflash|unrecognized.*dflash" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not have DFlash inference."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: vLLM speculative scheduling budget is too small."
                echo "           Lower MAX_NUM_SEQS/INFER_NUM_SPEC, or raise MAX_NUM_BATCHED_TOKENS."
            elif grep -qiE "m-?rope|multimodal.*spec|does not support" "$log"; then
                echo "DIAGNOSIS: this vLLM build likely lacks multimodal/M-RoPE spec support."
            fi
            return 1
        fi
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
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
    local port="$2"
    local log="$3"
    local draft_path="$4"
    local spec_tokens="$5"

    cleanup_server
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
        echo "ERROR: port $port already has a healthy server before starting $mode."
        echo "       Stop the old vLLM process or choose a different port."
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
        --port "$port"
    )

    if [ "$ENFORCE_EAGER" = "1" ]; then
        args+=(--enforce-eager)
    fi
    if [ -n "$ATTENTION_BACKEND" ]; then
        args+=(--attention-backend "$ATTENTION_BACKEND")
    fi

    args+=(
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$draft_path\",\"num_speculative_tokens\":$spec_tokens}"
    )

    echo
    echo "=== Starting $mode server ==="
    echo "  model:      $MODEL"
    echo "  draft:      $draft_path"
    echo "  num_spec:   $spec_tokens"
    echo "  port:       $port"
    echo "  devices:    $GPUS"
    echo "  max seqs:   $MAX_NUM_SEQS"
    echo "  media_root: $MEDIA_ROOT"
    echo "  log:        $log"
    printf '[cmd]'
    printf ' %q' env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}"
    echo

    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$port" "$log" "$mode"
}

run_client() {
    local mode="$1"
    local port="$2"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${port}/v1" \
        --data-jsonl "$MMSTAR_JSONL" \
        --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
        --summary-json "$RUN_DIR/${mode}_summary.json" \
        --num "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS"
}

parse_acceptance() {
    local mode="$1"
    local log="$2"
    local out="$RUN_DIR/${mode}_acceptance_from_log.txt"

    echo
    echo "=== ${mode} log acceptance parse ==="
    if ! python3 examples/evaluate/eval-guidellm/scripts/parse_logs.py "$log" \
        > "$out" 2>&1; then
        echo "(No parseable 'SpecDecoding metrics:' lines found. Showing spec-related log tail.)"
        grep -iE "spec|accept|draft" "$log" | tail -n 40 || true
    else
        cat "$out"
    fi
}

BASELINE_LOG="$RUN_DIR/baseline_vllm.log"
DFLASH_LOG="$RUN_DIR/dflash_vllm.log"

start_server native "$BASELINE_PORT" "$BASELINE_LOG" "$BASELINE_DRAFT" "$BASELINE_SPEC"
run_client baseline "$BASELINE_PORT"
cleanup_server
sleep 5

start_server trained "$DFLASH_PORT" "$DFLASH_LOG" "$DRAFT" "$DFLASH_SPEC"
run_client dflash "$DFLASH_PORT"
sleep 3

parse_acceptance native "$BASELINE_LOG"
parse_acceptance trained "$DFLASH_LOG"

echo
echo "=== Final comparison ==="
python3 - "$RUN_DIR/baseline_summary.json" "$RUN_DIR/dflash_summary.json" <<'PY'
import json
import sys
from pathlib import Path

native = json.loads(Path(sys.argv[1]).read_text())
trained = json.loads(Path(sys.argv[2]).read_text())

def fmt(value, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"

def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None

def spec(summary, key):
    return summary.get(key, 0) or 0

native_tps = native.get("output_tok_per_sec")
trained_tps = trained.get("output_tok_per_sec")
speedup = ratio(trained_tps, native_tps)

native_accept_rate = native.get("spec_token_acceptance_rate")
trained_accept_rate = trained.get("spec_token_acceptance_rate")
accept_rate_ratio = ratio(trained_accept_rate, native_accept_rate)

native_mean_accept = native.get("spec_mean_accepted_tokens_per_draft")
trained_mean_accept = trained.get("spec_mean_accepted_tokens_per_draft")
mean_accept_ratio = ratio(trained_mean_accept, native_mean_accept)

print(f"native completed:   {native['completed']}/{native['num_requested']}")
print(f"trained completed:  {trained['completed']}/{trained['num_requested']}")
print(f"native tok/s:       {fmt(native_tps)}")
print(f"trained tok/s:      {fmt(trained_tps)}")
print(f"trained/native:     {fmt(speedup)}")
print(f"native ref hit:     {fmt(native.get('reference_contains_rate'))}")
print(f"trained ref hit:    {fmt(trained.get('reference_contains_rate'))}")

print()
print("native DFlash spec metrics:")
print(f"  draft steps:      {fmt(spec(native, 'spec_draft_steps_total'))}")
print(f"  draft tokens:     {fmt(spec(native, 'spec_draft_tokens_total'))}")
print(f"  accepted:         {fmt(spec(native, 'spec_accepted_tokens_total'))}")
print(f"  token accept:     {fmt(native_accept_rate)}")
print(f"  first-pos accept: {fmt(native.get('spec_first_position_acceptance_rate'))}")
print(f"  mean accept/draft:{fmt(native_mean_accept):>8}")

print()
print("trained DFlash spec metrics:")
print(f"  draft steps:      {fmt(spec(trained, 'spec_draft_steps_total'))}")
print(f"  draft tokens:     {fmt(spec(trained, 'spec_draft_tokens_total'))}")
print(f"  accepted:         {fmt(spec(trained, 'spec_accepted_tokens_total'))}")
print(f"  token accept:     {fmt(trained_accept_rate)}")
print(f"  first-pos accept: {fmt(trained.get('spec_first_position_acceptance_rate'))}")
print(f"  mean accept/draft:{fmt(trained_mean_accept):>8}")

print()
print(f"accept-rate ratio:  {fmt(accept_rate_ratio)}")
print(f"mean-accept ratio:  {fmt(mean_accept_ratio)}")

print()
if native["completed"] == 0:
    print("VERDICT: BAD - native DFlash baseline accepted no requests.")
elif trained["completed"] == 0:
    print("VERDICT: BAD - trained DFlash candidate accepted no requests.")
elif spec(native, "spec_draft_tokens_total") <= 0:
    print("VERDICT: BAD - native baseline emitted no DFlash draft metrics.")
    print("         Check BASELINE_DRAFT, baseline_vllm.log, and vLLM DFlash support.")
elif spec(trained, "spec_draft_tokens_total") <= 0:
    print("VERDICT: BAD - trained candidate emitted no DFlash draft metrics.")
    print("         Check DRAFT, dflash_vllm.log, and vLLM DFlash support.")
elif spec(trained, "spec_accepted_tokens_total") <= 0:
    print("VERDICT: SUSPICIOUS - trained candidate drafted but accepted no tokens.")
elif speedup is not None and speedup > 1.0:
    print("VERDICT: IMPROVED - trained DFlash is faster than native DFlash.")
elif mean_accept_ratio is not None and mean_accept_ratio > 1.0:
    print("VERDICT: MIXED - trained DFlash accepts more, but throughput did not improve.")
else:
    print("VERDICT: NOT IMPROVED - trained DFlash did not beat native DFlash.")
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
