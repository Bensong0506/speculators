#!/bin/bash
# End-to-end MMStar smoke test for a trained Qwen3.5-9B DFlash checkpoint.
#
# This script answers a narrow question: "does my trained draft checkpoint load,
# run as a DFlash speculator, and produce non-trivial acceptance on held-out
# multimodal prompts?" It runs two servers, baseline then dflash, against the
# same MMStar prompts and writes all outputs under RUN_DIR.
#
# Usage:
#   bash examples/evaluate/test_dflash_mmstar_weights.sh
#
# Common overrides:
#   DRAFT=/path/to/checkpoint_best \
#   MMSTAR_SRC=/data/wenxuan/mmstar/mmstar_answers.json \
#   NUM_PROMPTS=128 \
#   bash examples/evaluate/test_dflash_mmstar_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
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
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
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
    python3 - "$MODEL" "$DRAFT" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

model = sys.argv[1]
draft = Path(sys.argv[2])
if not draft.exists():
    raise SystemExit(f"[fatal] DRAFT does not exist: {draft}")

cfg_path = draft / "config.json"
weights_path = draft / "model.safetensors"
if not cfg_path.exists():
    raise SystemExit(f"[fatal] missing config.json under DRAFT: {draft}")
if not weights_path.exists():
    raise SystemExit(f"[fatal] missing model.safetensors under DRAFT: {draft}")

cfg = json.loads(cfg_path.read_text())
if cfg.get("speculators_model_type") != "dflash":
    raise SystemExit(
        "[fatal] checkpoint is not a speculators DFlash model: "
        f"speculators_model_type={cfg.get('speculators_model_type')!r}"
    )

aux = cfg.get("aux_hidden_state_layer_ids") or cfg.get("dflash_config", {}).get(
    "target_layer_ids"
) or []
hidden = cfg.get("transformer_layer_config", {}).get("hidden_size") or cfg.get(
    "hidden_size"
)
block = int(cfg.get("block_size", 0))
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

if verifier and verifier != model:
    print(f"[warn] checkpoint verifier path differs: {verifier} != {model}")

print("Checkpoint sanity OK")
print(f"  draft:      {draft}")
print(f"  block_size: {block}")
print(f"  num_spec:   {max(0, block - 1)}")
print(f"  aux_layers: {aux}")
print(f"  hidden:     {hidden}")
print(f"  fc.weight:  {fc_shape}")
print(f"  layers.*:   {len(layer_keys)} tensors")
print(f"NUM_SPEC_TOKENS={max(1, block - 1)}")
PY
}

echo "=== Checkpoint sanity ==="
CHECKPOINT_INFO="$(checkpoint_json)"
printf '%s\n' "$CHECKPOINT_INFO" | tee "$RUN_DIR/checkpoint_sanity.txt"
DFLASH_SPEC="${DFLASH_SPEC:-$(printf '%s\n' "$CHECKPOINT_INFO" | awk -F= '/^NUM_SPEC_TOKENS=/{print $2}')}"

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
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "$mode server ready."
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 80 log lines:"
            tail -n 80 "$log" || true
            if grep -qiE "unknown.*dflash|method.*dflash" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not have DFlash inference."
            elif grep -qiE "m-?rope|multimodal.*spec|does not support" "$log"; then
                echo "DIAGNOSIS: this vLLM build likely lacks multimodal/M-RoPE spec support."
            fi
            return 1
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

    cleanup_server

    local args=(
        vllm serve "$MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --seed 42
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
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

    if [ "$mode" = "dflash" ]; then
        args+=(
            --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DFLASH_SPEC}"
        )
    fi

    echo
    echo "=== Starting $mode server ==="
    echo "  model:      $MODEL"
    echo "  draft:      ${DRAFT:-none}"
    echo "  port:       $port"
    echo "  devices:    $GPUS"
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

BASELINE_LOG="$RUN_DIR/baseline_vllm.log"
DFLASH_LOG="$RUN_DIR/dflash_vllm.log"

start_server baseline "$BASELINE_PORT" "$BASELINE_LOG"
run_client baseline "$BASELINE_PORT"
cleanup_server
sleep 5

start_server dflash "$DFLASH_PORT" "$DFLASH_LOG"
run_client dflash "$DFLASH_PORT"
sleep 3

echo
echo "=== DFlash log acceptance parse ==="
if ! python3 examples/evaluate/eval-guidellm/scripts/parse_logs.py "$DFLASH_LOG" \
    > "$RUN_DIR/dflash_acceptance_from_log.txt" 2>&1; then
    echo "(No parseable 'SpecDecoding metrics:' lines found. Showing spec-related log tail.)"
    grep -iE "spec|accept|draft" "$DFLASH_LOG" | tail -n 40 || true
else
    cat "$RUN_DIR/dflash_acceptance_from_log.txt"
fi

echo
echo "=== Final comparison ==="
python3 - "$RUN_DIR/baseline_summary.json" "$RUN_DIR/dflash_summary.json" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text())
dflash = json.loads(Path(sys.argv[2]).read_text())

def fmt(value, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"

b_tps = baseline.get("output_tok_per_sec")
d_tps = dflash.get("output_tok_per_sec")
speedup = d_tps / b_tps if b_tps and d_tps else None

print(f"baseline completed: {baseline['completed']}/{baseline['num_requested']}")
print(f"dflash completed:   {dflash['completed']}/{dflash['num_requested']}")
print(f"baseline tok/s:     {fmt(b_tps)}")
print(f"dflash tok/s:       {fmt(d_tps)}")
print(f"speedup:            {fmt(speedup)}")
print(f"baseline ref hit:   {fmt(baseline.get('reference_contains_rate'))}")
print(f"dflash ref hit:     {fmt(dflash.get('reference_contains_rate'))}")
print(f"spec draft steps:   {fmt(dflash.get('spec_draft_steps_total'))}")
print(f"spec draft tokens:  {fmt(dflash.get('spec_draft_tokens_total'))}")
print(f"spec accepted:      {fmt(dflash.get('spec_accepted_tokens_total'))}")
print(f"token accept rate:  {fmt(dflash.get('spec_token_acceptance_rate'))}")
print(f"first-pos accept:   {fmt(dflash.get('spec_first_position_acceptance_rate'))}")
print(f"mean accept/draft:  {fmt(dflash.get('spec_mean_accepted_tokens_per_draft'))}")

print()
if dflash["completed"] == 0:
    print("VERDICT: BAD - dflash server accepted no requests.")
elif dflash.get("spec_draft_tokens_total", 0) <= 0:
    print("VERDICT: INCONCLUSIVE - dflash served, but /metrics did not show draft counters.")
    print("         Check dflash_vllm.log and dflash_acceptance_from_log.txt.")
elif dflash.get("spec_accepted_tokens_total", 0) <= 0:
    print("VERDICT: SUSPICIOUS - draft counters advanced but no accepted draft tokens.")
elif speedup is not None and speedup < 1.0:
    print("VERDICT: LOADS BUT NOT EFFECTIVE - throughput regressed.")
    print("         Draft acceptance is too low to pay for DFlash overhead.")
else:
    print("VERDICT: PASS - checkpoint loads and improves throughput on this run.")
    print("         Use speedup/ref-hit as quality signals, not as strict pass/fail.")
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
