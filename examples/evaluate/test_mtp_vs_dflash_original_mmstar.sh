#!/bin/bash
# End-to-end MMStar baseline: native Qwen3.5 MTP vs original/raw DFlash.
#
# This script runs two vLLM servers against the same MMStar prompts:
#   1) native/built-in Qwen3.5 MTP
#   2) original downloaded DFlash draft
#
# By default it runs single-image MMStar prompts. Set IMAGES_PER_PROMPT=10 to
# reuse the grouped 10-image client.
#
# Usage:
#   bash examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh
#
# Common overrides:
#   MTP_SPEC=3 \
#   DFLASH_SPEC=7 \
#   NUM_PROMPTS=128 \
#   bash examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh
#
#   MTP_SPEC=3 \
#   DFLASH_SPEC=7 \
#   IMAGES_PER_PROMPT=10 \
#   NUM_GROUPS=16 \
#   bash examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh

set -euo pipefail

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
DFLASH_DRAFT="${DFLASH_DRAFT:-$DEFAULT_ROOT/Qwen3.5-9B-DFlash}"
MMSTAR_SRC="${MMSTAR_SRC:-$DEFAULT_MMSTAR_ROOT/mmstar_answers.json}"
MMSTAR_JSONL="${MMSTAR_JSONL:-$REPO_ROOT/data/mmstar/mmstar_eval.jsonl}"
MMSTAR_IMAGE_DIR="${MMSTAR_IMAGE_DIR:-$REPO_ROOT/data/mmstar/images}"
MEDIA_ROOT="${MEDIA_ROOT:-$DEFAULT_MMSTAR_ROOT/images}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_mtp_vs_dflash_original}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
mkdir -p "$RUN_DIR"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
MTP_PORT="${MTP_PORT:-8100}"
DFLASH_PORT="${DFLASH_PORT:-8101}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-1}"

if ! [[ "$IMAGES_PER_PROMPT" =~ ^[0-9]+$ ]] || [ "$IMAGES_PER_PROMPT" -le 0 ]; then
    echo "[fatal] IMAGES_PER_PROMPT must be a positive integer, got: $IMAGES_PER_PROMPT"
    exit 1
fi

if [ "$IMAGES_PER_PROMPT" -gt 1 ]; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_TOKENS="${MAX_TOKENS:-512}"
else
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_TOKENS="${MAX_TOKENS:-128}"
fi

NUM_PROMPTS="${NUM_PROMPTS:-128}"
NUM_GROUPS="${NUM_GROUPS:-16}"
MTP_METHOD="${MTP_METHOD:-qwen3_5_mtp}"
MTP_SPEC="${MTP_SPEC:-3}"
DFLASH_SPEC="${DFLASH_SPEC:-${INFER_NUM_SPEC:-7}}"

if ! [[ "$MTP_SPEC" =~ ^[0-9]+$ ]] || [ "$MTP_SPEC" -le 0 ]; then
    echo "[fatal] MTP_SPEC must be a positive integer, got: $MTP_SPEC"
    exit 1
fi
if ! [[ "$DFLASH_SPEC" =~ ^[0-9]+$ ]] || [ "$DFLASH_SPEC" -le 0 ]; then
    echo "[fatal] DFLASH_SPEC must be a positive integer, got: $DFLASH_SPEC"
    exit 1
fi

MAX_SPEC_TOKENS="$((MTP_SPEC > DFLASH_SPEC ? MTP_SPEC : DFLASH_SPEC))"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_SPEC_TOKENS))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mmstar-baseline}"
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

check_paths() {
    [ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
    [ -d "$DFLASH_DRAFT" ] || { echo "[fatal] DFLASH_DRAFT not found: $DFLASH_DRAFT"; exit 1; }
    [ -s "$MMSTAR_SRC" ] || { echo "[fatal] MMSTAR_SRC not found: $MMSTAR_SRC"; exit 1; }
    [ -d "$MEDIA_ROOT" ] || { echo "[fatal] MEDIA_ROOT not found: $MEDIA_ROOT"; exit 1; }
}

dflash_sanity() {
    python3 - "$MODEL" "$DFLASH_DRAFT" "$DFLASH_SPEC" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
draft = Path(sys.argv[2])
requested_spec = int(sys.argv[3])

cfg_path = draft / "config.json"
weights_path = draft / "model.safetensors"
if not cfg_path.exists():
    raise SystemExit(f"[fatal] missing config.json under original DFlash draft: {draft}")
if not weights_path.exists():
    raise SystemExit(f"[fatal] missing model.safetensors under original DFlash draft: {draft}")

cfg = json.loads(cfg_path.read_text())
is_speculators_dflash = cfg.get("speculators_model_type") == "dflash"
is_raw_dflash = bool(cfg.get("dflash_config")) and bool(cfg.get("block_size"))
if not is_speculators_dflash and not is_raw_dflash:
    raise SystemExit(
        "[fatal] DFLASH_DRAFT is not an accepted DFlash model: "
        f"{draft}\n        speculators_model_type={cfg.get('speculators_model_type')!r}, "
        f"has_dflash_config={bool(cfg.get('dflash_config'))}"
    )

block = int(cfg.get("block_size", 0))
max_spec = block - 1
if block <= 0:
    raise SystemExit(f"[fatal] invalid DFlash block_size={block}: {draft}")
if requested_spec <= 0:
    raise SystemExit(f"[fatal] DFLASH_SPEC must be positive: {requested_spec}")
if requested_spec > max_spec:
    raise SystemExit(
        f"[fatal] DFlash block_size={block} only supports "
        f"num_speculative_tokens <= {max_spec}, requested {requested_spec}."
    )

aux = cfg.get("aux_hidden_state_layer_ids") or cfg.get("dflash_config", {}).get(
    "target_layer_ids"
) or []
verifier = (
    cfg.get("speculators_config", {})
    .get("verifier", {})
    .get("name_or_path")
)
if verifier and verifier != model:
    print(f"[warn] DFlash verifier path differs: {verifier} != {model}")

print("Original DFlash sanity OK")
print(f"  draft:      {draft}")
print(f"  format:     {'speculators' if is_speculators_dflash else 'raw'}")
print(f"  block_size: {block}")
print(f"  max_spec:   {max_spec}")
print(f"  infer_spec: {requested_spec}")
print(f"  aux_layers: {aux}")
PY
}

write_run_config() {
    python3 - \
        "$RUN_DIR/run_config.json" \
        "$MODEL" \
        "$DFLASH_DRAFT" \
        "$MMSTAR_SRC" \
        "$MMSTAR_JSONL" \
        "$MEDIA_ROOT" \
        "$GPUS" \
        "$TP" \
        "$MTP_METHOD" \
        "$MTP_SPEC" \
        "$DFLASH_SPEC" \
        "$IMAGES_PER_PROMPT" \
        "$NUM_PROMPTS" \
        "$NUM_GROUPS" \
        "$MAX_TOKENS" \
        "$MAX_MODEL_LEN" \
        "$MAX_NUM_BATCHED_TOKENS" \
        "$MAX_NUM_SEQS" \
        "$DISABLE_CHUNKED_PREFILL" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path,
    model,
    dflash_draft,
    mmstar_src,
    mmstar_jsonl,
    media_root,
    gpus,
    tp,
    mtp_method,
    mtp_spec,
    dflash_spec,
    images_per_prompt,
    num_prompts,
    num_groups,
    max_tokens,
    max_model_len,
    max_num_batched_tokens,
    max_num_seqs,
    disable_chunked_prefill,
) = sys.argv[1:]

config = {
    "model": model,
    "dflash_draft": dflash_draft,
    "mmstar_src": mmstar_src,
    "mmstar_jsonl": mmstar_jsonl,
    "media_root": media_root,
    "gpus": gpus,
    "tp": int(tp),
    "mtp_method": mtp_method,
    "mtp_spec": int(mtp_spec),
    "dflash_spec": int(dflash_spec),
    "images_per_prompt": int(images_per_prompt),
    "num_prompts": int(num_prompts),
    "num_groups": int(num_groups),
    "max_tokens": int(max_tokens),
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "max_num_seqs": int(max_num_seqs),
    "disable_chunked_prefill": disable_chunked_prefill == "1",
}
Path(out_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
}

prepare_mmstar() {
    echo "=== Preparing MMStar conversations jsonl ==="
    if [ -s "$MMSTAR_JSONL" ]; then
        echo "reuse existing $MMSTAR_JSONL"
    else
        python3 scripts/mmstar_to_jsonl.py \
            --mmstar "$MMSTAR_SRC" \
            --out-jsonl "$MMSTAR_JSONL" \
            --image-dir "$MMSTAR_IMAGE_DIR"
    fi
}

wait_for_server() {
    local port="$1"
    local log="$2"
    local mode="$3"
    echo "Waiting for $mode server on :$port (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*(mtp|qwen3_5_mtp)|unsupported.*(mtp|qwen3_5_mtp)|unrecognized.*(mtp|qwen3_5_mtp)" "$log"; then
                echo "DIAGNOSIS: this vLLM build likely lacks native Qwen3.5 MTP support."
                echo "           Try MTP_METHOD=mtp if your build uses the generic method name."
            elif grep -qiE "unknown.*dflash|unsupported.*dflash|unrecognized.*dflash" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not have DFlash inference."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: vLLM speculative scheduling budget is too small."
                echo "           Lower MAX_NUM_SEQS/*_SPEC, or raise MAX_NUM_BATCHED_TOKENS."
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
    echo "ERROR: timed out waiting for $mode server. Last 100 log lines:"
    tail -n 100 "$log" || true
    return 1
}

start_server() {
    local mode="$1"
    local port="$2"
    local log="$3"
    local spec_config="$4"

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
        --limit-mm-per-prompt "{\"image\":$IMAGES_PER_PROMPT}"
        --generation-config vllm
        --host 0.0.0.0
        --port "$port"
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

    args+=(--speculative-config "$spec_config")

    echo
    echo "=== Starting $mode server ==="
    echo "  model:             $MODEL"
    echo "  spec_config:       $spec_config"
    echo "  images_per_prompt: $IMAGES_PER_PROMPT"
    echo "  max_model_len:     $MAX_MODEL_LEN"
    echo "  max batched toks:  $MAX_NUM_BATCHED_TOKENS"
    echo "  max seqs:          $MAX_NUM_SEQS"
    echo "  port:              $port"
    echo "  devices:           $GPUS"
    echo "  media_root:        $MEDIA_ROOT"
    echo "  log:               $log"
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

    if [ "$IMAGES_PER_PROMPT" -gt 1 ]; then
        python3 examples/evaluate/mmstar_10image_client.py \
            --endpoint "http://localhost:${port}/v1" \
            --data-jsonl "$MMSTAR_JSONL" \
            --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
            --summary-json "$RUN_DIR/${mode}_summary.json" \
            --num-groups "$NUM_GROUPS" \
            --images-per-prompt "$IMAGES_PER_PROMPT" \
            --max-tokens "$MAX_TOKENS"
    else
        python3 examples/evaluate/mmstar_weight_client.py \
            --endpoint "http://localhost:${port}/v1" \
            --data-jsonl "$MMSTAR_JSONL" \
            --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
            --summary-json "$RUN_DIR/${mode}_summary.json" \
            --num "$NUM_PROMPTS" \
            --max-tokens "$MAX_TOKENS"
    fi
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
        grep -iE "spec|accept|draft|mtp|dflash" "$log" | tail -n 60 || true
    else
        cat "$out"
    fi
}

final_comparison() {
    python3 - "$RUN_DIR/mtp_summary.json" "$RUN_DIR/dflash_original_summary.json" <<'PY'
import json
import sys
from pathlib import Path

mtp = json.loads(Path(sys.argv[1]).read_text())
dflash = json.loads(Path(sys.argv[2]).read_text())

def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)

def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None

def spec(summary, key):
    return summary.get(key, 0) or 0

def requested(summary):
    return summary.get("num_requested", summary.get("num_groups_requested"))

mtp_tps = mtp.get("output_tok_per_sec")
dflash_tps = dflash.get("output_tok_per_sec")
dflash_over_mtp = ratio(dflash_tps, mtp_tps)
mtp_over_dflash = ratio(mtp_tps, dflash_tps)

mtp_accept_rate = mtp.get("spec_token_acceptance_rate")
dflash_accept_rate = dflash.get("spec_token_acceptance_rate")
accept_rate_ratio = ratio(dflash_accept_rate, mtp_accept_rate)

mtp_mean_accept = mtp.get("spec_mean_accepted_tokens_per_draft")
dflash_mean_accept = dflash.get("spec_mean_accepted_tokens_per_draft")
mean_accept_ratio = ratio(dflash_mean_accept, mtp_mean_accept)

print(f"MTP completed:              {mtp['completed']}/{requested(mtp)}")
print(f"original DFlash completed:  {dflash['completed']}/{requested(dflash)}")
print(f"MTP tok/s:                  {fmt(mtp_tps)}")
print(f"original DFlash tok/s:      {fmt(dflash_tps)}")
print(f"DFlash/MTP tok/s:           {fmt(dflash_over_mtp)}")
print(f"MTP/DFlash tok/s:           {fmt(mtp_over_dflash)}")

if mtp.get("images_per_sec") is not None or dflash.get("images_per_sec") is not None:
    mtp_img_s = mtp.get("images_per_sec")
    dflash_img_s = dflash.get("images_per_sec")
    print(f"MTP images/s:               {fmt(mtp_img_s)}")
    print(f"original DFlash images/s:   {fmt(dflash_img_s)}")
    print(f"DFlash/MTP images/s:        {fmt(ratio(dflash_img_s, mtp_img_s))}")

print(f"MTP ref hit:                {fmt(mtp.get('reference_contains_rate'))}")
print(f"original DFlash ref hit:    {fmt(dflash.get('reference_contains_rate'))}")

print()
print("MTP spec metrics:")
print(f"  draft steps:              {fmt(spec(mtp, 'spec_draft_steps_total'))}")
print(f"  draft tokens:             {fmt(spec(mtp, 'spec_draft_tokens_total'))}")
print(f"  accepted:                 {fmt(spec(mtp, 'spec_accepted_tokens_total'))}")
print(f"  token accept:             {fmt(mtp_accept_rate)}")
print(f"  first-pos accept:         {fmt(mtp.get('spec_first_position_acceptance_rate'))}")
print(f"  mean accept/draft:        {fmt(mtp_mean_accept)}")

print()
print("original DFlash spec metrics:")
print(f"  draft steps:              {fmt(spec(dflash, 'spec_draft_steps_total'))}")
print(f"  draft tokens:             {fmt(spec(dflash, 'spec_draft_tokens_total'))}")
print(f"  accepted:                 {fmt(spec(dflash, 'spec_accepted_tokens_total'))}")
print(f"  token accept:             {fmt(dflash_accept_rate)}")
print(f"  first-pos accept:         {fmt(dflash.get('spec_first_position_acceptance_rate'))}")
print(f"  mean accept/draft:        {fmt(dflash_mean_accept)}")

print()
print(f"DFlash/MTP token-accept:    {fmt(accept_rate_ratio)}")
print(f"DFlash/MTP mean-accept:     {fmt(mean_accept_ratio)}")

print()
if mtp["completed"] == 0:
    print("VERDICT: BAD - MTP baseline accepted no requests.")
elif dflash["completed"] == 0:
    print("VERDICT: BAD - original DFlash accepted no requests.")
elif spec(mtp, "spec_draft_tokens_total") <= 0:
    print("VERDICT: BAD - MTP emitted no speculative metrics.")
    print("         Check MTP_METHOD/MTP_SPEC and vLLM MTP support.")
elif spec(dflash, "spec_draft_tokens_total") <= 0:
    print("VERDICT: BAD - original DFlash emitted no speculative metrics.")
    print("         Check DFLASH_DRAFT and vLLM DFlash support.")
elif dflash_over_mtp is not None and dflash_over_mtp > 1.0:
    print("VERDICT: original DFlash is faster than MTP on this MMStar slice.")
elif mtp_over_dflash is not None and mtp_over_dflash > 1.0:
    print("VERDICT: MTP is faster than original DFlash on this MMStar slice.")
else:
    print("VERDICT: throughput tie or inconclusive.")
PY
}

check_paths

echo "=== MTP vs original DFlash MMStar baseline ==="
echo "  model:             $MODEL"
echo "  dflash_draft:      $DFLASH_DRAFT"
echo "  mtp_method:        $MTP_METHOD"
echo "  mtp_spec:          $MTP_SPEC"
echo "  dflash_spec:       $DFLASH_SPEC"
echo "  images_per_prompt: $IMAGES_PER_PROMPT"
echo "  media_root:        $MEDIA_ROOT"
echo "  run_dir:           $RUN_DIR"

dflash_sanity | tee "$RUN_DIR/dflash_sanity.txt"
write_run_config
prepare_mmstar

MTP_LOG="$RUN_DIR/mtp_vllm.log"
DFLASH_LOG="$RUN_DIR/dflash_original_vllm.log"
MTP_SPEC_CONFIG="{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$MTP_SPEC,\"enforce_eager\":true}"
DFLASH_SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT\",\"num_speculative_tokens\":$DFLASH_SPEC}"

start_server mtp "$MTP_PORT" "$MTP_LOG" "$MTP_SPEC_CONFIG"
run_client mtp "$MTP_PORT"
cleanup_server
sleep 5

start_server dflash_original "$DFLASH_PORT" "$DFLASH_LOG" "$DFLASH_SPEC_CONFIG"
run_client dflash_original "$DFLASH_PORT"
sleep 3

parse_acceptance mtp "$MTP_LOG"
parse_acceptance dflash_original "$DFLASH_LOG"

echo
echo "=== Final MTP vs original DFlash comparison ==="
final_comparison | tee "$RUN_DIR/final_comparison.txt"

echo
echo "Artifacts:"
echo "  $RUN_DIR"
