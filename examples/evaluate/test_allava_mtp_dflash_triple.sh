#!/bin/bash
# End-to-end ALLaVA comparison:
#   native MTP vs original/raw DFlash vs trained DFlash.
#
# Defaults assume the A800 box uses /home/wexuan paths.
#
# Usage:
#   bash examples/evaluate/test_allava_mtp_dflash_triple.sh
#
# Common overrides:
#   DRAFT=/home/wexuan/speculators/output/.../checkpoints/14 \
#   ALLAVA_EVAL_SAMPLES=256 \
#   ALLAVA_SKIP_SAMPLES=100000 \
#   MTP_SPEC=7 \
#   DFLASH_SPEC=7 \
#   bash examples/evaluate/test_allava_mtp_dflash_triple.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/home/wexuan/Qwen3.5-9B}"
ORIGINAL_DFLASH="${ORIGINAL_DFLASH:-/home/wexuan/Qwen3.5-9B-DFlash}"
TRAINED_DRAFT_FIND_ROOT="${TRAINED_DRAFT_FIND_ROOT:-/home/wexuan/speculators/output}"

ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wexuan/ALLaVA-4V}"
ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"
ALLAVA_EVAL_SAMPLES="${ALLAVA_EVAL_SAMPLES:-128}"
ALLAVA_SKIP_SAMPLES="${ALLAVA_SKIP_SAMPLES:-100000}"
ALLAVA_STRIDE="${ALLAVA_STRIDE:-1}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_eval_skip${ALLAVA_SKIP_SAMPLES}_n${ALLAVA_EVAL_SAMPLES}.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-$ALLAVA_IMAGE_ROOT}"
FORCE_ALLAVA_EVAL_JSONL="${FORCE_ALLAVA_EVAL_JSONL:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/allava_triple_tests}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
mkdir -p "$RUN_DIR"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MTP_SPEC="${MTP_SPEC:-7}"
DFLASH_SPEC="${DFLASH_SPEC:-7}"
MAX_SPEC_TOKENS="$((MTP_SPEC > DFLASH_SPEC ? MTP_SPEC : DFLASH_SPEC))"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_SPEC_TOKENS))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-allava-triple-test}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MTP_ENFORCE_EAGER="${MTP_ENFORCE_EAGER:-$ENFORCE_EAGER}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
DTYPE="${DTYPE:-bfloat16}"

select_default_trained_draft() {
    python3 - "$TRAINED_DRAFT_FIND_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"[fatal] TRAINED_DRAFT_FIND_ROOT does not exist: {root}")

best = None
for results in sorted(
    root.glob("mmstar_checkpoint_sweeps/*/results.jsonl"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
):
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("exit_status") != 0:
            continue
        score = row.get("trained_tok_s")
        checkpoint = row.get("checkpoint")
        if score is None or not checkpoint:
            continue
        if best is None or score > best[0]:
            best = (score, Path(checkpoint), str(results))
    if best is not None and best[1].exists():
        print(best[1])
        print(
            f"[info] selected DRAFT from {best[2]} by trained_tok_s={best[0]}",
            file=sys.stderr,
        )
        raise SystemExit

candidates = []
for cfg in root.rglob("config.json"):
    ckpt = cfg.parent
    if not (ckpt / "model.safetensors").exists():
        continue
    text = cfg.read_text()
    if '"speculators_model_type"' not in text or '"dflash"' not in text:
        continue
    name = ckpt.name
    rank = 0
    if name == "checkpoint_best":
        rank = 2
    elif re.fullmatch(r"\d+", name):
        rank = 1
    else:
        continue
    candidates.append((rank, ckpt.stat().st_mtime, ckpt))

if not candidates:
    raise SystemExit(
        "[fatal] no trained speculators-format DFlash checkpoint found. "
        "Pass DRAFT=/path/to/checkpoint."
    )

rank, _, ckpt = max(candidates)
print(ckpt)
print(
    f"[info] selected {'checkpoint_best' if rank == 2 else 'latest numeric checkpoint'} fallback",
    file=sys.stderr,
)
PY
}

if [ -z "${DRAFT:-}" ]; then
    DRAFT="$(select_default_trained_draft)"
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
    raise SystemExit(f"[fatal] missing model.safetensors under {label} DFlash draft: {draft}")

cfg = json.loads(cfg_path.read_text())
speculators_type = cfg.get("speculators_model_type")
is_speculators_dflash = speculators_type == "dflash"
is_raw_dflash = bool(cfg.get("dflash_config")) and bool(cfg.get("block_size"))
if not is_speculators_dflash and not (label == "original" and is_raw_dflash):
    raise SystemExit(
        f"[fatal] {label} checkpoint is not an accepted DFlash model: {draft}\n"
        f"        speculators_model_type={speculators_type!r}, "
        f"has_dflash_config={bool(cfg.get('dflash_config'))}."
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
        raise SystemExit(f"[fatal] missing expected tensor(s): {missing}")
    fc_shape = tuple(handle.get_tensor("fc.weight").shape)
    layer_keys = [key for key in keys if key.startswith("layers.")]

if hidden and aux and fc_shape != (hidden, hidden * len(aux)):
    raise SystemExit(
        "fc.weight shape does not match aux hidden-state recipe: "
        f"fc_shape={fc_shape}, hidden={hidden}, aux={aux}"
    )
if not layer_keys:
    raise SystemExit("[fatal] no draft layer weights found under layers.*")
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
PY
}

echo "=== DFlash checkpoint sanity ==="
ORIGINAL_INFO="$(checkpoint_json original "$ORIGINAL_DFLASH" "$DFLASH_SPEC")"
TRAINED_INFO="$(checkpoint_json trained "$DRAFT" "$DFLASH_SPEC")"
{
    printf '%s\n' "$ORIGINAL_INFO"
    echo
    printf '%s\n' "$TRAINED_INFO"
} | tee "$RUN_DIR/checkpoint_sanity.txt"

echo "=== Preparing ALLaVA eval jsonl ==="
if [ "$FORCE_ALLAVA_EVAL_JSONL" = "1" ] || [ ! -s "$ALLAVA_JSONL" ]; then
    INPUT_ARGS=()
    for src in $ALLAVA_INPUTS; do
        INPUT_ARGS+=(--in "$src")
    done
    python3 scripts/allava_eval_to_jsonl.py \
        "${INPUT_ARGS[@]}" \
        --image-root "$ALLAVA_IMAGE_ROOT" \
        --out-jsonl "$ALLAVA_JSONL" \
        --max-samples "$ALLAVA_EVAL_SAMPLES" \
        --skip-samples "$ALLAVA_SKIP_SAMPLES" \
        --stride "$ALLAVA_STRIDE"
else
    echo "reuse existing $ALLAVA_JSONL"
fi

python3 - "$RUN_DIR/run_config.json" <<PY
import json
from pathlib import Path

config = {
    "model": "$MODEL",
    "original_dflash": "$ORIGINAL_DFLASH",
    "trained_dflash": "$DRAFT",
    "allava_jsonl": "$ALLAVA_JSONL",
    "allava_image_root": "$ALLAVA_IMAGE_ROOT",
    "allava_inputs": "$ALLAVA_INPUTS",
    "media_root": "$MEDIA_ROOT",
    "gpus": "$GPUS",
    "tp": int("$TP"),
    "mtp_spec": int("$MTP_SPEC"),
    "dflash_spec": int("$DFLASH_SPEC"),
    "allava_eval_samples": int("$ALLAVA_EVAL_SAMPLES"),
    "allava_skip_samples": int("$ALLAVA_SKIP_SAMPLES"),
    "allava_stride": int("$ALLAVA_STRIDE"),
    "max_tokens": int("$MAX_TOKENS"),
    "max_model_len": int("$MAX_MODEL_LEN"),
    "max_num_batched_tokens": int("$MAX_NUM_BATCHED_TOKENS"),
    "max_num_seqs": int("$MAX_NUM_SEQS"),
    "enforce_eager": "$ENFORCE_EAGER" == "1",
    "attention_backend": "$ATTENTION_BACKEND",
    "dtype": "$DTYPE",
}
Path("$RUN_DIR/run_config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
PY

wait_for_server() {
    local port="$1"
    local log="$2"
    local mode="$3"
    echo "Waiting for $mode server on :$port (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*qwen3_5_mtp|unknown.*mtp|unsupported.*mtp" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not support native Qwen3.5 MTP."
            elif grep -qiE "unknown.*dflash|unsupported.*dflash|unrecognized.*dflash" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not support DFlash inference."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: speculative scheduling budget is too small."
                echo "           Lower MAX_NUM_SEQS/spec tokens, or raise MAX_NUM_BATCHED_TOKENS."
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
    local log="$2"
    local spec_config="$3"

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting $mode."
        echo "       Stop the old vLLM process or choose PORT=...."
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
        --dtype "$DTYPE"
        --allowed-local-media-path "$MEDIA_ROOT"
        --limit-mm-per-prompt '{"image":1}'
        --generation-config vllm
        --host 0.0.0.0
        --port "$PORT"
    )

    if [ "$ENFORCE_EAGER" = "1" ]; then
        args+=(--enforce-eager)
    fi
    if [ -n "$ATTENTION_BACKEND" ]; then
        args+=(--attention-backend "$ATTENTION_BACKEND")
    fi
    args+=(--speculative-config "$spec_config")

    echo
    echo "=== Starting $mode server ==="
    echo "  model:             $MODEL"
    echo "  spec_config:       $spec_config"
    echo "  port:              $PORT"
    echo "  devices:           $GPUS"
    echo "  max_model_len:     $MAX_MODEL_LEN"
    echo "  max batched toks:  $MAX_NUM_BATCHED_TOKENS"
    echo "  max seqs:          $MAX_NUM_SEQS"
    echo "  media_root:        $MEDIA_ROOT"
    echo "  attention_backend: ${ATTENTION_BACKEND:-default}"
    echo "  enforce_eager:     $ENFORCE_EAGER"
    echo "  log:               $log"
    printf '[cmd]'
    printf ' %q' env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}"
    echo

    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$PORT" "$log" "$mode"
}

run_client() {
    local mode="$1"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${PORT}/v1" \
        --data-jsonl "$ALLAVA_JSONL" \
        --out-jsonl "$RUN_DIR/${mode}_responses.jsonl" \
        --summary-json "$RUN_DIR/${mode}_summary.json" \
        --num "$ALLAVA_EVAL_SAMPLES" \
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
        grep -iE "spec|accept|draft|mtp|dflash" "$log" | tail -n 60 || true
    else
        cat "$out"
    fi
}

mtp_eager_json=false
if [ "$MTP_ENFORCE_EAGER" = "1" ]; then
    mtp_eager_json=true
fi

MTP_LOG="$RUN_DIR/mtp_vllm.log"
ORIGINAL_LOG="$RUN_DIR/original_dflash_vllm.log"
TRAINED_LOG="$RUN_DIR/trained_dflash_vllm.log"

MTP_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP_SPEC,\"enforce_eager\":$mtp_eager_json}"
ORIGINAL_CONFIG="{\"method\":\"dflash\",\"model\":\"$ORIGINAL_DFLASH\",\"num_speculative_tokens\":$DFLASH_SPEC}"
TRAINED_CONFIG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DFLASH_SPEC}"

start_server mtp "$MTP_LOG" "$MTP_CONFIG"
run_client mtp
cleanup_server
sleep 5

start_server original_dflash "$ORIGINAL_LOG" "$ORIGINAL_CONFIG"
run_client original_dflash
cleanup_server
sleep 5

start_server trained_dflash "$TRAINED_LOG" "$TRAINED_CONFIG"
run_client trained_dflash
sleep 3

parse_acceptance mtp "$MTP_LOG"
parse_acceptance original_dflash "$ORIGINAL_LOG"
parse_acceptance trained_dflash "$TRAINED_LOG"

echo
echo "=== Final ALLaVA triple comparison ==="
python3 - \
    "$RUN_DIR/mtp_summary.json" \
    "$RUN_DIR/original_dflash_summary.json" \
    "$RUN_DIR/trained_dflash_summary.json" <<'PY'
import json
import sys
from pathlib import Path

mtp = json.loads(Path(sys.argv[1]).read_text())
original = json.loads(Path(sys.argv[2]).read_text())
trained = json.loads(Path(sys.argv[3]).read_text())

def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)

def ratio(num, den):
    return num / den if num is not None and den else None

def spec(summary, key):
    return summary.get(key, 0) or 0

rows = [
    ("mtp", mtp),
    ("original_dflash", original),
    ("trained_dflash", trained),
]

print("mode,completed,tok/s,ref_hit,draft_steps,draft_tokens,accepted,token_accept,first_pos,mean_accept_per_draft")
for name, summary in rows:
    print(
        ",".join(
            [
                name,
                f"{summary['completed']}/{summary['num_requested']}",
                fmt(summary.get("output_tok_per_sec")),
                fmt(summary.get("reference_contains_rate")),
                fmt(spec(summary, "spec_draft_steps_total")),
                fmt(spec(summary, "spec_draft_tokens_total")),
                fmt(spec(summary, "spec_accepted_tokens_total")),
                fmt(summary.get("spec_token_acceptance_rate")),
                fmt(summary.get("spec_first_position_acceptance_rate")),
                fmt(summary.get("spec_mean_accepted_tokens_per_draft")),
            ]
        )
    )

mtp_tps = mtp.get("output_tok_per_sec")
original_tps = original.get("output_tok_per_sec")
trained_tps = trained.get("output_tok_per_sec")

print()
print(f"original_dflash / mtp tok/s: {fmt(ratio(original_tps, mtp_tps))}")
print(f"trained_dflash / mtp tok/s:  {fmt(ratio(trained_tps, mtp_tps))}")
print(f"trained / original tok/s:    {fmt(ratio(trained_tps, original_tps))}")
print(
    "trained / original mean accept: "
    f"{fmt(ratio(trained.get('spec_mean_accepted_tokens_per_draft'), original.get('spec_mean_accepted_tokens_per_draft')))}"
)

print()
if mtp["completed"] == 0:
    print("VERDICT: BAD - MTP accepted no requests.")
elif original["completed"] == 0:
    print("VERDICT: BAD - original DFlash accepted no requests.")
elif trained["completed"] == 0:
    print("VERDICT: BAD - trained DFlash accepted no requests.")
elif trained_tps is not None and mtp_tps and trained_tps > mtp_tps:
    print("VERDICT: TRAINED_DFLASH_BEATS_MTP")
elif original_tps is not None and mtp_tps and original_tps > mtp_tps:
    print("VERDICT: ORIGINAL_DFLASH_BEATS_MTP")
else:
    print("VERDICT: MTP_STILL_FASTER")
PY

echo
echo "Artifacts:"
echo "  $RUN_DIR"
