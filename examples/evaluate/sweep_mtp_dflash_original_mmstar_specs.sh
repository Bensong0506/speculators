#!/bin/bash
# Sweep native Qwen3.5 MTP and original/raw DFlash speculative depths on MMStar.
#
# This runs six independent configs by default:
#   mtp:              spec = 3, 5, 7
#   dflash_original:  spec = 3, 5, 7
#
# Existing per-config summaries are skipped. The script also tries to seed results
# from previous test_mtp_vs_dflash_original_mmstar.sh runs, so an already completed
# MTP_SPEC=3 / DFLASH_SPEC=7 run does not need to be repeated.
#
# Usage:
#   bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh
#
# Common overrides:
#   MTP_SPECS="3 5 7" \
#   DFLASH_SPECS="3 5 7" \
#   NUM_PROMPTS=128 \
#   bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh
#
# Grouped 10-image variant:
#   IMAGES_PER_PROMPT=10 NUM_GROUPS=16 \
#   bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

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

MTP_METHOD="${MTP_METHOD:-qwen3_5_mtp}"
MTP_SPECS="${MTP_SPECS:-3 5 7}"
DFLASH_SPECS="${DFLASH_SPECS:-3 5 7}"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-1}"

if ! [[ "$IMAGES_PER_PROMPT" =~ ^[0-9]+$ ]] || [ "$IMAGES_PER_PROMPT" -le 0 ]; then
    echo "[fatal] IMAGES_PER_PROMPT must be a positive integer, got: $IMAGES_PER_PROMPT"
    exit 1
fi

if [ "$IMAGES_PER_PROMPT" -gt 1 ]; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_TOKENS="${MAX_TOKENS:-512}"
    NUM_GROUPS="${NUM_GROUPS:-16}"
    NUM_PROMPTS="${NUM_PROMPTS:-128}"
    DEFAULT_SUITE="group${IMAGES_PER_PROMPT}_g${NUM_GROUPS}_tok${MAX_TOKENS}"
else
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_TOKENS="${MAX_TOKENS:-128}"
    NUM_PROMPTS="${NUM_PROMPTS:-128}"
    NUM_GROUPS="${NUM_GROUPS:-16}"
    DEFAULT_SUITE="single_n${NUM_PROMPTS}_tok${MAX_TOKENS}"
fi

MAX_CONFIG_SPEC="$(python3 - "$MTP_SPECS" "$DFLASH_SPECS" <<'PY'
import sys

vals = []
for group in sys.argv[1:]:
    vals.extend(int(x) for x in group.replace(",", " ").split())
print(max(vals) if vals else 1)
PY
)"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_CONFIG_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-mmstar-spec-sweep}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SEED_PAIR_RESULTS="${SEED_PAIR_RESULTS:-1}"
STRICT_SEED_MATCH="${STRICT_SEED_MATCH:-0}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_mtp_vs_dflash_original}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_mtp_dflash_spec_sweeps}"
SUITE_NAME="${SUITE_NAME:-$DEFAULT_SUITE}"
SWEEP_DIR="${SWEEP_DIR:-$OUTPUT_ROOT/$SUITE_NAME}"
mkdir -p "$SWEEP_DIR"

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

validate_positive_int_list() {
    local name="$1"
    local values="$2"
    local value
    for value in $values; do
        if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
            echo "[fatal] $name must contain positive integers, got: $values"
            exit 1
        fi
    done
}

check_paths() {
    [ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
    [ -d "$DFLASH_DRAFT" ] || { echo "[fatal] DFLASH_DRAFT not found: $DFLASH_DRAFT"; exit 1; }
    [ -s "$MMSTAR_SRC" ] || { echo "[fatal] MMSTAR_SRC not found: $MMSTAR_SRC"; exit 1; }
    [ -d "$MEDIA_ROOT" ] || { echo "[fatal] MEDIA_ROOT not found: $MEDIA_ROOT"; exit 1; }
}

dflash_sanity() {
    python3 - "$MODEL" "$DFLASH_DRAFT" "$DFLASH_SPECS" "$SWEEP_DIR/dflash_sanity.txt" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
draft = Path(sys.argv[2])
requested = [int(x) for x in sys.argv[3].replace(",", " ").split()]
out = Path(sys.argv[4])

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
too_large = [x for x in requested if x > max_spec]
if too_large:
    raise SystemExit(
        f"[fatal] DFlash block_size={block} only supports "
        f"num_speculative_tokens <= {max_spec}, requested {too_large}."
    )

aux = cfg.get("aux_hidden_state_layer_ids") or cfg.get("dflash_config", {}).get(
    "target_layer_ids"
) or []
verifier = (
    cfg.get("speculators_config", {})
    .get("verifier", {})
    .get("name_or_path")
)
lines = [
    "Original DFlash sanity OK",
    f"  draft:      {draft}",
    f"  format:     {'speculators' if is_speculators_dflash else 'raw'}",
    f"  block_size: {block}",
    f"  max_spec:   {max_spec}",
    f"  requested:  {requested}",
    f"  aux_layers: {aux}",
]
if verifier and verifier != model:
    lines.append(f"  [warn] verifier path differs: {verifier} != {model}")
text = "\n".join(lines) + "\n"
out.write_text(text, encoding="utf-8")
print(text, end="")
PY
}

write_sweep_config() {
    python3 - \
        "$SWEEP_DIR/run_config.json" \
        "$MODEL" \
        "$DFLASH_DRAFT" \
        "$MMSTAR_SRC" \
        "$MMSTAR_JSONL" \
        "$MEDIA_ROOT" \
        "$GPUS" \
        "$TP" \
        "$PORT" \
        "$MTP_METHOD" \
        "$MTP_SPECS" \
        "$DFLASH_SPECS" \
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
    port,
    mtp_method,
    mtp_specs,
    dflash_specs,
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
    "port": int(port),
    "mtp_method": mtp_method,
    "mtp_specs": [int(x) for x in mtp_specs.replace(",", " ").split()],
    "dflash_specs": [int(x) for x in dflash_specs.replace(",", " ").split()],
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

seed_pair_results() {
    if [ "$SEED_PAIR_RESULTS" != "1" ] || [ ! -d "$PAIR_OUTPUT_ROOT" ]; then
        return 0
    fi

    python3 - \
        "$PAIR_OUTPUT_ROOT" \
        "$SWEEP_DIR" \
        "$MODEL" \
        "$DFLASH_DRAFT" \
        "$IMAGES_PER_PROMPT" \
        "$NUM_PROMPTS" \
        "$NUM_GROUPS" \
        "$MAX_TOKENS" \
        "$MTP_SPECS" \
        "$DFLASH_SPECS" \
        "$STRICT_SEED_MATCH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

(
    pair_root,
    sweep_dir,
    model,
    dflash_draft,
    images_per_prompt,
    num_prompts,
    num_groups,
    max_tokens,
    mtp_specs,
    dflash_specs,
    strict_seed_match,
) = sys.argv[1:]

pair_root_p = Path(pair_root)
sweep_dir_p = Path(sweep_dir)
images_per_prompt_i = int(images_per_prompt)
num_prompts_i = int(num_prompts)
num_groups_i = int(num_groups)
max_tokens_i = int(max_tokens)
mtp_spec_set = {int(x) for x in mtp_specs.replace(",", " ").split()}
dflash_spec_set = {int(x) for x in dflash_specs.replace(",", " ").split()}
strict_seed_match_b = strict_seed_match == "1"

def same_path(a: str | None, b: str) -> bool:
    if not a:
        return True
    return str(Path(a)) == str(Path(b))

def maybe_copy(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

seeded = []
for cfg_path in sorted(pair_root_p.glob("*/run_config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        continue
    if int(cfg.get("images_per_prompt", 1)) != images_per_prompt_i:
        continue
    if int(cfg.get("max_tokens", -1)) != max_tokens_i:
        continue
    if images_per_prompt_i > 1:
        if int(cfg.get("num_groups", -1)) != num_groups_i:
            continue
    else:
        if int(cfg.get("num_prompts", -1)) != num_prompts_i:
            continue
    if strict_seed_match_b:
        if not same_path(cfg.get("model"), model):
            continue
        if not same_path(cfg.get("dflash_draft"), dflash_draft):
            continue

    run_dir = cfg_path.parent
    mtp_spec = int(cfg.get("mtp_spec", -1))
    if mtp_spec in mtp_spec_set:
        dst_dir = sweep_dir_p / f"mtp_spec{mtp_spec}"
        dst_summary = dst_dir / "summary.json"
        src_summary = run_dir / "mtp_summary.json"
        if src_summary.exists() and not dst_summary.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_summary, dst_summary)
            maybe_copy(run_dir / "mtp_responses.jsonl", dst_dir / "responses.jsonl")
            maybe_copy(run_dir / "mtp_vllm.log", dst_dir / "vllm.log")
            maybe_copy(run_dir / "mtp_acceptance_from_log.txt", dst_dir / "acceptance_from_log.txt")
            (dst_dir / "status.json").write_text(
                json.dumps(
                    {"status": "seeded", "source_run": str(run_dir), "method": "mtp", "spec": mtp_spec},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            seeded.append(f"mtp@{mtp_spec} <- {run_dir}")

    dflash_spec = int(cfg.get("dflash_spec", -1))
    if dflash_spec in dflash_spec_set:
        dst_dir = sweep_dir_p / f"dflash_original_spec{dflash_spec}"
        dst_summary = dst_dir / "summary.json"
        src_summary = run_dir / "dflash_original_summary.json"
        if src_summary.exists() and not dst_summary.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_summary, dst_summary)
            maybe_copy(run_dir / "dflash_original_responses.jsonl", dst_dir / "responses.jsonl")
            maybe_copy(run_dir / "dflash_original_vllm.log", dst_dir / "vllm.log")
            maybe_copy(run_dir / "dflash_original_acceptance_from_log.txt", dst_dir / "acceptance_from_log.txt")
            (dst_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "seeded",
                        "source_run": str(run_dir),
                        "method": "dflash_original",
                        "spec": dflash_spec,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            seeded.append(f"dflash_original@{dflash_spec} <- {run_dir}")

for item in seeded:
    print(f"[seed] {item}")
PY
}

wait_for_server() {
    local log="$1"
    local mode="$2"
    echo "Waiting for $mode server on :$PORT (log: $log)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $mode vLLM server died during startup. Last 100 log lines:"
            tail -n 100 "$log" || true
            if grep -qiE "unknown.*(mtp|qwen3_5_mtp)|unsupported.*(mtp|qwen3_5_mtp)|unrecognized.*(mtp|qwen3_5_mtp)" "$log"; then
                echo "DIAGNOSIS: this vLLM build likely lacks native Qwen3.5 MTP support, or this MTP_SPEC is unsupported."
                echo "           Try MTP_METHOD=mtp if your build uses the generic method name."
            elif grep -qiE "unknown.*dflash|unsupported.*dflash|unrecognized.*dflash" "$log"; then
                echo "DIAGNOSIS: this vLLM build does not have DFlash inference."
            elif grep -qiE "max_num_scheduled_tokens|additional draft token slots" "$log"; then
                echo "DIAGNOSIS: vLLM speculative scheduling budget is too small."
                echo "           Lower MAX_NUM_SEQS/spec, or raise MAX_NUM_BATCHED_TOKENS."
            elif grep -qiE "m-?rope|multimodal.*spec|does not support" "$log"; then
                echo "DIAGNOSIS: this vLLM build likely lacks multimodal/M-RoPE spec support."
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
    local method="$1"
    local spec="$2"
    local log="$3"
    local spec_config

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting $method@$spec."
        echo "       Stop the old vLLM process or choose a different PORT."
        return 1
    fi

    if [ "$method" = "mtp" ]; then
        spec_config="{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$spec,\"enforce_eager\":true}"
    else
        spec_config="{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT\",\"num_speculative_tokens\":$spec}"
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

    args+=(--speculative-config "$spec_config")

    echo
    echo "=== Starting ${method}@${spec} server ==="
    echo "  model:             $MODEL"
    echo "  dflash_draft:      $DFLASH_DRAFT"
    echo "  spec_config:       $spec_config"
    echo "  images_per_prompt: $IMAGES_PER_PROMPT"
    echo "  max_model_len:     $MAX_MODEL_LEN"
    echo "  max batched toks:  $MAX_NUM_BATCHED_TOKENS"
    echo "  max seqs:          $MAX_NUM_SEQS"
    echo "  port:              $PORT"
    echo "  devices:           $GPUS"
    echo "  media_root:        $MEDIA_ROOT"
    echo "  log:               $log"
    printf '[cmd]'
    printf ' %q' env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}"
    echo

    env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$log" "${method}@${spec}"
}

run_client() {
    local out_dir="$1"

    if [ "$IMAGES_PER_PROMPT" -gt 1 ]; then
        python3 examples/evaluate/mmstar_10image_client.py \
            --endpoint "http://localhost:${PORT}/v1" \
            --data-jsonl "$MMSTAR_JSONL" \
            --out-jsonl "$out_dir/responses.jsonl" \
            --summary-json "$out_dir/summary.json" \
            --num-groups "$NUM_GROUPS" \
            --images-per-prompt "$IMAGES_PER_PROMPT" \
            --max-tokens "$MAX_TOKENS"
    else
        python3 examples/evaluate/mmstar_weight_client.py \
            --endpoint "http://localhost:${PORT}/v1" \
            --data-jsonl "$MMSTAR_JSONL" \
            --out-jsonl "$out_dir/responses.jsonl" \
            --summary-json "$out_dir/summary.json" \
            --num "$NUM_PROMPTS" \
            --max-tokens "$MAX_TOKENS"
    fi
}

parse_acceptance() {
    local log="$1"
    local out="$2"

    if ! python3 examples/evaluate/eval-guidellm/scripts/parse_logs.py "$log" \
        > "$out" 2>&1; then
        echo "(No parseable 'SpecDecoding metrics:' lines found. Showing spec-related log tail.)" | tee -a "$out"
        grep -iE "spec|accept|draft|mtp|dflash" "$log" | tail -n 60 | tee -a "$out" || true
    fi
}

write_status() {
    local out_dir="$1"
    local status="$2"
    local method="$3"
    local spec="$4"
    local message="${5:-}"
    python3 - "$out_dir/status.json" "$status" "$method" "$spec" "$message" <<'PY'
import json
import sys
from pathlib import Path

path, status, method, spec, message = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {"status": status, "method": method, "spec": int(spec), "message": message},
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_one() {
    local method="$1"
    local spec="$2"
    local out_dir="$SWEEP_DIR/${method}_spec${spec}"
    local log="$out_dir/vllm.log"
    local acceptance="$out_dir/acceptance_from_log.txt"

    mkdir -p "$out_dir"
    if [ "$FORCE_RERUN" != "1" ] && [ -s "$out_dir/summary.json" ]; then
        echo
        echo "=== Skipping ${method}@${spec}: existing $out_dir/summary.json ==="
        return 0
    fi

    echo
    echo "=== Running ${method}@${spec} ==="
    if ! start_server "$method" "$spec" "$log"; then
        write_status "$out_dir" "failed_startup" "$method" "$spec" "server failed during startup"
        cleanup_server
        return 1
    fi

    if ! run_client "$out_dir"; then
        write_status "$out_dir" "failed_client" "$method" "$spec" "client failed"
        cleanup_server
        return 1
    fi

    cleanup_server
    sleep 5
    parse_acceptance "$log" "$acceptance"
    write_status "$out_dir" "ok" "$method" "$spec"
    return 0
}

aggregate_results() {
    python3 - "$SWEEP_DIR" "$MTP_SPECS" "$DFLASH_SPECS" <<'PY'
import csv
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
mtp_specs = [int(x) for x in sys.argv[2].replace(",", " ").split()]
dflash_specs = [int(x) for x in sys.argv[3].replace(",", " ").split()]

rows = []
for method, specs in (("mtp", mtp_specs), ("dflash_original", dflash_specs)):
    for spec in specs:
        out_dir = sweep_dir / f"{method}_spec{spec}"
        summary_path = out_dir / "summary.json"
        status_path = out_dir / "status.json"
        status = "missing"
        message = ""
        if status_path.exists():
            try:
                status_obj = json.loads(status_path.read_text())
                status = status_obj.get("status", status)
                message = status_obj.get("message", "")
            except Exception:
                pass
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            status = "ok" if status == "missing" else status

        def get(key):
            return summary.get(key) if summary else None

        rows.append(
            {
                "method": method,
                "spec": spec,
                "status": status,
                "completed": get("completed"),
                "requested": get("num_requested") or get("num_groups_requested"),
                "tok_s": get("output_tok_per_sec"),
                "images_s": get("images_per_sec"),
                "mean_latency_s": get("mean_latency_sec"),
                "ref_hit": get("reference_contains_rate"),
                "draft_steps": get("spec_draft_steps_total"),
                "draft_tokens": get("spec_draft_tokens_total"),
                "accepted_tokens": get("spec_accepted_tokens_total"),
                "token_accept": get("spec_token_acceptance_rate"),
                "first_pos_accept": get("spec_first_position_acceptance_rate"),
                "mean_accept_per_draft": get("spec_mean_accepted_tokens_per_draft"),
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "log_path": str(out_dir / "vllm.log"),
                "message": message,
            }
        )

jsonl_path = sweep_dir / "results.jsonl"
with jsonl_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

csv_path = sweep_dir / "results.csv"
fieldnames = list(rows[0].keys()) if rows else []
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

ranked = sorted(
    [r for r in rows if r.get("tok_s") is not None],
    key=lambda r: r["tok_s"],
    reverse=True,
)

md_lines = [
    "# MMStar MTP vs Original DFlash Spec Sweep",
    "",
    "| rank | method | spec | status | tok/s | mean accept/draft | token accept | first-pos accept | completed |",
    "|---:|---|---:|---|---:|---:|---:|---:|---:|",
]
for rank, row in enumerate(ranked, 1):
    md_lines.append(
        "| {rank} | {method} | {spec} | {status} | {tok_s} | {mean_accept} | "
        "{token_accept} | {first_pos} | {completed}/{requested} |".format(
            rank=rank,
            method=row["method"],
            spec=row["spec"],
            status=row["status"],
            tok_s=fmt(row["tok_s"]),
            mean_accept=fmt(row["mean_accept_per_draft"]),
            token_accept=fmt(row["token_accept"]),
            first_pos=fmt(row["first_pos_accept"]),
            completed=fmt(row["completed"]),
            requested=fmt(row["requested"]),
        )
    )

missing = [r for r in rows if r.get("tok_s") is None]
if missing:
    md_lines.extend(["", "## Missing Or Failed", ""])
    for row in missing:
        md_lines.append(
            f"- {row['method']}@{row['spec']}: {row['status']} {row.get('message') or ''}".rstrip()
        )

if ranked:
    best = ranked[0]
    md_lines.extend(
        [
            "",
            "## Best Tok/S",
            "",
            f"`{best['method']}@{best['spec']}`: {fmt(best['tok_s'])} tok/s",
        ]
    )

md_path = sweep_dir / "results.md"
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print("\n".join(md_lines))
print()
print(f"results_jsonl={jsonl_path}")
print(f"results_csv={csv_path}")
print(f"results_md={md_path}")
PY
}

validate_positive_int_list MTP_SPECS "$MTP_SPECS"
validate_positive_int_list DFLASH_SPECS "$DFLASH_SPECS"
check_paths

echo "=== MMStar MTP/DFlash original spec sweep ==="
echo "  model:             $MODEL"
echo "  dflash_draft:      $DFLASH_DRAFT"
echo "  mtp_method:        $MTP_METHOD"
echo "  mtp_specs:         $MTP_SPECS"
echo "  dflash_specs:      $DFLASH_SPECS"
echo "  images_per_prompt: $IMAGES_PER_PROMPT"
echo "  num_prompts:       $NUM_PROMPTS"
echo "  num_groups:        $NUM_GROUPS"
echo "  max_tokens:        $MAX_TOKENS"
echo "  max_model_len:     $MAX_MODEL_LEN"
echo "  max batched toks:  $MAX_NUM_BATCHED_TOKENS"
echo "  max seqs:          $MAX_NUM_SEQS"
echo "  media_root:        $MEDIA_ROOT"
echo "  sweep_dir:         $SWEEP_DIR"
echo "  strict_seed_match: $STRICT_SEED_MATCH"

dflash_sanity
write_sweep_config
prepare_mmstar
seed_pair_results

overall_status=0
for spec in $MTP_SPECS; do
    if ! run_one mtp "$spec"; then
        overall_status=1
    fi
done

for spec in $DFLASH_SPECS; do
    if ! run_one dflash_original "$spec"; then
        overall_status=1
    fi
done

echo
echo "=== Aggregating sweep results ==="
aggregate_results | tee "$SWEEP_DIR/results.stdout.txt"

echo
echo "Artifacts:"
echo "  $SWEEP_DIR"

exit "$overall_status"
