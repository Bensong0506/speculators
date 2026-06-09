#!/bin/bash
# Evaluate a newly trained DFlash run against the established MMStar baselines.
#
# Flow:
#   1) Sweep all checkpoints under CHECKPOINT_FIND_ROOT with SELECT_SPEC=7.
#   2) Pick the best checkpoint by BEST_METRIC, default trained_tok_s.
#   3) Run that checkpoint with TRAINED_SPECS="3 5 7".
#   4) Merge those rows with the baseline six:
#      MTP@3/5/7 and original DFlash@3/5/7.
#
# Existing summaries are skipped, so this can be resumed safely.
#
# Usage:
#   CHECKPOINT_FIND_ROOT=/path/to/current/run/checkpoints \
#   bash examples/evaluate/eval_trained_dflash_best_vs_baselines.sh

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
BASELINE_DRAFT="${BASELINE_DRAFT:-$DEFAULT_ROOT/Qwen3.5-9B-DFlash}"
MMSTAR_SRC="${MMSTAR_SRC:-$DEFAULT_MMSTAR_ROOT/mmstar_answers.json}"
MMSTAR_JSONL="${MMSTAR_JSONL:-$REPO_ROOT/data/mmstar/mmstar_eval.jsonl}"
MMSTAR_IMAGE_DIR="${MMSTAR_IMAGE_DIR:-$REPO_ROOT/data/mmstar/images}"
MEDIA_ROOT="${MEDIA_ROOT:-$DEFAULT_MMSTAR_ROOT/images}"

CHECKPOINT_FIND_ROOT="${CHECKPOINT_FIND_ROOT:-}"
SELECT_SPEC="${SELECT_SPEC:-7}"
TRAINED_SPECS="${TRAINED_SPECS:-3 5 7}"
BEST_METRIC="${BEST_METRIC:-trained_tok_s}"

NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
BASELINE_PORT="${BASELINE_PORT:-8100}"
DFLASH_PORT="${DFLASH_PORT:-8101}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-trained-dflash-spec-sweep}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

MTP_SPECS="${MTP_SPECS:-3 5 7}"
DFLASH_SPECS="${DFLASH_SPECS:-3 5 7}"
SUITE_NAME="${SUITE_NAME:-single_n${NUM_PROMPTS}_tok${MAX_TOKENS}}"
BASELINE_SWEEP_DIR="${BASELINE_SWEEP_DIR:-$REPO_ROOT/output/mmstar_mtp_dflash_spec_sweeps/$SUITE_NAME}"
BASELINE_RESULTS_JSONL="${BASELINE_RESULTS_JSONL:-$BASELINE_SWEEP_DIR/results.jsonl}"
RUN_BASELINE_SWEEP_IF_MISSING="${RUN_BASELINE_SWEEP_IF_MISSING:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mmstar_trained_dflash_best_vs_baselines}"
RUN_TAG="${RUN_TAG:-}"
FORCE_CHECKPOINT_SWEEP="${FORCE_CHECKPOINT_SWEEP:-0}"
FORCE_TRAINED_SPEC_SWEEP="${FORCE_TRAINED_SPEC_SWEEP:-0}"
FORCE_BASELINE_SWEEP="${FORCE_BASELINE_SWEEP:-0}"

max_spec() {
    python3 - "$@" <<'PY'
import sys
vals = []
for group in sys.argv[1:]:
    vals.extend(int(x) for x in group.replace(",", " ").split())
print(max(vals) if vals else 1)
PY
}

MAX_REQUESTED_SPEC="$(max_spec "$TRAINED_SPECS" "$SELECT_SPEC" "$MTP_SPECS" "$DFLASH_SPECS")"
MIN_BATCHED_TOKENS="$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_REQUESTED_SPEC))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MIN_BATCHED_TOKENS}"
TRAINED_SWEEP_STATUS=0

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
    [ -d "$BASELINE_DRAFT" ] || { echo "[fatal] BASELINE_DRAFT not found: $BASELINE_DRAFT"; exit 1; }
    [ -s "$MMSTAR_SRC" ] || { echo "[fatal] MMSTAR_SRC not found: $MMSTAR_SRC"; exit 1; }
    [ -d "$MEDIA_ROOT" ] || { echo "[fatal] MEDIA_ROOT not found: $MEDIA_ROOT"; exit 1; }
}

discover_latest_checkpoint_root() {
    python3 - "$REPO_ROOT/output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
best = None
for cfg in root.rglob("config.json"):
    ckpt = cfg.parent
    if not (ckpt / "model.safetensors").exists():
        continue
    try:
        obj = json.loads(cfg.read_text())
    except Exception:
        continue
    if obj.get("speculators_model_type") != "dflash":
        continue
    score = max((ckpt / "model.safetensors").stat().st_mtime, ckpt.stat().st_mtime)
    checkpoints_dir = ckpt.parent if ckpt.parent.name == "checkpoints" else ckpt
    candidate = (score, checkpoints_dir)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is None:
    raise SystemExit("[fatal] could not auto-discover any DFlash checkpoint under output/")
print(best[1])
PY
}

safe_tag() {
    python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
parts = [p for p in path.parts if p not in ("/", "checkpoints")]
label = "_".join(parts[-3:]) if len(parts) >= 3 else path.name
label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
print(label[:120] or "trained_run")
PY
}

if [ -z "$CHECKPOINT_FIND_ROOT" ]; then
    CHECKPOINT_FIND_ROOT="$(discover_latest_checkpoint_root)"
    echo "[info] auto-discovered CHECKPOINT_FIND_ROOT=$CHECKPOINT_FIND_ROOT"
fi
if [ -z "$RUN_TAG" ]; then
    RUN_TAG="$(safe_tag "$CHECKPOINT_FIND_ROOT")"
fi

EVAL_ROOT="${EVAL_ROOT:-$OUTPUT_ROOT/$SUITE_NAME/$RUN_TAG}"
CHECKPOINT_SWEEP_ROOT="${CHECKPOINT_SWEEP_ROOT:-$EVAL_ROOT/checkpoint_sweep_spec${SELECT_SPEC}}"
TRAINED_SPEC_SWEEP_DIR="${TRAINED_SPEC_SWEEP_DIR:-$EVAL_ROOT/trained_best_spec_sweep}"
FINAL_RESULTS_JSONL="$EVAL_ROOT/final_results.jsonl"
FINAL_RESULTS_CSV="$EVAL_ROOT/final_results.csv"
FINAL_RESULTS_MD="$EVAL_ROOT/final_results.md"
BEST_INFO_JSON="$EVAL_ROOT/best_checkpoint.json"
mkdir -p "$EVAL_ROOT" "$TRAINED_SPEC_SWEEP_DIR"

run_baseline_sweep_if_needed() {
    if [ "$FORCE_BASELINE_SWEEP" = "1" ] || [ ! -s "$BASELINE_RESULTS_JSONL" ]; then
        if [ "$RUN_BASELINE_SWEEP_IF_MISSING" != "1" ] && [ "$FORCE_BASELINE_SWEEP" != "1" ]; then
            echo "[warn] baseline results missing: $BASELINE_RESULTS_JSONL"
            echo "       final comparison will include trained rows only."
            return 0
        fi
        echo
        echo "=== Baseline six-point sweep is missing; running it now ==="
        env \
            MODEL="$MODEL" \
            DFLASH_DRAFT="$BASELINE_DRAFT" \
            MMSTAR_SRC="$MMSTAR_SRC" \
            MMSTAR_JSONL="$MMSTAR_JSONL" \
            MMSTAR_IMAGE_DIR="$MMSTAR_IMAGE_DIR" \
            MEDIA_ROOT="$MEDIA_ROOT" \
            MTP_SPECS="$MTP_SPECS" \
            DFLASH_SPECS="$DFLASH_SPECS" \
            NUM_PROMPTS="$NUM_PROMPTS" \
            MAX_TOKENS="$MAX_TOKENS" \
            MAX_MODEL_LEN="$MAX_MODEL_LEN" \
            MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
            MAX_NUM_SEQS="$MAX_NUM_SEQS" \
            GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
            GPUS="$GPUS" \
            TP="$TP" \
            PORT="$PORT" \
            OUTPUT_ROOT="$REPO_ROOT/output/mmstar_mtp_dflash_spec_sweeps" \
            SUITE_NAME="$SUITE_NAME" \
            bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh
    else
        echo "[info] reusing baseline results: $BASELINE_RESULTS_JSONL"
    fi
}

run_checkpoint_sweep_if_needed() {
    if [ "$FORCE_CHECKPOINT_SWEEP" = "1" ]; then
        rm -rf "$CHECKPOINT_SWEEP_ROOT"
    fi
    if [ -s "$CHECKPOINT_SWEEP_ROOT/results.jsonl" ]; then
        echo "[info] reusing checkpoint sweep: $CHECKPOINT_SWEEP_ROOT/results.jsonl"
        return 0
    fi

    echo
    echo "=== Sweeping trained checkpoints to select best checkpoint ==="
    env \
        MODEL="$MODEL" \
        CHECKPOINT_FIND_ROOT="$CHECKPOINT_FIND_ROOT" \
        SWEEP_ROOT="$CHECKPOINT_SWEEP_ROOT" \
        INFER_NUM_SPEC="$SELECT_SPEC" \
        BASELINE_DRAFT="$BASELINE_DRAFT" \
        MMSTAR_SRC="$MMSTAR_SRC" \
        MMSTAR_JSONL="$MMSTAR_JSONL" \
        MMSTAR_IMAGE_DIR="$MMSTAR_IMAGE_DIR" \
        MEDIA_ROOT="$MEDIA_ROOT" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        MAX_TOKENS="$MAX_TOKENS" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
        GPUS="$GPUS" \
        TP="$TP" \
        BASELINE_PORT="$BASELINE_PORT" \
        DFLASH_PORT="$DFLASH_PORT" \
        bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
}

select_best_checkpoint() {
    python3 - "$CHECKPOINT_SWEEP_ROOT/results.jsonl" "$BEST_METRIC" "$BEST_INFO_JSON" <<'PY'
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
metric = sys.argv[2]
out = Path(sys.argv[3])

if not results.exists():
    raise SystemExit(f"[fatal] missing checkpoint sweep results: {results}")

aliases = {
    "tok_s": "trained_tok_s",
    "trained_tps": "trained_tok_s",
    "ratio": "trained_native_ratio",
    "accept": "trained_mean_accept_per_draft",
    "mean_accept": "trained_mean_accept_per_draft",
}
metric_key = aliases.get(metric, metric)

rows = []
for line in results.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("exit_status") != 0:
        continue
    value = row.get(metric_key)
    if value is None:
        continue
    rows.append((float(value), row))

if not rows:
    raise SystemExit(f"[fatal] no successful checkpoint rows with metric={metric_key}")

value, best = max(rows, key=lambda item: item[0])
obj = {
    "selection_metric": metric_key,
    "selection_value": value,
    "checkpoint": best["checkpoint"],
    "row": best,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
print(best["checkpoint"])
PY
}

check_dflash_spec_capacity() {
    local checkpoint="$1"
    python3 - "$checkpoint" "$TRAINED_SPECS" <<'PY'
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
specs = [int(x) for x in sys.argv[2].replace(",", " ").split()]
cfg_path = ckpt / "config.json"
if not cfg_path.exists():
    raise SystemExit(f"[fatal] missing config.json: {ckpt}")
cfg = json.loads(cfg_path.read_text())
if cfg.get("speculators_model_type") != "dflash":
    raise SystemExit(f"[fatal] best checkpoint is not speculators-format DFlash: {ckpt}")
block = int(cfg.get("block_size", 0))
max_spec = block - 1
too_large = [x for x in specs if x > max_spec]
if block <= 0 or too_large:
    raise SystemExit(
        f"[fatal] checkpoint block_size={block} supports num_spec <= {max_spec}, "
        f"requested {too_large or specs}"
    )
print(f"[info] best checkpoint block_size={block}, max_spec={max_spec}, requested={specs}")
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
            if grep -qiE "unknown.*dflash|unsupported.*dflash|unrecognized.*dflash" "$log"; then
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

start_trained_server() {
    local checkpoint="$1"
    local spec="$2"
    local log="$3"

    cleanup_server
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: port $PORT already has a healthy server before starting trained_dflash@$spec."
        echo "       Stop the old vLLM process or choose a different PORT."
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

    if [ "$ENFORCE_EAGER" = "1" ]; then
        args+=(--enforce-eager)
    fi
    if [ -n "$DTYPE" ]; then
        args+=(--dtype "$DTYPE")
    fi
    if [ -n "$ATTENTION_BACKEND" ]; then
        args+=(--attention-backend "$ATTENTION_BACKEND")
    fi

    args+=(
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$checkpoint\",\"num_speculative_tokens\":$spec}"
    )

    echo
    echo "=== Starting trained_dflash@$spec server ==="
    echo "  model:             $MODEL"
    echo "  checkpoint:        $checkpoint"
    echo "  num_spec:          $spec"
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
    wait_for_server "$log" "trained_dflash@$spec"
}

run_client() {
    local out_dir="$1"
    python3 examples/evaluate/mmstar_weight_client.py \
        --endpoint "http://localhost:${PORT}/v1" \
        --data-jsonl "$MMSTAR_JSONL" \
        --out-jsonl "$out_dir/responses.jsonl" \
        --summary-json "$out_dir/summary.json" \
        --num "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS"
}

parse_acceptance() {
    local log="$1"
    local out="$2"

    if ! python3 examples/evaluate/eval-guidellm/scripts/parse_logs.py "$log" \
        > "$out" 2>&1; then
        echo "(No parseable 'SpecDecoding metrics:' lines found. Showing spec-related log tail.)" | tee -a "$out"
        grep -iE "spec|accept|draft|dflash" "$log" | tail -n 60 | tee -a "$out" || true
    fi
}

write_status() {
    local out_dir="$1"
    local status="$2"
    local spec="$3"
    local message="${4:-}"
    python3 - "$out_dir/status.json" "$status" "$spec" "$message" <<'PY'
import json
import sys
from pathlib import Path

path, status, spec, message = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {"status": status, "method": "trained_dflash", "spec": int(spec), "message": message},
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_trained_spec_sweep() {
    local checkpoint="$1"
    local spec out_dir log acceptance
    local overall_status=0

    for spec in $TRAINED_SPECS; do
        out_dir="$TRAINED_SPEC_SWEEP_DIR/trained_dflash_spec${spec}"
        log="$out_dir/vllm.log"
        acceptance="$out_dir/acceptance_from_log.txt"
        mkdir -p "$out_dir"
        printf '%s\n' "$checkpoint" > "$out_dir/checkpoint_path.txt"

        if [ "$FORCE_TRAINED_SPEC_SWEEP" != "1" ] && [ -s "$out_dir/summary.json" ]; then
            echo
            echo "=== Skipping trained_dflash@$spec: existing $out_dir/summary.json ==="
            continue
        fi

        echo
        echo "=== Running trained_dflash@$spec ==="
        if ! start_trained_server "$checkpoint" "$spec" "$log"; then
            write_status "$out_dir" "failed_startup" "$spec" "server failed during startup"
            cleanup_server
            overall_status=1
            continue
        fi

        if ! run_client "$out_dir"; then
            write_status "$out_dir" "failed_client" "$spec" "client failed"
            cleanup_server
            overall_status=1
            continue
        fi

        cleanup_server
        sleep 5
        parse_acceptance "$log" "$acceptance"
        write_status "$out_dir" "ok" "$spec"
    done

    return "$overall_status"
}

aggregate_final_results() {
    python3 - \
        "$BASELINE_RESULTS_JSONL" \
        "$TRAINED_SPEC_SWEEP_DIR" \
        "$TRAINED_SPECS" \
        "$BEST_INFO_JSON" \
        "$FINAL_RESULTS_JSONL" \
        "$FINAL_RESULTS_CSV" \
        "$FINAL_RESULTS_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    baseline_jsonl,
    trained_dir,
    trained_specs,
    best_info_json,
    final_jsonl,
    final_csv,
    final_md,
) = sys.argv[1:]

baseline_path = Path(baseline_jsonl)
trained_root = Path(trained_dir)
trained_specs_i = [int(x) for x in trained_specs.replace(",", " ").split()]
best_info = json.loads(Path(best_info_json).read_text())
best_checkpoint = best_info["checkpoint"]

rows = []
if baseline_path.exists():
    for line in baseline_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "group": "baseline",
                "method": row.get("method"),
                "spec": row.get("spec"),
                "status": row.get("status"),
                "completed": row.get("completed"),
                "requested": row.get("requested"),
                "tok_s": row.get("tok_s"),
                "mean_accept_per_draft": row.get("mean_accept_per_draft"),
                "token_accept": row.get("token_accept"),
                "first_pos_accept": row.get("first_pos_accept"),
                "ref_hit": row.get("ref_hit"),
                "checkpoint": "",
                "summary_path": row.get("summary_path", ""),
            }
        )

for spec in trained_specs_i:
    out_dir = trained_root / f"trained_dflash_spec{spec}"
    summary_path = out_dir / "summary.json"
    status_path = out_dir / "status.json"
    status = "missing"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text()).get("status", status)
        except Exception:
            pass
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if status == "missing":
            status = "ok"

    def get(key):
        return summary.get(key) if summary else None

    rows.append(
        {
            "group": "trained",
            "method": "trained_dflash",
            "spec": spec,
            "status": status,
            "completed": get("completed"),
            "requested": get("num_requested"),
            "tok_s": get("output_tok_per_sec"),
            "mean_accept_per_draft": get("spec_mean_accepted_tokens_per_draft"),
            "token_accept": get("spec_token_acceptance_rate"),
            "first_pos_accept": get("spec_first_position_acceptance_rate"),
            "ref_hit": get("reference_contains_rate"),
            "checkpoint": best_checkpoint,
            "summary_path": str(summary_path) if summary_path.exists() else "",
        }
    )

Path(final_jsonl).parent.mkdir(parents=True, exist_ok=True)
with Path(final_jsonl).open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

fields = [
    "group",
    "method",
    "spec",
    "status",
    "completed",
    "requested",
    "tok_s",
    "mean_accept_per_draft",
    "token_accept",
    "first_pos_accept",
    "ref_hit",
    "checkpoint",
    "summary_path",
]
with Path(final_csv).open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
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
baseline_ranked = [r for r in ranked if r["group"] == "baseline"]
trained_ranked = [r for r in ranked if r["group"] == "trained"]
best_baseline = baseline_ranked[0] if baseline_ranked else None
best_trained = trained_ranked[0] if trained_ranked else None
mtp7 = next((r for r in rows if r["method"] == "mtp" and r["spec"] == 7), None)
dflash7 = next((r for r in rows if r["method"] == "dflash_original" and r["spec"] == 7), None)

def ratio(a, b):
    return a / b if a is not None and b else None

md = [
    "# MMStar Trained DFlash Best Checkpoint vs Baselines",
    "",
    f"Best checkpoint selected by `{best_info['selection_metric']}` at `spec={best_info['row']['infer_num_spec']}`:",
    "",
    f"`{best_checkpoint}`",
    "",
    f"Selection value: `{fmt(best_info['selection_value'])}`",
    "",
    "## Ranked Results",
    "",
    "| rank | group | method | spec | status | tok/s | mean accept/draft | token accept | first-pos accept | completed |",
    "|---:|---|---|---:|---|---:|---:|---:|---:|---:|",
]
for idx, row in enumerate(ranked, 1):
    md.append(
        "| {idx} | {group} | {method} | {spec} | {status} | {tok_s} | {mean_accept} | "
        "{token_accept} | {first_pos} | {completed}/{requested} |".format(
            idx=idx,
            group=row["group"],
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

md.extend(["", "## Key Comparisons", ""])
if best_trained:
    md.append(
        f"- Best trained config: `{best_trained['method']}@{best_trained['spec']}` = "
        f"`{fmt(best_trained['tok_s'])}` tok/s."
    )
if mtp7:
    md.append(
        f"- vs `MTP@7`: trained/MTP@7 = "
        f"`{fmt(ratio(best_trained['tok_s'] if best_trained else None, mtp7.get('tok_s')))}`."
    )
if dflash7:
    md.append(
        f"- vs `original DFlash@7`: trained/original = "
        f"`{fmt(ratio(best_trained['tok_s'] if best_trained else None, dflash7.get('tok_s')))}`."
    )
if best_baseline:
    md.append(
        f"- vs best baseline `{best_baseline['method']}@{best_baseline['spec']}`: trained/best-baseline = "
        f"`{fmt(ratio(best_trained['tok_s'] if best_trained else None, best_baseline.get('tok_s')))}`."
    )

md.extend(["", "## Verdict", ""])
if not best_trained:
    md.append("No trained DFlash spec run completed successfully.")
elif dflash7 and ratio(best_trained["tok_s"], dflash7.get("tok_s")) and ratio(best_trained["tok_s"], dflash7.get("tok_s")) > 1:
    md.append("Trained DFlash beats `original DFlash@7` on this MMStar slice.")
elif mtp7 and ratio(best_trained["tok_s"], mtp7.get("tok_s")) and ratio(best_trained["tok_s"], mtp7.get("tok_s")) > 1:
    md.append("Trained DFlash beats `MTP@7`, but does not beat `original DFlash@7`.")
else:
    md.append("Trained DFlash does not beat the main baselines on this MMStar slice.")

Path(final_md).write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print()
print(f"final_jsonl={final_jsonl}")
print(f"final_csv={final_csv}")
print(f"final_md={final_md}")
PY
}

validate_positive_int_list SELECT_SPEC "$SELECT_SPEC"
validate_positive_int_list TRAINED_SPECS "$TRAINED_SPECS"
validate_positive_int_list MTP_SPECS "$MTP_SPECS"
validate_positive_int_list DFLASH_SPECS "$DFLASH_SPECS"
check_paths

echo "=== Trained DFlash best-checkpoint eval vs baselines ==="
echo "  checkpoint_find_root: $CHECKPOINT_FIND_ROOT"
echo "  eval_root:            $EVAL_ROOT"
echo "  model:                $MODEL"
echo "  original_dflash:      $BASELINE_DRAFT"
echo "  select_spec:          $SELECT_SPEC"
echo "  trained_specs:        $TRAINED_SPECS"
echo "  best_metric:          $BEST_METRIC"
echo "  baseline_results:     $BASELINE_RESULTS_JSONL"
echo "  num_prompts:          $NUM_PROMPTS"
echo "  media_root:           $MEDIA_ROOT"

run_baseline_sweep_if_needed
run_checkpoint_sweep_if_needed
BEST_CHECKPOINT="$(select_best_checkpoint)"
echo
echo "=== Best checkpoint selected ==="
cat "$BEST_INFO_JSON"
echo
check_dflash_spec_capacity "$BEST_CHECKPOINT"

if ! run_trained_spec_sweep "$BEST_CHECKPOINT"; then
    TRAINED_SWEEP_STATUS=1
    echo "[warn] one or more trained spec runs failed; aggregating completed rows anyway."
fi

echo
echo "=== Aggregating trained-vs-baseline results ==="
aggregate_final_results | tee "$EVAL_ROOT/final_results.stdout.txt"

echo
echo "Artifacts:"
echo "  $EVAL_ROOT"
exit "$TRAINED_SWEEP_STATUS"
