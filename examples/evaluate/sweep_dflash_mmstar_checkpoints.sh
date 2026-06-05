#!/bin/bash
# Sweep MMStar DFlash evaluation over all trained checkpoint directories.
#
# This is an overnight runner around test_dflash_mmstar_weights.sh. It discovers
# speculators-format DFlash checkpoints, evaluates each one against the same
# native/raw DFlash baseline, and writes a compact results table.
#
# Usage:
#   INFER_NUM_SPEC=7 bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
#
# Common overrides:
#   CHECKPOINT_FIND_ROOT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_100k_continue_dflash \
#   BASELINE_DRAFT=/data/wenxuan/Qwen3.5-9B-DFlash \
#   NUM_PROMPTS=128 \
#   GPUS=0 \
#   bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
#
# Optional:
#   CHECKPOINT_LIST=/path/to/checkpoints.txt  # one checkpoint dir per line
#   CONTINUE_ON_FAIL=0                        # stop at the first failed run

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

TEST_SCRIPT="${TEST_SCRIPT:-$REPO_ROOT/examples/evaluate/test_dflash_mmstar_weights.sh}"
CHECKPOINT_FIND_ROOT="${CHECKPOINT_FIND_ROOT:-$REPO_ROOT/output}"
CHECKPOINT_LIST="${CHECKPOINT_LIST:-}"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/output/mmstar_checkpoint_sweeps/$STAMP}"
RESULTS_JSONL="$SWEEP_ROOT/results.jsonl"
RESULTS_CSV="$SWEEP_ROOT/results.csv"

INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
BASELINE_DRAFT="${BASELINE_DRAFT:-/data/wenxuan/Qwen3.5-9B-DFlash}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
BASELINE_PORT="${BASELINE_PORT:-8100}"
DFLASH_PORT="${DFLASH_PORT:-8101}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
INCLUDE_BEST="${INCLUDE_BEST:-1}"
DEDUP_REALPATH="${DEDUP_REALPATH:-1}"

mkdir -p "$SWEEP_ROOT"

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
    if name != "checkpoint_best" and not re.match(r"checkpoint[_-]?\d+$", name):
        continue
    cfg_path = path / "config.json"
    weights_path = path / "model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        continue
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        continue
    if cfg.get("speculators_model_type") != "dflash":
        continue
    match = re.search(r"checkpoint[_-]?(\d+)$", name)
    step = int(match.group(1)) if match else 10**18
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
        # Prefer a numbered checkpoint over checkpoint_best when both resolve
        # to the same directory.
        if current is None or (current["is_best"] and not item["is_best"]):
            deduped[item["key"]] = item
    items = list(deduped.values())

for item in sorted(items, key=lambda x: (x["parent"], x["step"], x["is_best"], x["path"])):
    print(item["path"])
PY
}

write_summary_row() {
    local checkpoint="$1"
    local run_dir="$2"
    local exit_status="$3"

    python3 - "$checkpoint" "$run_dir" "$exit_status" "$RESULTS_JSONL" "$RESULTS_CSV" "$INFER_NUM_SPEC" <<'PY'
import csv
import json
import sys
from pathlib import Path

checkpoint = sys.argv[1]
run_dir = Path(sys.argv[2])
exit_status = int(sys.argv[3])
results_jsonl = Path(sys.argv[4])
results_csv = Path(sys.argv[5])
infer_num_spec = int(sys.argv[6])

def load_json(name):
    path = run_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text())

def ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator

def get(summary, key):
    if not summary:
        return None
    return summary.get(key)

native = load_json("baseline_summary.json")
trained = load_json("dflash_summary.json")

native_tps = get(native, "output_tok_per_sec")
trained_tps = get(trained, "output_tok_per_sec")
native_mean = get(native, "spec_mean_accepted_tokens_per_draft")
trained_mean = get(trained, "spec_mean_accepted_tokens_per_draft")
native_accept = get(native, "spec_token_acceptance_rate")
trained_accept = get(trained, "spec_token_acceptance_rate")

speed_ratio = ratio(trained_tps, native_tps)
mean_ratio = ratio(trained_mean, native_mean)
accept_ratio = ratio(trained_accept, native_accept)

if exit_status != 0:
    verdict = "failed"
elif speed_ratio is not None and speed_ratio > 1:
    verdict = "improved"
elif mean_ratio is not None and mean_ratio > 1:
    verdict = "mixed_acceptance_up"
else:
    verdict = "not_improved"

tail = ""
stdout_log = run_dir / "sweep_stdout.log"
if exit_status != 0 and stdout_log.exists():
    lines = stdout_log.read_text(errors="replace").splitlines()
    tail = "\n".join(lines[-40:])

row = {
    "checkpoint": checkpoint,
    "run_dir": str(run_dir),
    "infer_num_spec": infer_num_spec,
    "exit_status": exit_status,
    "verdict": verdict,
    "native_tok_s": native_tps,
    "trained_tok_s": trained_tps,
    "trained_native_ratio": speed_ratio,
    "native_ref_hit": get(native, "reference_contains_rate"),
    "trained_ref_hit": get(trained, "reference_contains_rate"),
    "native_mean_accept_per_draft": native_mean,
    "trained_mean_accept_per_draft": trained_mean,
    "mean_accept_ratio": mean_ratio,
    "native_token_accept_rate": native_accept,
    "trained_token_accept_rate": trained_accept,
    "token_accept_ratio": accept_ratio,
    "native_first_pos_accept": get(native, "spec_first_position_acceptance_rate"),
    "trained_first_pos_accept": get(trained, "spec_first_position_acceptance_rate"),
    "native_draft_steps": get(native, "spec_draft_steps_total"),
    "trained_draft_steps": get(trained, "spec_draft_steps_total"),
    "native_draft_tokens": get(native, "spec_draft_tokens_total"),
    "trained_draft_tokens": get(trained, "spec_draft_tokens_total"),
    "native_accepted_tokens": get(native, "spec_accepted_tokens_total"),
    "trained_accepted_tokens": get(trained, "spec_accepted_tokens_total"),
    "error_tail": tail,
}

results_jsonl.parent.mkdir(parents=True, exist_ok=True)
with results_jsonl.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

fields = [
    "checkpoint",
    "run_dir",
    "infer_num_spec",
    "exit_status",
    "verdict",
    "native_tok_s",
    "trained_tok_s",
    "trained_native_ratio",
    "native_mean_accept_per_draft",
    "trained_mean_accept_per_draft",
    "mean_accept_ratio",
    "native_token_accept_rate",
    "trained_token_accept_rate",
    "token_accept_ratio",
    "native_first_pos_accept",
    "trained_first_pos_accept",
    "native_ref_hit",
    "trained_ref_hit",
    "native_draft_steps",
    "trained_draft_steps",
    "native_draft_tokens",
    "trained_draft_tokens",
    "native_accepted_tokens",
    "trained_accepted_tokens",
]
write_header = not results_csv.exists()
with results_csv.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow({field: row.get(field) for field in fields})

print(json.dumps({field: row.get(field) for field in fields}, ensure_ascii=False))
PY
}

if [ -n "$CHECKPOINT_LIST" ]; then
    if [ ! -s "$CHECKPOINT_LIST" ]; then
        echo "[fatal] CHECKPOINT_LIST is empty or missing: $CHECKPOINT_LIST"
        exit 1
    fi
    cp "$CHECKPOINT_LIST" "$SWEEP_ROOT/checkpoints.txt"
else
    discover_checkpoints > "$SWEEP_ROOT/checkpoints.txt"
fi

TOTAL="$(wc -l < "$SWEEP_ROOT/checkpoints.txt" | tr -d ' ')"
if [ "$TOTAL" = "0" ]; then
    echo "[fatal] no speculators-format DFlash checkpoints found under $CHECKPOINT_FIND_ROOT"
    echo "        Set CHECKPOINT_FIND_ROOT or CHECKPOINT_LIST explicitly."
    exit 1
fi

echo "=== MMStar DFlash checkpoint sweep ==="
echo "  checkpoints:       $TOTAL"
echo "  checkpoint list:   $SWEEP_ROOT/checkpoints.txt"
echo "  infer_num_spec:    $INFER_NUM_SPEC"
echo "  baseline_draft:    $BASELINE_DRAFT"
echo "  num_prompts:       $NUM_PROMPTS"
echo "  sweep root:        $SWEEP_ROOT"
echo "  results csv:       $RESULTS_CSV"
echo

index=0
while IFS= read -r checkpoint; do
    [ -n "$checkpoint" ] || continue
    index=$((index + 1))
    label="$(python3 - "$checkpoint" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
parent = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
name = path.name
label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{parent}_{name}")
print(label[:160])
PY
)"
    run_dir="$SWEEP_ROOT/$(printf '%03d' "$index")_$label"
    mkdir -p "$run_dir"
    printf '%s\n' "$checkpoint" > "$run_dir/checkpoint_path.txt"

    echo "=== [$index/$TOTAL] testing $checkpoint ==="
    env \
        DRAFT="$checkpoint" \
        RUN_DIR="$run_dir" \
        INFER_NUM_SPEC="$INFER_NUM_SPEC" \
        BASELINE_DRAFT="$BASELINE_DRAFT" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        MAX_TOKENS="$MAX_TOKENS" \
        GPUS="$GPUS" \
        TP="$TP" \
        BASELINE_PORT="$BASELINE_PORT" \
        DFLASH_PORT="$DFLASH_PORT" \
        bash "$TEST_SCRIPT" > "$run_dir/sweep_stdout.log" 2>&1
    status=$?

    echo "    exit_status=$status"
    write_summary_row "$checkpoint" "$run_dir" "$status" | tee "$run_dir/sweep_result.json"

    if [ "$status" != "0" ] && [ "$CONTINUE_ON_FAIL" != "1" ]; then
        echo "[fatal] stopping after failed checkpoint because CONTINUE_ON_FAIL=$CONTINUE_ON_FAIL"
        exit "$status"
    fi
    echo
done < "$SWEEP_ROOT/checkpoints.txt"

echo "=== Sweep complete ==="
echo "  results csv:   $RESULTS_CSV"
echo "  results jsonl: $RESULTS_JSONL"
echo "  sweep root:    $SWEEP_ROOT"
