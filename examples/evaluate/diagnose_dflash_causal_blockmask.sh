#!/bin/bash
# Phase-1 diagnostic (NO training): does a causal within-block mask help the
# ORIGINAL z-lab DFlash? Runs ONE validation pass (per-position top-1 acceptance)
# on the original draft, TWICE — bidirectional block mask vs causal block mask —
# and compares position_0/1/2 acc. No vLLM-serving patch needed: this uses the
# repo's own forward, which honours DFLASH_BLOCK_CAUSAL.
#
# WHY per-position top-1 acc: at temp 0, acceptance == (draft argmax == verifier
# argmax) per position, which is exactly what the DFlash val metrics report.
#
# IMPORTANT INTERPRETATION
#   z-lab DFlash was TRAINED bidirectional. Evaluating it under a causal mask is a
#   mask-MISMATCH probe. Read it as:
#     - causal pos-0 >= bidirectional pos-0  -> GREEN LIGHT: causal helps even
#       without retraining; retraining causal should help more. Go to Phase 2.
#     - causal pos-0 <  bidirectional pos-0  -> INCONCLUSIVE for the idea (it's
#       just train/infer mask mismatch); the real test is RETRAINING with causal,
#       not inference-only. Don't kill the idea on this alone.
#
# USAGE
#   BASELINE_DRAFT=/data/wenxuan/Qwen3.5-9B-DFlash \
#   VLLM_GPUS=0 TRAIN_GPUS=1 MAX_SAMPLES=2000 \
#   bash examples/evaluate/diagnose_dflash_causal_blockmask.sh
#
#   Needs 2 GPUs (verifier vLLM + draft forward). Runs the base launcher twice in
#   VALIDATE_ONLY mode (no training); keep MAX_SAMPLES small (val = last 10%).
#
# OUTPUT: output/dflash_causal_diag/<stamp>/summary.md

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
BASELINE_DRAFT="${BASELINE_DRAFT:-/data/wenxuan/Qwen3.5-9B-DFlash}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
VLLM_GPUS="${VLLM_GPUS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
USE_ALLAVA="${USE_ALLAVA:-1}"
LAUNCHER="${LAUNCHER:-examples/train/dflash_qwen3.5_9b_multimodal_online.sh}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/dflash_causal_diag/$STAMP}"
mkdir -p "$OUT_DIR"

[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$BASELINE_DRAFT" ] || { echo "[fatal] BASELINE_DRAFT (original DFlash) not found: $BASELINE_DRAFT"; exit 1; }
[ -f "$LAUNCHER" ] || { echo "[fatal] launcher not found: $LAUNCHER"; exit 1; }

run_one() {
    local tag="$1" causal="$2"
    local log="$OUT_DIR/${tag}_run.log"
    echo
    echo "============================================================"
    echo "=== [$tag] DFLASH_BLOCK_CAUSAL=$causal  (validate-only, original DFlash) ==="
    echo "============================================================"
    VALIDATE_ONLY=1 \
    DFLASH_BLOCK_CAUSAL="$causal" \
    FINETUNE_FROM="$BASELINE_DRAFT" \
    AUTO_CONVERT_DFLASH=1 \
    REQUIRE_PRETRAINED_WEIGHTS=1 \
    MODEL="$MODEL" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    EPOCHS=1 \
    USE_ALLAVA="$USE_ALLAVA" \
    VLLM_GPUS="$VLLM_GPUS" TRAIN_GPUS="$TRAIN_GPUS" VLLM_PORT="$VLLM_PORT" \
    OUTPUT_DIR="$OUT_DIR/${tag}_data" \
    SAVE_PATH="$OUT_DIR/${tag}_ckpt" \
    RUN_LOG_PATH="$log" \
    LOG_TO_FILE=1 \
    bash "$LAUNCHER" || echo "WARN: [$tag] launcher returned nonzero (see $log)"

    if grep -q "VALIDATE_ONLY_RESULT" "$log" 2>/dev/null; then
        grep "VALIDATE_ONLY_RESULT" "$log" | tail -1 | sed 's/^.*VALIDATE_ONLY_RESULT //' \
            > "$OUT_DIR/${tag}_metrics.json"
        echo "[$tag] metrics -> $OUT_DIR/${tag}_metrics.json"
    else
        echo '{}' > "$OUT_DIR/${tag}_metrics.json"
        echo "WARN: [$tag] no VALIDATE_ONLY_RESULT found in $log"
    fi
}

run_one bidirectional 0
run_one causal 1

echo
echo "============================================================"
echo "=== per-position comparison (bidirectional vs causal) ==="
echo "============================================================"
python3 - \
    "$OUT_DIR/bidirectional_metrics.json" \
    "$OUT_DIR/causal_metrics.json" \
    "$OUT_DIR/summary.md" \
    "$BASELINE_DRAFT" <<'PY' | tee "$OUT_DIR/summary.stdout.txt"
import json
import re
import sys
from pathlib import Path

bi = json.loads(Path(sys.argv[1]).read_text() or "{}")
ca = json.loads(Path(sys.argv[2]).read_text() or "{}")
out_md = Path(sys.argv[3])
draft = sys.argv[4]

pos_re = re.compile(r"position_(\d+)_acc")

def positions(d):
    out = {}
    for k, v in d.items():
        m = pos_re.search(k)
        if m and isinstance(v, (int, float)):
            out[int(m.group(1))] = float(v)
    return out

def loss(d):
    for k in ("loss_epoch", "loss"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v))

bp, cp = positions(bi), positions(ca)
all_pos = sorted(set(bp) | set(cp))

md = [
    "# DFlash causal vs bidirectional block mask — original z-lab DFlash (no training)",
    "",
    "Per-position top-1 acceptance from a single validation pass on the ORIGINAL "
    "DFlash, evaluated under each block mask. (z-lab was trained bidirectional, so "
    "the causal column is a mask-mismatch probe — see verdict.)",
    "",
    f"draft: `{draft}`  ",
    f"val loss: bidirectional {fmt(loss(bi))} / causal {fmt(loss(ca))}",
    "",
    "| position | bidirectional | causal | causal-bi |",
    "|---:|---:|---:|---:|",
]
for p in all_pos:
    b = bp.get(p)
    c = cp.get(p)
    d = (c - b) if (isinstance(b, float) and isinstance(c, float)) else None
    md.append(f"| {p} | {fmt(b)} | {fmt(c)} | {fmt(d)} |")

md += ["", "## verdict (pos-0 = first-position acceptance)", ""]
b0, c0 = bp.get(0), cp.get(0)
if not all_pos:
    md += ["No per-position metrics parsed — check the per-tag *_run.log "
           "(did both runs reach VALIDATE_ONLY_RESULT? were real weights loaded "
           "via AUTO_CONVERT_DFLASH=1?)."]
elif isinstance(b0, float) and isinstance(c0, float):
    d0 = c0 - b0
    if d0 >= 0.005:
        md += [f"GREEN LIGHT: causal pos-0 {fmt(c0)} > bidirectional {fmt(b0)} "
               f"(+{fmt(d0)}) even WITHOUT retraining. Retraining with the causal "
               f"mask should help more → proceed to Phase 2 (vLLM inference patch)."]
    elif d0 <= -0.005:
        md += [f"INCONCLUSIVE: causal pos-0 {fmt(c0)} < bidirectional {fmt(b0)} "
               f"({fmt(d0)}). Expected for a bidirectionally-trained draft under a "
               f"causal mask (train/infer mismatch). This does NOT kill the idea — "
               f"the real test is RETRAINING causal (Phase 1.5), not inference-only."]
    else:
        md += [f"NEUTRAL: causal pos-0 {fmt(c0)} ≈ bidirectional {fmt(b0)} "
               f"({fmt(d0)}). Inference-only causal neither helps nor hurts much; "
               f"causal's real value (num_spec-invariant pos-0) needs a retrain to show."]
else:
    md += ["pos-0 missing — see logs."]

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  summary:  $OUT_DIR/summary.md"
echo "  per-run:  $OUT_DIR/{bidirectional,causal}_run.log + _metrics.json"
