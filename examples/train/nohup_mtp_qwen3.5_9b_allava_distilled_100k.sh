#!/bin/bash
# MTP on 100k Qwen-distilled ALLaVA. Fine-tunes the verifier's NATIVE MTP head
# (extracted from MODEL) on 10x the data, matched to the DFlash 100k run for a
# fair MTP-vs-DFlash comparison. Thin wrapper over the validated MTP launcher.
#
#   bash examples/train/nohup_mtp_qwen3.5_9b_allava_distilled_100k.sh
#
# NOTES
# - Needs the merged 100k jsonl (data/allava/allava_qwen35_distill_100k.jsonl).
# - Same schedule as the DFlash 100k run (EPOCHS=10, fp32, LR 3e-5) so the two
#   are comparable. Each epoch regenerates the 100k hidden states online
#   (gen-dominated, hours/epoch); lower EPOCHS for a shorter run.
# - fp32 matches DFlash; if it OOMs, set HIDDEN_STATES_DTYPE=bfloat16 (the smoke
#   test ran fine in bf16).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DISTILLED_ALLAVA_JSONL="${DISTILLED_ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_100k.jsonl}"
export MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-float32}"
export LR="${LR:-3e-5}"
export EPOCHS="${EPOCHS:-10}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"
export STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"
export RUN_NAME="${RUN_NAME:-mtp_fp32_lr3e5_100k_$(date +%m%d_%H%M)}"

if [ ! -s "$DISTILLED_ALLAVA_JSONL" ]; then
    echo "[fatal] 100k jsonl not found: $DISTILLED_ALLAVA_JSONL"
    echo "        Build it first (see RUN.md 1d: 100k variant / two-machine split + cat)."
    exit 1
fi

echo "=== MTP 100k (fp32 + LR 3e-5, num_speculative_steps=$NUM_SPECULATIVE_STEPS) ==="
echo "  jsonl:    $DISTILLED_ALLAVA_JSONL"
echo "  samples:  $MAX_SAMPLES"
echo "  epochs:   $EPOCHS   (checkpoint_freq=$CHECKPOINT_FREQ)"
echo "  lr:       $LR    dtype: $HIDDEN_STATES_DTYPE"
echo "  run_name: $RUN_NAME"

exec bash "$SCRIPT_DIR/nohup_mtp_qwen3.5_9b_allava_distilled.sh"
