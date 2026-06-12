#!/bin/bash
# DFlash on 100k Qwen-distilled ALLaVA — the winning config (CE + fp32 + LR 3e-5).
#
# 10x the data of the 10k run, to fix the 10k overfitting (val peaked then
# declined) and lift the acceptance ceiling. Thin wrapper that sets the 100k
# defaults + winning config, then hands off to the validated 10k launcher (same
# warm-start from the open-source DFlash, same online multimodal pipeline).
#
#   bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_100k.sh
#
# NOTES
# - Needs the merged 100k jsonl (data/allava/allava_qwen35_distill_100k.jsonl).
# - Each epoch regenerates the 100k hidden states online (gen-dominated, hours/
#   epoch). EPOCHS=10 here ~= the 10k x 100 run's total sample passes; lower it
#   for a shorter run. Pick the best checkpoint by real acceptance (sweep), not
#   val loss.
# - fp32 (HIDDEN_STATES_DTYPE) is the precision that unlocked first-pos on 10k;
#   if it OOMs, set HIDDEN_STATES_DTYPE=bfloat16.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DISTILLED_ALLAVA_JSONL="${DISTILLED_ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_100k.jsonl}"
export MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export LOSS_FN="${LOSS_FN:-ce}"
export HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-float32}"
export LR_FT="${LR_FT:-3e-5}"
export EPOCHS="${EPOCHS:-10}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export RUN_NAME="${RUN_NAME:-dflash_ce_fp32_lr3e5_100k_$(date +%m%d_%H%M)}"

if [ ! -s "$DISTILLED_ALLAVA_JSONL" ]; then
    echo "[fatal] 100k jsonl not found: $DISTILLED_ALLAVA_JSONL"
    echo "        Build it first (see RUN.md 1d: 100k variant / two-machine split + cat)."
    exit 1
fi

echo "=== DFlash 100k (CE + fp32 + LR 3e-5) ==="
echo "  jsonl:    $DISTILLED_ALLAVA_JSONL"
echo "  samples:  $MAX_SAMPLES"
echo "  epochs:   $EPOCHS   (checkpoint_freq=$CHECKPOINT_FREQ)"
echo "  lr_ft:    $LR_FT    dtype: $HIDDEN_STATES_DTYPE   loss_fn: $LOSS_FN"
echo "  run_name: $RUN_NAME"

exec bash "$SCRIPT_DIR/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh"
