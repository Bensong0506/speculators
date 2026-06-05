#!/bin/bash
# Convert the z-lab DFlash checkpoint -> speculators format so it can be
# warm-started (FINETUNE_FROM loads its weights instead of training from scratch).
#
#   bash examples/train/convert_zlab_dflash.sh
#
# Override paths via env: SRC=... VERIFIER=... OUT=...

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC="${SRC:-/data/wenxuan/Qwen3.5-9B-DFlash}"
VERIFIER="${VERIFIER:-/data/wenxuan/Qwen3.5-9B}"
OUT="${OUT:-/data/wenxuan/Qwen3.5-9B-DFlash-spec}"

python3 scripts/convert_zlab_dflash_to_speculators.py \
    --src "$SRC" --verifier "$VERIFIER" --out "$OUT"

echo
echo "Done. Then TRUE warm-start (weights load because OUT is speculators format):"
echo "  FINETUNE_FROM=$OUT \\"
echo "    ALLAVA_INPUTS=\"/data/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json /data/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.json\" \\"
echo "    ALLAVA_IMAGE_ROOT=/data/wenxuan/ALLaVA-4V MAX_SAMPLES=100000 EPOCHS=2 \\"
echo "    bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh"
