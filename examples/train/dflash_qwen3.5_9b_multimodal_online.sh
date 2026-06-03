#!/bin/bash
# Online DFlash Training Script — Multimodal (Vision-Language) target
#
# Trains a DFlash speculator for a ~9B Qwen3.5 multimodal (VLM) verifier.
#
# IMPORTANT mental model:
#   The DFlash draft model is TEXT-ONLY. It never sees pixels. Multimodal
#   context reaches the drafter purely through the verifier's hidden states
#   (the VLM encodes the image, and the hidden states at every position —
#   including image-token positions — are what the drafter learns from).
#   The draft model is sized from `verifier_config.text_config`, so a VLM
#   "just works" as the verifier.
#
#   Therefore "multimodal DFlash" == normal DFlash, with three deltas:
#     1. --model / --verifier-name-or-path points at a VLM
#        (prepare_data + train auto-use the multimodal AutoProcessor).
#     2. The dataset contains image+text turns.
#     3. vLLM must be allowed to read the local images
#        (--allowed-local-media-path) when images are referenced by file path.
#
# Pipeline (same 3 steps as the text DFlash example):
#   1. prepare_data.py  — chat-template + tokenize with the VLM processor
#   2. launch_vllm.py   — serve the VLM, extract hidden states from target layers
#   3. train.py         — train the DFlash drafter against the live vLLM server
#
# Usage: edit the CONFIG block below, then:
#   bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
#
# Reference tutorial (text version):
#   docs/user_guide/tutorials/train_dflash_online.md

set -euo pipefail

# ===================== CONFIG — EDIT THESE ============================

# --- 1) Verifier (target) multimodal model --------------------------------
# Local path on the A800 box (or HF id). Must be a VLM, e.g. your Qwen3.5-9B.
MODEL="/home/models/Qwen3.5-9B"

# Some VLM processors/configs need remote code. Set to 1 if loading fails
# with "trust_remote_code" errors; harmless to leave on for Qwen-VL.
TRUST_REMOTE_CODE=1

# --- 2) Multimodal dataset -------------------------------------------------
# Option A (built-in sanity check): "sharegpt4v_coco"
#   -> pulls Lin-Chen/ShareGPT4V text + your local COCO 2017 train images.
#      Download: http://images.cocodataset.org/zips/train2017.zip
#      Set COCO_DIR to the folder that CONTAINS train2017/.
# Option B (your own data): path to a .jsonl file (see format note at bottom).
DATASET="sharegpt4v_coco"
export COCO_DIR="/path/to/coco"           # only used by sharegpt4v_coco

# Root directory that contains ALL images referenced by the dataset.
# vLLM will only load local images located under this path. For
# sharegpt4v_coco this is your COCO_DIR; for custom data set it to the
# common parent of your image files.
MEDIA_ROOT="${COCO_DIR}"

# Option C (quickest start — no COCO / no HF download): use a local MMStar set.
# Set USE_MMSTAR=1 and point MMSTAR_SRC at your MMStar data. Step 0 below
# extracts MMStar's inline images to disk and builds a conversations jsonl,
# then uses it as the dataset (overriding DATASET/MEDIA_ROOT above).
# NOTE: MMStar is a ~1.5k multiple-choice EVAL benchmark (single-letter
# answers) — ideal for verifying the multimodal pipeline runs end to end, but
# NOT a real training set. Use large, full-response data for real quality.
USE_MMSTAR="${USE_MMSTAR:-0}"   # smoke test only; set 1 (and USE_ALLAVA=0) to use it
# A JSON/JSONL of records with image FILE PATHS (a pre-extracted dump), or a HF
# MMStar dataset dir/parquet/id (inline images get extracted automatically).
MMSTAR_SRC="/home/wenxuan/mmstar/mmstar_answers.json"
MMSTAR_SPLIT="val"
# Folder vLLM is allowed to read images from (must be a prefix of the image
# paths). For the json-with-paths case set it to your images dir; leave EMPTY
# for the HF-extract case (the extraction dir is used instead).
MMSTAR_MEDIA_ROOT="/home/wenxuan/mmstar/images"

# Option D (REAL training): ALLaVA-4V (or any LLaVA-style json). Step 0 converts
# it to a conversations jsonl via scripts/llava_to_jsonl.py. Set the paths below.
USE_ALLAVA="${USE_ALLAVA:-1}"
# One or more ALLaVA json files, space-separated (caption + instruct subsets):
ALLAVA_INPUTS="${ALLAVA_INPUTS:-/data/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json /data/ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.json}"
# Dir that contains allava_laion/ , allava_vflan/ (where images.zip was extracted):
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/ALLaVA-4V}"

# --- 3) General training knobs --------------------------------------------
OUTPUT_DIR="./output/dflash_qwen3.5_9b_mm"
VLLM_PORT=8000
MAX_SAMPLES=5000        # 5k = sanity check only. Use 100k+ for real quality.
SEQ_LENGTH=8192         # raise if your image+text sequences are long
EPOCHS=5
LR=3e-4

# --- Experiment tracking (loss / acceptance curves) -----------------------
# tensorboard = local, intranet-friendly (view via SSH tunnel — see RUN.md /
# examples/train/view_tensorboard.sh). Use LOGGER=wandb if this box can reach
# wandb.ai (otherwise: WANDB_MODE=offline LOGGER=wandb ... then `wandb sync`).
LOGGER="${LOGGER:-tensorboard}"
RUN_NAME="${RUN_NAME:-dflash_qwen3.5_9b_mm}"
LOG_DIR="${LOG_DIR:-./train_logs}"

# --- 4) DFlash-specific ----------------------------------------------------
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8            # tokens drafted per block (one forward pass)
MAX_ANCHORS=3072        # max anchor positions sampled per step (memory knob)
NUM_LAYERS=5            # draft transformer layers (DFlash typically uses ~5)
DRAFT_VOCAB_SIZE=32000  # reduced draft vocab; auto-cleared (full vocab) when warm-starting

# --- Warm-start (continue-training) from a pretrained DFlash --------------
# Point at a DFlash checkpoint dir (e.g. /home/models/Qwen3.5-9B-DFlash) to
# FINE-TUNE it instead of training from scratch. The block below reads its
# config.json and auto-matches block_size / num_layers / draft_arch / aux
# target-layer-ids / mask_token_id / FULL vocab. Empty = train from scratch.
FINETUNE_FROM="${FINETUNE_FROM:-}"
LR_FT="${LR_FT:-1e-4}"   # lower LR used when warm-starting (vs 3e-4 from scratch)

# Target text-decoder layers to extract hidden states from. Leave EMPTY to
# auto-compute the repo-default "2  L/2  L-3" (L = text num_hidden_layers);
# launch_vllm appends the final layer L automatically. The SAME ids must be
# passed to both vLLM and train.py — this script guarantees that.
TARGET_LAYER_IDS=""

# --- 5) GPU layout on the A800 node ---------------------------------------
# Online training runs vLLM (serving) and the trainer (FSDP) on SEPARATE GPUs.
# Defaults assume an 8x A800 (80GB) node, split half/half.
VLLM_GPUS="0,1,2,3"
VLLM_DP=4               # vLLM data-parallel replicas (use TP instead if model is big)
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# ===========================================================================

TRC_FLAG=()
if [ "${TRUST_REMOTE_CODE}" = "1" ]; then
    TRC_FLAG=(--trust-remote-code)
fi

# Warm-start alignment: if FINETUNE_FROM is set, match the pretrained checkpoint
# exactly (mismatched block_size / aux layers / vocab => weights won't load or
# acceptance collapses). Reads everything from the checkpoint's config.json.
INCLUDE_LAST_FLAG=()
if [ -n "$FINETUNE_FROM" ]; then
    echo "=== Warm-start: aligning to $FINETUNE_FROM/config.json ==="
    [ -f "$FINETUNE_FROM/config.json" ] || { echo "[fatal] $FINETUNE_FROM/config.json not found"; exit 1; }
    eval "$(python3 - "$FINETUNE_FROM/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
df = c.get("dflash_config", {})
tli = df.get("target_layer_ids") or c.get("aux_hidden_state_layer_ids") or []
mt = df.get("mask_token_id", c.get("mask_token_id"))
print(f'BLOCK_SIZE={c.get("block_size", 16)}')
print(f'NUM_LAYERS={c.get("num_hidden_layers") or len(c.get("layer_types", [])) or 5}')
print(f'DRAFT_ARCH={c.get("model_type", "qwen3")}')
print('TARGET_LAYER_IDS="%s"' % " ".join(str(x) for x in tli))
print(f'MASK_TOKEN_ID={mt}' if mt is not None else 'MASK_TOKEN_ID=')
PY
)"
    DRAFT_VOCAB_SIZE=""                          # pretrained uses FULL vocab (no mapping)
    LR="$LR_FT"                                  # lower LR for fine-tuning
    OUTPUT_DIR="${OUTPUT_DIR}_ft"                # separate dir (avoid stale vocab maps)
    INCLUDE_LAST_FLAG=(--no-include-last-layer)  # aux layers == target_layer_ids exactly
    echo "    -> block_size=$BLOCK_SIZE num_layers=$NUM_LAYERS draft_arch=$DRAFT_ARCH"
    echo "    -> target_layer_ids='$TARGET_LAYER_IDS' mask_token_id='$MASK_TOKEN_ID' (full vocab, lr=$LR)"
fi

# Conditional trainer flags (mirror the TRC_FLAG pattern; empty arrays are no-ops)
FROM_FLAG=();      [ -n "$FINETUNE_FROM" ]     && FROM_FLAG=(--from-pretrained "$FINETUNE_FROM")
VOCAB_FLAG=();     [ -n "$DRAFT_VOCAB_SIZE" ]  && VOCAB_FLAG=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
DRAFTARCH_FLAG=(); [ -n "${DRAFT_ARCH:-}" ]    && DRAFTARCH_FLAG=(--draft-arch "$DRAFT_ARCH")
MASK_FLAG=();      [ -n "${MASK_TOKEN_ID:-}" ] && MASK_FLAG=(--mask-token-id "$MASK_TOKEN_ID")

# Auto-compute TARGET_LAYER_IDS from the verifier's *text* config if unset,
# so vLLM and the trainer always agree.
if [ -z "${TARGET_LAYER_IDS}" ]; then
    echo "=== Auto-computing --target-layer-ids from model config ==="
    TARGET_LAYER_IDS=$(python3 - "$MODEL" "${TRUST_REMOTE_CODE}" <<'PY'
import sys
from transformers import AutoConfig
model, trc = sys.argv[1], sys.argv[2] == "1"
cfg = AutoConfig.from_pretrained(model, trust_remote_code=trc)
cfg = getattr(cfg, "text_config", cfg)   # VLM -> text decoder
L = cfg.num_hidden_layers
print(2, L // 2, L - 3)
PY
)
    echo "    text num_hidden_layers based -> TARGET_LAYER_IDS = ${TARGET_LAYER_IDS}"
fi

# Step 0 (optional): build a `conversations` jsonl from the chosen data source.
# Idempotent: skips conversion if the target jsonl already exists (rm it to redo).
if [ "${USE_ALLAVA}" = "1" ]; then
    echo "=== Step 0: Converting ALLaVA-4V -> conversations jsonl ==="
    ALLAVA_JSONL="$(pwd)/data/allava/allava.jsonl"
    if [ -s "$ALLAVA_JSONL" ]; then
        echo "    reuse existing $ALLAVA_JSONL  (rm it to regenerate)"
    else
        IN_ARGS=()
        for j in $ALLAVA_INPUTS; do IN_ARGS+=(--in "$j"); done
        python3 scripts/llava_to_jsonl.py \
            "${IN_ARGS[@]}" \
            --image-root "$ALLAVA_IMAGE_ROOT" \
            --out-jsonl "$ALLAVA_JSONL" \
            --max-samples "$MAX_SAMPLES"
    fi
    DATASET="$ALLAVA_JSONL"
    MEDIA_ROOT="$ALLAVA_IMAGE_ROOT"
elif [ "${USE_MMSTAR}" = "1" ]; then
    echo "=== Step 0: Preparing MMStar (-> conversations jsonl) ==="
    MMSTAR_JSONL="$(pwd)/data/mmstar/mmstar.jsonl"
    MMSTAR_IMG_DIR="$(pwd)/data/mmstar/images"   # only used when extracting from HF
    if [ -s "$MMSTAR_JSONL" ]; then
        echo "    reuse existing $MMSTAR_JSONL  (rm it to regenerate)"
    else
        python3 scripts/mmstar_to_jsonl.py \
            --mmstar "$MMSTAR_SRC" \
            --split "$MMSTAR_SPLIT" \
            --out-jsonl "$MMSTAR_JSONL" \
            --image-dir "$MMSTAR_IMG_DIR" \
            --max-samples "$MAX_SAMPLES"
    fi
    DATASET="$MMSTAR_JSONL"
    if [ -n "${MMSTAR_MEDIA_ROOT}" ]; then
        MEDIA_ROOT="$MMSTAR_MEDIA_ROOT"   # json-with-paths: your images dir
    else
        MEDIA_ROOT="$MMSTAR_IMG_DIR"      # HF-extract: the extraction dir
    fi
fi

# Step 1: Prepare data ------------------------------------------------------
# The VLM's AutoProcessor is loaded automatically; image turns are tokenized
# with the proper <image> placeholders and an assistant loss mask is built.
echo "=== Step 1: Preparing data ==="
python3 scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    "${TRC_FLAG[@]}"

# Step 2: Launch vLLM server (verifier) in the background -------------------
# --allowed-local-media-path lets vLLM read the local images referenced by
# file path in the dataset. Everything after `--` is passed straight to vLLM.
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python3 scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    "${INCLUDE_LAST_FLAG[@]}" \
    -- --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --allowed-local-media-path "$MEDIA_ROOT" \
       --trust-remote-code \
       --max-model-len "$SEQ_LENGTH" \
       --max-num-batched-tokens "$SEQ_LENGTH" \
       --gpu-memory-utilization 0.85 \
       --limit-mm-per-prompt '{"image": 1}' \
       --enforce-eager &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train the DFlash drafter against the live vLLM server -------------
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    "${VOCAB_FLAG[@]}" \
    "${FROM_FLAG[@]}" \
    "${DRAFTARCH_FLAG[@]}" \
    "${MASK_FLAG[@]}" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --logger "$LOGGER" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --on-missing generate \
    --on-generate delete \
    "${TRC_FLAG[@]}"

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"

# ===========================================================================
# CUSTOM MULTIMODAL DATA FORMAT (Option B)
# ---------------------------------------------------------------------------
# Point DATASET at a .jsonl file where each line has a "conversations" field.
# Each turn is {"role": ..., "content": ...}; for multimodal turns, "content"
# is a LIST of parts. Images MUST be file paths or URLs (NOT base64/inline),
# and local paths must live under MEDIA_ROOT.
#
#   {"conversations": [
#       {"role": "user", "content": [
#           {"type": "image", "path": "/data/images/cat.jpg"},
#           {"type": "text",  "text": "What is in this image?"}
#       ]},
#       {"role": "assistant", "content": "A cat sitting on a sofa."}
#   ]}
#
# Then set:  DATASET="/data/my_multimodal.jsonl"   and
#            MEDIA_ROOT="/data/images"
# ===========================================================================
