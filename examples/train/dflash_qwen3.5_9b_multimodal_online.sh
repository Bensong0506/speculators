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
MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"

# Some VLM processors/configs need remote code. Set to 1 if loading fails
# with "trust_remote_code" errors; harmless to leave on for Qwen-VL.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# --- 2) Multimodal dataset -------------------------------------------------
# Option A (built-in sanity check): "sharegpt4v_coco"
#   -> pulls Lin-Chen/ShareGPT4V text + your local COCO 2017 train images.
#      Download: http://images.cocodataset.org/zips/train2017.zip
#      Set COCO_DIR to the folder that CONTAINS train2017/.
# Option B (your own data): path to a .jsonl file (see format note at bottom).
DATASET="${DATASET:-sharegpt4v_coco}"
export COCO_DIR="/path/to/coco"           # only used by sharegpt4v_coco

# Root directory that contains ALL images referenced by the dataset.
# vLLM will only load local images located under this path. For
# sharegpt4v_coco this is your COCO_DIR; for custom data set it to the
# common parent of your image files.
MEDIA_ROOT="${MEDIA_ROOT:-$COCO_DIR}"

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
MMSTAR_SRC="${MMSTAR_SRC:-/data/wenxuan/mmstar/mmstar_answers.json}"
MMSTAR_SPLIT="${MMSTAR_SPLIT:-val}"
# Folder vLLM is allowed to read images from (must be a prefix of the image
# paths). For the json-with-paths case set it to your images dir; leave EMPTY
# for the HF-extract case (the extraction dir is used instead).
MMSTAR_MEDIA_ROOT="${MMSTAR_MEDIA_ROOT:-/data/wenxuan/mmstar/images}"

# Option D (REAL training): ALLaVA-4V (or any LLaVA-style json). Step 0 converts
# it to a conversations jsonl via scripts/llava_to_jsonl.py. Set the paths below.
USE_ALLAVA="${USE_ALLAVA:-1}"
# Dir that contains allava_laion/ , allava_vflan/ (where images.zip was extracted):
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/wenxuan/ALLaVA-4V}"
# One or more ALLaVA json files, space-separated (caption + instruct subsets):
ALLAVA_INPUTS="${ALLAVA_INPUTS:-$ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json $ALLAVA_IMAGE_ROOT/allava_laion/ALLaVA-Instruct-LAION-4V.json}"

# --- 3) General training knobs --------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3.5_9b_mm}"
SAVE_PATH="${SAVE_PATH:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
NO_RESUME_FROM_CHECKPOINT="${NO_RESUME_FROM_CHECKPOINT:-0}"
AUTO_CONVERT_DFLASH="${AUTO_CONVERT_DFLASH:-0}"
CONVERTED_DFLASH_OUT="${CONVERTED_DFLASH_OUT:-}"
REQUIRE_PRETRAINED_WEIGHTS="${REQUIRE_PRETRAINED_WEIGHTS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"  # 5k = sanity check only. Use 100k+ for real quality.
SEQ_LENGTH="${SEQ_LENGTH:-4096}"    # Feeds vLLM --max-model-len /
                                    # --max-num-batched-tokens, and trainer
                                    # --total-seq-len. Raise only if you have
                                    # enough memory; long-image samples may drop.
PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-3584}"  # Conservative filter
                                                        # before vLLM expands MM
                                                        # inputs into prompt tokens.
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"  # set 1 to rebuild cached arrow data
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5}"  # save every N epochs
FORCE_EAGER="${FORCE_EAGER:-0}"          # set 1 to disable torch.compile in training
DFLASH_COMPILE="${DFLASH_COMPILE:-1}"    # keep on for flex-attention memory efficiency
VALIDATE_INITIAL="${VALIDATE_INITIAL:-0}" # set 1 to run true step-0 validation before training

# --- Experiment tracking (loss / acceptance curves) -----------------------
# tensorboard = local, intranet-friendly (view via SSH tunnel — see RUN.md /
# examples/train/view_tensorboard.sh). Use LOGGER=wandb with WANDB_BASE_URL for
# a self-hosted/internal W&B server.
LOGGER="${LOGGER:-tensorboard}"
RUN_NAME="${RUN_NAME:-dflash_qwen3.5_9b_mm}"
LOG_DIR="${LOG_DIR:-./train_logs}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
RUN_LOG_DIR="${RUN_LOG_DIR:-./run_logs}"
RUN_LOG_NAME="${RUN_NAME//\//_}"
RUN_LOG_PATH="${RUN_LOG_PATH:-$RUN_LOG_DIR/${RUN_LOG_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$LOG_TO_FILE" = "1" ]; then
    mkdir -p "$(dirname "$RUN_LOG_PATH")"
    exec > >(tee -a "$RUN_LOG_PATH") 2>&1
    echo "=== Writing full launcher log to $RUN_LOG_PATH ==="
fi

# --- 4) DFlash-specific ----------------------------------------------------
# SPECULATOR_TYPE=mtp trains the verifier's NATIVE multi-token-prediction head
# (extracted + fine-tuned). dflash-only knobs below (block_size/anchors/vocab/
# finetune-from) are then ignored by MTPDraftModel.
SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"   # MTP prediction steps
STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"           # MTP FastMTP step-weight decay
# Empty = use the warm-start checkpoint block size; from scratch falls back to 8.
BLOCK_SIZE="${BLOCK_SIZE:-}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"  # max anchor positions sampled per step (memory knob)
NUM_LAYERS=5            # draft transformer layers (DFlash typically uses ~5)
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE-32000}"  # empty = full vocab; default scratch uses reduced vocab

# --- DSpark-specific (used when SPECULATOR_TYPE=dspark) -------------------
# Repo convention: BLOCK_SIZE includes the anchor at position 0. So
# BLOCK_SIZE=8 means DSpark predicts gamma=7 speculative tokens.
MARKOV_RANK="${MARKOV_RANK:-256}"
CE_LOSS_ALPHA="${CE_LOSS_ALPHA:-0.1}"
L1_LOSS_ALPHA="${L1_LOSS_ALPHA:-0.9}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
CONFIDENCE_HEAD_WITH_MARKOV="${CONFIDENCE_HEAD_WITH_MARKOV:-1}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-4.0}"
# CE label source: ground_truth (paper-faithful, default) or target_argmax (old).
CE_TARGET="${CE_TARGET:-ground_truth}"

# --- Warm-start (continue-training) from a pretrained DFlash --------------
# Point at a DFlash checkpoint dir (e.g. /data/wenxuan/Qwen3.5-9B-DFlash-spec) to
# FINE-TUNE it instead of training from scratch. The block below reads its
# config.json and auto-matches block_size / num_layers / draft_arch / aux
# target-layer-ids / mask_token_id / FULL vocab. Empty = train from scratch.
FINETUNE_FROM="${FINETUNE_FROM:-}"
LR_FT="${LR_FT:-1e-5}"   # conservative LR used when warm-starting (vs 3e-4 from scratch)

# Target text-decoder layers to extract hidden states from. Leave EMPTY to
# auto-compute the repo-default "2  L/2  L-3" (L = text num_hidden_layers);
# launch_vllm appends the final layer L automatically. The SAME ids must be
# passed to both vLLM and train.py — this script guarantees that.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-}"

# --- 5) GPU layout on the A800 node ---------------------------------------
# Online training runs vLLM (serving) and the trainer (FSDP) on SEPARATE GPUs.
# Defaults assume an 8x A800 (80GB) node, split half/half. All overridable via env.
# Big verifiers (won't fit per-GPU) MUST use VLLM_TP (shards weights) instead of
# VLLM_DP (replicates the full model per replica). vLLM uses TP x DP GPUs total,
# which must equal the count in VLLM_GPUS.
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_TP="${VLLM_TP:-1}"      # tensor-parallel: shard weights across GPUs (use for big models)
VLLM_DP="${VLLM_DP:-4}"      # data-parallel: replicate weights (only for small models)
GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.85}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
# ===========================================================================

TRC_FLAG=()
if [ "${TRUST_REMOTE_CODE}" = "1" ]; then
    TRC_FLAG=(--trust-remote-code)
fi

# If FINETUNE_FROM is set, read the checkpoint's config.json to match its DFlash
# recipe (block_size / num_layers / draft_arch / aux target-layer-ids / mask /
# FULL vocab). Weights are LOADED only if it's a *speculators*-format checkpoint
# (has speculators_model_type). A raw DFlash ckpt (e.g. z-lab's) can't be loaded
# by this repo's from_pretrained, so we keep the recipe and train FROM SCRATCH.
FROM_FLAG=()
REQUESTED_BLOCK_SIZE="${BLOCK_SIZE:-}"
if [ -n "$FINETUNE_FROM" ]; then
    [ -f "$FINETUNE_FROM/config.json" ] || { echo "[fatal] $FINETUNE_FROM/config.json not found"; exit 1; }
    INITIAL_SPEC_FORMAT=$(python3 - "$FINETUNE_FROM/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(1 if "speculators_model_type" in c else 0)
PY
)
    if [ "$INITIAL_SPEC_FORMAT" != "1" ] && [ "$AUTO_CONVERT_DFLASH" = "1" ]; then
        CONVERTED_DFLASH_OUT="${CONVERTED_DFLASH_OUT:-${FINETUNE_FROM%/}-speculators}"
        echo "=== Converting raw DFlash checkpoint for warm-start ==="
        echo "    raw: $FINETUNE_FROM"
        echo "    out: $CONVERTED_DFLASH_OUT"
        if [ -f "$CONVERTED_DFLASH_OUT/config.json" ]; then
            echo "    reuse existing converted checkpoint"
        else
            python3 scripts/convert_zlab_dflash_to_speculators.py \
                --src "$FINETUNE_FROM" \
                --verifier "$MODEL" \
                --out "$CONVERTED_DFLASH_OUT"
        fi
        FINETUNE_FROM="$CONVERTED_DFLASH_OUT"
    fi

    echo "=== Aligning recipe to $FINETUNE_FROM/config.json ==="
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
print(f'SPEC_FORMAT={1 if "speculators_model_type" in c else 0}')
PY
)"
    PRETRAINED_BLOCK_SIZE="$BLOCK_SIZE"
    if [ -n "$REQUESTED_BLOCK_SIZE" ]; then
        BLOCK_SIZE="$REQUESTED_BLOCK_SIZE"
        BLOCK_SIZE_NOTE="checkpoint block_size=$PRETRAINED_BLOCK_SIZE; training override block_size=$BLOCK_SIZE"
    else
        BLOCK_SIZE_NOTE="checkpoint/training block_size=$BLOCK_SIZE"
    fi
    DRAFT_VOCAB_SIZE=""                          # match pretrained: FULL vocab (no mapping)
    LR="$LR_FT"                                  # lower LR
    if [ -z "$SAVE_PATH" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_ft"            # separate dir when save path is not explicit
    fi
    echo "    -> $BLOCK_SIZE_NOTE"
    echo "    -> num_layers=$NUM_LAYERS draft_arch=$DRAFT_ARCH"
    echo "    -> target_layer_ids='$TARGET_LAYER_IDS' mask_token_id='$MASK_TOKEN_ID' (full vocab, lr=$LR)"
    if [ "${SPEC_FORMAT:-0}" = "1" ]; then
        FROM_FLAG=(--from-pretrained "$FINETUNE_FROM")
        echo "    -> speculators-format checkpoint: LOADING its weights (true warm-start)"
    else
        echo "    -> NOTE: $FINETUNE_FROM is NOT a speculators-format checkpoint"
        echo "       (no speculators_model_type) -> this repo can't load its weights."
        echo "       Training FROM SCRATCH with this recipe (full vocab / block_size=$BLOCK_SIZE"
        echo "       / $DRAFT_ARCH / aux=[$TARGET_LAYER_IDS]). See me about a converter for true warm-start."
        if [ "$REQUIRE_PRETRAINED_WEIGHTS" = "1" ]; then
            echo "[fatal] REQUIRE_PRETRAINED_WEIGHTS=1 but $FINETUNE_FROM is not speculators-format."
            echo "        Set AUTO_CONVERT_DFLASH=1 or point FINETUNE_FROM at a converted checkpoint."
            exit 1
        fi
    fi
fi

if [ -z "$BLOCK_SIZE" ]; then
    BLOCK_SIZE=8
fi
if [ -z "$SAVE_PATH" ]; then
    SAVE_PATH="$OUTPUT_DIR/checkpoints"
fi

# Conditional trainer flags (mirror the TRC_FLAG pattern; empty arrays are no-ops)
VOCAB_FLAG=();     [ -n "$DRAFT_VOCAB_SIZE" ]  && VOCAB_FLAG=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
DRAFTARCH_FLAG=(); [ -n "${DRAFT_ARCH:-}" ]    && DRAFTARCH_FLAG=(--draft-arch "$DRAFT_ARCH")
MASK_FLAG=();      [ -n "${MASK_TOKEN_ID:-}" ] && MASK_FLAG=(--mask-token-id "$MASK_TOKEN_ID")
SPEC_FLAG=()
if [ "$SPECULATOR_TYPE" = "mtp" ]; then
    SPEC_FLAG=(--num-speculative-steps "$NUM_SPECULATIVE_STEPS" --step-weight-beta "$STEP_WEIGHT_BETA")
fi
DSPARK_FLAG=()
if [ "$SPECULATOR_TYPE" = "dspark" ]; then
    DSPARK_FLAG=(
        --markov-rank "$MARKOV_RANK"
        --ce-loss-alpha "$CE_LOSS_ALPHA"
        --l1-loss-alpha "$L1_LOSS_ALPHA"
        --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA"
        --loss-decay-gamma "$LOSS_DECAY_GAMMA"
        --ce-target "$CE_TARGET"
    )
    if [ "$CONFIDENCE_HEAD_WITH_MARKOV" = "1" ]; then
        DSPARK_FLAG+=(--confidence-head-with-markov)
    else
        DSPARK_FLAG+=(--no-confidence-head-with-markov)
    fi
fi
NO_RESUME_FLAG=()
if [ "$NO_RESUME_FROM_CHECKPOINT" = "1" ]; then
    NO_RESUME_FLAG=(--no-resume-from-checkpoint)
fi
FORCE_EAGER_FLAG=()
if [ "$FORCE_EAGER" = "1" ]; then
    FORCE_EAGER_FLAG=(--force-eager)
    DFLASH_COMPILE=0
fi
VALIDATE_INITIAL_FLAG=()
if [ "$VALIDATE_INITIAL" = "1" ]; then
    VALIDATE_INITIAL_FLAG=(--validate-initial)
fi
export SPECULATORS_DFLASH_COMPILE="$DFLASH_COMPILE"

# Auto-compute TARGET_LAYER_IDS from the verifier's *text* config if unset,
# so vLLM and the trainer always agree.
if [ -z "${TARGET_LAYER_IDS}" ]; then
    # MTP consumes ONLY the verifier's last hidden state (model target_layer_ids=[L]);
    # eagle3/dflash use aux layers "2  L/2  L-3". Auto-compute per type.
    echo "=== Auto-computing --target-layer-ids from model config (type=$SPECULATOR_TYPE) ==="
    TARGET_LAYER_IDS=$(python3 - "$MODEL" "${TRUST_REMOTE_CODE}" "$SPECULATOR_TYPE" <<'PY'
import sys
from transformers import AutoConfig
model, trc, stype = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
cfg = AutoConfig.from_pretrained(model, trust_remote_code=trc)
cfg = getattr(cfg, "text_config", cfg)   # VLM -> text decoder
L = cfg.num_hidden_layers
print(L) if stype == "mtp" else print(2, L // 2, L - 3)
PY
)
    echo "    type=$SPECULATOR_TYPE -> TARGET_LAYER_IDS = ${TARGET_LAYER_IDS}"
fi

if ! [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "[fatal] BLOCK_SIZE must be an integer, got '$BLOCK_SIZE'"
    exit 1
fi
if ! [[ "$MAX_ANCHORS" =~ ^[0-9]+$ ]]; then
    echo "[fatal] MAX_ANCHORS must be an integer, got '$MAX_ANCHORS'"
    exit 1
fi
if [ -z "$DRAFT_VOCAB_SIZE" ] && (( 10#$BLOCK_SIZE >= 16 && 10#$MAX_ANCHORS > 512 )); then
    echo "[fatal] Full-vocab DFlash with block_size=$BLOCK_SIZE and MAX_ANCHORS=$MAX_ANCHORS is likely to OOM."
    echo "        Use MAX_ANCHORS=512 or lower for the 4-GPU training split."
    exit 1
fi

echo "=== Resolved training limits ==="
echo "    seq_length: $SEQ_LENGTH"
echo "    preprocessing keep length: $PREPROCESS_SEQ_LENGTH"
echo "    dflash block_size: $BLOCK_SIZE"
echo "    dflash max_anchors: $MAX_ANCHORS"
echo "    dflash num_layers: $NUM_LAYERS"
echo "    dflash draft vocab: ${DRAFT_VOCAB_SIZE:-full}"
echo "    checkpoint_freq: $CHECKPOINT_FREQ"
echo "    save_path: $SAVE_PATH"
echo "    no_resume_from_checkpoint: $NO_RESUME_FROM_CHECKPOINT"
echo "    auto_convert_dflash: $AUTO_CONVERT_DFLASH"
echo "    require_pretrained_weights: $REQUIRE_PRETRAINED_WEIGHTS"
echo "    force_eager_training: $FORCE_EAGER"
echo "    dflash_compile_training: $DFLASH_COMPILE"
echo "    validate_initial: $VALIDATE_INITIAL"
echo "    target_layer_ids: $TARGET_LAYER_IDS"
if [ "$SPECULATOR_TYPE" = "dspark" ]; then
    echo "    dspark markov_rank: $MARKOV_RANK"
    echo "    dspark loss weights: ce=$CE_LOSS_ALPHA l1=$L1_LOSS_ALPHA confidence=$CONFIDENCE_HEAD_ALPHA gamma=$LOSS_DECAY_GAMMA"
    echo "    dspark confidence_head_with_markov: $CONFIDENCE_HEAD_WITH_MARKOV"
    echo "    dspark ce_target: $CE_TARGET"
fi

# Step 0 (optional): build a `conversations` jsonl from the chosen data source.
# Idempotent: skips conversion only when the source/image-root fingerprint matches.
DATASET_REBUILT=0
if [ "${USE_ALLAVA}" = "1" ]; then
    echo "=== Step 0: Converting ALLaVA-4V -> conversations jsonl ==="
    ALLAVA_JSONL="$(pwd)/data/allava/allava_${MAX_SAMPLES}.jsonl"  # name carries the count so changing MAX_SAMPLES regenerates
    ALLAVA_FINGERPRINT="${ALLAVA_JSONL}.fingerprint.json"
    CURRENT_ALLAVA_FINGERPRINT=$(python3 - "$ALLAVA_IMAGE_ROOT" "$MAX_SAMPLES" $ALLAVA_INPUTS <<'PY'
import json, sys
keys = ("image_root", "max_samples")
payload = dict(zip(keys, sys.argv[1:3]))
payload["inputs"] = sys.argv[3:]
payload["conversion_policy"] = "skip_missing_images_v1"
print(json.dumps(payload, sort_keys=True, indent=2))
PY
)
    build_allava_jsonl() {
        IN_ARGS=()
        for j in $ALLAVA_INPUTS; do IN_ARGS+=(--in "$j"); done
        python3 scripts/llava_to_jsonl.py \
            "${IN_ARGS[@]}" \
            --image-root "$ALLAVA_IMAGE_ROOT" \
            --out-jsonl "$ALLAVA_JSONL" \
            --max-samples "$MAX_SAMPLES"
        printf '%s\n' "$CURRENT_ALLAVA_FINGERPRINT" > "$ALLAVA_FINGERPRINT"
        DATASET_REBUILT=1
    }
    if [ -s "$ALLAVA_JSONL" ] && [ -f "$ALLAVA_FINGERPRINT" ] \
        && printf '%s\n' "$CURRENT_ALLAVA_FINGERPRINT" | cmp -s - "$ALLAVA_FINGERPRINT"; then
        echo "    reuse existing $ALLAVA_JSONL"
    else
        if [ -s "$ALLAVA_JSONL" ]; then
            echo "    ALLaVA source/root changed or fingerprint missing -> rebuilding $ALLAVA_JSONL"
        fi
        build_allava_jsonl
    fi
    DATASET="$ALLAVA_JSONL"
    MEDIA_ROOT="$ALLAVA_IMAGE_ROOT"
elif [ "${USE_MMSTAR}" = "1" ]; then
    echo "=== Step 0: Preparing MMStar (-> conversations jsonl) ==="
    MMSTAR_JSONL="$(pwd)/data/mmstar/mmstar_${MAX_SAMPLES}.jsonl"
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
echo "    vLLM/training max length: $SEQ_LENGTH"
echo "    preprocessing keep length: $PREPROCESS_SEQ_LENGTH"
PREPROCESS_FINGERPRINT="$OUTPUT_DIR/.preprocess_fingerprint.json"
CURRENT_PREPROCESS_FINGERPRINT=$(python3 - "$MODEL" "$DATASET" "$MAX_SAMPLES" "$PREPROCESS_SEQ_LENGTH" "$TRUST_REMOTE_CODE" <<'PY'
import json, sys
keys = ("model", "data", "max_samples", "seq_length", "trust_remote_code")
print(json.dumps(dict(zip(keys, sys.argv[1:])), sort_keys=True, indent=2))
PY
)
EXISTING_ARROW="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.arrow' -print -quit 2>/dev/null || true)"
PREPARE_DATA_ARGS=(
    --model "$MODEL"
    --data "$DATASET"
    --output "$OUTPUT_DIR"
    --max-samples "$MAX_SAMPLES"
    --seq-length "$PREPROCESS_SEQ_LENGTH"
    "${TRC_FLAG[@]}"
)
clear_preprocess_cache() {
    python3 - "$OUTPUT_DIR" "$PREPROCESS_FINGERPRINT" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
fingerprint = Path(sys.argv[2])
if not output.exists():
    raise SystemExit

for path in output.glob("*.arrow"):
    path.unlink()
for name in ("state.json", "dataset_info.json", "token_freq.pt"):
    path = output / name
    if path.exists():
        path.unlink()
if fingerprint.exists():
    fingerprint.unlink()
PY
}

if [ "$FORCE_PREPROCESS" = "1" ] || [ "${DATASET_REBUILT:-0}" = "1" ]; then
    if [ "$FORCE_PREPROCESS" = "1" ]; then
        echo "    FORCE_PREPROCESS=1 -> rebuilding cached arrow data"
    else
        echo "    converted dataset rebuilt -> rebuilding cached arrow data"
    fi
    clear_preprocess_cache
    python3 scripts/prepare_data.py "${PREPARE_DATA_ARGS[@]}"
    printf '%s\n' "$CURRENT_PREPROCESS_FINGERPRINT" > "$PREPROCESS_FINGERPRINT"
elif [ -n "$EXISTING_ARROW" ] && [ -f "$PREPROCESS_FINGERPRINT" ] \
    && printf '%s\n' "$CURRENT_PREPROCESS_FINGERPRINT" | cmp -s - "$PREPROCESS_FINGERPRINT"; then
    echo "    reuse existing preprocessed data in $OUTPUT_DIR"
elif [ -n "$EXISTING_ARROW" ] && [ ! -f "$PREPROCESS_FINGERPRINT" ]; then
    echo "    reuse existing preprocessed data in $OUTPUT_DIR"
    echo "    note: no fingerprint found; writing one for future parameter-change checks"
    printf '%s\n' "$CURRENT_PREPROCESS_FINGERPRINT" > "$PREPROCESS_FINGERPRINT"
else
    if [ -n "$EXISTING_ARROW" ]; then
        echo "    preprocessing parameters changed -> rebuilding cached arrow data"
        clear_preprocess_cache
        python3 scripts/prepare_data.py "${PREPARE_DATA_ARGS[@]}"
    else
        python3 scripts/prepare_data.py "${PREPARE_DATA_ARGS[@]}"
    fi
    printf '%s\n' "$CURRENT_PREPROCESS_FINGERPRINT" > "$PREPROCESS_FINGERPRINT"
fi

# Step 2: Launch vLLM server (verifier) in the background -------------------
# --allowed-local-media-path lets vLLM read the local images referenced by
# file path in the dataset. Everything after `--` is passed straight to vLLM.
echo "=== Step 2: Launching vLLM server ==="
filter_vllm_access_logs() {
    grep -v -E '^\(ApiServer_[^)]*\) INFO: .*"(POST /v1/chat/completions|GET /health|GET /v1/models) HTTP/1\.1" 200 OK$'
}
# The generation server re-tokenizes the FULL multimodal prompt (text + image
# token expansion + chat template), which can exceed the training sequence
# length SEQ_LENGTH and get rejected with HTTP 400 "Input length (N) exceeds
# model's maximum context length". Give the gen server headroom so those
# prompts are not dropped. Training is still capped at SEQ_LENGTH: --total-seq-len
# plus the collate slice_and_pad truncate everything back to SEQ_LENGTH.
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-$((SEQ_LENGTH + 2048))}"
echo "    gen vLLM max-model-len: $GEN_MAX_MODEL_LEN (training stays at $SEQ_LENGTH)"
echo "    vLLM parallelism: TP=$VLLM_TP x DP=$VLLM_DP on GPUs [$VLLM_GPUS], mem_util=$GEN_GPU_MEM_UTIL"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python3 scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" \
       --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --allowed-local-media-path "$MEDIA_ROOT" \
       --trust-remote-code \
       --max-model-len "$GEN_MAX_MODEL_LEN" \
       --max-num-batched-tokens "$GEN_MAX_MODEL_LEN" \
       --gpu-memory-utilization "$GEN_GPU_MEM_UTIL" \
       --limit-mm-per-prompt '{"image": 1}' \
       --enforce-eager \
       > >(filter_vllm_access_logs) \
       2> >(filter_vllm_access_logs >&2) &
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
    --save-path "$SAVE_PATH" \
    "${VOCAB_FLAG[@]}" \
    "${FROM_FLAG[@]}" \
    "${DRAFTARCH_FLAG[@]}" \
    "${MASK_FLAG[@]}" \
    "${NO_RESUME_FLAG[@]}" \
    "${FORCE_EAGER_FLAG[@]}" \
    "${VALIDATE_INITIAL_FLAG[@]}" \
    --epochs "$EPOCHS" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --lr "$LR" \
    --hidden-states-dtype "${HIDDEN_STATES_DTYPE:-bfloat16}" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    "${SPEC_FLAG[@]}" \
    "${DSPARK_FLAG[@]}" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --loss-fn "${LOSS_FN:-kl_div}" \
    --logger "$LOGGER" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --on-missing "${ON_MISSING:-generate}" \
    --on-generate "${ON_GENERATE:-delete}" \
    "${TRC_FLAG[@]}"

echo "Done. Checkpoints saved to $SAVE_PATH/"

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
