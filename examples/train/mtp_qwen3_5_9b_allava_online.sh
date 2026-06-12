#!/bin/bash
# Online MTP finetuning on the local Qwen-distilled ALLaVA dataset.
#
# Defaults match the intranet paths recorded in allava-qwen-distill-10k:RUN.md.
# Override env vars as needed, for example:
#   DATA_PATH=/home/wenxuan/speculators/data/allava/allava_qwen35_distill_100k.jsonl \
#   MAX_SAMPLES=100000 \
#   bash examples/train/mtp_qwen3_5_9b_allava_online.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/home/models/Qwen3.5-9B}"
export DATA_PATH="${DATA_PATH:-/home/wenxuan/speculators/data/allava/allava_qwen35_distill_10k.jsonl}"

# Put generated Arrow data, checkpoints, HF caches, and temp files on the same
# large filesystem as the ALLaVA jsonl. This avoids filling small checkout disks
# such as /home/wenxuan_prune during datasets.load_dataset().
DATA_REPO_ROOT="${DATA_REPO_ROOT:-$(dirname "$(dirname "$(dirname "$DATA_PATH")")")}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DATA_REPO_ROOT/output/mtp_qwen3.5_9b_allava_distilled_10k}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
export STITCHED_DIR="${STITCHED_DIR:-$OUTPUT_DIR/stitched}"
export HF_HOME="${HF_HOME:-$DATA_REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TMPDIR="${TMPDIR:-$DATA_REPO_ROOT/.cache/tmp}"

export MAX_SAMPLES="${MAX_SAMPLES:-10000}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export EPOCHS="${EPOCHS:-3}"
export LR="${LR:-1e-4}"
export NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"
export STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-32}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

export VLLM_GPUS="${VLLM_GPUS:-0}"
export TRAIN_GPUS="${TRAIN_GPUS:-1}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

# vLLM needs permission to read local image paths embedded in the ALLaVA jsonl.
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/home/wenxuan/ALLaVA-4V}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---allowed-local-media-path $ALLAVA_IMAGE_ROOT --trust-remote-code}"

mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"

bash "$SCRIPT_DIR/mtp_online_pipeline.sh"
