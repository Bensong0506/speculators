#!/bin/bash
# Eval the Qwen3.5-9B DFlash speculator on a RUNNING vLLM server.
#
# This does NOT start vLLM — use your proven launcher (run_qwen35_9b_onecase.sh)
# to serve, then point this client at it. Works on Ascend or CUDA (it's just an
# HTTP client) and measures single-stream output throughput + spec acceptance.
#
# 1) Start the server with your launcher (serves on port 8100):
#       RUN_MODE=dflash DFLASH_SPEC=5 bash run_qwen35_9b_onecase.sh
# 2) In another shell, benchmark it:
#       bash examples/evaluate/eval_qwen35_9b.sh
#
# REAL speedup = compare 'output throughput' between the two modes:
#       RUN_MODE=baseline bash run_qwen35_9b_onecase.sh   # bench -> tok/s = A
#       RUN_MODE=dflash   bash run_qwen35_9b_onecase.sh   # bench -> tok/s = B
#       speedup = B / A
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, so the python path below resolves

# served endpoint (your launcher's default port is 8100)
ENDPOINT="${ENDPOINT:-http://localhost:8100/v1}"

# Images MUST live under the server's --allowed-local-media-path. Your launcher
# uses MM_MEDIA_DIR=/home/wenxuan/multimodel_test, so point IMAGE_DIR there.
# (To eval on MMStar instead, serve with MM_MEDIA_DIR=/home/wenxuan/mmstar/images
#  and set DATA_JSONL=.../data/mmstar/mmstar.jsonl below.)
IMAGE_DIR="${IMAGE_DIR:-/home/wenxuan/multimodel_test}"
DATA_JSONL="${DATA_JSONL:-}"          # set this to use a conversations jsonl instead
QUESTION="${QUESTION:-Describe this image in detail.}"
NUM="${NUM:-32}"
MAX_TOKENS="${MAX_TOKENS:-256}"

ARGS=(--endpoint "$ENDPOINT" --num "$NUM" --max-tokens "$MAX_TOKENS")
if [ -n "$DATA_JSONL" ]; then
    ARGS+=(--data-jsonl "$DATA_JSONL")
else
    ARGS+=(--image-dir "$IMAGE_DIR" --question "$QUESTION")
fi

python3 examples/evaluate/bench_mm_speculative.py "${ARGS[@]}"
