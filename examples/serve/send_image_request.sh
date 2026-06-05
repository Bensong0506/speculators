#!/bin/bash
# Send ONE multimodal (image) chat request to the running vLLM server.
#
# Use it from a SECOND terminal, after the server is up
# (examples/serve/test_trained_dflash_mm_gpu.sh has printed "Application startup complete").
#
#   bash examples/serve/send_image_request.sh
#
# Override anything via env:
#   IMAGE=/path/under/allowed-media-path/x.jpg PROMPT="这是什么" \
#     PORT=8100 SERVED=qwen3.5-9b bash examples/serve/send_image_request.sh

set -euo pipefail

PORT="${PORT:-8100}"
SERVED="${SERVED:-qwen3.5-9b}"                 # must equal --served-model-name
IMAGE="${IMAGE:-/data/wenxuan/mmstar/images/mmstar_000000.jpg}"  # must be under --allowed-local-media-path
PROMPT="${PROMPT:-描述这张图}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0}"     # greedy → fair speculative-acceptance read

echo "[req] model=$SERVED  image=$IMAGE  prompt=$PROMPT  temperature=$TEMPERATURE"
echo

curl -sS "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED}\",\"temperature\":${TEMPERATURE},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file://${IMAGE}\"}},{\"type\":\"text\",\"text\":\"${PROMPT}\"}]}],\"max_tokens\":${MAX_TOKENS}}"
echo
