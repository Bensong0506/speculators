#!/usr/bin/env bash
# 原生 MTP vs 训好的 MTP —— 接受率对比 (NPU / vllm-ascend)
# 复用 eval-guidellm harness: vllm serve -> guidellm -> parse_logs (per-position 接受率)
#
# 训好的 MTP head 不能直接 serve(嵌在 verifier 权重里),所以:
#   - 你给原始 checkpoint -> 设 TRAINED_MTP_CKPT,脚本自动 stitch 进 MODEL(不用记有没 stitch 过)
#   - 已经 stitch 好的目录 -> 直接设 TRAINED_MTP_MODEL,跳过 stitch
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
EVAL="$REPO_ROOT/examples/evaluate/eval-guidellm/run_evaluation.sh"

: "${MODEL:?set MODEL=/abs/path/to/Qwen3.5-9B (基座 verifier = native MTP arm + stitch 用)}"
[[ -f "$EVAL" ]] || { echo "[ERROR] 缺 harness: $EVAL (checkout test_result_npu?)" >&2; exit 1; }
command -v guidellm >/dev/null || { echo "[ERROR] 缺 guidellm: pip install guidellm" >&2; exit 1; }

# --- trained arm:已 stitch 的目录优先;否则从原始 ckpt 自动 stitch ---
if [[ -z "${TRAINED_MTP_MODEL:-}" ]]; then
  : "${TRAINED_MTP_CKPT:?set TRAINED_MTP_CKPT=/abs/path/to/finetuned-mtp (原始训练产物;已 stitch 则改设 TRAINED_MTP_MODEL)}"
  STITCH_OUT="${STITCH_OUT:-$HERE/stitched/$(basename "$TRAINED_MTP_CKPT")-stitched}"
  if [[ -d "$STITCH_OUT" && -z "${FORCE_STITCH:-}" ]]; then
    echo "[stitch] 复用已存在: $STITCH_OUT  (FORCE_STITCH=1 重建)"
  else
    echo "[stitch] $TRAINED_MTP_CKPT  +  $MODEL  ->  $STITCH_OUT"
    python "$HERE/stitch_mtp.py" "$TRAINED_MTP_CKPT" "$MODEL" --output-path "$STITCH_OUT"
  fi
  export TRAINED_MTP_MODEL="$STITCH_OUT"
fi
export NATIVE_MTP_MODEL="$MODEL"
echo "[INFO] native=$NATIVE_MTP_MODEL"
echo "[INFO] trained=$TRAINED_MTP_MODEL  method=${MTP_METHOD:-mtp}  spec=${NUM_SPEC_TOKENS:-3}  tp=${TP:-1}"

pkill -f vllm 2>/dev/null || true
STAMP="$(date +%Y%m%d_%H%M%S)"; RUN="${RUN_DIR:-$HERE/results/$STAMP}"
export OUT_NATIVE="$RUN/native" OUT_TRAINED="$RUN/trained"
mkdir -p "$OUT_NATIVE" "$OUT_TRAINED"

echo "==== [1/2] NATIVE  MTP ===="
bash "$EVAL" -c "$HERE/configs/mtp-native.env";  pkill -f vllm 2>/dev/null || true; sleep 5
echo "==== [2/2] TRAINED MTP ===="
bash "$EVAL" -c "$HERE/configs/mtp-trained.env"; pkill -f vllm 2>/dev/null || true

echo ""; echo "############ 接受率对比 (per-position) ############"
echo "----- NATIVE  ($NATIVE_MTP_MODEL) -----"
cat "$OUT_NATIVE/acceptance_analysis.txt"  2>/dev/null || echo "(缺失;看 $OUT_NATIVE/vllm_server.log 是否有 SpecDecoding metrics)"
echo ""; echo "----- TRAINED ($TRAINED_MTP_MODEL) -----"
cat "$OUT_TRAINED/acceptance_analysis.txt" 2>/dev/null || echo "(缺失;看 $OUT_TRAINED/vllm_server.log)"
echo ""
echo "[sanity] 若 trained ≈ native:多半 stitch 的头没被加载 -> 重试 MTP_METHOD=qwen3_5_mtp"
echo "[INFO] 原始日志/结果: $RUN/{native,trained}/"
