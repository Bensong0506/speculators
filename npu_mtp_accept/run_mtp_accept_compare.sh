#!/usr/bin/env bash
# 原生 MTP vs 训好的 MTP —— 接受率对比 (NPU / vllm-ascend)
#
# 复用仓库自带的 eval-guidellm harness:每个 config 走
#   run_evaluation.sh = vllm serve -> guidellm 压测 -> parse_logs.py
# 把两次的 per-position 接受率打印出来对比。serve 在 NPU 上由 vllm-ascend 接管,
# guidellm 是纯客户端(命中 OpenAI 端点),与硬件无关。
#
# 用法:
#   export NATIVE_MTP_MODEL=/abs/path/to/native-9b        # 原生/baseline MTP
#   export TRAINED_MTP_MODEL=/abs/path/to/trained-9b-mtp  # 你训好的 MTP
#   export NUM_SPEC_TOKENS=3 TP=1                          # 按需
#   bash npu_mtp_accept/run_mtp_accept_compare.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
EVAL="$REPO_ROOT/examples/evaluate/eval-guidellm/run_evaluation.sh"

: "${NATIVE_MTP_MODEL:?set NATIVE_MTP_MODEL=/abs/path/to/native-9b}"
: "${TRAINED_MTP_MODEL:?set TRAINED_MTP_MODEL=/abs/path/to/trained-9b-mtp}"
[[ -f "$EVAL" ]] || { echo "[ERROR] 找不到 harness: $EVAL (是不是没 checkout test_result_npu?)" >&2; exit 1; }
command -v guidellm >/dev/null || { echo "[ERROR] 缺 guidellm:pip install guidellm" >&2; exit 1; }

pkill -f vllm 2>/dev/null || true
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN_DIR:-$HERE/results/$STAMP}"
export OUT_NATIVE="$RUN/native" OUT_TRAINED="$RUN/trained"
mkdir -p "$OUT_NATIVE" "$OUT_TRAINED"
echo "[INFO] 结果目录: $RUN   (spec=${NUM_SPEC_TOKENS:-3}, tp=${TP:-1})"

echo "==== [1/2] NATIVE  MTP: $NATIVE_MTP_MODEL ===="
bash "$EVAL" -c "$HERE/configs/mtp-native.env"
pkill -f vllm 2>/dev/null || true; sleep 5

echo "==== [2/2] TRAINED MTP: $TRAINED_MTP_MODEL ===="
bash "$EVAL" -c "$HERE/configs/mtp-trained.env"
pkill -f vllm 2>/dev/null || true

echo ""
echo "############################################################"
echo "#  接受率对比 (per-position) — native vs trained"
echo "############################################################"
echo "----- NATIVE  ($NATIVE_MTP_MODEL) -----"
cat "$OUT_NATIVE/acceptance_analysis.txt" 2>/dev/null || echo "(缺失;看 $OUT_NATIVE/vllm_server.log 是否有 'SpecDecoding metrics')"
echo ""
echo "----- TRAINED ($TRAINED_MTP_MODEL) -----"
cat "$OUT_TRAINED/acceptance_analysis.txt" 2>/dev/null || echo "(缺失;看 $OUT_TRAINED/vllm_server.log)"
echo ""
echo "[INFO] 原始日志/结果都在 $RUN/{native,trained}/"
