#!/usr/bin/env bash
# [test_result122B] 两台机器分别评 A/B 训好的 MTP best 的新接受率(各机一臂)。
# 包一层 run_mtp_accept_compare.sh:这台机器 serve native + 本臂 trained,出
# per-position 接受率 + mean-accept(ALLaVA in-domain + MMStar OOD)。
#
# 用法:
#   机器1(A · beta=0.6):
#     ARM=a CKPT=/data/wenxuan/speculators/output/mtp_122b_mm_distilled/mtp122b_50k_s5_b06/checkpoints/checkpoint_best \
#       bash mtp_accept/eval_mtp_ab.sh
#   机器2(B · beta=1.0):
#     ARM=b CKPT=.../mtp122b_50k_s5_b10/checkpoints/checkpoint_best \
#       bash mtp_accept/eval_mtp_ab.sh
# (CKPT 改成你实际的 RUN_NAME 路径;两台都先 git checkout test_result122B && git pull)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ARM:?set ARM=a|b(这台机器评哪一臂)}"
: "${CKPT:?set CKPT=/abs/path/to/this-arm/checkpoint_best}"
[ -d "$CKPT" ] || { echo "[fatal] CKPT 不存在: $CKPT"; exit 1; }

export MODEL="${MODEL:-/data/wenxuan/Qwen3.5-122B-A10B}"
export TRAINED_MTP_CKPT="$CKPT"
export ALLAVA_JSONL="${ALLAVA_JSONL:-$PWD/data/allava/allava_122b_distill_50k.jsonl}"
export ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-/data/wenxuan/ALLaVA-4V}"
export MMSTAR_JSONL="${MMSTAR_JSONL:-$PWD/data/mmstar/mmstar.jsonl}"
export MMSTAR_IMAGE_ROOT="${MMSTAR_IMAGE_ROOT:-/data/wenxuan/mmstar/images}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"   # 评测深度(与 122B 报告一致;deploy 深度后面 spec sweep 定。训的是 steps=5,想看"匹配深度"可设 5)
export TP="${TP:-4}" GPUS="${GPUS:-0,1,2,3}"
export RUN_DIR="${RUN_DIR:-$HERE/results/arm_${ARM}}"

echo "=== eval ARM=$ARM ==="
echo "  ckpt:  $CKPT"
echo "  spec:  $NUM_SPEC_TOKENS   tp=$TP   gpus=$GPUS"
echo "  out:   $RUN_DIR"
bash "$HERE/run_mtp_accept_compare.sh"

# 报告回传:给 summary 加 arm 前缀,免得两台撞名(output_log_debug 是目录才拷)
if [ -d "$PWD/output_log_debug" ]; then
  for f in "$RUN_DIR"/*_summary.json; do
    [ -f "$f" ] && cp -f "$f" "$PWD/output_log_debug/arm_${ARM}_$(basename "$f")" 2>/dev/null || true
  done
fi
echo "[INFO] ARM=$ARM 完成 -> $RUN_DIR"
echo "       两台都跑完:比 A vs B 的 *_trained_summary.json(mean-accept / 中后位 per-position);各自 vs native 已在本机表里。"
