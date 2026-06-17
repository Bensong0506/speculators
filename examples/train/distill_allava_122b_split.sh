#!/bin/bash
# 122B ALLaVA 蒸馏 —— 把已有 HAVE 条扩到 TARGET 条,跨 NUM_MACHINES 台机器均分「新增」部分。
# 每台机器各跑一份(各自 part 文件),全部跑完后 cat 合并。复用 distill_allava_122b.sh
# (TP=8 全 8 卡、CONCURRENCY=32、temp=0、RESUME=1 可断点续)。
#
# 用法(两台 A800,10k -> 50k,各 20k):
#   机器 A:  MACHINE=a  bash examples/train/distill_allava_122b_split.sh
#   机器 B:  MACHINE=b  bash examples/train/distill_allava_122b_split.sh
#   两台都跑完,在能拿到两份 part 文件的地方按顺序合并:
#     cat data/allava/allava_122b_distill_10k.jsonl \
#         data/allava/allava_122b_distill_part_a.jsonl \
#         data/allava/allava_122b_distill_part_b.jsonl \
#       > data/allava/allava_122b_distill_50k.jsonl
#
# ⚠️ 两台机器的 MODEL / ALLAVA_INPUTS / ALLAVA_IMAGE_ROOT 必须完全一致,
#    否则全局第 N 条样本对不上,分片会错位/重叠。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

HAVE="${HAVE:-10000}"              # 已经生成好的条数(全局前 HAVE 条,默认你那份 10k)
TARGET="${TARGET:-50000}"          # 目标总条数
NUM_MACHINES="${NUM_MACHINES:-2}"
MACHINE="${MACHINE:?set MACHINE=a|b|...(这台机器的序号:a=0, b=1, c=2 ...;也可直接给数字)}"

case "$MACHINE" in
  a|A|0) IDX=0 ;; b|B|1) IDX=1 ;; c|C|2) IDX=2 ;; d|D|3) IDX=3 ;;
  *) IDX="$MACHINE" ;;
esac

NEW=$(( TARGET - HAVE ))
[ "$NEW" -gt 0 ] || { echo "[fatal] TARGET=$TARGET <= HAVE=$HAVE,无需生成"; exit 1; }
PER=$(( NEW / NUM_MACHINES ))
REM=$(( NEW % NUM_MACHINES ))
THIS=$PER
[ "$IDX" -eq $(( NUM_MACHINES - 1 )) ] && THIS=$(( PER + REM ))   # 余数给最后一台,别漏样本
SKIP=$(( HAVE + IDX * PER ))                                      # 本机从全局第几条开始

OUT="${OUT_JSONL:-$PWD/data/allava/allava_122b_distill_part_${MACHINE}.jsonl}"

echo "=== 122B distill split: 机器 $MACHINE (idx=$IDX / $NUM_MACHINES 台) ==="
echo "  已有 HAVE=$HAVE   目标 TARGET=$TARGET   新增总数 NEW=$NEW"
echo "  >> 本机:SKIP_SAMPLES=$SKIP  MAX_SAMPLES=$THIS  -> $OUT"
echo "  合并(全部机器跑完后):cat 已有 + 各 part_* > 最终 $(( TARGET / 1000 ))k(见脚本头注)"
echo

export SKIP_SAMPLES="$SKIP"
export MAX_SAMPLES="$THIS"
export OUT_JSONL="$OUT"
export CONCURRENCY="${CONCURRENCY:-32}"
exec bash examples/train/distill_allava_122b.sh
