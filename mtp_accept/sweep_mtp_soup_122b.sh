#!/usr/bin/env bash
# [122B / test_result122B] MTP 权重汤(WiSE-FT / model-soup)alpha sweep。
#
# 对每个 alpha,缝合一个  头 = alpha*finetuned + (1-alpha)*native  的 verifier,
# 在 ALLaVA(in-domain)+ MMStar(OOD)上评接受率。目标:找一个 alpha **既保住
# 域内增益、又把域外拉回(甚至超过)原生**——零重训、零新数据,只混已有权重。
# (9B 上 alpha=0.5 把域外 -4.6% 救成 +1.4%,域内仍 +7.9%。)
#
# 机制:复用 run_mtp_accept_compare.sh。每个 alpha 先用打了 --alpha 的 stitch_mtp.py
# 预缝合到 SOUP_ROOT,再以 TRAINED_MTP_MODEL 传进去(跳过它自己的 alpha=1 缝合)。
# native 与 alpha 无关 → 只在**第一个 alpha**测一次(其余 SKIP_NATIVE=1),省 122B serve。
# 122B 省盘:每个 soup 只真拷贝带 MTP 头的 1~2 分片,其余 hardlink(不是 244GB)。
#
# 用法(机器上):
#   MODEL=/data/wenxuan/Qwen3.5-122B-A10B \
#   MTP_CKPT=$PWD/output/mtp_122b_mm_distilled/mtp122b_50k_s5_b06/checkpoints/checkpoint_best \
#   ALLAVA_JSONL=$PWD/data/allava/allava_122b_distill_50k.jsonl \
#   ALLAVA_IMAGE_ROOT=/data/wenxuan/ALLaVA-4V \
#   MMSTAR_JSONL=$PWD/data/mmstar/mmstar.jsonl MMSTAR_IMAGE_ROOT=/data/wenxuan/mmstar/images \
#   ALPHAS="0.5" TP=4 GPUS=0,1,2,3 NUM_SPEC_TOKENS=7 \
#   bash mtp_accept/sweep_mtp_soup_122b.sh
#
#   # 先一发(最省):ALPHAS="0.5"。要扫:ALPHAS="0.3 0.5 0.7"。
#   # 自检:ALPHAS="1.0" 必须复现 arm A(纯微调头),否则缝合/blend 有问题。
#   # ALLAVA_JSONL 必须是该 MTP 训练用的同一 jsonl(后 10% val 要对得上)。
# 产出:mtp_accept/results/soup_sweep/<时间戳>/soup_sweep_summary.md
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"; cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${MODEL:?set MODEL=/abs/path/to/Qwen3.5-122B-A10B}"
: "${MTP_CKPT:?set MTP_CKPT=/abs/path/to/<best>/checkpoint_best (= arm A, beta=0.6)}"
: "${ALLAVA_IMAGE_ROOT:?set ALLAVA_IMAGE_ROOT=/abs/path (= --allowed-local-media-path)}"
: "${ALLAVA_JSONL:?set ALLAVA_JSONL=/abs/path/to/full_allava.jsonl (自动切后 10% val)}"
[ -d "$MTP_CKPT" ] || { echo "[fatal] MTP_CKPT 不存在: $MTP_CKPT"; exit 1; }

ALPHAS="${ALPHAS:-0.5}"
MMSTAR_JSONL="${MMSTAR_JSONL:-}"; MMSTAR_IMAGE_ROOT="${MMSTAR_IMAGE_ROOT:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"; NUM_PROMPTS="${NUM_PROMPTS:-128}"
TP="${TP:-4}"; GPUS="${GPUS:-0,1,2,3}"
KEEP_SOUP="${KEEP_SOUP:-0}"
SOUP_ROOT="${SOUP_ROOT:-$HERE/stitched/soup}"
RUN="${RUN_DIR:-$HERE/results/soup_sweep/$STAMP}"
mkdir -p "$RUN" "$SOUP_ROOT"

echo "=== MTP soup (WiSE-FT) alpha sweep — 122B ==="
echo "  ckpt:   $MTP_CKPT"
echo "  model:  $MODEL"
echo "  alphas: $ALPHAS   (1.0 = 纯微调头, 0.0 = 原生)"
echo "  spec:   $NUM_SPEC_TOKENS   tp=$TP   gpus=$GPUS   prompts=$NUM_PROMPTS"
echo "  out:    $RUN"

soup_tag() { echo "a$(echo "$1" | tr -d '.')"; }

first=1
for a in $ALPHAS; do
  tag="$(soup_tag "$a")"
  soup="$SOUP_ROOT/$tag"
  echo; echo "############ alpha=$a  ->  $soup ############"
  rm -rf "$soup"
  if ! python3 "$HERE/stitch_mtp.py" "$MTP_CKPT" "$MODEL" --alpha "$a" --output-path "$soup"; then
    echo "[fatal] stitch (alpha=$a) 失败"; exit 1
  fi
  if [ "$first" = "1" ]; then skip_native=""; else skip_native="1"; fi   # native 只测一次
  TRAINED_MTP_MODEL="$soup" RUN_DIR="$RUN/soup_$tag" SKIP_NATIVE="$skip_native" \
    MODEL="$MODEL" \
    ALLAVA_JSONL="$ALLAVA_JSONL" ALLAVA_IMAGE_ROOT="$ALLAVA_IMAGE_ROOT" \
    MMSTAR_JSONL="$MMSTAR_JSONL" MMSTAR_IMAGE_ROOT="$MMSTAR_IMAGE_ROOT" \
    NUM_SPEC_TOKENS="$NUM_SPEC_TOKENS" NUM_PROMPTS="$NUM_PROMPTS" TP="$TP" GPUS="$GPUS" \
    bash "$HERE/run_mtp_accept_compare.sh" \
    || echo "[warn] alpha=$a 评测失败 (看 $RUN/soup_$tag/*_vllm.log)"
  [ "$KEEP_SOUP" = "1" ] || rm -rf "$soup"
  first=0
done

echo; echo "=== 汇总 soup sweep ==="
python3 - "$RUN" "$ALPHAS" "$MTP_CKPT" <<'PY' | tee "$RUN/soup_sweep_summary.stdout.txt"
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); alphas = sys.argv[2].split(); ckpt = sys.argv[3]

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
def tag(a): return "a" + a.replace(".", "")
KEYS = [("mean-acc","spec_mean_accepted_tokens_per_draft"),
        ("first-pos","spec_first_position_acceptance_rate"),
        ("tok/s","output_tok_per_sec")]
def cells(d): return [f"{d.get(k):.3f}" if isinstance(d.get(k),(int,float)) else "n/a" for _,k in KEYS]
def macc(d):
    v = d.get("spec_mean_accepted_tokens_per_draft"); return v if isinstance(v,(int,float)) else None

t0 = tag(alphas[0])  # native 测在第一个 alpha 的 run 里
nat_in  = load(run / f"soup_{t0}" / "allava_native_summary.json")
nat_ood = load(run / f"soup_{t0}" / "mmstar_native_summary.json")

md = [
    "# MTP 权重汤(WiSE-FT)alpha sweep — 122B(in-domain ALLaVA vs OOD MMStar)",
    "", f"ckpt: `{ckpt}`  ",
    "blend: `alpha*finetuned + (1-alpha)*native`  (alpha=1.0=纯微调, 0.0=原生)", "",
    "| alpha | in mean-acc | in first-pos | in tok/s | OOD mean-acc | OOD first-pos | OOD tok/s |",
    "|---|---:|---:|---:|---:|---:|---:|",
    "| native | " + " | ".join(cells(nat_in)+cells(nat_ood)) + " |",
]
for a in alphas:
    ti = load(run / f"soup_{tag(a)}" / "allava_trained_summary.json")
    to = load(run / f"soup_{tag(a)}" / "mmstar_trained_summary.json")
    label = f"{a} (纯微调)" if float(a) >= 1.0 else a
    md.append(f"| {label} | " + " | ".join(cells(ti)+cells(to)) + " |")

ni, no = macc(nat_in), macc(nat_ood)
picks = []
for a in alphas:
    im = macc(load(run / f"soup_{tag(a)}" / "allava_trained_summary.json"))
    om = macc(load(run / f"soup_{tag(a)}" / "mmstar_trained_summary.json"))
    if None not in (im, om, ni, no) and im > ni and om >= no:
        picks.append((a, im, om))
md += ["", "## 结论", ""]
if None in (ni, no):
    md += ["native 指标缺失 —— 检查第一个 alpha 的 *_native_summary.json(是否 0/128、图片路径)。"]
elif picks:
    a, im, om = max(picks, key=lambda x: x[2])
    md += [f"**alpha={a}**:域内保住(mean-acc {im:.3f} > 原生 {ni:.3f})且域外拉回 {om:.3f}(≥ 原生 {no:.3f})→ 全面型头,零重训。"]
else:
    md += [f"没有 alpha 同时满足(域内 > 原生 {ni:.3f})且(域外 ≥ 原生 {no:.3f})。权重汤通常把域外拉回接近原生、域内让一点;看表取最佳折中。域外真正**超过**原生多半要掺通用训练数据,而非只靠汤。"]
out = run / "soup_sweep_summary.md"; out.write_text("\n".join(md)+"\n", encoding="utf-8")
print("\n".join(md)); print(f"\n[written] {out}")
PY

echo; echo "产物:"
echo "  汇总:   $RUN/soup_sweep_summary.md"
echo "  每 alpha:$RUN/soup_a<XX>/{allava,mmstar}_{native,trained}_summary.json"
cp -f "$RUN/soup_sweep_summary.md" output_log_debug/ 2>/dev/null || true
