#!/bin/bash
# WiSE-FT / model-soup alpha sweep for the finetuned MTP head.
#
# For each alpha, stitch a head  =  alpha*finetuned + (1-alpha)*native  and
# evaluate it on BOTH in-domain (ALLaVA val) and OOD (MMStar). Goal: find an
# alpha that KEEPS the in-domain gain while lifting OOD back toward (or above)
# the native head -- i.e. a "comprehensive" head, with NO retraining and NO new
# data. This only blends weights you already have.
#
# Mechanics: it reuses the existing 2-arm eval scripts. It pre-stitches each
# soup into STITCHED_DIR and passes it in (FORCE_STITCH=0 -> the eval reuses it).
# The native arm is alpha-independent, so it is measured only on the FIRST alpha
# (SKIP_ORIGINAL=1 for the rest) to save server starts.
#
# USAGE
#   MTP_CKPT=/data/wenxuan/speculators/output/<run>/checkpoints/checkpoint_best \
#   ALLAVA_JSONL=/data/wenxuan/speculators/data/allava/allava_qwen35_distill_100k.jsonl \
#   ALPHAS="0.3 0.5 0.7 1.0" INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/sweep_mtp_soup_alpha.sh
#
#   # cheap first check (one blend): ALPHAS=0.5
#   # IMPORTANT: ALLAVA_JSONL must be the SAME jsonl the MTP trained on (the val
#   #            tail = last 10% must match). The 100k run trained on the 100k jsonl.
#
# Needs the mtp-training env (stitch_mtp.py imports speculators.convert.mtp).
# Disk: each soup is a full ~verifier-size copy (~18 GB for 9B); deleted after
#       each alpha unless KEEP_SOUP=1. Stitching is CPU-only (no GPU needed).
# OUTPUT: output/mtp_soup_sweep/<stamp>/soup_sweep_summary.md

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

# verifier path autodetect (mirror the eval scripts: /data first, then /home)
DEFAULT_ROOT="${DEFAULT_ROOT:-/data/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /home/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/home/wenxuan"
fi
MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"

MTP_CKPT="${MTP_CKPT:-}"
ALPHAS="${ALPHAS:-0.3 0.5 0.7 1.0}"
GPUS="${GPUS:-0}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mtp_soup_sweep}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
SOUP_ROOT="${SOUP_ROOT:-$REPO_ROOT/output/mtp_soup_stitched}"
KEEP_SOUP="${KEEP_SOUP:-0}"
mkdir -p "$RUN_DIR" "$SOUP_ROOT"

[ -n "$MTP_CKPT" ] || { echo "[fatal] set MTP_CKPT=/path/to/checkpoint_best"; exit 1; }
[ -d "$MTP_CKPT" ] || { echo "[fatal] MTP_CKPT not found: $MTP_CKPT"; exit 1; }
[ -d "$MODEL" ]    || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -s "$ALLAVA_JSONL" ] || { echo "[fatal] ALLAVA_JSONL missing/empty: $ALLAVA_JSONL"; exit 1; }

echo "=== MTP soup (WiSE-FT) alpha sweep ==="
echo "  ckpt:    $MTP_CKPT"
echo "  model:   $MODEL"
echo "  alphas:  $ALPHAS   (1.0 = pure finetuned, 0.0 = native)"
echo "  spec:    $INFER_NUM_SPEC   prompts: $NUM_PROMPTS   gpus: $GPUS"
echo "  allava:  $ALLAVA_JSONL"
echo "  out:     $RUN_DIR"

soup_tag() { echo "a$(echo "$1" | tr -d '.')"; }

first=1
for a in $ALPHAS; do
    tag="$(soup_tag "$a")"
    soup="$SOUP_ROOT/$tag"
    echo
    echo "############ alpha=$a  ->  $soup ############"
    rm -rf "$soup"
    if ! python3 scripts/stitch_mtp.py "$MTP_CKPT" "$MODEL" --alpha "$a" --output-path "$soup"; then
        echo "[fatal] stitch (alpha=$a) failed -- run from the mtp-training env"
        exit 1
    fi

    if [ "$first" = "1" ]; then skip_orig=0; else skip_orig=1; fi   # native measured once

    echo "---- in-domain ALLaVA (alpha=$a) ----"
    STITCHED_DIR="$soup" FORCE_STITCH=0 SKIP_ORIGINAL="$skip_orig" \
        MTP_CKPT="$MTP_CKPT" MODEL="$MODEL" GPUS="$GPUS" \
        INFER_NUM_SPEC="$INFER_NUM_SPEC" NUM_PROMPTS="$NUM_PROMPTS" \
        ALLAVA_JSONL="$ALLAVA_JSONL" RUN_DIR="$RUN_DIR/allava_$tag" \
        bash examples/evaluate/test_mtp_allava_orig_vs_trained.sh \
        || echo "[warn] allava eval failed at alpha=$a"

    echo "---- OOD MMStar (alpha=$a) ----"
    STITCHED_DIR="$soup" FORCE_STITCH=0 SKIP_ORIGINAL="$skip_orig" \
        MTP_CKPT="$MTP_CKPT" MODEL="$MODEL" GPUS="$GPUS" \
        INFER_NUM_SPEC="$INFER_NUM_SPEC" NUM_PROMPTS="$NUM_PROMPTS" \
        RUN_DIR="$RUN_DIR/mmstar_$tag" \
        bash examples/evaluate/test_mtp_mmstar_orig_vs_trained.sh \
        || echo "[warn] mmstar eval failed at alpha=$a"

    [ "$KEEP_SOUP" = "1" ] || rm -rf "$soup"
    first=0
done

echo
echo "=== Aggregating soup sweep ==="
python3 - "$RUN_DIR" "$ALPHAS" "$MTP_CKPT" <<'PY' | tee "$RUN_DIR/soup_sweep_summary.stdout.txt"
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
alphas = sys.argv[2].split()
ckpt = sys.argv[3]

def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def tag(a):
    return "a" + a.replace(".", "")

KEYS = [
    ("mean accept/draft", "spec_mean_accepted_tokens_per_draft"),
    ("first-pos", "spec_first_position_acceptance_rate"),
    ("tok/s", "output_tok_per_sec"),
]

def cells(d):
    out = []
    for _, k in KEYS:
        v = d.get(k)
        out.append(f"{v:.4f}" if isinstance(v, (int, float)) else "n/a")
    return out

def mean_acc(d):
    v = d.get("spec_mean_accepted_tokens_per_draft")
    return v if isinstance(v, (int, float)) else None

# native is measured on the first alpha's runs (SKIP_ORIGINAL=0 there)
t0 = tag(alphas[0])
nat_in = load(run_dir / f"allava_{t0}" / "original_mtp_summary.json")
nat_ood = load(run_dir / f"mmstar_{t0}" / "original_mtp_summary.json")

md = [
    "# MTP soup (WiSE-FT) alpha sweep -- in-domain vs OOD",
    "",
    f"ckpt: `{ckpt}`  ",
    "blend: `alpha*finetuned + (1-alpha)*native`  (alpha=1.0 = pure finetuned, 0.0 = native)",
    "",
    "| alpha | in mean-acc | in first-pos | in tok/s | OOD mean-acc | OOD first-pos | OOD tok/s |",
    "|---|---:|---:|---:|---:|---:|---:|",
    "| native (0.0) | " + " | ".join(cells(nat_in) + cells(nat_ood)) + " |",
]
for a in alphas:
    ti = load(run_dir / f"allava_{tag(a)}" / "trained_mtp_summary.json")
    to = load(run_dir / f"mmstar_{tag(a)}" / "trained_mtp_summary.json")
    label = f"{a} (pure ft)" if float(a) >= 1.0 else a
    md.append(f"| {label} | " + " | ".join(cells(ti) + cells(to)) + " |")

# verdict: alpha that keeps in-domain mean-acc > native AND lifts OOD >= native
ni, no = mean_acc(nat_in), mean_acc(nat_ood)
picks = []
for a in alphas:
    im = mean_acc(load(run_dir / f"allava_{tag(a)}" / "trained_mtp_summary.json"))
    om = mean_acc(load(run_dir / f"mmstar_{tag(a)}" / "trained_mtp_summary.json"))
    if None not in (im, om, ni, no) and im > ni and om >= no:
        picks.append((a, im, om))

md += ["", "## verdict", ""]
if None in (ni, no):
    md += ["Native metrics missing -- check the first alpha's original_mtp_summary.json."]
elif picks:
    a, im, om = max(picks, key=lambda x: x[2])
    md += [
        f"**alpha={a}** keeps in-domain up (mean-acc {im:.3f} > native {ni:.3f}) "
        f"AND lifts OOD to {om:.3f} (>= native {no:.3f}) -> comprehensive head, no retrain."
    ]
else:
    md += [
        f"No alpha cleared BOTH (in > native {ni:.3f}) and (OOD >= native {no:.3f}) on "
        "mean-accept. Soup usually recovers OOD toward native while trading some "
        "in-domain; read the table for the best trade-off. A true OOD *lift above "
        "native* most likely needs diverse training data, not just a soup.",
    ]

out = run_dir / "soup_sweep_summary.md"
out.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print(f"\n[written] {out}")
PY

echo
echo "Artifacts:"
echo "  sweep summary: $RUN_DIR/soup_sweep_summary.md"
echo "  per-alpha:     $RUN_DIR/{allava,mmstar}_a<XX>/{original_mtp,trained_mtp}_summary.json"
