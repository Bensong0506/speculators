#!/bin/bash
# One command, both datasets: compare mtp / original DFlash / trained DFlash on
#   - MMStar (OOD)        via test_dflash_mmstar_three_way.sh
#   - ALLaVA val (in-dom) via test_dflash_allava_val_weights.sh  (also runs a
#                          no-spec baseline, kept as a reference row)
# at the same spec (INFER_NUM_SPEC), then merge the per-method summaries into one
# cross-dataset table. Each sub-run writes under this run's dir.
#
# Usage:
#   DRAFT=/path/to/checkpoints/checkpoint_best \
#   INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/test_three_way_mmstar_allava.sh
#
# Notes:
#   - DRAFT = the trained DFlash checkpoint to test (e.g. the new fp32/3e-5 run's
#     checkpoint_best, or a sweep-picked epoch dir).
#   - ALLAVA_JSONL defaults to the distilled training jsonl so the ALLaVA val tail
#     matches training; override only if your path differs.
#   - Runs 7 vLLM servers total (3 MMStar + 4 ALLaVA), sequential ~25-30 min on one
#     card. Set GPUS to a free device.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -z "${DRAFT:-}" ]; then
    echo "[fatal] set DRAFT=/path/to/trained/checkpoint (e.g. .../checkpoints/checkpoint_best)"
    exit 1
fi
[ -d "$DRAFT" ] || { echo "[fatal] DRAFT not found: $DRAFT"; exit 1; }

INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/three_way_both}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$STAMP}"
MM_DIR="$RUN_DIR/mmstar"
AL_DIR="$RUN_DIR/allava"
mkdir -p "$MM_DIR" "$AL_DIR"

# Exported so both sub-scripts inherit them; dataset-specific paths are
# auto-detected by each sub-script (or pass MODEL/BASELINE_DRAFT/GPUS/... on the
# command line and they propagate through).
export DRAFT INFER_NUM_SPEC NUM_PROMPTS

echo "=== Three-way on BOTH datasets ==="
echo "  draft:           $DRAFT"
echo "  infer_num_spec:  $INFER_NUM_SPEC"
echo "  num_prompts:     $NUM_PROMPTS"
echo "  allava_jsonl:    $ALLAVA_JSONL"
echo "  output:          $RUN_DIR"

echo
echo "============================================================"
echo "=== [1/2] MMStar three-way (OOD) ==="
echo "============================================================"
RUN_DIR="$MM_DIR" bash examples/evaluate/test_dflash_mmstar_three_way.sh \
    || echo "WARN: MMStar sub-run returned nonzero (see $MM_DIR)"

echo
echo "============================================================"
echo "=== [2/2] ALLaVA four-way (in-domain) ==="
echo "============================================================"
ALLAVA_JSONL="$ALLAVA_JSONL" RUN_DIR="$AL_DIR" \
    bash examples/evaluate/test_dflash_allava_val_weights.sh \
    || echo "WARN: ALLaVA sub-run returned nonzero (see $AL_DIR)"

echo
echo "============================================================"
echo "=== Combined cross-dataset summary ==="
echo "============================================================"
python3 - "$MM_DIR" "$AL_DIR" "$RUN_DIR/combined_summary.md" "$DRAFT" <<'PY' | tee "$RUN_DIR/combined_summary.stdout.txt"
import json
import sys
from pathlib import Path

mm_dir = Path(sys.argv[1])
al_dir = Path(sys.argv[2])
out_md = Path(sys.argv[3])
draft = sys.argv[4]

datasets = [("MMStar (OOD)", mm_dir), ("ALLaVA val (in-domain)", al_dir)]
methods = ["mtp", "trained_dflash", "dflash_original"]

def load(d, m):
    p = d / f"{m}_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)

def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None

data = {dname: {m: load(d, m) for m in methods} for dname, d in datasets}

md = [
    "# Three-way comparison — MMStar (OOD) + ALLaVA (in-domain)",
    "",
    f"draft: `{draft}`",
    "",
    "| dataset | method | tok/s | mean accept/draft | token accept | first-pos | completed |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for dname, _ in datasets:
    for m in methods:
        s = data[dname][m]
        md.append(
            f"| {dname} | {m} | {fmt(s.get('output_tok_per_sec'))} | "
            f"{fmt(s.get('spec_mean_accepted_tokens_per_draft'))} | "
            f"{fmt(s.get('spec_token_acceptance_rate'))} | "
            f"{fmt(s.get('spec_first_position_acceptance_rate'))} | "
            f"{fmt(s.get('completed'))}/{fmt(s.get('num_requested'))} |"
        )

md += ["", "## Trained DFlash vs ... (per dataset)", ""]
for dname, _ in datasets:
    tr = data[dname]["trained_dflash"]
    orig = data[dname]["dflash_original"]
    mtp = data[dname]["mtp"]
    def rel(a_key, ref):
        return ratio(tr.get(a_key), ref.get(a_key))
    md.append(f"**{dname}**")
    md.append(
        f"- vs original: mean-accept `{fmt(rel('spec_mean_accepted_tokens_per_draft', orig))}`, "
        f"first-pos `{fmt(rel('spec_first_position_acceptance_rate', orig))}`, "
        f"tok/s `{fmt(rel('output_tok_per_sec', orig))}`"
    )
    md.append(
        f"- vs MTP: mean-accept `{fmt(rel('spec_mean_accepted_tokens_per_draft', mtp))}`, "
        f"first-pos `{fmt(rel('spec_first_position_acceptance_rate', mtp))}`, "
        f"tok/s `{fmt(rel('output_tok_per_sec', mtp))}`"
    )
    md.append("")

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts:"
echo "  combined: $RUN_DIR/combined_summary.md"
echo "  mmstar:   $MM_DIR/mmstar_three_way_summary.md"
echo "  allava:   $AL_DIR/allava_val_summary.md"
