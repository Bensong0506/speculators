#!/bin/bash
# Phase 2 (GPU vLLM 0.22): does CAUSAL within-block attention remove the first-pos
# drop with num_spec (you observed first-pos 3 > 5 > 7)?
#
# Causal is toggled at serve time via the DFLASH_CAUSAL env var. Editing the draft
# config.json (dflash_config.causal=true) was tried first and did NOT take effect
# (vLLM rebuilds dflash_config on load, dropping the manual key — confirmed by
# causal-ON == causal-OFF in the first run). So apply the one-time patch first:
#
#     bash examples/serve/patch_vllm_dflash_causal_env.sh
#
# Then this script serves the ORIGINAL DFlash with DFLASH_CAUSAL=0 vs 1 across
# num_spec and tabulates first-pos/mean-accept. Each cell's vLLM log prints a
# "[DFLASH] dflash_causal=..." line so you can confirm it actually engaged.
#
# INTERPRETATION (original DFlash was TRAINED bidirectional, so causal-ON is a
# mask-mismatch probe):
#   - The mechanism check: under causal-ON, first-pos should become ~FLAT across
#     num_spec (3≈5≈7), because pos-0 stops attending to later block tokens.
#   - If causal-ON first-pos is also >= causal-OFF -> GREEN LIGHT: retrain causal
#     (dflash-causal-block-mask branch) and ship with dflash_config.causal=true.
#   - If flat but lower -> mechanism confirmed; absolute value needs causal RETRAIN.
#
# USAGE
#   DRAFT=/data/wenxuan/Qwen3.5-9B-DFlash \
#   SPECS="3 5 7" NUM_PROMPTS=128 GPUS=0 \
#   bash examples/evaluate/serve_dflash_causal_spec_sweep.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_ROOT="${DEFAULT_ROOT:-/data/wenxuan}"
if [ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /home/wenxuan/Qwen3.5-9B ]; then
    DEFAULT_ROOT="/home/wenxuan"
fi
MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"
DRAFT="${DRAFT:-$DEFAULT_ROOT/Qwen3.5-9B-DFlash}"          # original z-lab DFlash
SPECS="${SPECS:-3 5 7}"

ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-$DEFAULT_ROOT/ALLaVA-4V}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_10k.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
ALLAVA_VAL_JSONL="${ALLAVA_VAL_JSONL:-$REPO_ROOT/data/allava/$(basename "${ALLAVA_JSONL%.jsonl}")_val_tail10pct.jsonl}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/dflash_causal_spec_sweep/$STAMP}"
mkdir -p "$OUT_DIR"

GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-dflash-causal-sweep}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"

# largest spec drives the batched-token budget
MAX_SPEC="$(echo $SPECS | tr ' ' '\n' | sort -n | tail -1)"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_MODEL_LEN + MAX_NUM_SEQS * MAX_SPEC))}"

[ -d "$MODEL" ] || { echo "[fatal] MODEL not found: $MODEL"; exit 1; }
[ -d "$DRAFT" ] || { echo "[fatal] DRAFT (original DFlash) not found: $DRAFT"; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "[fatal] $DRAFT/config.json not found"; exit 1; }
[ -d "$ALLAVA_IMAGE_ROOT" ] || { echo "[fatal] ALLAVA_IMAGE_ROOT not found: $ALLAVA_IMAGE_ROOT"; exit 1; }
[ -s "$ALLAVA_JSONL" ] || { echo "[fatal] ALLAVA_JSONL missing: $ALLAVA_JSONL"; exit 1; }

# Causal is toggled at serve time via the DFLASH_CAUSAL env var, which requires the
# one-time patch examples/serve/patch_vllm_dflash_causal_env.sh (editing the draft
# config.json was tried first and did NOT take effect — vLLM rebuilds dflash_config
# on load, dropping the manual key). Each cell sets DFLASH_CAUSAL=0/1 and the vLLM
# log prints a "[DFLASH] dflash_causal=..." line to confirm.

# ---- ALLaVA val tail (same as the other evals) ----
echo "=== Preparing ALLaVA val tail (last $VAL_RATIO of $(basename "$ALLAVA_JSONL")) ==="
python3 - "$ALLAVA_JSONL" "$ALLAVA_VAL_JSONL" "$VAL_RATIO" <<'PY'
import sys
from pathlib import Path
src, dst, ratio = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
val = lines[int(len(lines) * (1.0 - ratio)):]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(val) + "\n", encoding="utf-8")
print(f"[info] val rows={len(val)} (of {len(lines)})")
PY

# ---- serve/client helpers (mirror test_dflash_allava_val_weights.sh) ----
SERVER_PID=""
cleanup_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Ensure the port is actually released before the next cell. vLLM spawns
    # EngineCore/Worker subprocesses that can outlive the parent and keep holding
    # the port -> the next cell would hit EADDRINUSE. Wait for /health to go down,
    # then free the port as a last resort.
    for _ in $(seq 1 30); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || break
        sleep 2
    done
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    sleep 3
}
trap cleanup_server EXIT

wait_for_server() {
    local log="$1" tag="$2"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $tag server died. Last 80 log lines:"; tail -n 80 "$log" || true
            return 1
        fi
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo "$tag ready."; return 0; }
        sleep 5
    done
    echo "ERROR: $tag timed out."; tail -n 80 "$log" || true; return 1
}

run_cell() {
    local causal="$1" spec="$2"
    local tag="causal${causal}_spec${spec}"
    local dcausal=0; [ "$causal" = "on" ] && dcausal=1
    local log="$OUT_DIR/${tag}_vllm.log"
    local spec_cfg="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$spec}"

    cleanup_server
    echo
    echo "=== [$tag] serve dflash causal=$causal (DFLASH_CAUSAL=$dcausal) num_spec=$spec ==="
    env CUDA_VISIBLE_DEVICES="$GPUS" DFLASH_CAUSAL="$dcausal" vllm serve "$MODEL" \
        --served-model-name "$SERVED_MODEL_NAME" --seed 42 \
        --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" --trust-remote-code \
        --allowed-local-media-path "$ALLAVA_IMAGE_ROOT" --limit-mm-per-prompt '{"image":1}' \
        --generation-config vllm --enforce-eager --dtype "$DTYPE" \
        --attention-backend "$ATTENTION_BACKEND" --no-enable-chunked-prefill \
        --speculative-config "$spec_cfg" \
        --host 0.0.0.0 --port "$PORT" >"$log" 2>&1 &
    SERVER_PID=$!
    if wait_for_server "$log" "$tag"; then
        # confirm causal actually engaged (needs patch_vllm_dflash_causal_env.sh)
        if grep -m1 "\[DFLASH\] dflash_causal" "$log" >/dev/null 2>&1; then
            grep -m1 "\[DFLASH\] dflash_causal" "$log" | sed 's/^/    /'
        else
            echo "    WARN: no [DFLASH] line — apply examples/serve/patch_vllm_dflash_causal_env.sh first, else causal won't toggle."
        fi
        python3 examples/evaluate/mmstar_weight_client.py \
            --endpoint "http://localhost:${PORT}/v1" \
            --data-jsonl "$ALLAVA_VAL_JSONL" \
            --out-jsonl "$OUT_DIR/${tag}_responses.jsonl" \
            --summary-json "$OUT_DIR/${tag}_summary.json" \
            --num "$NUM_PROMPTS" --max-tokens "$MAX_TOKENS" || echo "WARN: client failed for $tag"
    else
        echo '{}' > "$OUT_DIR/${tag}_summary.json"
    fi
    cleanup_server
    sleep 5
}

echo "=== DFlash causal sweep (causal toggled via DFLASH_CAUSAL env) ==="
echo "  model:   $MODEL"
echo "  draft:   $DRAFT  (same draft both arms; causal off vs on via env)"
echo "  specs:   $SPECS   num_prompts: $NUM_PROMPTS   out: $OUT_DIR"
echo "  NOTE: requires examples/serve/patch_vllm_dflash_causal_env.sh applied once."

# Pre-flight: make sure PORT is free (a zombie from a previous run would block us).
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "WARN: something is already serving on :$PORT — freeing it."
    command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 3
fi

for causal in off on; do
    for spec in $SPECS; do
        run_cell "$causal" "$spec"
    done
done

# ---- tabulate first-pos / mean-accept across (causal × spec) ----
python3 - "$OUT_DIR" "$OUT_DIR/causal_spec_sweep_summary.md" "$SPECS" <<'PY' | tee "$OUT_DIR/causal_spec_sweep_summary.stdout.txt"
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_md = Path(sys.argv[2])
specs = [int(x) for x in sys.argv[3].split()]

def load(causal, spec):
    p = out_dir / f"causal{causal}_spec{spec}_summary.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v))

def cell(causal, spec, key):
    return load(causal, spec).get(key)

FP = "spec_first_position_acceptance_rate"
MA = "spec_mean_accepted_tokens_per_draft"
TS = "output_tok_per_sec"

def table(key, title):
    md = [f"### {title}", "", "| num_spec | causal OFF (bi) | causal ON | ON-OFF |",
          "|---:|---:|---:|---:|"]
    for s in specs:
        off, on = cell("off", s, key), cell("on", s, key)
        d = (on - off) if (isinstance(off, float) and isinstance(on, float)) else None
        md.append(f"| {s} | {fmt(off)} | {fmt(on)} | {fmt(d)} |")
    return md

def spread(causal, key):
    vals = [cell(causal, s, key) for s in specs]
    vals = [v for v in vals if isinstance(v, float)]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None

md = ["# DFlash causal vs bidirectional — first-pos across num_spec (original DFlash, GPU serve)",
      "",
      "Original z-lab DFlash (trained bidirectional). causal-ON via "
      "`dflash_config.causal=true` (no vLLM patch). Question: does causal flatten "
      "the first-pos-vs-num_spec curve you saw (3>5>7)?",
      ""]
md += table(FP, "first-pos acceptance"); md += [""]
md += table(MA, "mean accept / draft"); md += [""]
md += table(TS, "tok/s"); md += [""]

fp_off_spread = spread("off", FP)
fp_on_spread = spread("on", FP)
md += ["## verdict", ""]
md += [f"- first-pos spread across num_spec — OFF: `{fmt(fp_off_spread)}`, ON: `{fmt(fp_on_spread)}` "
       "(smaller = flatter; causal should be ~flat)."]
fp0_off = cell("off", specs[0], FP)
fp_hi_on = cell("on", specs[-1], FP)
fp_hi_off = cell("off", specs[-1], FP)
if isinstance(fp_on_spread, float) and isinstance(fp_off_spread, float):
    if fp_on_spread <= fp_off_spread * 0.5 + 1e-9:
        md += ["- ✅ MECHANISM CONFIRMED: causal flattens first-pos across num_spec "
               "(spread roughly halved or better)."]
    else:
        md += ["- ⚠️ causal did NOT clearly flatten the curve — inspect per-cell logs "
               "(did causal=true actually take effect? check the served draft config)."]
if isinstance(fp_hi_on, float) and isinstance(fp_hi_off, float):
    d = fp_hi_on - fp_hi_off
    tag = "GREEN LIGHT (retrain causal)" if d >= 0.005 else ("comparable" if d > -0.005 else "lower (mask mismatch; needs causal retrain)")
    md += [f"- at num_spec={specs[-1]}: causal ON first-pos {fmt(fp_hi_on)} vs OFF {fmt(fp_hi_off)} ({fmt(d)}) -> {tag}"]

out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
PY

echo
echo "Artifacts: $OUT_DIR/causal_spec_sweep_summary.md"
