#!/bin/bash
# Dump how the INSTALLED vLLM handles DFlash causal/config, so we can write a patch
# that matches YOUR vLLM (the stock pip 0.22.0 differs from the local vllm-fork).
#
# Just run it; it writes everything to vllm_dflash_probe.out at the repo root.
# Then:  git add vllm_dflash_probe.out && git commit -m probe && git push
# (or paste the file contents back).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$REPO_ROOT/vllm_dflash_probe.out"

VLLM="$(python3 -c 'import os,vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null)"

{
echo "### vllm version + location"
python3 -c 'import vllm; print("version:", getattr(vllm,"__version__","?"))' 2>/dev/null
echo "dir: $VLLM"
[ -n "$VLLM" ] && [ -d "$VLLM" ] || { echo "[fatal] cannot import vllm"; exit 0; }

echo
echo "### all files mentioning dflash"
grep -rln -i dflash "$VLLM" 2>/dev/null

echo
echo "### every file's causal / dflash_config / non_causal lines (with context)"
for f in $(grep -rln -iE "dflash" "$VLLM" 2>/dev/null); do
    if grep -qiE "causal|dflash_config|non_causal" "$f" 2>/dev/null; then
        echo "==================== $f ===================="
        grep -n -B2 -A3 -iE "causal|dflash_config\b|non_causal|use_non_causal" "$f"
        echo
    fi
done

echo
echo "### how the DFlash proposer reads config (spec_decode/dflash.py if present)"
for cand in \
    "$VLLM/v1/spec_decode/dflash.py" \
    "$VLLM/v1/spec_decode/dflash_proposer.py" \
    "$VLLM/model_executor/models/qwen3_dflash.py"; do
    if [ -f "$cand" ]; then
        echo "==================== $cand ===================="
        grep -n -iE "self\.config|hf_config|dflash_config|num_speculative_tokens|\.get\(|class .*Proposer|def __init__|causal|attn|mask" "$cand" | head -50
        echo
    fi
done

echo
echo "### grep get_dflash_causal / dflash_causal anywhere"
grep -rn -iE "get_dflash_causal|dflash_causal" "$VLLM" 2>/dev/null
} 2>&1 | tee "$OUT"

echo
echo "Wrote: $OUT  ($(wc -l < "$OUT") lines)"
echo "Now:   git add vllm_dflash_probe.out && git commit -m probe && git push"
