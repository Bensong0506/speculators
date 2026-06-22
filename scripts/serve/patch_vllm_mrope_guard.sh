#!/bin/bash
# Neutralize vLLM 0.22.0's hard M-RoPE guard so a DFlash draft can serve a
# Qwen-VL (M-RoPE) verifier.
#
# Root cause (from the traceback): vllm/v1/spec_decode/llm_base_proposer.py
# calls self._raise_if_mrope() in __init__, which raises
#   NotImplementedError: ... does not support M-RoPE yet
# on the stable 0.22.0 build. This script comments out that CALL (idempotent,
# backs up the file once).
#
#   bash examples/serve/patch_vllm_mrope_guard.sh
#   # then re-run: bash examples/serve/test_trained_dflash_gpu.sh
#
# CAVEAT: this only removes the *guard*. Text serving should then work (M-RoPE
# degenerates to standard positions), which is enough to verify the trained
# weights run + measure text speedup. Full image/video serving wants the real
# fallback that vLLM nightly ships:
#   uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
# Revert anytime: cp <file>.bak <file>

set -euo pipefail

F=$(python3 -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "v1", "spec_decode", "llm_base_proposer.py"))')
echo "Target: $F"
[ -f "$F" ] || { echo "[fatal] not found — is vllm importable in this python3?"; exit 1; }

[ -f "$F.bak" ] || cp "$F" "$F.bak"

python3 - "$F" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "patched-out _raise_if_mrope" in src:
    print("already patched.")
elif "self._raise_if_mrope()" in src:
    src = re.sub(
        r'(\n[ \t]*)self\._raise_if_mrope\(\)',
        r'\1pass  # patched-out _raise_if_mrope (allow M-RoPE; text fallback)',
        src,
    )
    open(path, "w", encoding="utf-8").write(src)
    print("OK: neutralized the self._raise_if_mrope() call.")
else:
    print("WARN: 'self._raise_if_mrope()' not found — vLLM version may differ.")
    print("      Open the file and remove/return-early the _raise_if_mrope guard manually.")
PY

echo "Done. Restart vLLM:  bash examples/serve/test_trained_dflash_gpu.sh"
echo "Revert:              cp \"$F.bak\" \"$F\""
