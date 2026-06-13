#!/bin/bash
# Make the INSTALLED vLLM honour a DFLASH_CAUSAL env var for the DFlash within-block
# attention, and print the resolved mode at startup so you can SEE it engage.
#
# WHY: setting dflash_config.causal=true in the draft's config.json did NOT take
# effect (vLLM rebuilds/loads dflash_config so the manual key was dropped). This
# patches the two runtime read points directly, so causal becomes a serve-time env
# toggle, independent of config format:
#   - vllm/v1/spec_decode/dflash.py                        (self.dflash_causal)
#   - vllm/v1/worker/gpu/spec_decode/dflash/utils.py        (get_dflash_causal)
#
# Idempotent + backs up each file once (.dflashcausal.bak). Re-run safely.
#
# USAGE
#   bash examples/serve/patch_vllm_dflash_causal_env.sh          # apply
#   DFLASH_CAUSAL=1 vllm serve ...   # then causal ON;  unset/0 = OFF (default)
#
# To revert: restore the .dflashcausal.bak files (the script prints their paths).

set -uo pipefail

VLLM_DIR="$(python3 -c 'import os,vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null)"
[ -n "$VLLM_DIR" ] && [ -d "$VLLM_DIR" ] || { echo "[fatal] could not locate installed vllm (python3 -c 'import vllm')"; exit 1; }
echo "vllm install: $VLLM_DIR"

python3 - "$VLLM_DIR" <<'PY'
import sys
from pathlib import Path

vllm_dir = Path(sys.argv[1])
MARKER = "DFLASH_CAUSAL"

targets = [
    (
        vllm_dir / "v1" / "spec_decode" / "dflash.py",
        '        self.dflash_causal = self.dflash_config.get("causal", False)',
        (
            '        _env = __import__("os").environ.get("DFLASH_CAUSAL")  # DFLASH_CAUSAL override\n'
            '        self.dflash_causal = (_env not in ("0", "", "false", "False")) '
            'if _env is not None else self.dflash_config.get("causal", False)\n'
            '        print(f"[DFLASH] dflash_causal={self.dflash_causal} '
            '(env DFLASH_CAUSAL={_env})", flush=True)'
        ),
    ),
    (
        vllm_dir / "v1" / "worker" / "gpu" / "spec_decode" / "dflash" / "utils.py",
        '    return dflash_config.get("causal", False)',
        (
            '    _env = __import__("os").environ.get("DFLASH_CAUSAL")  # DFLASH_CAUSAL override\n'
            '    if _env is not None:\n'
            '        return _env not in ("0", "", "false", "False")\n'
            '    return dflash_config.get("causal", False)'
        ),
    ),
]

fail = False
for path, anchor, replacement in targets:
    if not path.exists():
        print(f"[fatal] not found: {path}")
        fail = True
        continue
    text = path.read_text()
    if MARKER in text:
        print(f"[skip] already patched: {path}")
        continue
    if anchor not in text:
        print(f"[fatal] anchor not found in {path}")
        print(f"        expected line:\n        {anchor.strip()}")
        print("        (installed vLLM differs from expected 0.22 layout — paste the")
        print("         function and I'll adjust the patch.)")
        fail = True
        continue
    if text.count(anchor) != 1:
        print(f"[fatal] anchor appears {text.count(anchor)}x in {path} (expected 1)")
        fail = True
        continue
    bak = path.with_suffix(path.suffix + ".dflashcausal.bak")
    if not bak.exists():
        bak.write_text(text)
    path.write_text(text.replace(anchor, replacement))
    print(f"[ok] patched {path}  (backup: {bak})")

if fail:
    raise SystemExit(1)
print("\nDone. Serve with DFLASH_CAUSAL=1 (causal) or unset/0 (bidirectional).")
print("Look for a '[DFLASH] dflash_causal=...' line in the vLLM log to confirm.")
PY