#!/bin/bash
# Dump a DFlash checkpoint's config + safetensors weight keys.
# Used to assess whether a z-lab/raw DFlash checkpoint can be converted to the
# speculators format for true warm-start (need its weight-key names).
#
#   bash examples/train/dump_ckpt_keys.sh                       # default z-lab path
#   CKPT=/home/models/Qwen3.5-9B-DFlash bash examples/train/dump_ckpt_keys.sh

set -uo pipefail
CKPT="${CKPT:-/home/models/Qwen3.5-9B-DFlash}"

echo "=== checkpoint dir: $CKPT ==="
ls -la "$CKPT" 2>/dev/null
echo

echo "=== config.json ==="
python3 - "$CKPT" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "config.json")
try:
    print(json.dumps(json.load(open(p)), indent=2))
except Exception as e:  # noqa: BLE001
    print("could not read config.json:", e)
PY
echo

echo "=== safetensors weight keys ==="
python3 - "$CKPT" <<'PY'
import glob, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
if not files:
    print("no *.safetensors in", d, "(maybe pytorch_model.bin?)")
    files = sorted(glob.glob(os.path.join(d, "*.bin")))
keys = []
try:
    from safetensors import safe_open
    for fp in files:
        if fp.endswith(".safetensors"):
            with safe_open(fp, framework="pt") as f:
                keys += list(f.keys())
        else:
            import torch
            keys += list(torch.load(fp, map_location="cpu", weights_only=True).keys())
except Exception as e:  # noqa: BLE001
    print("read error:", e)
print(f"\n{len(keys)} tensors across {len(files)} file(s). All keys:")
for k in keys:
    print("  ", k)
PY
