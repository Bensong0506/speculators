#!/bin/bash
# Locate ALLaVA-4V json files + image dirs and tell you what to set for training.
#
#   bash examples/train/find_allava.sh           # searches /home/wenxuan
#   ROOT=/some/dir bash examples/train/find_allava.sh
#
# Use the output to fill ALLAVA_INPUTS (the json paths) and ALLAVA_IMAGE_ROOT
# (the dir that CONTAINS allava_laion/ / allava_vflan/ / images/).

set -uo pipefail   # not -e: find/grep/head may exit nonzero harmlessly

ROOT="${ROOT:-/home/wenxuan}"
echo "Searching under: $ROOT"

echo
echo "=== ALLaVA json files (-> ALLAVA_INPUTS) ==="
find "$ROOT" -maxdepth 6 -iname "*allava*.json" 2>/dev/null | sort

echo
echo "=== image dirs (their parent -> ALLAVA_IMAGE_ROOT) ==="
find "$ROOT" -maxdepth 6 -type d \( -iname images -o -iname "allava_laion" -o -iname "allava_vflan" \) 2>/dev/null | sort

echo
echo "=== sample image present? (must be downloaded, not just the json) ==="
find "$ROOT" -maxdepth 7 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | head -1
echo -n "rough image count: "
find "$ROOT" -maxdepth 7 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | head -200000 | wc -l

echo
echo "=== first record of the first json (format + image path) ==="
J=$(find "$ROOT" -maxdepth 6 -iname "*allava*.json" 2>/dev/null | sort | head -1)
if [ -n "$J" ]; then
    echo "file: $J"
    python3 - "$J" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    r = d[0] if isinstance(d, list) else d
    print("keys :", list(r)[:12])
    print("image:", r.get("image"))
    convs = r.get("conversations", [])
    if convs:
        t = convs[0]
        print("turn0:", {k: (v[:80] if isinstance(v, str) else v) for k, v in t.items()})
except Exception as e:  # noqa: BLE001
    print("could not parse:", e)
PY
else
    echo "(no *allava*.json found under $ROOT — check the path / extraction)"
fi

echo
echo "Then set, e.g.:"
echo "  ALLAVA_INPUTS=\"<json1> <json2>\"   ALLAVA_IMAGE_ROOT=\"<dir containing allava_laion/ or images/>\""
