#!/bin/bash
# Extract ALLaVA-4V image_chunks/*.zip into the images/ dirs the json expects.
#
# ALLaVA ships LAION/VFLAN images as many zip chunks under
#   <root>/allava_laion/image_chunks/*.zip   (and allava_vflan/...)
# while the json references  allava_laion/images/<id>.jpeg  relative to the root.
# This unzips the chunks into the right place (idempotent: -n, resumable).
#
#   ALLAVA_ROOT=/home/wenxuan/ALLaVA-4V bash examples/train/extract_allava_images.sh
#
# Needs `unzip` (apt-get install -y unzip). LAION is large — this can take a while.

set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${ALLAVA_ROOT:-/home/wenxuan/ALLaVA-4V}"
command -v unzip >/dev/null || { echo "[fatal] 'unzip' not found — apt-get install -y unzip"; exit 1; }

extract_subset() {
    local sub="$1"
    local chunks="$ROOT/$sub/image_chunks"
    local imgdir="$ROOT/$sub/images"
    [ -d "$chunks" ] || { echo "[$sub] no image_chunks/ — skip"; return; }
    shopt -s nullglob
    local zips=("$chunks"/*.zip)
    if [ "${#zips[@]}" -eq 0 ]; then echo "[$sub] no .zip in image_chunks/ — skip"; return; fi
    echo "[$sub] ${#zips[@]} zip chunks found -> target images dir: $imgdir"
    mkdir -p "$imgdir"

    # Detect layout from the first zip: does it already contain an images/ prefix?
    local first; first=$(unzip -Z1 "${zips[0]}" 2>/dev/null | grep -m1 -iE '\.(jpe?g|png|webp)$' || true)
    echo "   sample entry: ${first:-<none>}"
    local target="$imgdir"                 # default: zip is flat -> extract into images/
    case "$first" in
        images/*|*/images/*) target="$ROOT/$sub" ;;   # zip carries images/ prefix
    esac
    echo "   extracting into: $target"

    local i=0
    for z in "${zips[@]}"; do
        i=$((i + 1))
        printf '   [%d/%d] %s\n' "$i" "${#zips[@]}" "$(basename "$z")"
        unzip -n -q -d "$target" "$z" || echo "   ! failed on $z"
    done
    local n; n=$(find "$imgdir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | head -3000000 | wc -l)
    echo "[$sub] done -> $imgdir  (images now: $n)"
}

extract_subset allava_laion
extract_subset allava_vflan

echo
echo "=== Sanity check: do the json image paths resolve now? ==="
for j in "$ROOT/allava_laion/ALLaVA-Caption-LAION-4V.json" "$ROOT/allava_vflan/ALLaVA-Caption-VFLAN-4V.json"; do
    [ -f "$j" ] || continue
    echo "--- probing $(basename "$j") ---"
    python3 scripts/llava_to_jsonl.py --in "$j" --image-root "$ROOT" \
        --out-jsonl /tmp/allava_probe.jsonl --max-samples 200 || true
done
echo "(look at the 'missing' counts above — 0 missing = images resolved, ready to train)"
