#!/bin/bash
# Validate the client STEP-1 distilled jsonl before launching STEP 2 MTP training.
#
# Default behavior:
#   1. Check the multimodal distill file exists and has EXPECTED_ROWS rows.
#   2. Validate every JSONL row has speculators-compatible conversations.
#   3. Require image parts to be present and local image paths to exist.
#   4. Start examples/train/nohup_mtp_client_122b.sh only if validation passes.
#
# Usage:
#   bash examples/train/check_client_distill_and_train_122b.sh
#
# Useful overrides:
#   CHECK_ONLY=1 bash examples/train/check_client_distill_and_train_122b.sh
#   EXPECTED_ROWS=8137 CHECK_IMAGES=0 bash examples/train/check_client_distill_and_train_122b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CLIENT_MODEL="${CLIENT_MODEL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B}"
CLIENT_DISTILL_JSONL="${CLIENT_DISTILL_JSONL:-/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/client_122b_distill_multimodal_8137.jsonl}"
CLIENT_IMAGE_ROOT="${CLIENT_IMAGE_ROOT:-/mnt/tidal-alsh01}"
EXPECTED_ROWS="${EXPECTED_ROWS:-8137}"
MIN_IMAGE_ROWS="${MIN_IMAGE_ROWS:-1}"
REQUIRE_ALL_ROWS_WITH_IMAGES="${REQUIRE_ALL_ROWS_WITH_IMAGES:-0}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# Training defaults for the real run. Override normally via env if needed.
export CLIENT_MODEL
export CLIENT_DISTILL_JSONL
export CLIENT_IMAGE_ROOT
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-3e-5}"
export NUM_SPECULATIVE_STEPS="${NUM_SPECULATIVE_STEPS:-3}"
export STEP_WEIGHT_BETA="${STEP_WEIGHT_BETA:-0.6}"

echo "=== Check client distill data ==="
echo "  jsonl:        $CLIENT_DISTILL_JSONL"
echo "  expected:     $EXPECTED_ROWS rows"
echo "  image_root:   $CLIENT_IMAGE_ROOT"
echo "  check_images: $CHECK_IMAGES"

[ -d "$CLIENT_MODEL" ] || { echo "[fatal] CLIENT_MODEL not found: $CLIENT_MODEL"; exit 1; }
[ -s "$CLIENT_DISTILL_JSONL" ] || { echo "[fatal] CLIENT_DISTILL_JSONL missing/empty: $CLIENT_DISTILL_JSONL"; exit 1; }
[ -d "$CLIENT_IMAGE_ROOT" ] || { echo "[fatal] CLIENT_IMAGE_ROOT not found: $CLIENT_IMAGE_ROOT"; exit 1; }

python3 - "$CLIENT_DISTILL_JSONL" "$EXPECTED_ROWS" "$CLIENT_IMAGE_ROOT" "$MIN_IMAGE_ROWS" "$REQUIRE_ALL_ROWS_WITH_IMAGES" "$CHECK_IMAGES" <<'PY'
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

jsonl = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
image_root = Path(sys.argv[3]).resolve()
min_image_rows = int(sys.argv[4])
require_all_rows_with_images = sys.argv[5] == "1"
check_images = sys.argv[6] == "1"

errors = []
warnings = []
assistant_chars = []
role_counts = Counter()
rows = 0
image_rows = 0
image_parts = 0
missing_images = []
literal_image_token_rows = []
empty_assistant_rows = []
text_only_rows = []
sample_preview = None


def add_error(msg):
    if len(errors) < 50:
        errors.append(msg)


def iter_parts(content):
    if isinstance(content, list):
        for part in content:
            yield part


with jsonl.open(encoding="utf-8") as handle:
    for lineno, line in enumerate(handle, 1):
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            row = json.loads(line)
        except Exception as exc:
            add_error(f"line {lineno}: invalid JSON: {exc}")
            continue

        conv = row.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            add_error(f"line {lineno}: conversations must be a list with at least 2 turns")
            continue

        last = conv[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            add_error(f"line {lineno}: last turn must be assistant")
            continue
        answer = last.get("content")
        if not isinstance(answer, str) or not answer.strip():
            empty_assistant_rows.append(lineno)
            add_error(f"line {lineno}: empty assistant answer")
            continue
        assistant_chars.append(len(answer))

        saw_user = False
        row_image_parts = 0
        row_literal_image_token = False
        for turn_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                add_error(f"line {lineno}: turn {turn_idx} is not an object")
                continue
            role = turn.get("role")
            role_counts[str(role)] += 1
            if role == "user":
                saw_user = True
            if role not in ("system", "user", "assistant"):
                add_error(f"line {lineno}: unexpected role {role!r}")

            content = turn.get("content")
            if isinstance(content, str):
                if "<image>" in content:
                    row_literal_image_token = True
                continue
            if isinstance(content, list):
                for part in iter_parts(content):
                    if not isinstance(part, dict):
                        add_error(f"line {lineno}: non-object content part")
                        continue
                    ptype = part.get("type")
                    if ptype == "image":
                        row_image_parts += 1
                        path = part.get("path") or part.get("url")
                        if not isinstance(path, str) or not path:
                            add_error(f"line {lineno}: image part missing path/url")
                            continue
                        if check_images and "://" not in path:
                            p = Path(path)
                            if not p.is_absolute():
                                add_error(f"line {lineno}: image path is not absolute: {path}")
                                continue
                            if not p.exists():
                                if len(missing_images) < 20:
                                    missing_images.append((lineno, path))
                            else:
                                try:
                                    p.resolve().relative_to(image_root)
                                except ValueError:
                                    add_error(f"line {lineno}: image outside image root: {path}")
                    elif ptype == "text":
                        text = part.get("text", "")
                        if isinstance(text, str) and "<image>" in text:
                            row_literal_image_token = True
                    else:
                        add_error(f"line {lineno}: unknown content part type {ptype!r}")
                continue
            add_error(f"line {lineno}: content must be string or list")

        if not saw_user:
            add_error(f"line {lineno}: no user turn")
        if row_image_parts:
            image_rows += 1
            image_parts += row_image_parts
        else:
            text_only_rows.append(lineno)
        if row_literal_image_token:
            literal_image_token_rows.append(lineno)
        if sample_preview is None:
            sample_preview = {
                "line": lineno,
                "turns": len(conv),
                "image_parts": row_image_parts,
                "assistant_chars": len(answer),
                "assistant_preview": answer[:160].replace("\n", " "),
            }

if rows != expected_rows:
    add_error(f"row count mismatch: got {rows}, expected {expected_rows}")
if image_rows < min_image_rows:
    add_error(f"multimodal check failed: image rows {image_rows} < MIN_IMAGE_ROWS {min_image_rows}")
if require_all_rows_with_images and text_only_rows:
    add_error(f"{len(text_only_rows)} row(s) have no image parts; first rows: {text_only_rows[:20]}")
if literal_image_token_rows:
    add_error(f"{len(literal_image_token_rows)} row(s) still contain literal <image>; first rows: {literal_image_token_rows[:20]}")
if missing_images:
    add_error("missing image file(s): " + "; ".join(f"line {ln}: {p}" for ln, p in missing_images[:20]))

print("Data summary")
print(f"  rows:              {rows}")
print(f"  role_counts:       {dict(role_counts)}")
print(f"  image_rows:        {image_rows}")
print(f"  image_parts:       {image_parts}")
print(f"  text_only_rows:    {len(text_only_rows)}")
if assistant_chars:
    sorted_chars = sorted(assistant_chars)
    p95 = sorted_chars[int(0.95 * (len(sorted_chars) - 1))]
    print(f"  assistant_chars:   min={min(sorted_chars)} avg={statistics.mean(sorted_chars):.1f} p95={p95} max={max(sorted_chars)}")
if sample_preview:
    print("Sample preview")
    print(f"  line:              {sample_preview['line']}")
    print(f"  turns:             {sample_preview['turns']}")
    print(f"  image_parts:       {sample_preview['image_parts']}")
    print(f"  assistant_chars:   {sample_preview['assistant_chars']}")
    print(f"  assistant_preview: {sample_preview['assistant_preview']}")

if warnings:
    print("Warnings")
    for warning in warnings:
        print(f"  [warn] {warning}")
if errors:
    print("Validation errors")
    for error in errors:
        print(f"  [fatal] {error}")
    raise SystemExit(1)

print("Validation passed.")
PY

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY=1 -> not starting training."
    exit 0
fi

echo
echo "=== Starting STEP 2 training ==="
echo "  CLIENT_MODEL=$CLIENT_MODEL"
echo "  CLIENT_DISTILL_JSONL=$CLIENT_DISTILL_JSONL"
echo "  CLIENT_IMAGE_ROOT=$CLIENT_IMAGE_ROOT"
echo "  EPOCHS=$EPOCHS LR=$LR NUM_SPECULATIVE_STEPS=$NUM_SPECULATIVE_STEPS STEP_WEIGHT_BETA=$STEP_WEIGHT_BETA"

exec bash examples/train/nohup_mtp_client_122b.sh
