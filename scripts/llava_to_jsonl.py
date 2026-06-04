#!/usr/bin/env python3
"""Convert LLaVA-style multimodal data into the speculators `conversations` jsonl.

Works for ALLaVA-4V, LLaVA, ShareGPT4V, etc. — any dataset whose records look like:

    {"id": ...,
     "image": "allava_laion/images/100277305.jpeg",          # relative path
     "conversations": [
         {"from": "human", "value": "<image>\\nPlease depict the image."},
         {"from": "gpt",   "value": "The image displays a silver ring ..."}]}

We: resolve `image` against --image-root, split each human turn's value on the
literal "<image>" placeholder into image+text content parts, map human->user /
gpt->assistant, and emit one {"conversations": [...]} per line — the format
prepare_data.py expects. Text-only records (no image / no <image>) pass through.

The resolved image paths must sit under whatever you give vLLM as
--allowed-local-media-path (i.e. under --image-root).

Usage (combine several ALLaVA subsets into one jsonl):
    python3 scripts/llava_to_jsonl.py \
        --in /home/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json \
        --in /home/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.json \
        --image-root /home/wenxuan/ALLaVA-4V \
        --out-jsonl ./data/allava/allava.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLE = {
    "human": "user",
    "user": "user",
    "system": "system",
    "gpt": "assistant",
    "assistant": "assistant",
}


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "annotations", "items", "records"):
            if isinstance(data.get(k), list):
                return data[k]
    raise SystemExit(f"{path}: no list of records found (top-level {type(data).__name__})")


def to_parts(value: str, image_abs: str | None) -> tuple[list, int]:
    """Split a turn value on '<image>' into image+text content parts."""
    segs = value.split("<image>")
    parts: list[dict] = []
    used = 0
    for i, seg in enumerate(segs):
        if i > 0 and image_abs is not None:
            parts.append({"type": "image", "path": image_abs})
            used += 1
        seg = seg.strip()
        if seg:
            parts.append({"type": "text", "text": seg})
    if not parts:
        parts = [{"type": "text", "text": value.strip()}]
    return parts, used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in", dest="inputs", action="append", required=True,
        help="ALLaVA/LLaVA json or jsonl file (repeatable to merge subsets)",
    )
    ap.add_argument(
        "--image-root", required=True,
        help="Dir that relative image paths are resolved against (e.g. the ALLaVA root)",
    )
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=None, help="Total cap across inputs")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--conv-key", default="conversations")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    missing_examples: list[str] = []
    with out.open("w", encoding="utf-8") as f:
        for src in args.inputs:
            records = load_records(Path(src))
            for r in records:
                if args.max_samples and written >= args.max_samples:
                    break
                img_rel = r.get(args.image_key)
                image_abs = None
                if img_rel:
                    p = Path(img_rel)
                    image_abs = str(p if p.is_absolute() else (image_root / img_rel))
                    if not Path(image_abs).exists():
                        missing += 1
                        if len(missing_examples) < 5:
                            missing_examples.append(image_abs)
                        continue

                conv_out: list[dict] = []
                saw_image = False
                for turn in r.get(args.conv_key) or []:
                    role = ROLE.get(str(turn.get("from", turn.get("role", ""))).lower())
                    val = (turn.get("value") if "value" in turn else turn.get("content")) or ""
                    if role is None:
                        continue
                    if role in ("user", "system"):
                        parts, used = to_parts(val, image_abs)
                        saw_image = saw_image or used > 0
                        conv_out.append({"role": role, "content": parts})
                    else:
                        conv_out.append({"role": "assistant", "content": val.strip()})

                # image present but no <image> placeholder -> prepend to first user turn
                if image_abs and not saw_image:
                    for t in conv_out:
                        if t["role"] == "user":
                            base = t["content"] if isinstance(t["content"], list) \
                                else [{"type": "text", "text": t["content"]}]
                            t["content"] = [{"type": "image", "path": image_abs}, *base]
                            saw_image = True
                            break

                if conv_out:
                    f.write(json.dumps({"conversations": conv_out}, ensure_ascii=False) + "\n")
                    written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f"Wrote {written} samples -> {out}")
    if missing:
        print(f"WARNING: skipped {missing} image path(s) missing on disk under {image_root} "
              "— check --image-root / that images.zip is extracted.")
        for p in missing_examples:
            print(f"  missing: {p}")
    if written == 0:
        raise SystemExit(
            f"No samples were written to {out}. Check --image-root and whether "
            "ALLaVA images are extracted."
        )


if __name__ == "__main__":
    main()
