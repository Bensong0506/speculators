#!/usr/bin/env python3
"""Build a small ALLaVA evaluation jsonl in speculators conversations format.

This is intentionally close to scripts/llava_to_jsonl.py, but adds an offset so
the eval slice can avoid the first N samples used by training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llava_to_jsonl import ROLE, load_records, to_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="ALLaVA/LLaVA json or jsonl file. Repeat to merge subsets.",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Dir that relative image paths are resolved against.",
    )
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=100000,
        help="Skip this many valid image records before writing eval samples.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="After skipping, keep every Nth valid record. Default: 1.",
    )
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--conv-key", default="conversations")
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.skip_samples < 0:
        raise SystemExit("--skip-samples must be non-negative")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")

    image_root = Path(args.image_root)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_valid = 0
    after_skip_valid = 0
    written = 0
    missing = 0
    missing_examples: list[str] = []

    with out.open("w", encoding="utf-8") as handle:
        for src in args.inputs:
            records = load_records(Path(src))
            for record in records:
                if written >= args.max_samples:
                    break

                img_rel = record.get(args.image_key)
                if not img_rel:
                    continue
                img_path = Path(img_rel)
                image_abs = str(
                    img_path if img_path.is_absolute() else image_root / img_rel
                )
                if not Path(image_abs).exists():
                    missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(image_abs)
                    continue

                if seen_valid < args.skip_samples:
                    seen_valid += 1
                    continue
                seen_valid += 1

                if after_skip_valid % args.stride != 0:
                    after_skip_valid += 1
                    continue
                after_skip_valid += 1

                conv_out: list[dict] = []
                saw_image = False
                for turn in record.get(args.conv_key) or []:
                    role = ROLE.get(
                        str(turn.get("from", turn.get("role", ""))).lower()
                    )
                    value = (
                        turn.get("value")
                        if "value" in turn
                        else turn.get("content")
                    ) or ""
                    if role is None:
                        continue
                    if role in ("user", "system"):
                        parts, used = to_parts(str(value), image_abs)
                        saw_image = saw_image or used > 0
                        conv_out.append({"role": role, "content": parts})
                    else:
                        conv_out.append(
                            {"role": "assistant", "content": str(value).strip()}
                        )

                if image_abs and not saw_image:
                    for turn in conv_out:
                        if turn["role"] != "user":
                            continue
                        base = (
                            turn["content"]
                            if isinstance(turn["content"], list)
                            else [{"type": "text", "text": turn["content"]}]
                        )
                        turn["content"] = [
                            {"type": "image", "path": image_abs},
                            *base,
                        ]
                        saw_image = True
                        break

                if conv_out and saw_image:
                    handle.write(
                        json.dumps({"conversations": conv_out}, ensure_ascii=False)
                        + "\n"
                    )
                    written += 1

            if written >= args.max_samples:
                break

    print(f"Wrote {written} ALLaVA eval samples -> {out}")
    print(f"Skipped valid image records before eval slice: {args.skip_samples}")
    if missing:
        print(f"WARNING: skipped {missing} missing image path(s) under {image_root}")
        for path in missing_examples:
            print(f"  missing: {path}")
    if written == 0:
        raise SystemExit(
            f"No samples were written to {out}. Check --image-root and inputs."
        )


if __name__ == "__main__":
    main()
