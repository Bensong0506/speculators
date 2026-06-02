#!/usr/bin/env python3
"""Convert MMStar into a speculators multimodal `conversations` jsonl.

Why this is needed:
    MMStar (e.g. `Lin-Chen/MMStar`) stores images INLINE as a parquet `Image`
    feature. The speculators online pipeline extracts hidden states by sending
    the conversation to a live vLLM server, which requires images to be on-disk
    files referenced by path/URL (see data_generation/preprocessing.py — inline
    / base64 images are explicitly rejected). So we dump each embedded image to
    `--image-dir` and emit a jsonl whose turns reference those file paths.

Output format (one JSON object per line), matching what prepare_data.py expects
(the `conversations` column):

    {"conversations": [
        {"role": "user", "content": [
            {"type": "image", "path": "/abs/path/mmstar_3.jpg"},
            {"type": "text",  "text": "<question, with options>"}
        ]},
        {"role": "assistant", "content": "C"}
    ]}

NOTE / caveat:
    MMStar is a ~1.5k multiple-choice EVAL benchmark; its answers are single
    letters. This is great for SMOKE-TESTING that the multimodal training
    pipeline runs end to end (VLM served, image-conditioned hidden states
    extracted, draft model trains), but it is NOT a good training set for a
    real speculator: it is tiny and has ~1 trainable token per sample. For real
    quality, use a large dataset with full-length assistant responses (or run
    response regeneration).

Usage:
    python3 scripts/mmstar_to_jsonl.py \
        --mmstar /home/models/MMStar \
        --out-jsonl ./data/mmstar/mmstar.jsonl \
        --image-dir ./data/mmstar/images
"""

import argparse
import json
from pathlib import Path


def load_mmstar(src: str, split: str):
    """Load MMStar from a local dir/parquet/json, or from a HF hub id."""
    from datasets import (  # noqa: PLC0415
        Dataset,
        DatasetDict,
        load_dataset,
        load_from_disk,
    )

    p = Path(src)
    if p.exists():
        # 1) a `save_to_disk` directory
        try:
            ds = load_from_disk(src)
        except Exception:  # noqa: BLE001
            # 2) a parquet/json file, or a dir of them
            if p.is_file():
                builder = "parquet" if p.suffix == ".parquet" else "json"
                ds = load_dataset(builder, data_files=src)
            else:
                ds = load_dataset(src)
    else:
        # 3) a HuggingFace hub id (respects HF_ENDPOINT mirror)
        ds = load_dataset(src)

    if isinstance(ds, DatasetDict):
        ds = ds[split] if split in ds else ds[next(iter(ds.keys()))]
    assert isinstance(ds, Dataset)  # noqa: S101
    return ds


def to_pil(img):
    """Coerce a datasets image cell to a PIL.Image.

    Handles both a decoded PIL image and the raw `{"bytes": ..., "path": ...}`
    form you get when loading parquet without the Image feature applied.
    """
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if img is None:
        return None
    if hasattr(img, "convert"):  # already a PIL image
        return img
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(io.BytesIO(img["bytes"]))
        if img.get("path"):
            return Image.open(img["path"])
    raise TypeError(f"Unsupported image cell type: {type(img)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mmstar",
        default="Lin-Chen/MMStar",
        help="Local MMStar dir/parquet/json, or a HF hub id (default: Lin-Chen/MMStar)",
    )
    ap.add_argument("--split", default="val", help="Split to use (default: val)")
    ap.add_argument("--out-jsonl", required=True, help="Output .jsonl path")
    ap.add_argument("--image-dir", required=True, help="Where to dump images")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of samples (default: all)",
    )
    ap.add_argument(
        "--image-key", default="image", help="Image column name (default: image)"
    )
    ap.add_argument(
        "--question-key",
        default="question",
        help="Question column name (default: question)",
    )
    ap.add_argument(
        "--answer-key", default="answer", help="Answer column name (default: answer)"
    )
    args = ap.parse_args()

    ds = load_mmstar(args.mmstar, args.split)
    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")
    for key in (args.image_key, args.question_key):
        if key not in ds.column_names:
            raise SystemExit(
                f"Column '{key}' not found. Available: {ds.column_names}. "
                "Pass --image-key/--question-key/--answer-key to match your data."
            )

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    img_dir = Path(args.image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            img = to_pil(ex.get(args.image_key))
            if img is None:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            idx = ex.get("index", i)
            img_path = (img_dir / f"mmstar_{idx}.jpg").absolute()
            img.save(img_path, "JPEG", quality=95)

            question = str(ex.get(args.question_key) or "").strip()
            answer = str(ex.get(args.answer_key) or "").strip()

            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": str(img_path)},
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": answer},
            ]
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples -> {out}")
    print(f"Images           -> {img_dir}")
    print(f"Set in your training script:  DATASET={out}  MEDIA_ROOT={img_dir}")


if __name__ == "__main__":
    main()
