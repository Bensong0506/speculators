#!/usr/bin/env python3
"""Convert MMStar into a speculators multimodal `conversations` jsonl.

Supports TWO input forms:

  (1) A JSON / JSONL file of flat records, each with an image FILE PATH plus a
      question and answer. This matches a pre-extracted MMStar dump, e.g.:
          {"id": 0, "answer": "A",
           "question": "...\\nOptions: A: ... B: ...",
           "image": "/home/wenxuan/mmstar/images/mmstar_000000.jpg",
           "category": "...", "l2_category": "...", "meta_info": {...}}
      Images are referenced IN PLACE (nothing is copied). Relative image paths
      are resolved against the json file's directory.

  (2) A HuggingFace MMStar dataset (a save_to_disk dir, a parquet file, or a hub
      id like `Lin-Chen/MMStar`) whose `image` column holds INLINE images. Those
      are extracted to --image-dir as jpgs and then referenced by path.

By default the multiple-choice items are rewritten to OPEN-ENDED form: the
"Options: A: .. B: .." list is stripped from the question and the letter answer
is replaced by the full text of the chosen option. This gives the draft model a
natural, multi-token target instead of a single letter. Pass --no-open-ended to
keep the raw multiple-choice question + letter answer.

Output (one JSON object per line) is the `conversations` format that
prepare_data.py expects (its `load_raw_dataset` reads .json/.jsonl and the
preprocessing reads the `conversations` column):

    {"conversations": [
        {"role": "user", "content": [
            {"type": "image", "path": "/abs/img.jpg"},
            {"type": "text",  "text": "<question>"}]},
        {"role": "assistant", "content": "<answer>"}]}

Caveat: MMStar is a ~1.5k EVAL benchmark; even open-ended, answers are short
(one option's worth of text). Good for SMOKE-TESTING the multimodal pipeline,
but for real speculator quality use large data with full-length responses.

Usage (your pre-extracted dump):
    python3 scripts/mmstar_to_jsonl.py \
        --mmstar /home/wenxuan/mmstar/mmstar_answers.json \
        --out-jsonl ./data/mmstar/mmstar.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

# Split the question stem from the trailing "Options: ..." block.
_OPTIONS_SPLIT_RE = re.compile(r"\n?\s*options?\s*[:：]\s*", re.IGNORECASE)
# Match each "<LETTER>: <text>" option, ending at the next ", <LETTER>:" or EOS.
# Restricting the label to a single A-H avoids matching letters/periods that
# appear inside the option text itself (e.g. "...water.").
_OPT_RE = re.compile(r"([A-H])\s*[:.．、]\s*(.+?)(?=,\s*[A-H]\s*[:.．、]|$)", re.DOTALL)


def _to_open_ended(question: str, answer: str) -> tuple[str, str]:
    """Turn a multiple-choice MMStar item into an open question + full answer.

    'What is X?\\nOptions: A: foo, B: bar' + 'B'  ->  ('What is X?', 'bar').
    Falls back to the original (question, answer) if it can't parse cleanly.
    """
    parts = _OPTIONS_SPLIT_RE.split(question, maxsplit=1)
    if len(parts) < 2:
        return question.strip(), answer.strip()
    stem = parts[0].strip()
    options = {
        m.group(1).upper(): m.group(2).strip().rstrip(",").strip()
        for m in _OPT_RE.finditer(parts[1])
    }
    key = next((c for c in answer.upper() if c in "ABCDEFGH"), "")
    full = options.get(key)
    if stem and full:
        return stem, full
    return question.strip(), answer.strip()


def _load_flat_json(path: Path) -> list[dict]:
    """Load a JSON array (or JSONL, or a dict wrapping a list) of records."""
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
    raise SystemExit(
        f"Could not find a list of records in {path}. "
        f"Top-level type is {type(data).__name__}."
    )


def _to_pil(cell):
    """Coerce a datasets image cell to PIL (decoded image or {bytes,path})."""
    from PIL import Image  # noqa: PLC0415

    if hasattr(cell, "convert"):  # already a PIL image
        return cell
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"]))
        if cell.get("path"):
            return Image.open(cell["path"])
    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def _load_hf(src: str, split: str):
    """Load a HF MMStar dataset from a dir / parquet / hub id."""
    from datasets import (  # noqa: PLC0415
        DatasetDict,
        load_dataset,
        load_from_disk,
    )

    p = Path(src)
    if p.exists():
        try:
            ds = load_from_disk(src)
        except Exception:  # noqa: BLE001
            if p.is_file():
                builder = "parquet" if p.suffix == ".parquet" else "json"
                ds = load_dataset(builder, data_files=src)
            else:
                ds = load_dataset(src)
    else:
        ds = load_dataset(src)  # hub id; respects HF_ENDPOINT mirror

    if isinstance(ds, DatasetDict):
        ds = ds[split] if split in ds else ds[next(iter(ds.keys()))]
    return ds


def _make_conv(img_path: str, question: str, answer: str) -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def main():  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mmstar",
        default="Lin-Chen/MMStar",
        help="JSON/JSONL file of records (image=path), or a HF dataset dir/parquet/id",
    )
    ap.add_argument("--split", default="val", help="HF split to use (default: val)")
    ap.add_argument("--out-jsonl", required=True, help="Output .jsonl path")
    ap.add_argument(
        "--image-dir",
        default=None,
        help="Where to dump images (only used for HF inline-image input)",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="Cap #samples")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument(
        "--no-open-ended",
        dest="open_ended",
        action="store_false",
        default=True,
        help="Keep raw multiple-choice questions + letter answers "
        "(default: rewrite to open-ended question + full-text answer).",
    )
    args = ap.parse_args()

    def emit(f, img_path: str, question: str, answer: str) -> None:
        if args.open_ended:
            question, answer = _to_open_ended(question, answer)
        f.write(
            json.dumps(_make_conv(img_path, question, answer), ensure_ascii=False)
            + "\n"
        )

    src = Path(args.mmstar)
    is_flat_json = src.is_file() and src.suffix in (".json", ".jsonl")

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0

    with out.open("w", encoding="utf-8") as f:
        if is_flat_json:
            # --- Form (1): records already reference image files on disk -----
            records = _load_flat_json(src)
            if args.max_samples:
                records = records[: args.max_samples]
            for r in records:
                img = r.get(args.image_key)
                if not img:
                    continue
                img_path = Path(img)
                if not img_path.is_absolute():
                    # make relative paths absolute against the json's dir; keep
                    # already-absolute paths verbatim so they match MEDIA_ROOT
                    img_path = (src.parent / img).resolve()
                if not img_path.exists():
                    missing += 1
                emit(
                    f,
                    str(img_path),
                    str(r.get(args.question_key) or "").strip(),
                    str(r.get(args.answer_key) or "").strip(),
                )
                written += 1
        else:
            # --- Form (2): HF dataset with inline images -> extract to disk ---
            if args.image_dir is None:
                raise SystemExit("--image-dir is required for HF inline-image input")
            img_dir = Path(args.image_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            ds = _load_hf(args.mmstar, args.split)
            print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")
            for key in (args.image_key, args.question_key):
                if key not in ds.column_names:
                    raise SystemExit(
                        f"Column '{key}' not found. Available: {ds.column_names}."
                    )
            if args.max_samples:
                ds = ds.select(range(min(args.max_samples, len(ds))))
            for i, ex in enumerate(ds):
                cell = ex.get(args.image_key)
                if cell is None:
                    continue
                pil = _to_pil(cell)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                idx = ex.get("index", ex.get("id", i))
                img_path = (img_dir / f"mmstar_{idx}.jpg").resolve()
                pil.save(img_path, "JPEG", quality=95)
                emit(
                    f,
                    str(img_path),
                    str(ex.get(args.question_key) or "").strip(),
                    str(ex.get(args.answer_key) or "").strip(),
                )
                written += 1

    print(f"Wrote {written} samples -> {out}  (open_ended={args.open_ended})")
    if missing:
        print(
            f"WARNING: {missing} image path(s) in {src} do not exist on disk. "
            "Check the 'image' paths / your images directory."
        )


if __name__ == "__main__":
    main()
