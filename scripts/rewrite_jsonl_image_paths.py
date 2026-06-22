#!/usr/bin/env python3
"""Rewrite local image paths inside speculators conversation JSONL files.

Distilled ALLaVA files often contain absolute image paths from the machine that
created them (for example /data/wenxuan/ALLaVA-4V/...). This helper rewrites
those paths to the local image root used by the current training machine without
touching the source JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rewrite_path(path: str, image_root: Path) -> tuple[str, bool]:
    if "://" in path:
        return path, False

    root = str(image_root).rstrip("/")
    if path == root or path.startswith(f"{root}/"):
        return path, False

    source = Path(path)
    if not source.is_absolute():
        rewritten = str(image_root / source)
        return rewritten, rewritten != path

    parts = source.parts
    root_name = image_root.name
    if root_name in parts:
        idx = parts.index(root_name)
        rewritten = str(image_root.joinpath(*parts[idx + 1 :]))
        return rewritten, rewritten != path

    return path, False


def _rewrite_obj(obj: Any, image_root: Path, stats: dict[str, int]) -> Any:
    if isinstance(obj, list):
        return [_rewrite_obj(item, image_root, stats) for item in obj]

    if not isinstance(obj, dict):
        return obj

    rewritten = {}
    for key, value in obj.items():
        rewritten[key] = _rewrite_obj(value, image_root, stats)

    if rewritten.get("type") == "image" and isinstance(rewritten.get("path"), str):
        stats["image_paths"] += 1
        new_path, changed = _rewrite_path(rewritten["path"], image_root)
        rewritten["path"] = new_path
        stats["rewritten" if changed else "unchanged"] += 1

    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    args = parser.parse_args()

    input_path = Path(args.in_jsonl)
    output_path = Path(args.out_jsonl)
    image_root = Path(args.image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"records": 0, "image_paths": 0, "rewritten": 0, "unchanged": 0}
    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            record = _rewrite_obj(record, image_root, stats)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["records"] += 1

    print(
        "Rewrote image paths: "
        f"records={stats['records']} image_paths={stats['image_paths']} "
        f"rewritten={stats['rewritten']} unchanged={stats['unchanged']} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
