#!/usr/bin/env python3
"""Distill the client's SFT jsonl (`messages` + `images`) with the served verifier.

The client data (小红书 "问一问" RAG search assistant SFT set) is already in
OpenAI chat shape, one record per line:

    {"messages": [{"role": "system",    "content": "<big instruction>"},
                  {"role": "user",       "content": "...<image>...<image>..."},
                  {"role": "assistant",  "content": "<GT answer — DROPPED>"}],
     "images":   ["/abs/path/a.jpg", "/abs/path/b.jpg", ...]}

The `<image>` tokens in the message content map, in order, onto `images`
(count(<image>) == len(images)). We keep the system + user turns (dropping the
GT assistant answer), ask the served *post-SFT* verifier to regenerate the
answer, and write a speculators-trainer record:

    {"conversations": [ ...prompt turns..., {"role": "assistant", "content": answer} ]}

This matches scripts/distill_allava_with_qwen.py's OUTPUT exactly, so the
training (STEP 2) and eval (STEP 3) scripts consume it unchanged. The only thing
that differs is the INPUT reader + native multi-image support.

MODES
  --multimodal (default): keep images; each <image> becomes an {"type":"image","path":...}
      part (the verifier must be served with --allowed-local-media-path covering
      the image root and --limit-mm-per-prompt '{"image":N}', N>=max images/row).
      Missing local image files are dropped by default so the remaining prompt
      can still be distilled.
  --text-only: strip <image> tokens, drop images, distill TEXT ONLY.
      Useful only as a quick smoke path.

USAGE
  python3 scripts/distill_client_messages.py \
      --endpoint http://localhost:8100/v1 \
      --in /mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/train.jsonl \
      --out-jsonl data/client/client_122b_distill.jsonl \
      --max-samples 8137 --max-tokens 600 --temperature 0 \
      --concurrency 16 --resume            # add --text-only only for smoke tests
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

IMAGE_TOKEN = "<image>"


def _load_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _prompt_key(turns: list[dict]) -> str:
    """Stable content key for a prompt, computed identically from an input-built
    prompt or an already-written output row. Keys on the system+user TEXT only
    (images and <image> tokens ignored), so a row keys the same whether or not its
    images were present/dropped — which is exactly what lets us tell which inputs
    are already distilled vs. still missing."""
    chunks: list[str] = []
    for t in turns:
        if t.get("role") not in ("system", "user"):
            continue
        c = t.get("content", "")
        if isinstance(c, list):
            text = " ".join(
                p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = str(c)
        chunks.append(text)
    canon = re.sub(r"\s+", " ", " ".join(chunks).replace(IMAGE_TOKEN, "")).strip()
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def _load_done_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            keys.add(_prompt_key(row.get("conversations") or row.get("messages") or []))
    return keys


def _local_image_missing(path: str) -> bool:
    if "://" in path:
        return False
    return not Path(path).exists()


def _expand_content(
    content: str,
    images: deque[str],
    text_only: bool,
    missing_image_policy: str,
    dropped_missing_images: list[str],
) -> list[dict] | str:
    """Turn a content string with <image> tokens into trainer content parts.

    text_only -> drop the tokens, return a plain string.
    multimodal -> interleave {"type":"image","path":...} parts pulled (in order)
                  from the shared `images` deque, with {"type":"text","text":...}.
    """
    if text_only or IMAGE_TOKEN not in content:
        return content.replace(IMAGE_TOKEN, "").strip() if text_only else content

    parts: list[dict] = []
    segs = content.split(IMAGE_TOKEN)
    for i, seg in enumerate(segs):
        if i > 0:
            if not images:
                if missing_image_policy == "skip":
                    raise ValueError("more <image> tokens than image paths")
                dropped_missing_images.append("<missing image path>")
            else:
                image_path = images.popleft()
                if _local_image_missing(image_path):
                    if missing_image_policy == "skip":
                        raise ValueError(f"missing image file: {image_path}")
                    dropped_missing_images.append(image_path)
                else:
                    parts.append({"type": "image", "path": image_path})
        seg = seg.strip()
        if seg:
            parts.append({"type": "text", "text": seg})
    return parts


def _build_prompt(
    record: dict,
    text_only: bool,
    missing_image_policy: str,
) -> tuple[list[dict] | None, str | None, list[str]]:
    """Keep system + user turns up to (and including) the last user turn; drop
    the assistant answer(s). Returns (prompt_turns, skip_reason)."""
    messages = record.get("messages") or []
    images = deque(record.get("images") or [])
    n_img_tokens = sum(str(m.get("content", "")).count(IMAGE_TOKEN) for m in messages)
    if not text_only and n_img_tokens != len(images):
        return None, f"image/token mismatch: {len(images)} images vs {n_img_tokens} <image>", []

    # index of the last user turn — we generate a fresh answer after it
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    if last_user < 0:
        return None, "no user turn", []

    prompt: list[dict] = []
    dropped_missing_images: list[str] = []
    for m in messages[: last_user + 1]:
        role = m.get("role")
        if role not in ("system", "user"):
            continue
        try:
            content = _expand_content(
                str(m.get("content", "")),
                images,
                text_only,
                missing_image_policy,
                dropped_missing_images,
            )
        except ValueError as exc:
            return None, str(exc), []
        prompt.append({"role": role, "content": content})
    return prompt, None, dropped_missing_images


def _to_openai(turn: dict) -> dict:
    content = turn["content"]
    if isinstance(content, list):
        oai = []
        for p in content:
            if p.get("type") == "image":
                path = p["path"]
                url = path if "://" in path else f"file://{path}"
                oai.append({"type": "image_url", "image_url": {"url": url}})
            else:
                oai.append({"type": "text", "text": p.get("text", "")})
        content = oai
    return {"role": turn["role"], "content": content}


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def _bounded_parallel_map(fn, items, concurrency):
    def _wrapped(it):
        try:
            return fn(it)
        except Exception as exc:  # noqa: BLE001
            return exc

    if concurrency <= 1:
        for it in items:
            yield it, _wrapped(it)
        return
    src = iter(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: deque = deque()
        for _ in range(concurrency):
            try:
                nxt = next(src)
                pending.append((nxt, pool.submit(_wrapped, nxt)))
            except StopIteration:
                break
        while pending:
            it, fut = pending.popleft()
            yield it, fut.result()
            try:
                nxt = next(src)
                pending.append((nxt, pool.submit(_wrapped, nxt)))
            except StopIteration:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:8100/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--in", dest="inputs", action="append", required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--skip-samples", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--missing-image-policy", choices=("drop", "skip"), default="drop",
                    help="for multimodal rows with missing local images: drop the missing image part or skip the row")
    ap.add_argument("--allow-partial", action="store_true",
                    help="allow exiting successfully with fewer than --max-samples rows")
    ap.add_argument("--complete-missing", action="store_true",
                    help="APPEND only the input records not already present in --out-jsonl "
                         "(matched by prompt-text key). Use to fill gaps left by a prior run "
                         "that skipped rows (e.g. images that were inaccessible before). Does "
                         "NOT redo or reorder existing rows; implies --allow-partial.")
    ap.add_argument("--resume", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--text-only", dest="text_only", action="store_true",
                      help="strip images, distill text only")
    mode.add_argument("--multimodal", dest="text_only", action="store_false",
                      help="keep images (default; server needs --allowed-local-media-path + --limit-mm-per-prompt)")
    mode.set_defaults(text_only=False)
    args = ap.parse_args()

    # SAFETY: never write onto an input file (a mis-set --out-jsonl with mode "w"
    # would truncate the client's original train.jsonl). Refuse if out == any in.
    out_rp = args.out_jsonl.resolve()
    for src in args.inputs:
        if out_rp == Path(src).resolve():
            raise SystemExit(f"[fatal] --out-jsonl resolves to an input file ({src}); refusing to overwrite source data")

    import openai  # noqa: PLC0415

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    append_mode = args.resume or args.complete_missing
    existing = _count_existing(args.out_jsonl) if append_mode else 0
    allow_partial = args.allow_partial or args.complete_missing
    done_keys: set[str] = set()
    if args.complete_missing:
        done_keys = _load_done_keys(args.out_jsonl)
        print(f"[complete-missing] {len(done_keys)} rows already present in {args.out_jsonl}; "
              "will only distill+append the inputs not yet there.")
    elif existing >= args.max_samples:
        print(f"Already have {existing} rows in {args.out_jsonl}; nothing to do.")
        return

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=1)
    model_name = args.model or client.models.list().data[0].id

    print("Client messages distillation")
    print(f"  endpoint:    {args.endpoint}")
    print(f"  model:       {model_name}")
    print(f"  mode:        {'TEXT-ONLY' if args.text_only else 'MULTIMODAL'}")
    print(f"  out_jsonl:   {args.out_jsonl}")
    print(f"  max_samples: {args.max_samples}  skip: {args.skip_samples}  resume rows: {existing}")
    print(f"  max_tokens:  {args.max_tokens}  temperature: {args.temperature}  concurrency: {args.concurrency}")
    print(f"  missing_image_policy: {args.missing_image_policy}  allow_partial: {args.allow_partial}")

    stats = {
        "skips": 0,
        "skip_reasons": {},
        "rows_with_missing_images_dropped": 0,
        "missing_images_dropped": 0,
        "missing_image_examples": [],
    }

    def _prompts() -> Iterator[list[dict]]:
        seen = 0          # valid prompts produced before skip/resume filters
        emitted = 0
        skipped_existing = 0
        for src in args.inputs:
            for rec in _load_records(Path(src)):
                prompt, reason, dropped_missing_images = _build_prompt(
                    rec, args.text_only, args.missing_image_policy
                )
                if prompt is None:
                    stats["skips"] += 1
                    stats["skip_reasons"][reason or "unknown"] = (
                        stats["skip_reasons"].get(reason or "unknown", 0) + 1
                    )
                    continue
                if dropped_missing_images:
                    stats["rows_with_missing_images_dropped"] += 1
                    stats["missing_images_dropped"] += len(dropped_missing_images)
                    for path in dropped_missing_images:
                        if len(stats["missing_image_examples"]) < 10:
                            stats["missing_image_examples"].append(path)
                if seen < args.skip_samples:
                    seen += 1
                    continue
                seen += 1
                if args.complete_missing:
                    # skip inputs already distilled (by prompt-text key), not by count
                    if _prompt_key(prompt) in done_keys:
                        continue
                elif skipped_existing < existing:
                    skipped_existing += 1
                    continue
                yield prompt
                emitted += 1
                # complete-missing fills ALL gaps (don't cap by a count budget);
                # the normal path caps at the requested new-row count.
                if not args.complete_missing and emitted >= args.max_samples - existing:
                    return

    def _generate(prompt: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[_to_openai(t) for t in prompt],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.request_timeout,
        )
        return (resp.choices[0].message.content or "").strip()

    written = existing
    started = time.perf_counter()
    with args.out_jsonl.open("a" if append_mode else "w", encoding="utf-8") as out:
        for prompt, result in _bounded_parallel_map(_generate, _prompts(), args.concurrency):
            if isinstance(result, Exception):
                print(f"[warn] request failed at output row {written + 1}: {result}")
                continue
            if not result:
                print(f"[warn] empty answer at output row {written + 1}; skipped")
                continue
            row = {"conversations": [*prompt, {"role": "assistant", "content": result}]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if written % 25 == 0:
                rate = (written - existing) / max(time.perf_counter() - started, 1e-9)
                print(f"[{written}/{args.max_samples}] rate={rate:.3f} samples/s")
            if written >= args.max_samples:
                break

    if stats["missing_images_dropped"]:
        print(
            "[warn] dropped "
            f"{stats['missing_images_dropped']} missing image part(s) from "
            f"{stats['rows_with_missing_images_dropped']} row(s)"
        )
        print(f"[warn] missing image examples: {stats['missing_image_examples']}")
    if stats["skips"]:
        print(f"[warn] skipped {stats['skips']} records: {stats['skip_reasons']}")
    if written == existing:
        if args.complete_missing:
            print(f"Nothing missing: all inputs already present in {args.out_jsonl} ({existing} rows).")
            return
        raise SystemExit("No distilled samples were written.")
    if written < args.max_samples and not allow_partial:
        raise SystemExit(
            f"[fatal] only wrote {written}/{args.max_samples} rows. "
            "Do not train on this partial file; fix the warnings above and rerun, "
            "or pass --allow-partial explicitly."
        )
    print(f"Done. Wrote {written} rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
