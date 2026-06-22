#!/usr/bin/env python3
"""Distill ALLaVA prompts with a Qwen verifier served by vLLM.

This script reads ALLaVA/LLaVA-style records, keeps the image + user prompt, drops
the original assistant/GT answer, asks the served Qwen model to generate a new
assistant response, and writes a speculators-compatible conversations jsonl.

The output can be used directly as DATASET for
examples/train/dflash_qwen3.5_9b_multimodal_online.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from llava_to_jsonl import ROLE, load_records, to_parts


def _training_part_to_openai(part: dict[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "image":
        path = part.get("path") or part.get("url")
        if not path:
            raise ValueError(f"Image part missing path/url: {part}")
        url = path if "://" in path else f"file://{path}"
        return {"type": "image_url", "image_url": {"url": url}}
    return part


def _turn_to_openai(turn: dict[str, Any]) -> dict[str, Any]:
    content = turn.get("content", "")
    if isinstance(content, list):
        content = [_training_part_to_openai(part) for part in content]
    return {"role": turn["role"], "content": content}


def _first_prompt_conversation(
    record: dict[str, Any],
    image_root: Path,
    image_key: str,
    conv_key: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    img_rel = record.get(image_key)
    image_abs = None
    if img_rel:
        image_path = Path(img_rel)
        image_abs = str(image_path if image_path.is_absolute() else image_root / img_rel)
        if not Path(image_abs).exists():
            return None, image_abs

    prompt: list[dict[str, Any]] = []
    saw_image = False
    saw_user = False

    for turn in record.get(conv_key) or []:
        role = ROLE.get(str(turn.get("from", turn.get("role", ""))).lower())
        if role is None:
            continue
        if role == "assistant":
            # Stop before the original GT answer. Later user turns often depend on it.
            break
        if role not in ("system", "user"):
            continue

        value = (turn.get("value") if "value" in turn else turn.get("content")) or ""
        parts, used = to_parts(str(value), image_abs)
        saw_image = saw_image or used > 0
        saw_user = saw_user or role == "user"
        prompt.append({"role": role, "content": parts})

    if image_abs and not saw_image:
        for turn in prompt:
            if turn["role"] != "user":
                continue
            content = turn["content"]
            base = content if isinstance(content, list) else [{"type": "text", "text": content}]
            turn["content"] = [{"type": "image", "path": image_abs}, *base]
            saw_image = True
            break

    if not prompt or not saw_user:
        return None, None
    return prompt, None


def _iter_prompt_records(
    inputs: list[str],
    image_root: Path,
    image_key: str,
    conv_key: str,
):
    missing = 0
    missing_examples: list[str] = []
    for src in inputs:
        for record in load_records(Path(src)):
            prompt, missing_path = _first_prompt_conversation(
                record, image_root, image_key, conv_key
            )
            if missing_path:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(missing_path)
                continue
            if prompt:
                yield {"prompt": prompt}

    if missing:
        print(f"WARNING: skipped {missing} missing image path(s) under {image_root}")
        for path in missing_examples:
            print(f"  missing: {path}")


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_model_name(endpoint: str, requested_model: str | None) -> str:
    import openai  # noqa: PLC0415

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=1)
    if requested_model:
        return requested_model
    return client.models.list().data[0].id


def _bounded_parallel_map(fn, items, concurrency):
    """Apply ``fn`` over ``items`` keeping up to ``concurrency`` calls in flight,
    yielding ``(item, result)`` in the original order.

    Order preservation matters: writes stay in input order, so count-based
    ``--resume`` keeps mapping the first N outputs to the first N inputs.
    Exceptions raised by ``fn`` are yielded as the result value (not raised) so
    the caller can skip a failed item and continue, matching the prior loop.
    """

    def _wrapped(it):
        try:
            return fn(it)
        except Exception as exc:  # noqa: BLE001
            return exc

    if concurrency <= 1:
        for it in items:
            yield it, _wrapped(it)
        return

    from collections import deque  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    src = iter(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: deque = deque()
        for _ in range(concurrency):  # prime the window
            try:
                nxt = next(src)
            except StopIteration:
                break
            pending.append((nxt, pool.submit(_wrapped, nxt)))
        while pending:
            it, fut = pending.popleft()
            yield it, fut.result()
            try:
                nxt = next(src)
            except StopIteration:
                continue
            pending.append((nxt, pool.submit(_wrapped, nxt)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--endpoint", default="http://localhost:8100/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--in", dest="inputs", action="append", required=True)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--total-samples",
        type=int,
        default=None,
        help=(
            "Global post-skip sample budget before sharding. When set with "
            "--num-shards/--shard-index, each shard covers its own slice of "
            "this global budget."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--conv-key", default="conversations")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of in-flight generation requests. >1 sends requests in "
            "parallel via a thread pool while preserving input/write order, so "
            "count-based --resume stays correct. Match it to the server's "
            "--max-num-seqs to saturate the vLLM batch."
        ),
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.skip_samples < 0:
        raise SystemExit("--skip-samples must be non-negative")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.total_samples is not None and args.total_samples <= 0:
        raise SystemExit("--total-samples must be positive when set")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")

    import openai  # noqa: PLC0415

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_existing(args.out_jsonl) if args.resume else 0
    target_rows = args.max_samples
    if args.total_samples is not None:
        selected_total = (args.total_samples + args.stride - 1) // args.stride
        remaining = selected_total - args.shard_index
        target_rows = 0 if remaining <= 0 else (remaining + args.num_shards - 1) // args.num_shards
    if existing >= target_rows:
        print(f"Already have {existing} rows in {args.out_jsonl}; nothing to do.")
        return

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=1)
    model_name = _load_model_name(args.endpoint, args.model)

    print("ALLaVA Qwen distillation")
    print(f"  endpoint:     {args.endpoint}")
    print(f"  model:        {model_name}")
    print(f"  image_root:   {args.image_root}")
    print(f"  out_jsonl:    {args.out_jsonl}")
    print(f"  max_samples:  {args.max_samples}")
    print(f"  total_samples:{args.total_samples}")
    print(f"  skip_samples: {args.skip_samples}")
    print(f"  shard:        {args.shard_index}/{args.num_shards}")
    print(f"  target_rows:  {target_rows}")
    print(f"  resume rows:  {existing}")
    print(f"  concurrency:  {args.concurrency}")

    prompts = _iter_prompt_records(
        args.inputs, args.image_root, args.image_key, args.conv_key
    )

    def _selected_prompts():
        """Yield prompts passing skip/stride/shard/resume filters, in order."""
        skipped_valid = 0
        after_skip = 0
        skipped_existing = 0
        for item in prompts:
            if skipped_valid < args.skip_samples:
                skipped_valid += 1
                continue
            global_idx = after_skip
            after_skip += 1

            if args.total_samples is not None and global_idx >= args.total_samples:
                return
            if global_idx % args.stride != 0:
                continue
            sharded_idx = global_idx // args.stride
            if sharded_idx % args.num_shards != args.shard_index:
                continue

            if skipped_existing < existing:
                skipped_existing += 1
                continue

            yield item["prompt"]

    def _generate(prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=[_turn_to_openai(turn) for turn in prompt],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.request_timeout,
        )
        return (response.choices[0].message.content or "").strip()

    written = existing
    generated_this_run = 0
    started = time.perf_counter()

    mode = "a" if args.resume else "w"
    with args.out_jsonl.open(mode, encoding="utf-8") as out:
        for prompt, result in _bounded_parallel_map(
            _generate, _selected_prompts(), args.concurrency
        ):
            if isinstance(result, Exception):
                print(f"[warn] request failed at output row {written + 1}: {result}")
                continue
            answer = result
            if not answer:
                print(f"[warn] empty answer at output row {written + 1}; skipped")
                continue

            row = {"conversations": [*prompt, {"role": "assistant", "content": answer}]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            written += 1
            generated_this_run += 1
            if written % 25 == 0 or written == args.max_samples:
                elapsed = time.perf_counter() - started
                rate = generated_this_run / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{written}/{target_rows}] "
                    f"generated_this_run={generated_this_run} "
                    f"rate={rate:.3f} samples/s"
                )
            if written >= target_rows:
                break

    if written == 0:
        raise SystemExit("No distilled samples were written.")
    print(f"Done. Wrote {written} distilled samples -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
