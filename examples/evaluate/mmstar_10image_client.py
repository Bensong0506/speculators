#!/usr/bin/env python3
"""Run grouped 10-image MMStar prompts against a vLLM OpenAI endpoint.

This is a throughput/acceptance probe, not a strict MMStar accuracy evaluator.
It groups consecutive single-image MMStar records into one chat request with
multiple image parts, asks for short numbered answers, and records vLLM
speculative-decoding metric deltas.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _first_user_and_answer(conv: list[dict[str, Any]]) -> tuple[str | None, str, str]:
    user = next((turn for turn in conv if turn.get("role") == "user"), None)
    assistant = next((turn for turn in conv if turn.get("role") == "assistant"), None)
    if not user:
        return None, "", ""

    image: str | None = None
    text = ""
    content = user.get("content", "")
    if isinstance(content, str):
        text = content
    else:
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                image = part.get("path") or part.get("url")
            elif part.get("type") == "text":
                text = part.get("text", "")

    answer = ""
    if assistant:
        answer_content = assistant.get("content", "")
        answer = answer_content if isinstance(answer_content, str) else ""
    return image, text, answer


def load_items(jsonl: Path, limit_items: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with jsonl.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if limit_items is not None and len(items) >= limit_items:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image, text, answer = _first_user_and_answer(
                record.get("conversations", [])
            )
            if not text or not image:
                continue
            items.append(
                {
                    "index": line_idx,
                    "image": image,
                    "text": text,
                    "answer": answer,
                }
            )
    return items


def make_groups(
    items: list[dict[str, Any]], group_size: int, num_groups: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for start in range(0, len(items), group_size):
        group = items[start : start + group_size]
        if len(group) < group_size:
            break
        groups.append(group)
        if len(groups) >= num_groups:
            break
    return groups


def scrape_metrics(endpoint: str) -> dict[str, float]:
    root = endpoint.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = root + "/metrics"
    metrics: dict[str, float] = {}
    try:
        raw = urllib.request.urlopen(url, timeout=10).read().decode()  # noqa: S310
    except Exception as exc:  # noqa: BLE001
        print(f"(could not fetch {url}: {exc})")
        return metrics

    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        lowered = line.lower()
        if "spec_decode" not in lowered and "speculative" not in lowered:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            metrics[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return metrics


def _metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def _metric_name(key: str) -> str:
    return key.split("{", 1)[0]


def _sum_metric(metrics: dict[str, float], name: str) -> float:
    return sum(value for key, value in metrics.items() if _metric_name(key) == name)


def _sum_pos_metric(metrics: dict[str, float], name: str) -> dict[int, float]:
    totals: dict[int, float] = {}
    pattern = re.compile(r'position="(\d+)"')
    for key, value in metrics.items():
        if _metric_name(key) != name:
            continue
        match = pattern.search(key)
        if not match:
            continue
        pos = int(match.group(1))
        totals[pos] = totals.get(pos, 0.0) + value
    return totals


def _image_part(image: str) -> dict[str, Any]:
    url = image if "://" in image else f"file://{image}"
    return {"type": "image_url", "image_url": {"url": url}}


def _build_content(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_count = len(group)
    header = (
        f"You will see {image_count} images. Answer each numbered question "
        f"briefly. Return exactly {image_count} numbered lines when possible.\n\n"
    )
    content.append({"type": "text", "text": header})
    for idx, item in enumerate(group, 1):
        content.append(_image_part(item["image"]))
        content.append(
            {
                "type": "text",
                "text": f"\nQuestion {idx}: {item['text']}\n",
            }
        )
    return content


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--data-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--images-per-prompt", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    import openai  # noqa: PLC0415

    needed = args.num_groups * args.images_per_prompt
    items = load_items(args.data_jsonl, needed)
    groups = make_groups(items, args.images_per_prompt, args.num_groups)
    if not groups:
        raise SystemExit(f"No complete {args.images_per_prompt}-image groups found")

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=1)
    served_model = args.model or client.models.list().data[0].id
    print(
        f"Endpoint={args.endpoint} model={served_model} groups={len(groups)} "
        f"images_per_prompt={args.images_per_prompt} max_tokens={args.max_tokens}"
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    before_metrics = scrape_metrics(args.endpoint)
    latencies: list[float] = []
    total_completion_tokens = 0
    completed = 0
    contains_ref = 0
    ref_count = 0
    total_images = 0

    wall_start = time.perf_counter()
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for req_idx, group in enumerate(groups, 1):
            request_start = time.perf_counter()
            error = None
            output = ""
            completion_tokens = 0
            try:
                response = client.chat.completions.create(
                    model=served_model,
                    messages=[
                        {
                            "role": "user",
                            "content": _build_content(group),
                        }
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                latency = time.perf_counter() - request_start
                output = response.choices[0].message.content or ""
                if response.usage:
                    completion_tokens = int(response.usage.completion_tokens or 0)
                completed += 1
                latencies.append(latency)
                total_completion_tokens += completion_tokens
                total_images += len(group)
            except Exception as exc:  # noqa: BLE001
                latency = time.perf_counter() - request_start
                error = str(exc)

            output_norm = _normalize_text(output)
            item_results: list[dict[str, Any]] = []
            for item_idx, item in enumerate(group, 1):
                answer = item.get("answer", "")
                answer_norm = _normalize_text(answer)
                matched = False
                if answer_norm:
                    ref_count += 1
                    matched = answer_norm in output_norm
                    contains_ref += int(matched)
                item_results.append(
                    {
                        "item_index": item_idx,
                        "source_index": item.get("index"),
                        "image": item.get("image"),
                        "question": item.get("text", ""),
                        "reference_answer": answer,
                        "reference_contained": matched,
                    }
                )

            row = {
                "request_index": req_idx,
                "images_per_prompt": len(group),
                "items": item_results,
                "output": output,
                "completion_tokens": completion_tokens,
                "latency_sec": latency,
                "error": error,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = "ok" if error is None else "FAILED"
            print(
                f"[{req_idx}/{len(groups)}] {status} images={len(group)} "
                f"tokens={completion_tokens} latency={latency:.2f}s"
            )

    wall_sec = time.perf_counter() - wall_start
    after_metrics = scrape_metrics(args.endpoint)
    spec_delta = _metric_delta(before_metrics, after_metrics)
    draft_steps = _sum_metric(spec_delta, "vllm:spec_decode_num_drafts_total")
    draft_tokens = _sum_metric(spec_delta, "vllm:spec_decode_num_draft_tokens_total")
    accepted_tokens = _sum_metric(
        spec_delta, "vllm:spec_decode_num_accepted_tokens_total"
    )
    accepted_by_pos = _sum_pos_metric(
        spec_delta, "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    )
    token_acceptance_rate = (
        accepted_tokens / draft_tokens if draft_tokens > 0 else None
    )
    first_position_acceptance_rate = (
        accepted_by_pos.get(0, 0.0) / draft_steps if draft_steps > 0 else None
    )
    mean_accepted_tokens_per_draft = (
        accepted_tokens / draft_steps if draft_steps > 0 else None
    )

    summary: dict[str, Any] = {
        "endpoint": args.endpoint,
        "model": served_model,
        "num_groups_requested": len(groups),
        "images_per_prompt": args.images_per_prompt,
        "completed": completed,
        "failed": len(groups) - completed,
        "total_images_completed": total_images,
        "completion_tokens": total_completion_tokens,
        "wall_sec": wall_sec,
        "output_tok_per_sec": (
            total_completion_tokens / wall_sec if wall_sec > 0 else None
        ),
        "images_per_sec": total_images / wall_sec if wall_sec > 0 else None,
        "mean_latency_sec": statistics.mean(latencies) if latencies else None,
        "reference_contains_rate": (
            contains_ref / ref_count if ref_count > 0 else None
        ),
        "reference_count": ref_count,
        "spec_metrics_delta": spec_delta,
        "spec_draft_steps_total": draft_steps,
        "spec_draft_tokens_total": draft_tokens,
        "spec_accepted_tokens_total": accepted_tokens,
        "spec_accepted_tokens_by_position": accepted_by_pos,
        "spec_token_acceptance_rate": token_acceptance_rate,
        "spec_first_position_acceptance_rate": first_position_acceptance_rate,
        "spec_mean_accepted_tokens_per_draft": mean_accepted_tokens_per_draft,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
