#!/usr/bin/env python3
"""Benchmark a RUNNING vLLM OpenAI endpoint on multimodal prompts.

Hardware-agnostic client (works against your Ascend `run_qwen35_9b_onecase.sh`
server, or any vLLM). Measures end-to-end output throughput (tokens/s) and
per-request latency, and scrapes the server's /metrics for speculative-decoding
acceptance. Sends requests SEQUENTIALLY (matches --max-num-seqs 1), so tokens/s
is a clean single-stream number you can compare across modes.

Workflow for the real speedup:
    1) start baseline server, run this, note tokens/s
    2) start dflash server,   run this, note tokens/s
    speedup = dflash_tok_s / baseline_tok_s

Prompt source (pick one):
    --image-dir DIR     use every image under DIR (must be under the server's
                        --allowed-local-media-path) with --question
    --data-jsonl FILE   use image+text from a `conversations` jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def prompts_from_jsonl(jsonl: str, n: int) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if len(out) >= n:
                break
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line).get("conversations", [])
            user = next((t for t in conv if t.get("role") == "user"), None)
            if not user:
                continue
            content = user["content"]
            img, text = None, ""
            if isinstance(content, str):
                text = content
            else:
                for part in content:
                    if part.get("type") == "image":
                        img = part.get("path") or part.get("url")
                    elif part.get("type") == "text":
                        text = part.get("text", "")
            out.append((img, text))
    return out


def prompts_from_dir(image_dir: str, question: str, n: int) -> list[tuple[str, str]]:
    imgs = sorted(
        p for p in Path(image_dir).rglob("*") if p.suffix.lower() in _IMG_EXTS
    )
    return [(str(p), question) for p in imgs[:n]]


def scrape_spec_metrics(endpoint: str) -> dict[str, float]:
    """GET <root>/metrics and return spec-decode-related counters."""
    root = endpoint.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = root + "/metrics"
    out: dict[str, float] = {}
    try:
        text = urllib.request.urlopen(url, timeout=10).read().decode()  # noqa: S310
    except Exception as e:  # noqa: BLE001
        print(f"(could not fetch {url}: {e})")
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        low = line.lower()
        if "spec_decode" in low or ("accept" in low and "token" in low):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def _sum_by_substr(metrics: dict[str, float], *subs: str) -> float | None:
    hits = [v for k, v in metrics.items() if all(s in k.lower() for s in subs)]
    return sum(hits) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8100/v1")
    ap.add_argument("--model", default=None, help="served model id (auto-detected)")
    ap.add_argument("--data-jsonl", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--question", default="Describe this image in detail.")
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    import openai  # noqa: PLC0415

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=1)
    model = args.model or client.models.list().data[0].id

    if args.image_dir:
        prompts = prompts_from_dir(args.image_dir, args.question, args.num)
    elif args.data_jsonl:
        prompts = prompts_from_jsonl(args.data_jsonl, args.num)
    else:
        raise SystemExit("Provide --image-dir or --data-jsonl")
    if not prompts:
        raise SystemExit("No prompts found.")

    print(f"Endpoint={args.endpoint}  model={model}  prompts={len(prompts)}")
    before = scrape_spec_metrics(args.endpoint)

    total_out_tokens = 0
    latencies: list[float] = []
    t0 = time.perf_counter()
    for i, (img, text) in enumerate(prompts):
        content: list[dict] = []
        if img:
            url = img if "://" in img else f"file://{img}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": text or args.question})
        r0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(prompts)}] FAILED: {e}")
            continue
        latencies.append(time.perf_counter() - r0)
        if resp.usage:
            total_out_tokens += resp.usage.completion_tokens
    wall = time.perf_counter() - t0

    after = scrape_spec_metrics(args.endpoint)

    print("\n" + "=" * 60)
    print("Throughput (single-stream)")
    print("=" * 60)
    print(f"  requests completed : {len(latencies)}/{len(prompts)}")
    print(f"  output tokens      : {total_out_tokens}")
    print(f"  wall time          : {wall:.1f} s")
    if wall > 0:
        print(f"  output throughput  : {total_out_tokens / wall:.1f} tok/s")
    if latencies:
        print(f"  mean latency/req   : {sum(latencies) / len(latencies):.2f} s")

    print("\n" + "=" * 60)
    print("Speculative-decoding acceptance (from /metrics delta)")
    print("=" * 60)
    if not after:
        print("  (no spec-decode metrics exposed — baseline mode, or /metrics off)")
    else:
        acc = _sum_by_substr(after, "spec_decode", "accept") or 0.0
        draft = _sum_by_substr(after, "spec_decode", "draft") or 0.0
        acc_b = _sum_by_substr(before, "spec_decode", "accept") or 0.0
        draft_b = _sum_by_substr(before, "spec_decode", "draft") or 0.0
        d_acc, d_draft = acc - acc_b, draft - draft_b
        if d_draft > 0:
            rate = d_acc / d_draft
            print(f"  accepted draft tokens : {d_acc:.0f}")
            print(f"  proposed draft tokens : {d_draft:.0f}")
            print(f"  draft acceptance rate : {rate:.3f}")
            print(f"  ~mean accepted length : {1 + rate * (d_draft / max(1, len(latencies))):.2f} (rough)")
        else:
            print("  (acceptance counters didn't advance — see raw lines below)")
        print("  raw spec-decode /metrics lines:")
        for k, v in sorted(after.items()):
            print(f"    {k} = {v}")

    print("\nTip: run this against baseline AND dflash servers; the ratio of")
    print("'output throughput' is your real end-to-end speedup.")


if __name__ == "__main__":
    main()
