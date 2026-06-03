#!/usr/bin/env python3
"""Send a few multimodal chat requests to a vLLM endpoint.

Used by eval_dflash_mmstar.sh to exercise the served speculator so vLLM emits
"SpecDecoding metrics:" log lines (parsed by eval-guidellm/scripts/parse_logs.py).
Prompts are read from a `conversations` jsonl (the one Step 0 produced); each
user turn's image + text is forwarded as an OpenAI-style multimodal request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prompts(jsonl: str, n: int) -> list[tuple[str | None, str]]:
    prompts: list[tuple[str | None, str]] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= n:
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
            prompts.append((img, text))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--model", default=None, help="Served model id (auto-detected)")
    args = ap.parse_args()

    import openai  # noqa: PLC0415

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=1)
    model = args.model or client.models.list().data[0].id
    prompts = load_prompts(args.data_jsonl, args.num)
    print(f"Sending {len(prompts)} requests to {args.endpoint} (model={model})")

    ok = 0
    for i, (img, text) in enumerate(prompts):
        content: list[dict] = []
        if img:
            url = img if "://" in img else f"file://{img}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": text or "Describe the image."})
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            ok += 1
            n_chars = len(resp.choices[0].message.content or "")
            print(f"[{i + 1}/{len(prompts)}] ok ({n_chars} chars)")
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(prompts)}] FAILED: {e}")

    print(f"Done: {ok}/{len(prompts)} succeeded. Now parse the server log for metrics.")


if __name__ == "__main__":
    main()
