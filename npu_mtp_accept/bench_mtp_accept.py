#!/usr/bin/env python3
"""Measure MTP draft acceptance rate on Ascend NPU via vllm-ascend (offline, V1).

Runs your trained 9B (with MTP heads) through vLLM speculative decoding and
reports acceptance rate / acceptance length from V1 SpecDecodingStats.

Assumes `vllm` + `vllm_ascend` are ALREADY installed in the env (no reinstall).
All paths/knobs come from env vars -- see RUN.md. Acceptance rate is independent
of CUDA/ACL-graph mode, so this defaults to eager for a robust first run.
"""
import json
import os
import sys
import time

# ---- config via env (fill MTP_MODEL on the NPU box) ----
MODEL = os.environ.get("MTP_MODEL")             # required: 9B ckpt WITH trained MTP heads
DRAFT = os.environ.get("MTP_DRAFT", "")         # optional: separate MTP/draft ckpt if not fused in MODEL
NUM_SPEC = int(os.environ.get("NUM_SPEC_TOKENS", "3"))   # MUST match your trained MTP depth
TP = int(os.environ.get("TP", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "64"))
TEMP = float(os.environ.get("TEMP", "0"))       # greedy -> clean, reproducible acceptance
DTYPE = os.environ.get("DTYPE", "bfloat16")     # MTP must be bf16 (project note)
GMU = float(os.environ.get("GPU_MEM_UTIL", "0.9"))
MAXLEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
DATASET = os.environ.get("DATASET", "")         # optional jsonl, one {"prompt": "..."} per line
EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"

if not MODEL:
    sys.exit("ERROR: set MTP_MODEL=/abs/path/to/9b_with_mtp  (see RUN.md)")

os.environ.setdefault("VLLM_USE_V1", "1")       # acceptance metrics live in the V1 engine

from vllm import LLM, SamplingParams  # noqa: E402


def load_prompts():
    if DATASET and os.path.exists(DATASET):
        ps = []
        with open(DATASET) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                p = o.get("prompt") or o.get("text") or ""
                if p:
                    ps.append(p)
        ps = ps[:NUM_PROMPTS]
        if ps:
            print(f"[data] {len(ps)} prompts from {DATASET}")
            return ps
    base = [
        "Explain how speculative decoding speeds up LLM inference.",
        "Write a Python function that merges two sorted lists and dedups them.",
        "Summarize the main causes of the French Revolution in five bullet points.",
        "What is the difference between TCP and UDP, and when would you use each?",
        "Describe step by step how to fine-tune a small language model on a custom dataset.",
        "Prove that the square root of 2 is irrational.",
        "Write a short story about a robot that learns to paint.",
        "How does PagedAttention manage the KV cache in vLLM, and why does it help throughput?",
    ]
    out = [base[i % len(base)] for i in range(NUM_PROMPTS)]
    print(f"[data] {len(out)} built-in prompts (set DATASET=xx.jsonl to override)")
    return out


spec = {"method": "mtp", "num_speculative_tokens": NUM_SPEC}
if DRAFT:
    spec["model"] = DRAFT

print(f"[cfg] model={MODEL}")
print(f"[cfg] draft={DRAFT or '(built-in MTP heads in MODEL)'} num_spec={NUM_SPEC} "
      f"tp={TP} dtype={DTYPE} eager={EAGER} prompts={NUM_PROMPTS} max_tokens={MAX_TOKENS} temp={TEMP}")

llm = LLM(
    model=MODEL,
    tensor_parallel_size=TP,
    dtype=DTYPE,
    speculative_config=spec,
    gpu_memory_utilization=GMU,
    max_model_len=MAXLEN,
    enforce_eager=EAGER,
    trust_remote_code=True,
    disable_log_stats=False,   # also prints "Spec decode" lines to the engine log
)

prompts = load_prompts()
sp = SamplingParams(temperature=TEMP, max_tokens=MAX_TOKENS, ignore_eos=True)

t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0
gen_toks = sum(len(o.outputs[0].token_ids) for o in outs)


def report():
    try:
        metrics = llm.get_metrics()
    except Exception as e:
        print(f"[warn] llm.get_metrics() unavailable ({e}).")
        print("       -> read the 'Spec decode metrics' / acceptance lines in the engine log above.")
        return
    acc = draft = drafts = None
    per_pos = None
    for m in metrics:
        name = getattr(m, "name", "")
        key = name.split("spec_decode_")[-1]
        val = getattr(m, "value", None)
        if key == "num_accepted_tokens":
            acc = val
        elif key == "num_draft_tokens":
            draft = val
        elif key == "num_drafts":
            drafts = val
        elif key == "num_accepted_tokens_per_pos":
            per_pos = getattr(m, "values", None) or val
    print("=" * 56)
    if draft:
        print(f"  draft tokens    : {draft}")
        print(f"  accepted tokens : {acc}")
        print(f"  ACCEPTANCE RATE : {acc / draft:.4f}   (accepted / draft)")
        if drafts:
            print(f"  ACCEPT LENGTH   : {acc / drafts + 1:.3f} tok/step (accepted/step + 1 bonus)")
        if per_pos:
            print(f"  per-position    : {per_pos}")
    else:
        print("  [warn] no draft tokens recorded -- spec decode may be off.")
        print("         Check: MODEL has MTP heads? NUM_SPEC_TOKENS matches trained depth?")
        print("         Also read the engine 'Spec decode' log lines above.")
    print("=" * 56)


report()
print(f"[tput] {gen_toks} gen tokens in {dt:.1f}s = {gen_toks / dt:.1f} tok/s (with MTP)")
