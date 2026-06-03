# RUN — Qwen3.5-9B Multimodal DFlash (copy-paste)

Single source of truth for running this on the intranet. `git pull`, then copy a
block and run it. The assistant keeps this file in sync with every script change.

Paths assume the proven layout — override the env vars if yours differ:
- target model: `/home/models/Qwen3.5-9B`
- repo clone:   `/home/wenxuan/speculators`
- MMStar:       `/home/wenxuan/mmstar/mmstar_answers.json` + `/home/wenxuan/mmstar/images`

---

## 0. One-time setup

```bash
cd /home/wenxuan/speculators

# speculators editable install — makes repo scripts match the package
# (fixes the "load_and_preprocess_dataset got unexpected kwarg trust_remote_code" error)
python3 -m pip install -e . --no-deps --no-build-isolation

# GPU vLLM with DFlash + M-RoPE support (nightly, per z-lab/Qwen3.5-9B-DFlash card)
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

---

## 1. Train the DFlash speculator (multimodal, online)

Edit the CONFIG block at the top of the script if paths differ (MODEL,
MMSTAR_SRC, MMSTAR_MEDIA_ROOT, GPU layout), then:

```bash
cd /home/wenxuan/speculators
bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
# -> checkpoints in ./output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best
# (to regenerate the MMStar jsonl after data changes: rm -f data/mmstar/mmstar.jsonl)
```

---

## 2. Serve on GPU (verifier + OUR trained draft)

```bash
cd /home/wenxuan/speculators
RUN_MODE=dflash DFLASH_SPEC=5 \
  MODEL_PATH=/home/models/Qwen3.5-9B \
  DFLASH_DRAFT_PATH="$(pwd)/output/dflash_qwen3.5_9b_mm/checkpoints/checkpoint_best" \
  CUDA_VISIBLE_DEVICES=0 \
  MM_MEDIA_DIR=/home/wenxuan/multimodel_test \
  bash examples/serve/run_qwen35_9b_gpu.sh
```

Other modes: `RUN_MODE=baseline` (no spec) / `RUN_MODE=mtp MTP_SPEC=3`.
If vLLM rejects `--attention-backend`: `export VLLM_ATTENTION_BACKEND=FLASH_ATTN` first.

---

## 3. Eval — throughput + acceptance (baseline vs dflash = speedup)

In a second shell, after the server is up on `:8100`:

```bash
cd /home/wenxuan/speculators
bash examples/evaluate/eval_qwen35_9b.sh          # images from MM_MEDIA_DIR
# or against the MMStar jsonl (serve must allow that image dir):
# DATA_JSONL="$(pwd)/data/mmstar/mmstar.jsonl" bash examples/evaluate/eval_qwen35_9b.sh
```

Real speedup = compare `output throughput (tok/s)`:

```bash
RUN_MODE=baseline ... bash examples/serve/run_qwen35_9b_gpu.sh   # then eval -> tok/s = A
RUN_MODE=dflash   ... bash examples/serve/run_qwen35_9b_gpu.sh   # then eval -> tok/s = B
# speedup = B / A
```

---

## Files

| What | Path |
|---|---|
| Train (multimodal DFlash, online) | `examples/train/dflash_qwen3.5_9b_multimodal_online.sh` |
| MMStar → conversations jsonl | `scripts/mmstar_to_jsonl.py` |
| Serve on GPU | `examples/serve/run_qwen35_9b_gpu.sh` |
| Eval client (throughput + acceptance) | `examples/evaluate/eval_qwen35_9b.sh` + `examples/evaluate/bench_mm_speculative.py` |
