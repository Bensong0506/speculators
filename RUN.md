# RUN — Qwen3.5-9B Multimodal DFlash (copy-paste)

Single source of truth for running this on the intranet. `git pull`, then copy a
block and run it. The assistant keeps this file in sync with every script change.

Paths assume the proven layout — override the env vars if yours differ:
- target model:      `/home/models/Qwen3.5-9B`
- open-source draft: `/home/models/Qwen3.5-9B-DFlash`  (warm-start base)
- repo clone:        `/home/wenxuan/speculators`
- ALLaVA-4V:         `/home/wenxuan/ALLaVA-4V`  (LAION images extracted; VFLAN has none → LAION jsons only)
- MMStar:            `/home/wenxuan/mmstar/...`  (smoke test only)

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

### If DFlash serving errors with `... does not support M-RoPE yet`

Confirmed on vLLM **0.22.0**: `vllm/v1/spec_decode/llm_base_proposer.py` calls
`self._raise_if_mrope()` in `__init__` and raises. Your trained draft is fine
(`Resolved architecture: DFlashDraftModel`); it's only this build's hard guard.

1. **Quick patch (offline, this box's 0.22.0)** — neutralize the guard, re-test:
   ```bash
   bash examples/serve/patch_vllm_mrope_guard.sh
   bash examples/serve/test_trained_dflash_gpu.sh
   ```
   Reversible; gets text serving up (enough to verify the weights + text speedup).
2. **Proper fix — vLLM nightly** (real M-RoPE fallback; needed for image/video):
   ```bash
   uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
   ```
3. **Reuse your NPU vLLM source** (`/home/wenxuan/2012/vllm`) built for CUDA — it
   already has the fix (vllm-ascend is only the NPU backend).

---

## 1. Train the DFlash speculator (multimodal, online)

### 1a. One-time: extract the ALLaVA images (they ship as zip chunks)
```bash
cd /home/wenxuan/speculators
ALLAVA_ROOT=/home/wenxuan/ALLaVA-4V bash examples/train/extract_allava_images.sh
# LAION -> allava_laion/images/ (~484k imgs; probe at the end should say 0 missing).
# VFLAN has no image_chunks here -> use the LAION jsons only.
# (lost the paths? `bash examples/train/find_allava.sh` locates jsons/images.)
```

### 1b. Warm-start from the open-source DFlash — RECOMMENDED (not from scratch)
`FINETUNE_FROM` auto-reads the checkpoint's `config.json` and matches block_size /
num_layers / draft_arch / aux target-layer-ids / mask_token_id / FULL vocab (and
uses a lower LR plus a separate `..._ft` dir). vLLM still appends the verifier's
final text layer for `verifier_last_hidden_states`. Step 0
auto-converts ALLaVA → a conversations jsonl (`scripts/llava_to_jsonl.py`).
```bash
FINETUNE_FROM=/home/models/Qwen3.5-9B-DFlash \
  ALLAVA_INPUTS="/home/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json /home/wenxuan/ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.json" \
  ALLAVA_IMAGE_ROOT=/home/wenxuan/ALLaVA-4V \
  MAX_SAMPLES=100000 EPOCHS=2 \
  bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
# -> ./output/dflash_qwen3.5_9b_mm_ft/checkpoints/checkpoint_best
# Default hard token limit is SEQ_LENGTH=4096 to avoid OOM.
# Preprocessing keeps only PREPROCESS_SEQ_LENGTH=3584 tokens by default, leaving
# margin for vLLM's multimodal prompt expansion. DFlash training also defaults
# to MAX_ANCHORS=512 because full-vocab block-16 logits are large.
# The launcher prints a "Resolved training limits" block before preprocessing;
# if MAX_ANCHORS is accidentally overridden above 512 in full-vocab block-16 mode,
# it now exits immediately instead of running until CUDA OOM.
# Full stdout/stderr is also saved by default under ./run_logs/*.log; override with
# RUN_LOG_PATH=/abs/train.log or disable with LOG_TO_FILE=0.
# Data preprocessing is cached in OUTPUT_DIR and keyed by model/data/max_samples/
# preprocessing seq length/trust-remote-code. Rebuild it explicitly with
# FORCE_PREPROCESS=1; this clears only preprocessing artifacts and keeps checkpoints.
# vLLM request access logs are filtered by the shell so repeated 200 OK lines do
# not hide errors, without relying on version-specific vLLM CLI flags.
# the run prints "Warm-start: aligning ..." -> eyeball block_size=16 / 5 layers /
#   qwen3 / aux=[1,8,15,22,29] / mask=248070 / full vocab before it continues.
# NOTE: MAX_SAMPLES caps TOTAL in input order (Caption first). The two local
#   LAION files are 468,670 caption + 468,670 instruct = 937,340 rows total, so
#   MAX_SAMPLES must be >468670 to include instruct. Re-convert after changing
#   data by either using FORCE_PREPROCESS=1 or removing data/allava/allava_*.jsonl.
```
(From scratch instead: drop `FINETUNE_FROM`. MMStar smoke test:
`USE_ALLAVA=0 USE_MMSTAR=1 bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh`.)

**TRUE warm-start (inherit z-lab's weights, not just its recipe):** the raw z-lab
checkpoint isn't speculators-format, so `FINETUNE_FROM` on it only copies the
recipe and trains from scratch. Convert it once, then point `FINETUNE_FROM` at the
converted dir (weights then load):
```bash
bash examples/train/convert_zlab_dflash.sh        # -> /home/models/Qwen3.5-9B-DFlash-spec
# then in 1b's command use:  FINETUNE_FROM=/home/models/Qwen3.5-9B-DFlash-spec
```

### 1c. Detached longer ALLaVA run
After W&B login, this starts a nohup run over all local LAION caption+instruct rows
and writes `run_logs/<run_name>.nohup.log` plus a PID file:
```bash
bash examples/train/nohup_dflash_qwen3.5_9b_allava_full.sh
tail -f run_logs/dflash_qwen35_9b_allava_full_*.nohup.log
```
Defaults: `MAX_SAMPLES=937340 EPOCHS=1000 CHECKPOINT_FREQ=5 MAX_ANCHORS=512 LOGGER=wandb`.

### 1d. Watch training (loss + per-position acceptance)
```bash
bash examples/train/view_tensorboard.sh
# from your laptop:  ssh -N -L 6006:localhost:6006 <user>@<gpu-box>  -> http://localhost:6006
```
Prefer wandb? Install/login once, then add `LOGGER=wandb` to the training command:
```bash
python3 -m pip install wandb
wandb login --host http://<internal-wandb-host>:<port>

WANDB_BASE_URL=http://<internal-wandb-host>:<port> \
  WANDB_PROJECT=speculators \
  LOGGER=wandb \
  bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
```
First time for tensorboard only: `pip install tensorboard`.

---

## 2. Serve on GPU (verifier + OUR trained draft)

**Quickest test (text)** — serve our trained draft and print a ready-to-paste curl:
```bash
bash examples/serve/test_trained_dflash_gpu.sh
```

**Multimodal (image) test** on stock vLLM 0.22.0 — patch the M-RoPE guard, then serve:
```bash
bash examples/serve/patch_vllm_mrope_guard.sh
bash examples/serve/test_trained_dflash_mm_gpu.sh   # serves (uses --enforce-eager)
```
Once it prints "Application startup complete", send an image request from a SECOND terminal:
```bash
bash examples/serve/send_image_request.sh
# override: IMAGE=/abs/img.jpg PROMPT="这是什么" bash examples/serve/send_image_request.sh
```

Full launcher (modes baseline/mtp/dflash, multimodal flags):
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

### Sanity-check with the published z-lab draft (instead of ours)

The z-lab draft is block-16, so use `DFLASH_SPEC=15`. Download it (small), then
reuse the same launcher:

```bash
export HF_ENDPOINT=https://hf-mirror.com    # intranet mirror
huggingface-cli download z-lab/Qwen3.5-9B-DFlash --local-dir /home/models/Qwen3.5-9B-DFlash
RUN_MODE=dflash DFLASH_SPEC=15 \
  DFLASH_DRAFT_PATH=/home/models/Qwen3.5-9B-DFlash \
  CUDA_VISIBLE_DEVICES=0 \
  bash examples/serve/run_qwen35_9b_gpu.sh
```

This is exactly the z-lab card's command (target + dflash draft + flash_attn +
max-num-batched-tokens 32768), just via the launcher with local paths. For 27B:
download `Qwen/Qwen3.5-27B` + `z-lab/Qwen3.5-27B-DFlash`, set `MODEL_PATH` to the
27B, and use `TP=2` (54 GB bf16 won't leave much room on one 80 GB card).

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

### Overnight MMStar sweep over all checkpoints

This tests every speculators-format DFlash checkpoint under the training output
directory against the native/raw DFlash baseline, using `INFER_NUM_SPEC=7`.
Results are written to `output/mmstar_checkpoint_sweeps/<timestamp>/results.csv`
and `results.jsonl`.

```bash
cd /data/wenxuan/speculators

INFER_NUM_SPEC=7 \
CHECKPOINT_FIND_ROOT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_100k_continue_dflash \
bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
```

### 10-image MMStar grouped comparison

This compares original/native DFlash
`/data/wenxuan/Qwen3.5-9B-DFlash` against the best trained checkpoint selected
from the latest MMStar checkpoint sweep result. It groups 10 MMStar images into
each request, so this is mainly a multi-image throughput/acceptance probe.

```bash
cd /data/wenxuan/speculators

INFER_NUM_SPEC=7 \
NUM_GROUPS=16 \
IMAGES_PER_PROMPT=10 \
bash examples/evaluate/test_dflash_mmstar_10image_weights.sh
```

To pin a specific trained checkpoint instead of auto-selecting:

```bash
DRAFT=/data/wenxuan/speculators/output/.../checkpoints/14 \
INFER_NUM_SPEC=7 \
bash examples/evaluate/test_dflash_mmstar_10image_weights.sh
```

### Native MTP vs original DFlash baseline

This compares native Qwen3.5 MTP against the original/downloaded DFlash draft on
the same MMStar prompts. Defaults prefer `/home/wenxuan/...` paths and fall back
to `/data/wenxuan/...` if needed. `MTP_SPEC=3` is kept separate from
`DFLASH_SPEC=7` because native MTP and DFlash do not necessarily support the same
draft depth.

```bash
cd /home/wenxuan/speculators

MTP_SPEC=3 \
DFLASH_SPEC=7 \
NUM_PROMPTS=128 \
bash examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh
```

For the grouped 10-image probe:

```bash
cd /home/wenxuan/speculators

MTP_SPEC=3 \
DFLASH_SPEC=7 \
IMAGES_PER_PROMPT=10 \
NUM_GROUPS=16 \
bash examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh
```

Spec sweep over `mtp@3/5/7` and `original_dflash@3/5/7`:

```bash
cd /home/wenxuan/speculators

MTP_SPECS="3 5 7" \
DFLASH_SPECS="3 5 7" \
NUM_PROMPTS=128 \
bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh
```

The sweep writes stable, resumable results under
`output/mmstar_mtp_dflash_spec_sweeps/single_n128_tok128/`. Existing
`summary.json` files are skipped, and matching results from previous
`test_mtp_vs_dflash_original_mmstar.sh` runs are reused automatically.

### Trained DFlash best checkpoint vs the six baselines

This first sweeps the current trained checkpoints with `SELECT_SPEC=7`, selects
the best checkpoint by trained `tok/s`, then evaluates that checkpoint at
`num_spec=3/5/7` and merges the rows with the six baseline rows
(`MTP@3/5/7`, `original_dflash@3/5/7`). Checkpoint folders named `0`, `4`, `9`,
`14`, ... and `checkpoint_best` are both supported.

```bash
cd /home/wenxuan/speculators
git checkout test_result
git pull

CHECKPOINT_FIND_ROOT=/home/wenxuan/speculators/output/<your_run>/checkpoints \
EVAL_GPU_GROUPS="0 1 2 3 4 5 6 7" \
SELECT_SPEC=7 \
TRAINED_SPECS="3 5 7" \
NUM_PROMPTS=128 \
bash examples/evaluate/eval_trained_dflash_best_vs_baselines.sh
```

If `CHECKPOINT_FIND_ROOT` is omitted, the script auto-discovers the newest
speculators-format DFlash checkpoints under `output/`. Results are written under
`output/mmstar_trained_dflash_best_vs_baselines/single_n128_tok128/<run_tag>/`,
especially `final_results.md`, `final_results.csv`, and `final_results.jsonl`.
Use `FORCE_CHECKPOINT_SWEEP=1`, `FORCE_TRAINED_SPEC_SWEEP=1`, or
`FORCE_BASELINE_SWEEP=1` to rerun cached stages.
`EVAL_GPU_GROUPS` is whitespace-separated CUDA device groups. For this 9B eval,
`"0 1 2 3 4 5 6 7"` runs eight one-GPU workers; for tensor parallel groups use
values such as `"0,1 2,3 4,5 6,7"` with `TP=2`.

### ALLaVA validation-tail original vs trained DFlash

This uses the same split convention as training: first 90% train, last 10% val.
Use this to check whether a continued DFlash checkpoint improves in-domain
acceptance/throughput on the ALLaVA distilled validation tail, even if it does
not transfer to MMStar.

```bash
cd /home/wenxuan/speculators
git checkout test_result
git pull

DRAFT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609_012511/checkpoints/6 \
ALLAVA_JSONL="$(pwd)/data/allava/allava_10000.jsonl" \
INFER_NUM_SPEC=7 \
NUM_PROMPTS=128 \
bash examples/evaluate/test_dflash_allava_val_weights.sh
```

If the distilled training jsonl has a different path, replace `ALLAVA_JSONL`.
Results are written to `output/allava_val_weight_tests/<timestamp>/`, especially
`allava_val_summary.md`.

---

## Files

| What | Path |
|---|---|
| Train (multimodal DFlash, online + warm-start) | `examples/train/dflash_qwen3.5_9b_multimodal_online.sh` |
| ALLaVA/LLaVA → conversations jsonl | `scripts/llava_to_jsonl.py` |
| ALLaVA image extractor · finder | `examples/train/extract_allava_images.sh` · `examples/train/find_allava.sh` |
| MMStar → conversations jsonl | `scripts/mmstar_to_jsonl.py` |
| MMStar original-vs-trained DFlash eval | `examples/evaluate/test_dflash_mmstar_weights.sh` |
| MMStar 10-image DFlash eval | `examples/evaluate/test_dflash_mmstar_10image_weights.sh` |
| MMStar native MTP-vs-original DFlash eval | `examples/evaluate/test_mtp_vs_dflash_original_mmstar.sh` |
| MMStar MTP/DFlash spec sweep | `examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh` |
| MMStar MTP/DFlash spec sweep summary | `examples/evaluate/mmstar_mtp_dflash_spec_sweep_summary.md` |
| MMStar trained-best-vs-baselines eval | `examples/evaluate/eval_trained_dflash_best_vs_baselines.sh` |
| ALLaVA val original-vs-trained DFlash eval | `examples/evaluate/test_dflash_allava_val_weights.sh` |
| Training curves (TensorBoard) | `examples/train/view_tensorboard.sh` |
| Serve on GPU (baseline/mtp/dflash) | `examples/serve/run_qwen35_9b_gpu.sh` |
| Quick serve test (text · image) | `examples/serve/test_trained_dflash_gpu.sh` · `examples/serve/test_trained_dflash_mm_gpu.sh` |
| vLLM 0.22 M-RoPE guard patch · send image req | `examples/serve/patch_vllm_mrope_guard.sh` · `examples/serve/send_image_request.sh` |
| Eval client (throughput + acceptance) | `examples/evaluate/eval_qwen35_9b.sh` + `examples/evaluate/bench_mm_speculative.py` |
