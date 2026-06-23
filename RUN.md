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
  MEDIA_ROOT=/home/wenxuan/ALLaVA-4V \
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

### 1e. Domino-style causal correction head(NEW — 仅训练侧)

Domino = 在 DFlash 并行块草稿上加一个**轻量 GRU 因果修正头**(teacher-forced prefix + base-anchored λ 课程),目标:DFlash 的速度 + 接近 MTP 的接受率(论文/SpecForge:Domino > DFlash/DART/EAGLE-3)。默认关,开关 `ENABLE_DOMINO=1`。架构默认已对齐 SpecForge(GRU 1024 / 低秩 256 / teacher-forced)。

> ⚠️ **只做了训练,没做 serving**:trained 头(`prefix_gru`+`embed_proj`)目前只在训练 forward 生效;**vLLM DFlash proposer 不认识它** → 现在**量不到真实 vLLM 接受率、不能部署**。判断有没有用看训练/val metrics **`domino_final_acc` vs `domino_base_acc`**(teacher-forced 下修正头 top-1 vs 纯 DFlash top-1;final 稳 > base = 有效)。值了再做 serve 侧。

先跑单测(GPU 机;本地 torch 太老跑不了):
```bash
pytest tests/unit/models/test_dflash_domino.py tests/unit/train/test_cli_args.py -q
```

Smoke(50 条;warm-start 自你**最好的已训 DFlash**,base 越强课程越省):
> ⚠️ **数据怎么喂**:这个 launcher 用 `DATASET`/`USE_ALLAVA` 喂数据,**不认 `DISTILLED_ALLAVA_JSONL`**;且 `MAX_SAMPLES` 默认 **5000**、`USE_ALLAVA=1` 默认会拿 **raw** ALLaVA。要训**蒸馏 100k** 必须**四件套**:`USE_ALLAVA=0` + `DATASET=<蒸馏 jsonl>` + `MEDIA_ROOT=<图片根,如 /home/wenxuan/ALLaVA-4V>` + `MAX_SAMPLES=100000`。⚠️ `USE_ALLAVA=0` 下 **`ALLAVA_IMAGE_ROOT` 被忽略**,`--allowed-local-media-path` 只认 **`MEDIA_ROOT`**(不设就 fallback 到 `/path/to/coco` → `Invalid --allowed-local-media-path` 崩);`MEDIA_ROOT` 必须是 jsonl 里图片绝对路径的**父目录**。数据**对齐你 base DFlash 训练用的那种**(蒸馏就蒸馏,raw 就 raw)。

SMOKE(50 条,先确认 domino 跑通):
```bash
ENABLE_DOMINO=1 DOMINO_LOSS_DECAY_GAMMA=4 BLOCK_SIZE=8 \
NO_RESUME_FROM_CHECKPOINT=1 OUTPUT_DIR=$PWD/output/dflash_domino_smoke \
FINETUNE_FROM=$PWD/output/<你最好的 dflash run>/checkpoints/checkpoint_best \
MODEL=/home/wenxuan/Qwen3.5-9B \
USE_ALLAVA=0 DATASET=$PWD/data/allava/allava_qwen35_distill_100k.jsonl \
MEDIA_ROOT=/home/wenxuan/ALLaVA-4V \
VLLM_GPUS=0 VLLM_TP=1 VLLM_DP=1 TRAIN_GPUS=4,5,6,7 NUM_TRAIN_GPUS=4 \
MAX_SAMPLES=50 EPOCHS=1 \
bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
```
日志确认:`domino_enabled: 1` + `domino lambda_base: 1.0 -> 0` + metrics 有 `domino_final_acc/base_acc`,且 "Resolved training limits" 里 `block_size=8`、样本数=50。

全量 **100k**(smoke 过后;把 `MAX_SAMPLES=50 EPOCHS=1` 换成 `MAX_SAMPLES=100000 EPOCHS=2`,其余不变):
```bash
ENABLE_DOMINO=1 DOMINO_LOSS_DECAY_GAMMA=4 BLOCK_SIZE=8 \
NO_RESUME_FROM_CHECKPOINT=1 OUTPUT_DIR=$PWD/output/dflash_domino_100k \
FINETUNE_FROM=$PWD/output/<你最好的 dflash run>/checkpoints/checkpoint_best \
MODEL=/home/wenxuan/Qwen3.5-9B \
USE_ALLAVA=0 DATASET=$PWD/data/allava/allava_qwen35_distill_100k.jsonl \
MEDIA_ROOT=/home/wenxuan/ALLaVA-4V \
VLLM_GPUS=0 VLLM_TP=1 VLLM_DP=1 TRAIN_GPUS=4,5,6,7 NUM_TRAIN_GPUS=4 \
MAX_SAMPLES=100000 EPOCHS=2 \
bash examples/train/dflash_qwen3.5_9b_multimodal_online.sh
```
> 要 raw ALLaVA 100k 而非蒸馏:去掉 `USE_ALLAVA=0 DATASET=...`、保留 `MAX_SAMPLES=100000`。
> ⚠️ **每个 run 必带 `NO_RESUME_FROM_CHECKPOINT=1` + 独立 `OUTPUT_DIR`**:否则 full 会撞上 smoke 在同一 `OUTPUT_DIR` 留下的 checkpoint 去 resume → `Missing optimizer state for 'verifier_lm_head.weight'` 崩(base launcher 默认 resume=on)。

- **gamma 按 block_size**:bs8→4 / bs10→5 / bs16→7。**你的 DFlash 是 bs=8**(`_full` wrapper 默认 `BLOCK_SIZE=8`,`train.py` 会覆盖 z-lab 的 16)→ **gamma=4(launcher 默认即可)**。不放心就核一眼:`grep -o '"block_size":[^,]*' <ckpt>/config.json`。
- 旋钮(默认对齐 SpecForge):`DOMINO_GRU_HIDDEN_DIM=1024` · `DOMINO_EMB_DIM=256` · `DOMINO_PURE_DRAFT_PREFIX_LEN=1` · `DOMINO_LAMBDA_BASE_START=1.0`→0 · `DOMINO_LAMBDA_BASE_DECAY_RATIO=1.0`(全程衰减;想后段才转 corrected 调小,如 0.5)。
- **盯**:`domino_final_acc` 是否稳 > `domino_base_acc`(尤其中后位)。是 → 有效,再投 vLLM serve 侧;否 → 调 `DOMINO_LAMBDA_BASE_DECAY_RATIO` / gamma。

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

---

## Files

| What | Path |
|---|---|
| Train (multimodal DFlash, online + warm-start) | `examples/train/dflash_qwen3.5_9b_multimodal_online.sh` |
| ALLaVA/LLaVA → conversations jsonl | `scripts/llava_to_jsonl.py` |
| ALLaVA image extractor · finder | `examples/train/extract_allava_images.sh` · `examples/train/find_allava.sh` |
| MMStar → conversations jsonl | `scripts/mmstar_to_jsonl.py` |
| Training curves (TensorBoard) | `examples/train/view_tensorboard.sh` |
| Serve on GPU (baseline/mtp/dflash) | `examples/serve/run_qwen35_9b_gpu.sh` |
| Quick serve test (text · image) | `examples/serve/test_trained_dflash_gpu.sh` · `examples/serve/test_trained_dflash_mm_gpu.sh` |
| vLLM 0.22 M-RoPE guard patch · send image req | `examples/serve/patch_vllm_mrope_guard.sh` · `examples/serve/send_image_request.sh` |
| Eval client (throughput + acceptance) | `examples/evaluate/eval_qwen35_9b.sh` + `examples/evaluate/bench_mm_speculative.py` |
