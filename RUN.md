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
Defaults: `MAX_SAMPLES=100000 EPOCHS=20 CHECKPOINT_FREQ=1 LR_FT=1e-5 MAX_ANCHORS=512 LOGGER=wandb`.
These defaults are meant for an early-checkpoint sweep: evaluate each saved
checkpoint with `INFER_NUM_SPEC=7` instead of trusting training validation loss
or `checkpoint_best` alone.

### 1d. Build 10k Qwen-distilled ALLaVA data
This removes the original ALLaVA/GT assistant answers, asks Qwen3.5-9B to answer
the same image prompts, and writes a training-ready conversations jsonl.

```bash
cd /home/wenxuan/speculators

bash examples/train/distill_allava_qwen35_10k_8gpu.sh
```

Default output:

```bash
/home/wenxuan/speculators/data/allava/allava_qwen35_distill_10k.jsonl
```

The 8-GPU launcher starts one verifier server per GPU on ports `8100..8107`,
writes shard files under `data/allava/allava_qwen35_distill_10k_shards/`, then
merges them back into the default output path. It is resumable by default.

**100k 变体（治过拟合 — 10× 数据）。** `MAX_SAMPLES` 是**总量**（跨卡取模分片），用 100k 专属输出路径别覆盖 10k。

**两台机器各 8 卡 = 16 卡并行（时间减半）** —— 两台不互通，所以各跑**不重叠的一半**再拼。`SKIP_SAMPLES` 是分片前的全局偏移：

机器 A（前 50k）：
```bash
cd /home/wenxuan/speculators
git checkout allava-qwen-distill-10k && git pull
SKIP_SAMPLES=0 MAX_SAMPLES=50000 \
FINAL_JSONL="$(pwd)/data/allava/allava_qwen35_distill_100k_partA.jsonl" \
SHARD_ROOT="$(pwd)/data/allava/allava_qwen35_distill_100k_partA_shards" \
bash examples/train/distill_allava_qwen35_10k_8gpu.sh
```
机器 B（后 50k）：
```bash
cd /home/wenxuan/speculators
git checkout allava-qwen-distill-10k && git pull
SKIP_SAMPLES=50000 MAX_SAMPLES=50000 \
FINAL_JSONL="$(pwd)/data/allava/allava_qwen35_distill_100k_partB.jsonl" \
SHARD_ROOT="$(pwd)/data/allava/allava_qwen35_distill_100k_partB_shards" \
bash examples/train/distill_allava_qwen35_10k_8gpu.sh
```
两边都跑完，在**训练那台机器**上合并（先把另一台的 part 拷过来）：
```bash
scp <另一台>:/home/wenxuan/speculators/data/allava/allava_qwen35_distill_100k_partB.jsonl data/allava/
cat data/allava/allava_qwen35_distill_100k_partA.jsonl \
    data/allava/allava_qwen35_distill_100k_partB.jsonl \
    > data/allava/allava_qwen35_distill_100k.jsonl
wc -l data/allava/allava_qwen35_distill_100k.jsonl   # ≈ 100000
```

**单机 8 卡（不拆）**：`MAX_SAMPLES=100000 FINAL_JSONL=...100k.jsonl SHARD_ROOT=...100k_shards bash examples/train/distill_allava_qwen35_10k_8gpu.sh`

要点：
- 100k = 前 100k 条 prompt = 全 Caption-LAION（和 10k 同分布，只是 10×）。
- A/B 各 8 卡 × 6.25k，**可断点续跑**（重跑即续）；端口 8100-8107（两台各自用，不冲突）。
- 拼完训练：1e 命令前加 `DISTILLED_ALLAVA_JSONL="$(pwd)/data/allava/allava_qwen35_distill_100k.jsonl" MAX_SAMPLES=100000`（两个都要，否则只用前 10k）。

Single-GPU fallback:

```bash
bash examples/train/distill_allava_qwen35_10k.sh
```

To reuse an existing verifier server:

```bash
START_SERVER=0 \
ENDPOINT=http://localhost:8100/v1 \
bash examples/train/distill_allava_qwen35_10k.sh
```

### 1e. Train on 10k Qwen-distilled ALLaVA
```bash
cd /home/wenxuan/speculators
git pull origin allava-qwen-distill-10k        # get latest fixes/toggles

bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
```

Defaults: `MAX_SAMPLES=10000 EPOCHS=20 CHECKPOINT_FREQ=1 LR_FT=1e-5` (bf16, warm-start from open-source DFlash).

Env toggles:

| var | default | what |
|---|---|---|
| `CONTROL_LR0` | `0` | `1` = force LR to 0 (no weight update). Diagnostic: teacher-forced val MUST stay flat across epochs; any drift = bug. |
| `HIDDEN_STATES_DTYPE` | `bfloat16` | `float32` = train params + AdamW state in fp32 (fixes small-LR bf16 rounding; more GPU mem). |
| `GEN_MAX_MODEL_LEN` | `SEQ_LENGTH+2048` | gen-server context; headroom so image-expanded prompts >4096 aren't rejected (training still caps at `SEQ_LENGTH`). |
| `ON_GENERATE` | `delete` | `cache` = keep generated hidden states across epochs (needs a writable `hidden_states/` dir). |
| `LOSS_FN` | `kl_div` | `ce` = cross-entropy on the verifier's argmax (targets top-1 / acceptance, vs KL matching the soft distribution). |

#### test1 — LR=0 control (is the val dip a bug?)
```bash
CONTROL_LR0=1 EPOCHS=2 MAX_SAMPLES=500 \
  bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
```
Watch the teacher-forced val `position_1_acc` at `initial_val / epoch0 / epoch1`:
- all ≈ equal (within ~0.005) → **no bug**; the dip-then-recover is warm-start optimization dynamics.
- still drifts (e.g. 0.78→0.75) → **bug**: something mutates the model outside `optimizer.step()`.

#### fp32 training (only if you decide bf16 precision is the issue)
```bash
HIDDEN_STATES_DTYPE=float32 \
  bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
```

#### switch training loss to CE (target top-1, not KL) — current experiment
```bash
LOSS_FN=ce bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
```
KL lowers the soft-distribution distance but may not move top-1 (= what acceptance needs). CE pushes the verifier's argmax toward top-1. Watch **train top-1** (`position_1_acc`), not loss: rising = the objective was the problem; still flat = a real ceiling (capacity / features / data / bf16).

#### ⭐ 过夜长跑实验：CE + fp32 + LR 3e-5 + warmup（攻中段位置 pos2-4）
目标：验证「纯 bf16 舍入 / 小 LR 把提升卡住」的假设，并把**中段位置**的接受率往上推（pos2-4 条件率最低 ~0.63、headroom 最大；first-pos 已 0.73 接近天花板）。改动：`HIDDEN_STATES_DTYPE=float32`（小更新不被 bf16 ULP 舍掉）+ `LR_FT=1e-5→3e-5` + warmup（默认 linear、约 1 epoch）+ **不限 20 epoch，过夜长跑**。
```bash
cd /home/wenxuan/speculators
git pull origin allava-qwen-distill-10k

LOSS_FN=ce \
HIDDEN_STATES_DTYPE=float32 \
LR_FT=3e-5 \
EPOCHS=100 \
CHECKPOINT_FREQ=5 \
RUN_NAME="dflash_ce_fp32_lr3e5_$(date +%m%d_%H%M)" \
bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh
tail -f run_logs/dflash_ce_fp32_lr3e5_*.nohup.log
```
- 盯 wandb 的 `position_2/3/4_acc`（+ `position_1_acc` / `full_acc`）：中段爬升 = fp32/LR 起效；pos1 动了算白捡。
- checkpoint：`output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/<RUN_NAME>/checkpoints/`，每 5 epoch 一个（bf16 存盘）。**20 个 checkpoint 含 optimizer ≈ 150GB**，磁盘紧就把 `CHECKPOINT_FREQ` 调大。
- 早上：切 `test_result`，用 `sweep_dflash_allava_checkpoints.sh` 扫这些 checkpoint 按真实接受率挑最优（`CHECKPOINT_FIND_ROOT=.../<RUN_NAME>/checkpoints`），再跑 4-way / 三路对比原始 + MTP。
- fp32 更费显存、更慢；若 OOM 降 batch 或回 `bfloat16`。想显式控制 warmup/总步数可加 `--scheduler-warmup-steps/--scheduler-total-steps`（train.py 支持，launcher 暂未透传）。

### 1f. Watch training (loss + per-position acceptance)
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

---

## Files

| What | Path |
|---|---|
| Train (multimodal DFlash, online + warm-start) | `examples/train/dflash_qwen3.5_9b_multimodal_online.sh` |
| Distill 10k ALLaVA with Qwen | `examples/train/distill_allava_qwen35_10k.sh` |
| Distill 10k ALLaVA with Qwen on 8 GPUs | `examples/train/distill_allava_qwen35_10k_8gpu.sh` |
| Train on Qwen-distilled 10k ALLaVA | `examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_10k.sh` |
| ALLaVA Qwen distillation client | `scripts/distill_allava_with_qwen.py` |
| ALLaVA/LLaVA → conversations jsonl | `scripts/llava_to_jsonl.py` |
| ALLaVA image extractor · finder | `examples/train/extract_allava_images.sh` · `examples/train/find_allava.sh` |
| MMStar → conversations jsonl | `scripts/mmstar_to_jsonl.py` |
| Training curves (TensorBoard) | `examples/train/view_tensorboard.sh` |
| Serve on GPU (baseline/mtp/dflash) | `examples/serve/run_qwen35_9b_gpu.sh` |
| Quick serve test (text · image) | `examples/serve/test_trained_dflash_gpu.sh` · `examples/serve/test_trained_dflash_mm_gpu.sh` |
| vLLM 0.22 M-RoPE guard patch · send image req | `examples/serve/patch_vllm_mrope_guard.sh` · `examples/serve/send_image_request.sh` |
| Eval client (throughput + acceptance) | `examples/evaluate/eval_qwen35_9b.sh` + `examples/evaluate/bench_mm_speculative.py` |
