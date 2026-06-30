# install — vLLM 0.22.0 on the client PAI box

The client box (`lshb-reservedpool-byzo-14`, 8× H800 80GB, conda base py3.10) has
vLLM **0.13.0**. conda is fine; we only need a fresh vLLM in an isolated env.

## Install (detached — survives SSH drops)

```bash
cd <repo>            # where you git pull'd this branch
nohup bash install/setup_vllm022.sh > ~/setup_vllm022.log 2>&1 &
tail -f ~/setup_vllm022.log
```

Check progress / completion after a reconnect:

```bash
grep -E "VLLM_VERSION|SETUP_DONE|supported" ~/setup_vllm022.log
```

If it died before `SETUP_DONE`, rerun the same `nohup` line — the env-create is
skipped if it exists and finished wheels are reused from `~/.cache/pip`.

## What it does

1. fresh conda env `vllm022` (py3.10) — does **not** touch `base`/ray.
2. `pip install vllm==0.22.0` from the aliyun mirror, long timeout + 30 retries.
3. prints the installed vLLM/torch/CUDA versions.
4. **arch check** — whether this vLLM registers `Qwen3_5MoeForConditionalGeneration`
   (the client model's architecture). This is the gate for serving — see below.

Knobs (env overrides): `CONDA_SH`, `ENV_NAME`, `PY_VER`, `VLLM_VER`, `PIP_INDEX`.

## If serving dies with "NVIDIA driver too old (found version 12080)"

vLLM 0.22.0 pulled a torch built for CUDA newer than this box's 12.8 driver. Fix by
reinstalling torch as cu128:

```bash
nohup bash install/fix_torch_cu128.sh > ~/fix_torch.log 2>&1 &
tail -f ~/fix_torch.log
grep -E "CUDA_OK|CUDA_FAIL|FIX_DONE" ~/fix_torch.log
```

Look for `CUDA_OK` + `VLLM <ver>`. Then retry the serve / STEP 1.

## After install — the decisive check

The client model is `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` (VL MoE,
native MTP head present). vLLM uses its OWN model implementations, so `--trust-remote-code`
will NOT make an unregistered arch work. The arch-check line in the log tells you:

- `supported: True`  → serve with vLLM 0.22.0 (see `RUN_CLIENT.md` / the serve cmd).
- `supported: False` → 0.22.0 predates this arch; install a newer vLLM
  (`VLLM_VER=<newer> bash install/setup_vllm022.sh`, or `pip install -U vllm`).

## Serve (8× H800, full-precision weights — NOT the quantized dir)

```bash
conda activate vllm022
M=/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B   # full-precision SFT HF weights
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "$M" \
  --served-model-name qwen3.5-vl-122b-sft \
  --tensor-parallel-size 8 --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --generation-config vllm --host 0.0.0.0 --port 8100 \
  2>&1 | tee serve_qwen35vl_122b.log
```

Use the full-precision `qwen3.5-vl-122B/`, not the Huawei-msModelSlim-quantized
`Qwen3.5-122B-A10B/` (GPU vLLM can't load that format).
