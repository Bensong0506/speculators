# [GPU / test_result] MTP best checkpoint:原生 vs 训好的 (ALLaVA + MMStar)

GPU 走 `test_result`(本目录),NPU 走 `test_result_npu:npu_mtp_accept/`,两边隔离。
方式 = test_result 多模态评测:serve(`qwen3_5_mtp` + `--attention-backend flash_attn` +
`--allowed-local-media-path` + `--limit-mm-per-prompt '{"image":1}'`)+ `mmstar_weight_client.py`
→ `{arm}_summary.json`。native 臂 = serve 基座;trained 臂 = serve 自动 stitch 的目录。

## 0. checkout + 依赖
```bash
cd /path/to/speculators
git checkout test_result && git pull
pip install -q typer rich huggingface_hub 2>/dev/null || true   # stitch 用
chmod +x mtp_accept/*.sh mtp_accept/stitch_mtp.py
```

## 1. 跑 MTP best(自动 stitch + 两数据集 native/trained 各一臂)
```bash
export MODEL=/home/wenxuan/Qwen3.5-9B
export TRAINED_MTP_CKPT=/home/wenxuan/.../checkpoints/checkpoint_best   # MTP best
export ALLAVA_JSONL=$PWD/data/allava/allava_qwen35_distill_100k.jsonl  # 完整集,脚本自动切后 10% 当 val(无泄漏)
export ALLAVA_IMAGE_ROOT=/home/wenxuan/ALLaVA-4V
# 想改比例:VAL_RATIO=0.1(默认);已有现成 val 则改给 ALLAVA_VAL_JSONL 跳过切分
export MMSTAR_JSONL=$PWD/data/mmstar/mmstar.jsonl MMSTAR_IMAGE_ROOT=/home/wenxuan/mmstar/images  # 可选 OOD
export NUM_SPEC_TOKENS=7 TP=1 GPUS=0
bash mtp_accept/run_mtp_accept_compare.sh
```
- 已 stitch 的目录 → 改设 `TRAINED_MTP_MODEL=/path/to/stitched`,跳过 stitch。
- 末尾打印 native vs trained 对比表(mean-accept / token-accept / first-pos / tok/s + ratio);
  数据在 `mtp_accept/results/<时间戳>/`,汇总也进 `output_log_debug/`。

## 注意
- 接受率来自 server `/metrics` 的 spec_decode delta;图片必须在 `--allowed-local-media-path` 下。
- sanity:trained ≈ native → `export MTP_METHOD=mtp` 重跑。
- 代理坑已修(脚本内 `no_proxy=localhost`),`/health` 不会再超时。
- ALLaVA val jsonl 没有就用 `scripts/llava_to_jsonl.py` 转;MMStar 用 `scripts/mmstar_to_jsonl.py`。

---
DFlash best checkpoint(另一台 GPU)走仓库自带脚本,不在本目录,见 README/§ 下方命令:
`DRAFT=.../checkpoints/checkpoint_best INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 bash examples/evaluate/test_three_way_mmstar_allava.sh`
