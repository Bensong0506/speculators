# [122B / test_result122B] MTP & DFlash best-checkpoint 接受率 (ALLaVA + MMStar)

122B 专用分支(与 9B 的 `test_result` / `test_result_npu` 隔开)。**两个评测都在这个文件里,别从聊天抄。**
两者都同时跑 **ALLaVA(in-domain)+ MMStar(OOD)**,完整 jsonl 自动切后 10% 当 val(无泄漏)。

- **MTP best** → 本目录 `mtp_accept/`(serve native vs stitch 后的 trained,`qwen3_5_mtp`)
- **DFlash best** → 仓库自带 `examples/evaluate/test_three_way_mmstar_allava.sh`(三路:mtp / trained_dflash / 原始 dflash)

⚠️ **122B**:MoE ~244GB,**必须 TP∈{4,8}**(heads=32,不能 1/6);4 卡 → TP=4。MTP 的 stitch 已省盘(只真拷贝带 MTP 头的 1~2 分片,其余 hardlink,不再 244GB 全拷)。

## 0. checkout + 依赖(两台都做)
```bash
cd /path/to/speculators
git checkout test_result122B && git pull
pip install -q guidellm typer rich huggingface_hub 2>/dev/null || true
```

## 1. MTP best(机器 1) —— 跑 ALLaVA + MMStar
```bash
export MODEL=/data/wenxuan/Qwen3.5-122B-A10B            # 122B verifier;路径按机器
export TRAINED_MTP_CKPT=/home/wenxuan/.../checkpoint_best   # MTP best(自动 stitch)
export ALLAVA_JSONL=$PWD/data/allava/<完整>.jsonl       # 完整集,自动切后 10%
export ALLAVA_IMAGE_ROOT=/data/wenxuan/ALLaVA-4V        # = --allowed-local-media-path
export MMSTAR_JSONL=$PWD/data/mmstar/mmstar.jsonl       # ← 设了才跑 MMStar(OOD)
export MMSTAR_IMAGE_ROOT=/data/wenxuan/mmstar/images
export NUM_SPEC_TOKENS=7 TP=4 GPUS=0,1,2,3
bash mtp_accept/run_mtp_accept_compare.sh
```
末尾打印 native vs trained 的 per-position 对比表;结果在 `mtp_accept/results/<时间戳>/` + `output_log_debug/`。
- 已 stitch 好 → 改设 `TRAINED_MTP_MODEL=/path/to/stitched` 跳过。
- sanity:trained ≈ native → `export MTP_METHOD=mtp` 重跑。

## 2. DFlash best(机器 2) —— ALLaVA + MMStar 已内置
`test_three_way_mmstar_allava.sh` 一次跑 **MMStar(OOD)+ ALLaVA(in-domain)**,三路对比,**MMStar 自动包含**,无需额外开关:
```bash
export MODEL=/data/wenxuan/Qwen3.5-122B-A10B            # 122B verifier
export BASELINE_DRAFT=<你下载的原始 122B DFlash 真实路径>   # ⚠️ 占位!这是 dflash_original 对照臂,找不到会 [fatal] 直接退
#   就是你 warm-start 用的那份 z-lab 122B DFlash;不知道路径:
#   find /data/wenxuan /home/wenxuan /data/models /home/models -maxdepth 3 -iname '*122[Bb]*DFlash*' -type d 2>/dev/null
#   (确实没有原始 122B DFlash → 告诉我,我把对照臂改成可选,只比 trained_dflash vs mtp)
export DRAFT=/home/wenxuan/.../122b-dflash/checkpoint_best  # 你训好的 122B DFlash
export ALLAVA_JSONL=$PWD/data/allava/<完整>.jsonl       # 自动切后 10%
export MMSTAR_ROOT=/data/wenxuan/mmstar                 # MMStar 根(默认 $DEFAULT_ROOT/mmstar,不对就设)
export TP=4 GPUS=0,1,2,3 INFER_NUM_SPEC=7 NUM_PROMPTS=128
bash examples/evaluate/test_three_way_mmstar_allava.sh
```
产出 `output/three_way_both/<时间戳>/combined_summary.md`(MMStar + ALLaVA 一张表),也复制进 `output_log_debug/`。

## 注意
- 接受率:MTP 从 `/metrics` spec_decode delta;DFlash 脚本自带统计 → `*_summary.json`(`spec_mean_accepted_tokens_per_draft` / `token_acceptance` / `first_position` / `output_tok_per_sec`)。
- 图片必须在 `--allowed-local-media-path`(`ALLAVA_IMAGE_ROOT` / MMStar images)下,否则 404、接受率全 0。
- 路径前缀 `/home` vs `/data` 按机器调;不通 HF 没事,全吃本地。
- 8 卡换 `TP=8 GPUS=0,1,2,3,4,5,6,7`。
