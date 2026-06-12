# DFlash + MTP 多模态蒸馏 — Session 交接（2026-06-12）

> **怎么读这份文档**：在内网克隆里 `git checkout test_result && git pull`，本文件在仓库根目录；
> 或不切分支直接 `git show test_result:HANDOFF.md`。
>
> **已校验（2026-06-12，对本地克隆核对）**：分支 tip 全部一致 —
> `allava-qwen-distill-10k 9734833` · `test_result 8fb467b` · `mtp-training e924572` · `main a9d0a1e`；
> remote：`origin = Bensong0506/speculators`、`upstream = vllm-project/speculators`。
> ⚠️ 注意分支命名：MTP 工作在 **`mtp-training`**（连字符，e924572）；另有一条独立分支 `training_mtp`（下划线，7dc482b，"Use python3…"），**别混**。

## 项目目标

提升投机解码 draft 在 VLM（Qwen3.5-9B verifier）上的接受率 → 吞吐；在 ALLaVA 域内打过开源 draft。
本 session 两大进展：① DFlash 域内全面反超原版；② 从无到有打通了 MTP 训练。

## 仓库 / 分支 / 路径

- 内网：`/home/wenxuan/speculators`（跑 GPU；Claude 在 Mac 跑不了 GPU）
- remote：`origin = Bensong0506/speculators`（SSH）；本 session 新增 `upstream = vllm-project/speculators`（取 MTP 用）
- 分支（最新 tip）：

| 分支 | tip | 说明 |
| --- | --- | --- |
| `allava-qwen-distill-10k` | `9734833` | DFlash 训练分支 + DFlash 100k 脚本 |
| `test_result` | `8fb467b` | 所有评测脚本 + 报告（本 HANDOFF 也在这） |
| `mtp-training` | `e924572` | MTP 集成（已 merge upstream/main）+ MTP 脚本 |
| `main` | `a9d0a1e` | `debug_error_from_inside`（错误/log 回传通道） |

- **错误回传约定**：用户把 log 更新到分支根目录的 `output_debug` 或 `main:debug_error_from_inside`，Claude `git show` 读。
- **工作流**：Claude（Mac）改 → push → 用户内网 pull + 跑 → `output_debug` 回传 → Claude debug。

## 关键结果

### ① DFlash 域内反超 + 两数据集 tok/s 最快（核心成果）

两步修复：**CE 损失**（替代 `kl_div`，推 top-1）+ **fp32** + **LR 3e-5**（替代 bf16/1e-5）。
最新 `dflash_ce_fp32_lr3e5` checkpoint_best @spec7：

| 指标 | trained | 原版 | MTP |
| --- | --- | --- | --- |
| ALLaVA mean-accept | **2.369** | 1.938 | 2.807 |
| ALLaVA first-pos | **0.775** | 0.728 | 0.818 |
| ALLaVA tok/s | **71.4（最快）** | 63.2 | 65.9 |
| MMStar mean / first-pos | 2.162 / 0.758 | 2.099 / 0.766 | 2.822 / 0.828 |

- trained DFlash 在两个数据集上 **tok/s 都最快**（超原版 + 超 MTP，因为 DFlash draft 每步更便宜）。
- MMStar 上 ≥ 原版（**无遗忘**）。
- **关键技术发现**：first-pos 长期卡死 = 纯 bf16 把小于 ULP 的更新舍掉（LR 3e-5 下尤甚）；fp32 解锁 → first-pos 0.727 → 0.775。
- **但 10k 过拟合**（train acc → 0.9，val 0.46 → 0.55 见顶回落）→ 需要更多数据。
- 报告：`test_result:examples/evaluate/dflash_ce_finetune_report.md`（汇报版）。

### ② MTP 训练打通（从无到有）

- **关键认知**：speculators 的 MTP「训练」= 抽取并微调 verifier 自带的**原生 MTP 头**（Qwen3.5-9B 有，就是 benchmark 里 `qwen3_5_mtp` serve 的那个），**不是**从零训新 draft。
- `mtp-training` = `allava-qwen-distill-10k` + merge `upstream/main`，把 MTP 接进多模态管线（保留你的 dataloader 设计）。
- 冒烟测试通过：loss 1.03 → 0.81 下降、checkpoint 存盘、干净退出。
- **踩过并修好的坑**：
  1. merge 冲突（`train.py` / `data.py` 取 fork 版）
  2. VLM 的 `text_config`（`vocab_size` 嵌套）
  3. 单层 hidden-state 宽度（MTP 只用最后一层，`[:, :-1]` 会切空）
  4. fp32 破坏 MTP `lm_head`（dtype mismatch）→ **MTP 用 bf16**（DFlash 保持 fp32）

### ③ 100k 数据 + 训练脚本（当前在跑/待跑）

- **100k 蒸馏**：给 8 卡脚本加了 `SKIP_SAMPLES`，两台不互通机器各跑 50k
  （A：`SKIP_SAMPLES=0`，B：`SKIP_SAMPLES=50000`）再 `cat` → `data/allava/allava_qwen35_distill_100k.jsonl`。用户已做好这份数据。
- **当前状态**：用户正要/正在跑两个 100k 训练。MTP 第一次跑 fp32 崩了（lm_head dtype），已改 bf16，待重跑。

## 可复制命令

**DFlash 100k**（`allava-qwen-distill-10k`，CE + fp32 + 3e-5，EPOCHS=10）：

```bash
cd /home/wenxuan/speculators && git checkout allava-qwen-distill-10k && git pull
bash examples/train/nohup_dflash_qwen3.5_9b_allava_distilled_100k.sh
```

**MTP 100k**（`mtp-training`，bf16 + 3e-5，EPOCHS=10）：

```bash
cd /home/wenxuan/speculators && git checkout mtp-training && git pull
bash examples/train/nohup_mtp_qwen3.5_9b_allava_distilled_100k.sh
```

**评测**（双数据集三路，mtp / trained / original 一把出，`test_result` 分支）：

```bash
DRAFT=<checkpoint_best> INFER_NUM_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_three_way_mmstar_allava.sh
```

**按真实接受率选 best checkpoint**：

```bash
# CHECKPOINT_FIND_ROOT 指向 run 的 checkpoints
bash examples/evaluate/sweep_dflash_allava_checkpoints.sh
```

## 各分支关键新文件

- `test_result/examples/evaluate/`：`test_three_way_mmstar_allava.sh`（一键双数据集三路）、`test_dflash_mmstar_three_way.sh`、`sweep_dflash_allava_checkpoints.sh`、`dflash_ce_finetune_report.md`、`allava_val_four_way_summary.md`
- `allava-qwen-distill-10k/examples/train/`：`nohup_dflash_qwen3.5_9b_allava_distilled_100k.sh`
- `mtp-training/examples/train/`：`nohup_mtp_qwen3.5_9b_allava_distilled.sh`、`nohup_mtp_qwen3.5_9b_allava_distilled_100k.sh`；MTP 钩子在 `scripts/train.py` + `src/speculators/train/data.py`

## 下一步（待办）

1. 跑完 100k → 用 sweep 按**真实接受率**选 best → 双数据集三路对比 trained-DFlash vs trained-MTP vs original。
2. **评测 trained-MTP**：需确认 vLLM 怎么挂微调后的 MTP 头（之前 `qwen3_5_mtp` 用的是 verifier 自带的，spec-config 没 model 路径）—— 这是 MTP 评测的**待解点**。
3. （可选）修 MTP fp32 的 `lm_head` / 各层 dtype 一致性，让 MTP 也能用 fp32。
4. 若想再提升：pos-1 加权 CE / 加 draft 容量 / on-policy replay。

## 评测标杆（原版 / native @spec7，记牢）

- **MMStar**：first-pos ~0.764，mean ~2.09
- **ALLaVA**：first-pos ~0.728，mean ~1.94
- **MTP@7**：ALLaVA 0.818 / 2.807，MMStar 0.828 / 2.822
- **判据**：trained 在域内（ALLaVA）要 > 原版；MMStar 是遗忘检查不指望赢。tok/s 有 ±数% 噪声，信 first-pos / mean-accept。

## 关键事实（debug 用）

- `checkpoint_best` 按 val-loss 选（`trainer.py`）；选最优应按**真实接受率**（sweep），别只看 val。
- 在线管线每 epoch 重生成 hidden states（gen 为主，100k 每 epoch 数小时；缓存不现实 ~TB）。
- MTP 钩子要点：`--speculator-type mtp` + `--num-speculative-steps` / `--step-weight-beta`；MTP 用 verifier 的 `text_config` + 只取最后一层 hidden state；**MTP 必须 bf16**。
