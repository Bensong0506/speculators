# DFlash 多模态 Draft 微调 — 100k 数据结果报告 (2026-06-15)

> 接续 [`dflash_ce_finetune_report.md`](dflash_ce_finetune_report.md)（2026-06-11，10k 蒸馏数据）。
> 那份报告的收尾计划是「把蒸馏数据扩到 100k（10×）」治过拟合、抬高接受率天花板——本报告即该计划的结果。

## 执行摘要（汇报用）

**目标**：让 DFlash 投机解码 draft 在多模态（VLM, Qwen3.5-9B）场景下更强，至少在 ALLaVA 域内**超过开源原版 draft**，提升端到端吞吐；并验证扩数据能否进一步缩小与原生 MTP 的差距、且不伤泛化。

**结论（100k 蒸馏数据，CE + fp32 + LR 3e-5，`checkpoint_best`，@spec=7，n=128）**：

- **域内（ALLaVA）相对原版 DFlash 大幅领先**：mean-accept **+37%**（1.945→2.666）、first-pos **+10.5%**（0.733→0.810）、tok/s **+20%**（三项全胜，且较 10k 进一步拉大）。
- **OOD（MMStar）从「持平」转为「真正反超原版」**：mean-accept **+15%**（2.102→2.416）、first-pos **+4.6%**、tok/s **+12%**。10k 时 OOD 仅约等于原版（1.03×、first-pos 还略输），扩数据后 OOD 出现明确正收益——**没有过拟合 / 灾难性遗忘，泛化反而变好**。
- **两个数据集上都是吞吐（tok/s）最快的方案**，超过原版 DFlash 和原生 MTP（DFlash draft 每步更便宜，即便接受率略低于 MTP，墙钟吞吐仍最高）。
- **与最强的 MTP 的接受率差距继续收窄**：域内 mean-accept 比从 10k 的 0.844× → **0.914×**（差距由 ~16% 缩到 ~9%），同时吞吐对 MTP 的领先从 1.084× → **1.124×**。

**一句话**：扩数据 10k→100k 在**域内、域外、吞吐每一项上都把 trained DFlash 又抬高了一截**，并把 OOD 从「不退化」推进到「反超原版」，与 MTP 的接受率差距收窄、吞吐领先扩大。10k 报告里担心的过拟合，被 100k 数据解决了。

---

## 最新结果（100k 蒸馏数据，checkpoint_best，@spec=7，n=128）

draft = `dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_ce_fp32_lr3e5_100k_0612_0831/checkpoints/checkpoint_best`

### ALLaVA val（域内，四路）

| rank | method | tok/s | mean accept/draft | token accept | first-pos | completed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **trained DFlash** | **73.779** | 2.666 | 0.381 | 0.810 | 128/128 |
| 2 | MTP | 65.634 | **2.916** | **0.417** | **0.838** | 128/128 |
| 3 | original DFlash | 61.375 | 1.945 | 0.278 | 0.733 | 128/128 |
| 4 | baseline (no spec) | 30.139 | n/a | n/a | n/a | 128/128 |

- trained vs original：mean-accept **1.370×**、first-pos **1.105×**、tok/s **1.202×**（三项全胜）
- trained vs MTP：mean-accept 0.914×、first-pos 0.967×、tok/s **1.124×**（接受率仍小输，吞吐反超）
- trained vs baseline：tok/s **2.448×**

### MMStar（OOD，泛化 / 遗忘检查，三路）

| rank | method | tok/s | mean accept/draft | token accept | first-pos | completed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **trained DFlash** | **76.106** | 2.416 | 0.345 | 0.801 | 128/128 |
| 2 | MTP | 72.556 | **2.822** | **0.403** | **0.828** | 128/128 |
| 3 | original DFlash | 67.904 | 2.102 | 0.300 | 0.766 | 128/128 |

- trained vs original：mean-accept **1.150×**、first-pos **1.046×**、tok/s **1.121×**（≥ 原版，且首次在 OOD 三项全胜）
- trained vs MTP：mean-accept 0.856×、first-pos 0.967×、tok/s **1.049×**

> 排序（两数据集一致）：吞吐 `trained DFlash > MTP > original DFlash > baseline`；接受率 `MTP > trained DFlash > original DFlash`。

---

## 关键进展：10k → 100k（扩数据的收益）

trained DFlash 自身随蒸馏数据量的变化（CE + fp32 + LR 3e-5 固定，只变数据量）：

| 数据集 | 指标 | 10k | 100k | 变化 |
|---|---|---:|---:|---:|
| ALLaVA（域内） | mean-accept | 2.369 | **2.666** | **+12.5%** |
| ALLaVA（域内） | first-pos | 0.775 | **0.810** | **+4.5%** |
| ALLaVA（域内） | tok/s | 71.437 | **73.779** | **+3.3%** |
| MMStar（OOD） | mean-accept | 2.162 | **2.416** | **+11.7%** |
| MMStar（OOD） | first-pos | 0.758 | **0.801** | **+5.7%** |
| MMStar（OOD） | tok/s | 70.910 | **76.106** | **+7.3%** |

相对基线（原版 DFlash / MTP）的倍率变化：

| 对比 | 指标 | 10k | 100k | 走向 |
|---|---|---:|---:|---|
| vs original（域内） | mean-accept | 1.222× | **1.370×** | 领先扩大 |
| vs original（OOD） | mean-accept | 1.030× | **1.150×** | 持平 → 真反超 |
| vs original（OOD） | first-pos | 0.990× | **1.046×** | 略输 → 反超 |
| vs MTP（域内） | mean-accept | 0.844× | **0.914×** | 差距收窄 |
| vs MTP（OOD） | mean-accept | 0.766× | **0.856×** | 差距收窄 |
| vs MTP（域内） | tok/s | 1.084× | **1.124×** | 吞吐领先扩大 |

**要点**：扩数据在**每一项上都是正向**，且对 OOD 的提升（+11.7% mean-accept）与域内（+12.5%）几乎同幅——说明 100k 学到的是更通用的 draft 能力，而非过拟合 ALLaVA。这正面回答了 10k 报告留下的过拟合疑问。

---

## 怎么读这些数（为什么 DFlash 接受率低于 MTP，却更快）

- **接受率（mean-accept / token-accept / first-pos）**：MTP 仍是最强，trained DFlash 次之，原版 DFlash 最低。这是「draft 猜得有多准」。
- **吞吐（tok/s）**：trained DFlash 最快。DFlash 的 draft 头**每步计算比 MTP 便宜**，所以即使每次 draft 接受的 token 略少，单位时间内的有效产出仍最高——**端到端用户实际感受到的是吞吐**，这正是项目目标。
- 因此两条结论并不矛盾：**MTP = 接受率天花板；trained DFlash = 吞吐天花板**。本轮把 trained DFlash 的接受率推得更接近 MTP，同时保住并扩大了吞吐优势。

---

## 诚实的注意事项

- **tok/s 有 run 间噪声**（单流离线基准 ±数%）。trained 对 MTP 的吞吐领先（1.05–1.12×）应理解为「持平到小幅领先」；最稳健的结论以**接受率**为准（接受率指标 run 间稳定）。对**原版 DFlash** 的领先（+15%~+37% 接受率）远超噪声，结论稳。
- **baseline 仅在 ALLaVA 测了**（30.139 tok/s）；MMStar 本轮未测 no-spec baseline，故 OOD 的「×baseline」未列。
- **基线可复现性（sanity anchor）**：原版 DFlash 在 10k / 100k 两次独立 run 下数值基本一致（域内 mean-accept 1.938 vs 1.945、OOD 2.099 vs 2.102），说明评测台稳定，trained 的增量可信。
- 本轮按 **min-val-loss 的 `checkpoint_best`** 选点（CE 下 val-loss 跟踪 top-1，与接受率一致）；未做 per-epoch 真实接受率 sweep，最优 epoch 可能还能再抠一点。

---

## 结论与下一步

**结论**：项目目标——「在 ALLaVA 域内超过开源 DFlash draft 并提升吞吐」——已**稳健达成且优势随数据扩大显著增强**；OOD 由「不退化」推进到「反超」；与 MTP 的接受率差距收窄、吞吐持续领先。trained DFlash 是当前两数据集上**吞吐最快**的方案。

**下一步（优先级从高到低）**
1. **per-epoch 接受率 sweep 选 checkpoint**：用真实接受率（而非 val-loss）在 100k 训练曲线上挑最优点，可能再抬一截。
2. **针对 first-position 加权 CE**：把 gate 位置（pos-1）再往 MTP 靠，这是与 MTP 接受率差距的主要来源。
3. **真实服务下复核吞吐**：在端到端在线服务（非 128-prompt 离线基准）下确认 tok/s 领先成立。
4. **（搁置）causal DFlash**：stock vLLM 0.22 的 DFlash 写死非因果、无 causal 开关（仅本 fork 有），serve 端暂做不了，待 fork 侧打通后再评。

---

## 复现

```bash
# 双数据集三路（mtp / trained / original，MMStar + ALLaVA 一把出）
DRAFT=.../dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_ce_fp32_lr3e5_100k_0612_0831/checkpoints/checkpoint_best \
ALLAVA_JSONL="$(pwd)/data/allava/allava_qwen35_distill_10k.jsonl" \
INFER_NUM_SPEC=7 MTP_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_three_way_mmstar_allava.sh
```

- 来源日志：本分支 `test_result:output_log`
- 本轮 artifact：`output/three_way_both/20260615_020313/`
  - combined：`.../combined_summary.md`
  - mmstar：`.../mmstar/mmstar_three_way_summary.md`
  - allava：`.../allava/allava_val_summary.md`
- checkpoint：`dflash_ce_fp32_lr3e5_100k_0612_0831`（100k 蒸馏数据，从 10k continue，CE + fp32 + LR 3e-5）
- 回传：combined summary 已 copy 到内网 `output_log_debug`（内网只能 pull 不能 push，结果走该文件回传）
