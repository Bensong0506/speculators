# DFlash 多模态 Draft 微调 — 进展报告 (2026-06-11)

## 执行摘要（汇报用）

**目标**：让 DFlash 投机解码 draft 在多模态（VLM, Qwen3.5-9B）场景下更强，至少在 ALLaVA 域内**超过开源原版 draft**，提升端到端吞吐。

**结论**：经过「① 换损失函数（KL→CE）→ ② 修优化（纯 bf16 → fp32、LR 1e-5→3e-5）」两步，训练出的 draft：

- **域内（ALLaVA）全面超过原版 DFlash**：mean-accept **+22%**（1.94→2.37）、first-position **+6.4%**（0.728→0.775）、tok/s **+13%**。
- **在两个数据集上都是吞吐（tok/s）最快的方案**，超过原版 DFlash 和原生 MTP（DFlash draft 每步更便宜）。
- **OOD（MMStar）不退化**：≥ 原版，无灾难性遗忘（早期 KL 方案曾退化到 0.87×）。
- 与最强的 MTP 相比，接受率仍有差距（mean-accept 域内 0.84×），但差距已从 0.71× 收窄，且吞吐已反超 MTP。

**关键技术发现**：first-position 接受率长期卡死，根因是**纯 bf16 训练把小于精度的权重更新舍掉了**；切到 fp32 + 提高 LR 后 first-pos 第一次明显上升（0.727→0.775），验证了这是优化瓶颈而非模型容量上限。

**当前瓶颈与下一步**：本结果来自 10k 蒸馏数据，已观察到过拟合（val 见顶回落）。**正在把蒸馏数据扩到 100k（10×，双机 16 卡并行）**，预期进一步抬高接受率、继续缩小与 MTP 的差距。

---

## 最新结果（CE + fp32 + LR 3e-5，checkpoint_best，@spec=7，n=128）

### ALLaVA val（域内）

| method | tok/s | mean accept/draft | token accept | first-pos |
|---|---:|---:|---:|---:|
| **trained DFlash** | **71.437** | 2.369 | 0.338 | 0.775 |
| MTP | 65.876 | **2.807** | **0.401** | **0.818** |
| original DFlash | 63.249 | 1.938 | 0.277 | 0.728 |

- trained vs original: mean-accept **1.222×**, first-pos **1.064×**, tok/s **1.129×**（三项全胜）
- trained vs MTP: mean-accept 0.844×, first-pos 0.948×, tok/s **1.084×**（吞吐反超 MTP）

### MMStar（OOD，泛化/遗忘检查）

| method | tok/s | mean accept/draft | token accept | first-pos |
|---|---:|---:|---:|---:|
| **trained DFlash** | **70.910** | 2.162 | 0.309 | 0.758 |
| original DFlash | 67.564 | 2.099 | 0.300 | 0.766 |
| MTP | 67.309 | **2.822** | **0.403** | **0.828** |

- trained vs original: mean-accept **1.030×**, first-pos 0.990×, tok/s 1.050×（≥ 原版，无遗忘）
- trained vs MTP: mean-accept 0.766×, first-pos 0.916×, tok/s 1.053×

> **tok/s 注意**：单流吞吐有 ±数% 的 run 间噪声，trained 对 MTP 的吞吐领先（1.05–1.08×）属于「持平到小幅领先」。稳健结论以**接受率**为准（接受率指标 run 间稳定）。

---

## 进展轨迹（ALLaVA 域内，mean-accept / first-pos；原版 ≈ 1.94 / 0.728）

| 阶段 | mean-accept | first-pos | vs 原版 (mean) | 结论 |
|---|---:|---:|---:|---|
| KL 损失 (bf16, 1e-5) | 1.615 | 0.672 | 0.83× | 输给原版 |
| CE 损失 (bf16, 1e-5) | 1.993 | 0.727 | 1.03× | mean 反超、first-pos 持平 |
| **CE + fp32 + 3e-5** | **2.369** | **0.775** | **1.22×** | **全面反超，first-pos 终于动** |

两个关键修复：
1. **KL → CE**：KL 只压软分布距离、不推 top-1；而 @temp0 的接受率只认 top-1，所以 KL 训练的 draft 一直输。CE 直接对 verifier 的 argmax 做交叉熵，top-1/接受率随之上升。
2. **bf16 → fp32 + 提 LR**：纯 bf16 下，AdamW 的单步更新（~1e-5）远小于权重的 bf16 ULP（~1e-3），被舍入归零——尤其打击已接近收敛的 first-position。fp32 master 权重保留小更新，first-pos 第一次明显上升。

---

## 下一步计划

**短期（进行中）**
1. **扩数据到 100k（10×）**：双机各 8 卡（16 卡并行）蒸馏 on-policy ALLaVA 数据，治 10k 过拟合、抬高接受率天花板。完成后用 CE + fp32 + 3e-5 重训。
2. **按真实接受率选 checkpoint**：用 per-epoch sweep（而非 val-loss）挑最优 epoch，避免选到过拟合点。
3. **正则**：按 100k 的 train/val 曲线决定是否加 weight_decay（需新增开关）/ early-stop。

**中期（缩小与 MTP 的接受率差距）**
4. **针对 first-position 加权**：position-1 加权 CE，把 gate 位置再往上抠。
5. **容量 / 特征**：必要时增加 draft 层或调整 aux hidden-state 层（pos≥2 预测更难）。
6. **on-policy 数据 / replay**：用 verifier 自身输出做训练分布，进一步贴合解码时上下文。

**验证**
7. **真实服务下复核吞吐**：确认 tok/s 领先在端到端服务（非 128-prompt 离线基准）下成立。

---

## 复现

```bash
# 双数据集三路（mtp / trained / original，MMStar + ALLaVA 一把出）
DRAFT=.../checkpoints/checkpoint_best INFER_NUM_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_three_way_mmstar_allava.sh
```

来源日志 `origin/main:debug_error_from_inside`；本轮 artifact `output/three_way_both/20260611_015741/combined_summary.md`；checkpoint `dflash_ce_fp32_lr3e5_0610_0922`（10k 蒸馏数据，fp32 + LR 3e-5）。
