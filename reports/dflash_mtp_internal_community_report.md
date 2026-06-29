# Qwen3.5 多模态投机解码加速实验总结: DFlash 与 MTP 接受率提升

**日期**: 2026-06-29  
**范围**: Qwen3.5-9B 多模态模型与 Qwen3.5-122B-A10B 多模态 MoE 模型  
**主题**: 在 vLLM 推理框架下，通过 DFlash draft 训练与 MTP 头微调提升多模态投机解码的接受率与端到端吞吐  

## 摘要

投机解码(speculative decoding)的目标是在不改变主模型输出分布的前提下提升解码吞吐。其基本流程是由轻量 draft 机制一次提出多个候选 token，再由主模型 verifier 并行校验。只要 verifier 接受的 token 数增多，每次主模型前向能够产出的有效 token 就增多，端到端吞吐通常随平均接受长度提升。

本文总结了截至目前在 Qwen3.5 多模态 9B 与 122B-A10B 上围绕 DFlash 与 MTP 两条路线完成的实验。核心结论如下:

1. **投机解码在 122B 多模态模型上已经形成可用闭环**。在 ALLaVA 域内评测上，MTP 相对无投机 baseline 达到约 **2.9x** 输出吞吐；训练后的 DFlash 达到约 **2.6x** 输出吞吐。该加速由 verifier 逐 token 校验保证无损。
2. **DFlash 的提升幅度最大**。9B 上，100k 自蒸馏训练后的 DFlash 在 ALLaVA 上相对原版 DFlash 达到平均接受长度 **+37%**、QPS **+20%**；在 MMStar 域外也达到平均接受长度 **+15%**、QPS **+12%**。122B 上，同样观察到训练后 DFlash 在 ALLaVA 上平均接受长度 **+30.2%**、QPS **+16.0%**，在 MMStar 上平均接受长度 **+8.4%**、QPS **+6.7%**。
3. **MTP 的绝对接受率更高，提升幅度相对较小但更适合作为强基线**。9B 上，微调原生 MTP 头使 ALLaVA 平均接受长度 **+7.7%**、QPS **+6.1%**；经 WiSE-FT 权重插值后，ALLaVA **+7.9%**、MMStar **+1.4%**，实现两域同时为正。122B 上，`beta=0.6` 步权重训练使 ALLaVA 平均接受长度 **+11.5%**，`alpha=0.5` 权重插值后 ALLaVA **+10.2%**、MMStar **+2.3%**。
4. **两条路线的收益来源不同**。DFlash 开源 draft 原本缺少多模态数据分布，自蒸馏训练补齐了图文场景下的 verifier 分布，因此提升显著；MTP 原生头已有多模态训练基础，起点更高、可挖增量更小，但接受率上限更高。
5. **后续高并发场景的瓶颈会转向 verifier 校验**。当 draft 侧已经足够便宜时，减少每轮 verify token 数比继续提升 draft 速度更关键。D-Cut 可作为下一阶段优化方向: 例如 spec=7 时若平均仅校验 4 个 token，则 verify token 数约减少 **43%**。目前 GPU 路径已跑通，下一步是迁移至 NPU。

## 1. 指标与实验口径

### 1.1 指标定义

| 指标 | 单位 | 含义 | 解读 |
|---|---:|---|---|
| QPS / output tok/s | token/s | 端到端输出 token 吞吐 | 用户侧直接感知的速度指标 |
| mean accept/draft | token/draft | 每轮投机平均被 verifier 接受的 token 数 | 本文主要称为“接受率指标”或“平均接受长度” |
| token acceptance | ratio | 被接受 token 数 / draft token 数 | spec=7 时约等于 mean accept/draft 除以 7 |
| first-position acceptance | ratio | draft 第 1 个 token 被接受的概率 | 链式接受的入口位置，杠杆最高 |

注意: 本文中的“接受率”主要指 **mean accept/draft**，其单位是 token/draft，并非 0 到 1 之间的概率。first-position acceptance 与 token acceptance 才是比例指标。

### 1.2 统一评测设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3.5-9B 多模态；Qwen3.5-122B-A10B 多模态 MoE，约 10B active |
| 框架 | vLLM speculative decoding |
| 投机深度 | spec=7 |
| 解码 | greedy，temperature=0 |
| 样本数 | 每个数据集 128 prompts，所有主实验均 128/128 完成 |
| 域内数据 | ALLaVA validation tail，训练集后 10% held-out，避免数据泄漏 |
| 域外数据 | MMStar，用于检查泛化与灾难性遗忘 |
| 训练数据 | verifier 自蒸馏或 Qwen 蒸馏的 ALLaVA 图文数据，规模从 10k 扩展到 100k；122B MTP 使用自身蒸馏 50k |
| 122B 并行 | TP=4，用于训练与评测 |

## 2. 方法概览: DFlash 与 MTP

| 维度 | DFlash draft | MTP head |
|---|---|---|
| 形态 | 独立 draft 模型 / draft 模块 | verifier 原生 multi-token prediction 头 |
| 草稿方式 | 并行预测整块候选 token | 自回归式多 token 预测 |
| 主要优势 | draft 侧计算便宜，吞吐潜力高 | 接受率更高，强基线更稳 |
| 主要瓶颈 | 原版缺少多模态分布，需蒸馏对齐 | 起点高，进一步提升空间较小 |
| 当前定位 | 高吞吐与高并发路径的重点优化对象 | 当前接受率与 122B 加速的首选方案 |

两者不是互斥关系，而是投机解码速度与草稿质量前沿上的两个设计点。DFlash 可以在 draft 成本上占优；MTP 通常在接受长度上占优。实际端到端 QPS 取决于“接受长度”和“draft 代价”共同作用。

## 3. DFlash 提升实验

### 3.1 问题定位

原版 DFlash 在文本场景中可用，但多模态场景下与 verifier 的真实输出分布不完全一致，尤其缺少图文输入下的多模态条件分布。因此，直接使用原版 DFlash 时，平均接受长度明显低于原生 MTP。

我们的核心策略是使用 verifier 或同族模型在 ALLaVA 图文输入上生成蒸馏目标，让 DFlash 学习主模型在多模态上下文中的 greedy 输出分布。由于投机解码在 temperature=0 时接受与否主要由 top-1 一致性决定，训练目标需要直接优化 top-1，而不是只匹配软分布。

### 3.2 9B DFlash: 从 KL 到 CE、从 bf16 到 fp32、从 10k 到 100k

早期实验表明，仅使用 KL loss 与 bf16 优化并不能有效提升 greedy 接受率。关键转折来自两个修正:

1. **KL loss 改为 CE loss**: KL 更关注软分布距离，但 greedy 接受只关心 verifier 的 top-1 token；CE 直接推高正确 token 的概率，更匹配投机解码的验收机制。
2. **纯 bf16 改为 fp32 master + 更高学习率**: DFlash 中较小的 AdamW 更新量在纯 bf16 下会被舍入吞掉，尤其影响已经接近收敛的 first-position。fp32 master 权重保留小更新后，first-position acceptance 首次明显上升。

ALLaVA 域内优化轨迹如下:

| 阶段 | 训练配置 | QPS(token/s) | mean accept/draft(token/draft) | first-position acceptance(ratio) | 结论 |
|---|---|---:|---:|---:|---|
| 原版 DFlash | 未微调 | 63.249 | 1.938 | 0.728 | 多模态接受率偏低 |
| 早期训练 | KL, bf16, LR=1e-5 | 未记录 | 1.615 | 0.672 | 低于原版 |
| 换损失 | CE, bf16, LR=1e-5 | 未记录 | 1.993 | 0.727 | mean accept 小幅超过原版 |
| 修优化 | CE, fp32, LR=3e-5, 10k | 71.437 | 2.369 | 0.775 | 全面超过原版 |
| 扩数据 | CE, fp32, LR=3e-5, 100k | 73.779 | 2.666 | 0.810 | 域内域外同时提升 |

10k 到 100k 的扩数据收益如下:

| 数据集 | 指标 | 10k | 100k | 相对变化 |
|---|---|---:|---:|---:|
| ALLaVA 域内 | mean accept/draft(token/draft) | 2.369 | 2.666 | +12.5% |
| ALLaVA 域内 | first-position acceptance(ratio) | 0.775 | 0.810 | +4.5% |
| ALLaVA 域内 | QPS(token/s) | 71.437 | 73.779 | +3.3% |
| MMStar 域外 | mean accept/draft(token/draft) | 2.162 | 2.416 | +11.7% |
| MMStar 域外 | first-position acceptance(ratio) | 0.758 | 0.801 | +5.7% |
| MMStar 域外 | QPS(token/s) | 70.910 | 76.106 | +7.3% |

扩数据不仅提升了 ALLaVA 域内结果，也显著改善 MMStar 域外结果。这说明 100k 自蒸馏数据没有带来明显过拟合，反而学到了更通用的多模态 draft 能力。

### 3.3 9B DFlash 最终对比

| 数据集 | 方法 | QPS(token/s) | mean accept/draft(token/draft) | token acceptance(ratio) | first-position acceptance(ratio) |
|---|---|---:|---:|---:|---:|
| ALLaVA 域内 | baseline(no spec) | 30.139 | n/a | n/a | n/a |
| ALLaVA 域内 | 原版 DFlash | 61.375 | 1.945 | 0.278 | 0.733 |
| ALLaVA 域内 | 训练后 DFlash | **73.779** | 2.666 | 0.381 | 0.810 |
| ALLaVA 域内 | MTP 参考 | 65.634 | **2.916** | **0.417** | **0.838** |
| MMStar 域外 | 原版 DFlash | 67.904 | 2.102 | 0.300 | 0.766 |
| MMStar 域外 | 训练后 DFlash | **76.106** | 2.416 | 0.345 | 0.801 |
| MMStar 域外 | MTP 参考 | 72.556 | **2.822** | **0.403** | **0.828** |

关键观察:

- ALLaVA 上，训练后 DFlash 相对原版 DFlash: mean accept **+37.0%**，first-position **+10.5%**，QPS **+20.2%**。
- MMStar 上，训练后 DFlash 相对原版 DFlash: mean accept **+15.0%**，first-position **+4.6%**，QPS **+12.1%**。
- 训练后 DFlash 的接受长度仍低于 MTP，但 9B 上端到端 QPS 高于 MTP。这说明在 9B 规模下，DFlash 的低 draft 成本足以抵消部分接受率差距。
- ALLaVA 上相对 no-spec baseline，训练后 DFlash 达到 **2.45x** 输出吞吐。

### 3.4 122B DFlash: 大模型 verifier 上的迁移验证

在 Qwen3.5-122B-A10B 上，DFlash 蒸馏训练同样有效:

| 数据集 | 方法 | QPS(token/s) | mean accept/draft(token/draft) | token acceptance(ratio) | first-position acceptance(ratio) |
|---|---|---:|---:|---:|---:|
| ALLaVA 域内 | 原版 DFlash | 26.645 | 1.753 | 0.250 | 0.691 |
| ALLaVA 域内 | 训练后 DFlash | **30.918** | **2.283** | **0.326** | **0.745** |
| ALLaVA 域内 | MTP 参考 | 34.375 | 3.209 | 0.458 | 0.865 |
| MMStar 域外 | 原版 DFlash | 28.513 | 1.967 | 0.281 | 0.710 |
| MMStar 域外 | 训练后 DFlash | **30.415** | **2.133** | **0.305** | **0.750** |
| MMStar 域外 | MTP 参考 | 33.590 | 3.238 | 0.463 | 0.847 |

相对原版 DFlash:

| 数据集 | QPS 提升 | mean accept 提升 | first-position 提升 |
|---|---:|---:|---:|
| ALLaVA 域内 | +16.0% | +30.2% | +7.8% |
| MMStar 域外 | +6.7% | +8.4% | +5.6% |

122B 上的结论与 9B 一致: DFlash 自蒸馏训练可以稳定提升多模态接受率，并且域外不会崩溃。但不同于 9B，122B 上 MTP 的接受率与 QPS 仍然同时领先 DFlash。一个合理解释是: 122B verifier 的主模型前向代价更高，MTP 的高接受长度在大模型上带来的收益更显著；同时 122B 原生 MTP 头本身更强，DFlash 仍需进一步缩小多模态分布差距。

## 4. MTP 提升实验

### 4.1 方法

MTP 路线不是训练一个独立 draft，而是在 verifier 原生 multi-token prediction head 的基础上继续微调。这样做有三个优点:

1. 起点高: 原生 MTP 头已经学习过主模型的隐状态与后续 token 关系。
2. 部署路径短: serve 时仍使用 `qwen3_5_mtp` 路径，只需将微调头缝合回完整 verifier 权重。
3. 接受率上限高: MTP 与 verifier 共享更强的表示，通常比独立 draft 更接近 verifier 分布。

工程上，vLLM 不能直接挂载裸 MTP 头 checkpoint，因此实现了 stitch 流程，将微调后的 MTP 层缝合回完整 verifier 权重，再进行 speculative serving。MTP 训练使用 bf16；此前验证过 fp32 会破坏 MTP 的 lm_head 行为。

### 4.2 9B MTP: 域内涨、域外轻微回退

9B 原生 MTP 头在 100k Qwen 自蒸馏 ALLaVA 上微调后的结果如下:

| 数据集 | 指标 | 原生 MTP | 微调 MTP | 相对变化 |
|---|---|---:|---:|---:|
| ALLaVA 域内 | mean accept/draft(token/draft) | 2.916 | **3.142** | +7.7% |
| ALLaVA 域内 | QPS(token/s) | 66.4 | **70.4** | +6.1% |
| ALLaVA 域内 | first-position acceptance(ratio) | 0.838 | **0.844** | +0.7% |
| MMStar 域外 | mean accept/draft(token/draft) | 2.822 | 2.694 | -4.6% |
| MMStar 域外 | QPS(token/s) | 69.8 | 65.9 | -5.6% |
| MMStar 域外 | first-position acceptance(ratio) | 0.828 | 0.824 | -0.5% |

该结果呈现典型的域内专化现象: ALLaVA 上收益明确，MMStar 上平均接受长度小幅下降，但 first-position 几乎持平，说明 next-token 层面的能力没有崩溃，回退主要发生在更深位置。

### 4.3 WiSE-FT 权重插值: 9B 两域同时为正

为降低域内微调带来的域外回退，使用 WiSE-FT 风格的权重插值:

```text
theta_soup = (1 - alpha) * theta_native + alpha * theta_finetuned
```

当 `alpha=0.5` 时，9B 上得到如下接受长度:

| 数据集 | 原生 MTP | 纯微调 MTP(alpha=1.0) | 权重插值(alpha=0.5) |
|---|---:|---:|---:|
| ALLaVA mean accept/draft(token/draft) | 2.916 | 3.142(+7.7%) | **3.147(+7.9%)** |
| MMStar mean accept/draft(token/draft) | 2.822 | 2.694(-4.6%) | **2.861(+1.4%)** |

权重插值几乎完整保留域内收益，同时把 MMStar 从负收益修复为正收益。该结果说明，MTP 微调学到的域内增量与原生头的通用能力可以通过参数空间插值兼容。

### 4.4 122B MTP: 步权重 A/B 与权重插值

122B MTP 使用自身蒸馏的 ALLaVA 50k 数据，训练 verifier 原生 MTP 头。由于投机接受是链式过程，早位 token 的价值远高于深位 token: 只有当前面 token 都被接受时，后面的 token 才有机会被 verifier 接受。因此我们对训练中的 step weight 做了 A/B:

- `beta=0.6`: 衰减权重，更偏重早位和中位。
- `beta=1.0`: 等权，对深位相对更友好。

结果如下:

| 指标 | native | beta=0.6 | beta=1.0 |
|---|---:|---:|---:|
| ALLaVA mean accept/draft(token/draft) | 3.066 | **3.419(+11.5%)** | 3.376(+10.1%) |
| ALLaVA QPS(token/s) | 34.3 | **36.82(+7.3%)** | 36.47 |
| ALLaVA first-position acceptance(ratio) | 0.848 | 0.866 | **0.873** |
| MMStar mean accept/draft(token/draft) | 2.930 | **2.862(-2.4%)** | 2.764(-5.7%) |
| MMStar first-position acceptance(ratio) | 0.831 | **0.828** | 0.827 |

虽然 `beta=1.0` 在 ALLaVA first-position 上略高，但整体 mean accept 不如 `beta=0.6`，域外回退也更大。实验推翻了“等权能靠深位收益抬高整体接受长度”的假设。由于深位只有在浅位全部通过时才会生效，训练权重优先分配给早中位更划算。

随后对 122B MTP 做 `alpha=0.5` 权重插值，得到两域同时为正:

| 数据集 | 指标 | native | 权重插值(alpha=0.5) | 相对变化 |
|---|---|---:|---:|---:|
| ALLaVA 域内 | mean accept/draft(token/draft) | 3.008 | **3.314** | +10.2% |
| ALLaVA 域内 | QPS(token/s) | 34.248 | **35.806** | +4.5% |
| ALLaVA 域内 | first-position acceptance(ratio) | 0.848 | **0.864** | +1.9% |
| MMStar 域外 | mean accept/draft(token/draft) | 2.930 | **2.997** | +2.3% |
| MMStar 域外 | QPS(token/s) | 33.405 | **33.798** | +1.2% |
| MMStar 域外 | first-position acceptance(ratio) | 0.831 | **0.839** | +1.0% |

需要注意的是，122B A/B run 与 soup run 是不同评测轮次，ALLaVA native 从 3.066 到 3.008 的差异属于独立 run 的服务波动；百分比均按同轮 native 计算。

### 4.5 Self-forcing 负结果

为缓解中后位 exposure bias，我们实现过 self-forcing/on-policy MTP 训练: 训练时 draft 每一步条件于自己上一步预测 token，而不是 gold token。机制上它确实会把接受分布向深位移动，但 9B 实验没有带来净收益:

| 数据集 | 基线微调头 | self-forcing | 相对变化 |
|---|---:|---:|---:|
| ALLaVA mean accept/draft(token/draft) | 3.142 | 3.070 | -2.3% |
| MMStar mean accept/draft(token/draft) | 2.694 | 2.671 | -0.8% |

原因仍然是链式接受。早位每轮都会生效，深位只有在前缀全部接受后才会生效；如果 self-forcing 损失了早位，即使深位有所提升，也难以补回总平均接受长度。本方向当前 ROI 有限，暂缓。若后续重启，应固定 `beta=0.6`，小比例引入 self-forcing，例如 `sf in {0.1, 0.25}`，并采用从 0 逐步增加的 ramp。

## 5. 横向对比

### 5.1 可直接比较 QPS 的主结果

| 模型 | 方法 | ALLaVA QPS(token/s) | ALLaVA mean accept/draft(token/draft) | MMStar QPS(token/s) | MMStar mean accept/draft(token/draft) |
|---|---|---:|---:|---:|---:|
| 9B | 训练后 DFlash | **73.779** | 2.666 | **76.106** | 2.416 |
| 9B | 原生 MTP 参考 | 65.634 | **2.916** | 72.556 | **2.822** |
| 122B | 训练后 DFlash | 30.918 | 2.283 | 30.415 | 2.133 |
| 122B | MTP 权重插值(alpha=0.5) | **35.806** | **3.314** | **33.798** | **2.997** |

注: 9B MTP 的 `alpha=0.5` 权重插值记录了接受长度(ALLaVA 3.147, MMStar 2.861)，但当时未单独记录 QPS，因此 QPS 横向表中使用原生 MTP 参考。

### 5.2 对 no-spec baseline 的整体加速

| 模型 | 数据集 | 方法 | QPS(token/s) | 相对 no-spec baseline |
|---|---|---|---:|---:|
| 9B | ALLaVA | baseline(no spec) | 30.139 | 1.00x |
| 9B | ALLaVA | 原版 DFlash | 61.375 | 2.04x |
| 9B | ALLaVA | 训练后 DFlash | 73.779 | **2.45x** |
| 9B | ALLaVA | MTP 参考 | 65.634 | 2.18x |
| 122B | ALLaVA | baseline(no spec) | 11.756 | 1.00x |
| 122B | ALLaVA | 原版 DFlash | 26.645 | 2.27x |
| 122B | ALLaVA | 训练后 DFlash | 30.918 | 2.63x |
| 122B | ALLaVA | MTP 参考 | 34.375 | **2.92x** |

### 5.3 主要结论

1. **DFlash 的训练收益更大**。这是因为原版 DFlash 的多模态分布缺口更明显，蒸馏数据直接补齐了图文输入下 verifier 的输出分布。
2. **MTP 的绝对接受率更高**。原生 MTP 头起点高，微调后提升幅度没有 DFlash 大，但最终接受长度仍是当前最高。
3. **吞吐不只由接受率决定**。9B 上训练后 DFlash 接受长度低于 MTP，但 QPS 更高，说明 DFlash draft 成本更低；122B 上 MTP 同时领先 QPS 与接受长度，说明大模型 verifier 下高接受长度带来的收益更强。
4. **权重插值是 MTP 通用服务的关键技巧**。纯域内微调可能导致 OOD 小幅回退，而 `alpha=0.5` 在 9B 与 122B 上均把域外结果修复为正，同时保留大部分域内收益。
5. **早位权重比深位权重更重要**。`beta=0.6` 优于 `beta=1.0`，self-forcing 也因损失早位而没有净收益。这与链式接受机制一致。

## 6. 工程沉淀

本轮实验不只是离线训练，还打通了 vLLM 下的训练、缝合、serve 与评测闭环。关键工程产出包括:

| 工程项 | 解决的问题 |
|---|---|
| MTP stitch 流程 | 将微调后的 MTP 头写回完整 verifier 权重，使 vLLM 可以直接 serve |
| 122B TP 训练路径 | 122B bf16 权重无法数据并行整模复制，改为 verifier TP=4 与 trainer 分配 |
| MoE 配置 fallback | 兼容 122B MoE 的 `moe_intermediate_size`，避免构建层配置失败 |
| 蒸馏并发与跨机分片 | 将蒸馏从串行改为并发生成，并支持 SKIP/MAX 分片 |
| 验证尾自动切分 | ALLaVA 完整集自动切后 10% 作为 held-out validation，避免泄漏 |
| A/B 评测脚本 | 支持跨机 native 基线一致性检查与逐臂评测 |
| serve 稳定化 | 固化 vLLM health check、media path、`no_proxy`、进程清理等部署细节 |

这些工程改动使得后续优化可以按“改训练策略、内网 GPU 验证、回传 log、汇总结论”的方式快速迭代。

## 7. 局限性

1. **样本数为 128 prompts**。接受率指标在多次 run 中较稳定，但 QPS 仍存在单次服务波动，通常需要按百分之几的噪声理解。
2. **部分 QPS 未记录**。例如 9B MTP `alpha=0.5` 权重插值记录了接受长度，但未独立记录 QPS，因此横向 QPS 表使用原生 MTP 参考。
3. **离线评测不完全等价于在线高并发服务**。DFlash 在高并发下可能更有优势，但需要真实并发 workload 复核。
4. **DFlash 122B 仍落后 MTP**。虽然相对原版 DFlash 明显提升，但在 122B 上接受率与 QPS 均尚未超过 MTP。
5. **D-Cut、Domino-style 训练与 causal DFlash 尚未纳入主结果**。这些方向已经有工程探索或初步实现，但仍需统一评测后再进入正式结论。

## 8. 下一步工作

### 8.1 MTP

1. **122B 正式采用 `beta=0.6` 作为训练权重策略**。域内定向场景可优先部署纯微调头，获得 ALLaVA 平均接受长度 +11.5% 与 QPS +7% 到 +10%。
2. **通用服务优先使用 `alpha=0.5` 权重插值**。该配置在 122B 上 ALLaVA +10.2%、MMStar +2.3%，在 9B 上也验证了两域同时为正。
3. **继续扫 `alpha` 与 serve-time spec**。建议 `alpha in {0.3, 0.4, 0.5, 0.6, 0.7, 1.0}`，spec in `{3, 5, 7}`，寻找不同负载下的最优点。
4. **若重启 self-forcing，需要做干净消融**。固定 `beta=0.6`，小比例 self-forcing，先保住 early token 再尝试提升深位。

### 8.2 DFlash

1. **按真实接受率做 checkpoint sweep**。当前主要按 validation loss 选点，后续可按 per-epoch acceptance 直接选最优 checkpoint。
2. **对 first-position 加权**。first-position 是链式接受入口，提升它对整体 mean accept 的杠杆最大。
3. **在在线高并发服务中复核吞吐**。9B 离线结果显示 DFlash QPS 最高，需要在真实并发 workload 中确认优势是否扩大。
4. **继续研究 Domino-style 修正头与 causal DFlash**。这些方向与 D-Cut、MTP 微调基本正交，可作为进一步提升 draft 质量或降低冗余计算的候选路径，但目前不计入已验证主结论。

### 8.3 D-Cut 与高并发 verify 优化

在高并发场景下，draft 侧一旦足够便宜，投机推理的瓶颈会转向 verifier 校验。D-Cut 的目标不是让 draft 猜得更多，而是减少每轮需要 verifier 校验的 token 数。

一个直观例子是: 当前 spec=7，如果策略判断平均只需要校验 4 个 token，则每轮 verify token 数从 7 降到 4，约减少 **43%**。对于 MoE 或 NPU 场景，这类 verify token 削减可能比继续压低 draft latency 更重要。当前 GPU 路径已经跑通，下一步是迁移到 NPU 并与 DFlash 高并发服务结合评测。

## 9. 可复现来源

本文数据整理自当前分支下的以下报告与 PPT:

- `reports/spec_decoding_customer_report_huawei.html`
- `reports/spec_decoding_customer_report_2026-06-22.md`
- `reports/dflash_100k_three_way_report.md`
- `reports/dflash_ce_finetune_report.md`
- `reports/mtp_finetune_report.md`
- `reports/mtp_dflash_122b_report.md`

主要原始输出目录:

- 9B DFlash 100k: `output/three_way_both/20260615_020313/`
- 9B MTP: `output/mtp_orig_vs_trained/20260613_020839/` 与 `output/mtp_mmstar_orig_vs_trained/20260613_023232/`
- 122B DFlash: `output/three_way_both/20260617_025446/`
- 122B MTP: `mtp_accept/results/20260617_024459/`

推荐复核命令口径:

```bash
INFER_NUM_SPEC=7 MTP_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_three_way_mmstar_allava.sh
```

MTP 路线需先完成 head stitch，再以同一 verifier、同一 spec、同一 validation tail 对 native 与 trained/soup 进行 A/B 评测。
