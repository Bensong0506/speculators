# 投机解码微调总览 — 报告合集 + MTP vs DFlash 分析 (2026-06-15)

本文是这条线全部评测报告的**合集与索引**,并回答一个核心问题:
**为什么 MTP 头微调的收益,看起来不如 DFlash draft 微调?**

模型统一为 Qwen3.5-9B 多模态 verifier;域内 = ALLaVA(蒸馏自蒸的 held-out 尾巴),
域外 = MMStar(从未训练);投机步数统一 `@spec=7`,n=128。

---

## 一、报告索引(合集)

按时间线,两条平行的微调线:DFlash draft(开源 draft 适配本 VLM)与 MTP 头(模型自带头适配目标域)。

| # | 报告 | 日期 | 范围 | 一句话结论 | 位置 |
|---|---|---|---|---|---|
| 1 | `mmstar_dflash_spec_sweep_analysis.md` | 06-09 | DFlash·spec 长度 sweep | KL 期:trained DFlash 在 spec7/spec3 都**输给** native | `test_result` |
| 2 | `mmstar_10k_checkpoint_sweep_summary.md` | 06-09 | DFlash·10k ckpt sweep | KL 期:**没有任何 checkpoint** 超过 native(best 0.869×) | `test_result` |
| 3 | `allava_val_four_way_summary.md` | 06-10 | DFlash·域内四路 | **转折点**:换 CE + 正确 jsonl + checkpoint_best,域内首次**小幅反超** native | `test_result` |
| 4 | `dflash_ce_finetune_report.md` | 06-11 | DFlash·10k 微调 | **关键突破**:bf16→fp32 解锁 first-pos,域内全面反超(mean +22%) | `test_result` |
| 5 | `dflash_100k_three_way_report.md` | 06-15 | DFlash·100k 三路 | 扩到 100k:域内 **+37%**、OOD 由持平转 **+15%**、两数据集 **tok/s 最快** | `test_result` |
| 6 | `mtp_finetune_report.md` | 06-13 | MTP·100k 微调 | 域内 **+7.7%**、OOD 小幅回退 **−4.6%**(bf16 训练) | `mtp-training` |
| 7 | `speculative_decoding_overview.md`(本文) | 06-15 | 合集 + 分析 | MTP vs DFlash 全景对比与归因 | `test_result` |

---

## 二、全景数据(所有方法同台)

> ✅ 标注「同 run」的行来自同一次基准,可直接比;其余为跨 run 补入(tok/s 有 ±数% run 间噪声,接受率跨 run 稳定)。

### 域内 ALLaVA(越高越好)

| 方法 | mean accept/draft | first-pos | tok/s | 来源 |
|---|---:|---:|---:|---|
| **微调 MTP** | **3.142** 🏆 | **0.844** 🏆 | 70.4 | 报告6 |
| 原生 MTP | 2.916 | 0.838 | 65.6 | 报告5(同 run) |
| **微调 DFlash · 100k** | 2.666 | 0.810 | **73.779** 🏆 | 报告5(同 run) |
| 微调 DFlash · 10k | 2.369 | 0.775 | 71.437 | 报告4 |
| 原版 DFlash | 1.945 | 0.733 | 61.375 | 报告5(同 run) |
| baseline(无投机) | n/a | n/a | 30.139 | 报告5(同 run) |

### 域外 MMStar(泛化 / 遗忘检查)

| 方法 | mean accept/draft | first-pos | tok/s | 来源 |
|---|---:|---:|---:|---|
| 原生 MTP | **2.822** 🏆 | **0.828** 🏆 | 72.556 | 报告5(同 run) |
| 微调 MTP | 2.694 | 0.824 | 65.9 | 报告6 |
| **微调 DFlash · 100k** | 2.416 | 0.801 | **76.106** 🏆 | 报告5(同 run) |
| 微调 DFlash · 10k | 2.162 | 0.758 | 70.910 | 报告4 |
| 原版 DFlash | 2.102 | 0.766 | 67.904 | 报告5(同 run) |

**先看清两个事实(否则容易得出错误结论):**
- **接受率上 MTP 是最强的**(域内 3.142 > DFlash 2.666;域外原生 MTP 2.822 也最高)。MTP 不是"更差"。
- **吞吐(tok/s)上微调 DFlash 最快**(域内 73.8 > 微调 MTP 70.4 > 原生 MTP 65.6)。因为 **DFlash draft 每步更便宜**——即使接受的 token 少,墙钟产出仍最高。这正是用户端真正感受到的指标。

> 所以一句话:**MTP = 接受率天花板;DFlash = 吞吐天花板。** 谁"更好"取决于看哪个指标。

---

## 三、训练收益对比(这才是"MTP 不如 DFlash"的真正所指)

把"微调相对各自起点提升了多少"并排看:

| 维度 | DFlash(原版 → 100k 微调) | MTP(原生 → 微调) |
|---|---:|---:|
| 域内 mean-accept | **+37.0%** | +7.7% |
| 域内 first-pos | **+10.5%** | +0.7% |
| 域内 tok/s | **+20.2%** | +6.1% |
| 域外 mean-accept | **+15.0%**(反超) | **−4.6%**(回退) |
| 域外 first-pos | +4.6%(反超) | −0.5%(持平) |

**DFlash 微调是一次大且干净的胜利(域内域外双涨);MTP 微调是小幅域内涨 + 域外回退。** 这就是直观上"MTP 训练结果不如 DFlash"的来源。下面分析为什么。

---

## 四、核心分析:为什么 MTP 微调收益远小于 DFlash?

### ① 优化精度:MTP 用了 bf16,而 DFlash 最大的突破恰恰是 bf16→fp32 ⭐(最可操作)

- MTP 微调:**bf16** + LR 3e-5(报告6 设置表)。
- DFlash 报告4 明确写道:first-pos 长期卡死的**根因是纯 bf16 把小于精度的权重更新舍掉了**;切到 **fp32** + 提 LR 后,first-pos **第一次明显上升**(0.727→0.775),这是 DFlash 反超的关键一步。
- **症状吻合**:MTP 的 first-pos 几乎没动(0.838→0.844,**+0.7%**)——正是"bf16 卡 first-pos"的典型签名(first-pos 已接近收敛,单步 AdamW 更新 ~1e-5 远小于权重 bf16 的 ULP ~1e-3,被舍入归零)。
- **结论**:MTP 大概率**还有没吃到的收益**——把 DFlash 学到的 fp32 经验搬过来重训,first-pos / mean-accept 很可能继续抬。这是第一优先级建议。

### ② 起点高度:MTP 起步就接近天花板,DFlash 起步远低于

- 原生 MTP mean-accept **2.916**:它是模型**自带、与模型联合训练**过的头,本就很强。
- 原版 DFlash mean-accept **1.945**:一个**通用开源 draft,完全没适配**这个 VLM,起点低一大截。
- DFlash 有约 50% 的绝对上涨空间;在弱起点上拿大百分比,比在强起点上再抠容易得多(边际递减)。2.916→3.142 本就比 1.945→2.666 难。

### ③ 特化代价:同样 100k ALLaVA,MTP 伤了 OOD,DFlash 反而帮了 OOD

- MTP 已经是通用的;只喂 ALLaVA 蒸馏数据 → 把它的分布**往 ALLaVA 拉偏**,域外随之回退(−4.6%)。典型的特化/遗忘,且**没掺通用数据**兜底。
- DFlash 起点是"没适配的通用 draft",学 verifier 的真实续写分布 → 学到的是**更通用的 VLM-draft 能力**,连 OOD 一起涨(+15%)。同样的数据,因起点不同得到相反的 OOD 效果。

### ④ 容量:MTP 头很小,DFlash draft 更大

- MTP 头是轻量的单头(从 hidden state 预测后续 token),可调容量有限。
- DFlash 是更实在的 draft 模块,吸收新分布的容量更大。
- (此条为结构性推断,证据强度弱于 ①②;但与"小头小收益"一致。)

### ⑤ 不是数据量的锅

两条线**都用了 100k 蒸馏 ALLaVA**。DFlash 从 10k→100k 的大涨(报告5)发生在 DFlash 内部;MTP 也已是 100k。所以差异**不来自数据规模**,而来自上面 ①–④。

---

## 五、方法学缺口:两个最强模型还没真正同台 ⚠️

当前所有"MTP vs DFlash"的对比其实是**错配的**:

- 报告5 的三路里,MTP 用的是**原生 MTP**(2.916),不是微调 MTP(3.142)。
- 报告6 里,微调 MTP 只和**原生 MTP** 比,没和微调 DFlash 比。

**= 微调 DFlash(2.666 / 73.8)和微调 MTP(3.142 / 70.4)从未在同一次 run 里同台。** 跨 run 的 tok/s 差(73.8 vs 70.4 = 1.05×)落在噪声带内,严格说现在只能讲"接受率 MTP 明显高、吞吐两者接近、DFlash 略快"。要给"到底谁强"下定论,**必须跑一次包含两者 + 双 baseline 的四路同 run**(两份报告的"下一步"都列了这条,但还没跑)。

---

## 六、结论与下一步

**结论**:
- "MTP 不如 DFlash"准确说法是:**MTP 这次微调的收益(+7.7% 且伤 OOD)远小于 DFlash 微调(+37% 且帮 OOD),且端到端吞吐被 DFlash 反超**——但 **MTP 仍是接受率最强的方法**,潜力没被这次 bf16 训练榨干。
- DFlash 的成功很大程度来自两条已被验证的经验:**CE 损失** + **fp32 优化**;前者 MTP 用了(它本就 CE 向),**后者 MTP 没用**。

**下一步(优先级从高到低):**
1. **MTP 改 fp32 重训** ⭐:直接套用 DFlash 验证过的 fp32 + 提 LR,目标解锁卡住的 first-pos。预期 MTP 接受率进一步抬,可能把吞吐也带到与 DFlash 持平/反超。
2. **跑一次四路同 run**:微调 DFlash + 微调 MTP + 原生 MTP + 原版 DFlash + baseline,在同一基准上给出公平裁决(补上第五节的缺口)。
3. **MTP 掺通用数据**:训练集混入少量通用样本,收窄 −4.6% 的 OOD 回退(若要做通用 serve 这是必需)。
4. **pos-0 加权 CE**:两条线的共同短板都在更深位置 / first-pos;对 gate 位置加权,继续抬平均接受长度。

---

## 七、附:关键术语与读数口径

- **mean accept/draft(平均接受长度)**:每次 draft 被 verifier 接受的 token 数,衡量"draft 猜得准不准"。接受率类指标 run 间稳定,是稳健结论的依据。
- **tok/s(吞吐)**:端到端每秒输出 token,用户实际感受。受"接受率 × draft 每步成本 × verify 成本"共同决定——**DFlash 接受率低却更快,因为它 draft 每步更便宜**。单流离线基准有 ±数% run 间噪声。
- **first-pos(首位接受率)**:draft 第一个 token 的接受率,最接近收敛、最难再抬;bf16 训练会把它卡死,fp32 才能撬动。
- **域内/域外**:域内 = ALLaVA held-out(测泛化非记忆);域外 = MMStar(从未训练,查遗忘)。
