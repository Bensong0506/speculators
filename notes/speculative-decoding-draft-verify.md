# 投机推理:draft + verify 流程详解

> 以 Qwen3.5 **9B(draft)+ 122B(verify)** 两模型两服务为例,讲清楚 speculative
> decoding 的一轮到底怎么跑;并说明它和本仓库训练的耦合式 draft(DFlash / EAGLE / MTP)、
> 以及 SPECTRE 远程投机的关系。
>
> 这是一份概念笔记(为什么 / 怎么跑),不是 API 文档。

## TL;DR

- **投机推理** = 让便宜的小模型(draft)猜 γ 个 token,贵的大模型(verify)**一次前向并行核对**,
  通过的直接采纳 → 大模型每跑一次提交多个 token,从而加速。**无损**(输出分布严格等于大模型自己)。
- **两种 draft**:独立小模型(token 进 / 出,可远程)vs 耦合头(EAGLE / DFlash / MTP,吃 verifier
  hidden state,必须同驻)。本仓库训的是后者。
- **单模型部署** → 用耦合 draft(便宜、看得见图、接受率高)。**多模型集群、有闲置小模型** → 才考虑
  独立小模型远程当 draft(SPECTRE)。

---

## 1. 两种 draft 范式(先分清,别混)

|        | 独立 draft(standalone)  | 耦合 draft(coupled head)    |
| ------ | ----------------------- | --------------------------- |
| 例子   | 一个完整小模型(Qwen3.5-9B) | EAGLE / DFlash / MTP        |
| 输入   | 只要 token id            | verifier 的 hidden state     |
| 部署   | 可拆成**远程两服务**       | **必须和 verifier 同驻**      |
| 本仓库 | —                       | ✅ 训的就是这种               |

本文主讲**独立 draft 的两服务流程**(也就是 SPECTRE / 经典 Leviathan'23 那套),因为它最能把
"draft 和 verify 各干什么"摊开看清楚。耦合 draft 的核对逻辑完全一样,区别只在 draft 怎么产生
(它多吃一份 verifier 的 hidden state,且和 verify 同卡)。

---

## 2. 架构:两个模型,两个服务

```
   ┌────────────────────┐                          ┌────────────────────┐
   │  DRAFT 服务         │ ──── d₁ d₂ d₃ d₄ ────►   │  VERIFY 服务        │
   │  Qwen3.5-9B        │      (4 个候选 token id)   │  Qwen3.5-122B-MoE  │
   │  1×GPU,便宜         │                          │  8×GPU,贵           │
   │  自回归吐 γ 个 token │ ◄──── L + bonus ─────    │  一次前向验证 γ 个    │
   └────────────────────┘   (接受几个 + 修正 token)   └────────────────────┘
             ▲                                                  ▲
             └───────────────── 同一张图喂给两边 ────────────────┘
                              (两个都原生多模态)
```

**跨网络只传 token id + 接受长度 L + bonus token**,全是几十字节的整数。
→ 不传 hidden state——这正是"独立 draft"能拆成远程两服务的原因,也是耦合式 DFlash 拆不了的原因
(它要 hidden state)。

---

## 3. 一轮(round)到底发生了什么 —— 用具体 token 走一遍

设图里是只橘猫,已提交前缀 `…这 只 猫 是`,γ=4。

**Step 1 — 9B 自回归吐 4 个草稿**(它原生多模态、看得见图,所以敢猜"橘色"):

```
d₁=橘   d₂=色   d₃=的   d₄=。
```

**Step 2 — 把这 4 个 token 发给 122B,122B 一次前向并行验证。**
把 `前缀 + [d₁ d₂ d₃ d₄]` 当一个序列喂进去,因果注意力让**每个位置都同时吐出"122B 自己认为的
下一个 token"**——一次前向就拿到全部 4 个位置的"标准答案":

```
喂给 122B:   …这 只 猫 是 │ 橘    色    的    。
                         │ d₁    d₂    d₃    d₄
122B 在每个位置的标准答案:
   "是" 之后 → 橘    ⟺ d₁=橘   ✓
   d₁   之后 → 色    ⟺ d₂=色   ✓
   d₂   之后 → 的    ⟺ d₃=的   ✓
   d₃   之后 → ，    ⟺ d₄=。   ✗   ← 第一个不一致,到此为止
```

**Step 3 — 接受最长一致前缀 + 用 122B 自己的 token 当修正(bonus)。**

```
接受 橘 色 的(前 3 个一致),d₄=。被拒 → 用 122B 的「，」顶上
本轮提交:  橘 色 的 ，   = 4 个 token,只花了 1 次 122B 前向
新前缀:    …这只猫是橘色的，   → 下一轮 9B 从「，」继续
```

**三种结局**(决定一次 122B 前向赚多少):

| 结局                  | 提交 token 数                       |
| --------------------- | ---------------------------------- |
| 全中(d₁…d₄ 全一致)   | γ+1 = 5(d₄ 之后再白送一个 bonus)   |
| 部分中(本例)         | 接受数 + 1                          |
| 全错(d₁ 就不对)      | 1(退化成普通自回归那一步,9B 草稿白算,但 122B 没多花) |

→ 一次 122B 前向提交 **1 ~ γ+1** 个 token,平均 = **接受长度 L**。verify 永远至少吐 1 个自己的
token,保证前进 + 无损(temp>0 时把"==判断"换成 **rejection sampling**,数学上严格等于 122B
自己的分布)。

---

## 4. 两种编排:Ordinary(串行) vs Parallel(重叠)

**Ordinary —— 谁干活,另一个闲着:**

```
9B :  ██draft██            ········            ██draft██
              ╲ 发 4 个 token                       ╲
122B: ········  ██verify██            ········  ██verify██
                       ╱ 回 L + bonus
      串行,有气泡;但每次 verify 提交 L 个 token
time ───────────────────────────────────────────────►
```

**Parallel —— 9B 不等结果,领先一段一路抢跑:**

```
9B :  ██d₁██ ██d₂██ ██d₃██ ██d₄██   ← 基于自己的续写往前抢跑
122B:        ██v₁██ ██v₂██ ██v₃██   ← 同时不停验证
                  ↑ 若 v₁ 拒了几个,基于旧前缀抢跑的 d₂ 作废 → rollback 重做
      draft 与 verify 时间重叠 → 更快,但有作废(rollback)风险
```

**两个容易混的"串 / 并行"是不同的轴:**

- **轴 A(本节,跨轮协同)**:draft 这步和 verify 这步在时间上要不要重叠 → 这就是 ordinary / parallel。
- **轴 B(draft 内部出 token)**:小模型自己吐 γ 个 token 是自回归串行(小模型 / MTP)还是
  一次前向并行(DFlash 块式)。

> 两者**正交**。SPECTRE 切的是轴 A,和 draft 内部结构无关。

SPECTRE 的 **hybrid** 就是按实测 rollback 比例 `r̂` 和阈值 `r*` 在轴 A 上**逐轮切**:作废少用
parallel(吃重叠红利),作废多退回 ordinary(别白抢跑)。

```
r* = (γ−1)·L·T_D / [ (L−1)·(T_T + (γ−1)·T_D) ]
mode = Parallel  if  r̂ ≤ r* ;   Ordinary  if  r̂ > r*
```

---

## 5. 为什么这样就快

> 贵的 122B **每轮只前向一次,却提交 L 个 token**(而不是每个 token 都要它跑一次);代价是便宜的
> 9B 多算几步草稿 + 偶尔白算。

```
加速 ≈ L / (1 + (γ−1)·T_D/T_T)        # ordinary
加速 ≈ L / max(1, γ·T_D/T_T)          # parallel(重叠,忽略 rollback)
```

- `L` = 接受长度,越高越好 → 取决于 draft 和 target 的分布一致程度。
- `T_D / T_T` = draft 单步 / verify 单轮 的延迟比,越小越好 → draft 要足够便宜。
- **MoE 坑**:122B 是 MoE,单 token verify 成本 ∝ **激活参数**(不是总参)。若激活 ~20–30B,则
  9B 只比它便宜 ~3×,`T_D/T_T ≈ 0.3–0.4`,收益被压。验证 γ 个 token 还会路由到更多专家,
  verify 不像 dense 那样"几乎免费"。

---

## 6. 什么时候用哪种 draft(决策)

| 场景                          | 选择                            | 为什么                                              |
| ----------------------------- | ------------------------------- | --------------------------------------------------- |
| **单模型专用部署**(压单流延迟) | **耦合 draft(DFlash / MTP)**   | 1 层就够、几乎不占算力、靠 hidden state 看得见图、接受率高 |
| **多模型集群,小模型常闲置**    | **独立小模型远程当 draft(SPECTRE)** | 复用闲置 GPU = 边际成本≈0;大模型那 8 张卡只干验证      |

**两条独立的"便宜":**

- 耦合 draft 便宜 = **FLOPs 少**,但花在最贵、最抢手的 verifier GPU 上。
- 闲置独立 draft 便宜 = **边际成本≈0**,因为那张卡本来就闲着,且活儿在 verifier 关键路径之外。

> SPECTRE 的指标是 `$/GPU/1000s` 而不是 acceptance,正因为它比的是"让哪块硬件干这点活最划算",
> 不是"谁是更好的 draft"。

**多模态特别注意**:独立的小 **text** draft 看不见图 → 图相关 token 接受率崩。要么用**同源多模态**
小模型当 draft,要么用耦合 draft(它从 VLM verifier 的 hidden state 免费继承图理解——这正是本仓库
DFlash 对多模态友好的原因)。

---

## 7. 三种 draft 在两根轴上的对照

| draft              | 块内出 token   | 跨块依赖 verifier hidden | 能否远程独立服务 | parallel 重叠收益 | 适配 SPECTRE |
| ------------------ | ------------- | ----------------------- | -------------- | --------------- | ------------ |
| 独立小模型(0.6B/9B) | 自回归(串行)  | 否                       | ✅             | 大(有延迟可藏)   | ✅ 量身定制    |
| MTP 头             | 链式(串行)    | 是                       | ❌             | 被耦合依赖阻断    | ❌           |
| DFlash             | 一次前向(并行) | 是                       | ❌             | ≈0(没延迟可藏)   | ❌           |

---

## 参考

- SPECTRE: *Parallel Speculative Decoding with a Multi-Tenant Remote Drafter*, arXiv:2605.08151
- 经典投机解码: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML'23
- EAGLE-3 / DFlash 算法说明见 `docs/user_guide/algorithms/`
