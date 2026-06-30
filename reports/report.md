# DeepSpec / DSpark 投机解码汇报: 从 DFlash/MTP 到聪明验证

**日期**: 2026-06-30  
**范围**: DeepSeek-AI DeepSpec 仓库、DSpark 论文与开源实现；对照我们已有的 Qwen3.5 多模态 MTP / DFlash / D-Cut 工作  
**目标**: 给内部论坛/哈桑汇报使用，讲清楚 DeepSeek 新开源的投机推理仓库做了什么、为什么有效、和我们现有路线怎么合入

## 导读: 这不是又一个 draft 模型，而是下一阶段 serving 方案

上一篇内部报告《Qwen3.5 多模态投机解码接受率提升: 从 DFlash 到 MTP》主要回答的是一个问题: **draft 怎么猜得更准**。我们用多模态自蒸馏把 DFlash 的接受率做上去，也验证了 MTP 微调 + 权重汤能继续提升 verifier 原生 MTP 头。

DeepSeek 这次开源的 DeepSpec / DSpark 接的是下一个问题: **当 draft 已经能便宜地产生较长候选块后，系统到底应该校验多少 token**。

一句话概括:

> **DSpark = DFlash 的并行骨架 + MTP 的局部依赖 + D-Cut 的聪明验证。**

它不是推翻 DFlash，而是在 DFlash 最有价值的地方继续加两块:

1. **Markov Head / Semi-Autoregressive**: 保留 DFlash 一次 forward 出整块的并行能力，只在 logits 侧加一个很轻的串行修正，让后一个 token 能看到前一个 token，缓解 "of problem" 这类 mode collision。
2. **Confidence-Scheduled Verification**: 给每个 draft 位置预测“如果前缀已被接受，这个位置继续活下来的概率”，再结合硬件吞吐曲线决定每条请求应该验几个 token，高并发时自动剪掉低价值尾巴。

这和我们正在做的 D-Cut 很近: **D-Cut 是不用训练的 serving 侧 MVP；DSpark 是训练版、校准版、硬件感知版的最终形态。**

## 0. 最终结论先看

### 0.1 DSpark 的核心贡献

| 问题 | DFlash / MTP 里的现象 | DSpark 的解法 |
|---|---|---|
| DFlash 后缀容易崩 | U1、U2、U3 并行出来，U2 不知道 x1 实际采成什么 | Markov head 给 U2 加 B(x1)，用极小串行成本补局部依赖 |
| MTP 接受率高但草稿更串行 | 自回归依赖强，但 draft 成本随步数涨 | 只把依赖放在轻量 logits 修正里，重 backbone 仍并行 |
| 固定 spec=7 会浪费 verify | 高并发时，低置信尾 token 占用 target batch capacity | Confidence head 预测 prefix survival，scheduler 只验正收益前缀 |
| 静态阈值不适合生产 | 轻载多验没事，重载多验会挤占别的请求 | 用 SPS(B) 硬件曲线最大化系统吞吐 `Theta = expected_accepts * SPS(B)` |

### 0.2 论文主结果

离线 benchmark 上，DSpark 相对两类强基线都有稳定提升:

| Target | 相对 Eagle3 accepted length | 相对 DFlash accepted length |
|---|---:|---:|
| Qwen3-4B | +30.9% | +16.3% |
| Qwen3-8B | +26.7% | +18.4% |
| Qwen3-14B | +30.0% | +18.3% |

线上生产场景里，DeepSeek 把 DSpark-5 部署到 DeepSeek-V4-Flash / V4-Pro preview。相对之前的 MTP-1 baseline:

| 场景 | 结论 |
|---|---|
| V4-Flash 中等 SLA | aggregate throughput +51% |
| V4-Pro 中等 SLA | aggregate throughput +52% |
| matched throughput | V4-Flash 单用户生成速度 +60% 到 +85% |
| matched throughput | V4-Pro 单用户生成速度 +57% 到 +78% |

这里最重要的不是单点数字，而是 DSpark 把 serving Pareto frontier 往外推了: 在相同吞吐下用户更快，在严格交互 SLA 下系统不再因为固定多 token verify 而崩掉。

### 0.3 对我们当前工作的判断

1. **DeepSpec 目前有现成 DSpark/DFlash/Eagle3 权重，但不是 Qwen3.5 多模态系列**。公开 checkpoint 覆盖 Qwen3-4B/8B/14B 和 Gemma4-12B-it；没有直接可用的 Qwen3.5 多模态、Qwen3.5-122B-A10B 权重。
2. **完整 DSpark 必须训练**。Markov head 和 confidence head 都是新增参数，不能直接从我们现有 DFlash checkpoint 推出来。
3. **静态 D-Cut 可以不训练先落地**。这仍然是我们近期最现实的路径: 不动主模型、不重训 draft，先把 variable verify length、metadata、NPU/vLLM runner 跑通。
4. **后续合入 DSpark 的关键不是“多一个 head”这么简单**。必须保证 rejection sampling 用的是 Markov 修正后的 draft probability；scheduler 不能看未来 token；confidence 需要校准；runtime 要能处理每条请求不同 verify 长度。

## 1. 背景: 为什么 DFlash/MTP 之后还需要 DSpark

投机解码的吞吐可以粗略看成:

```text
per-token latency ~= (draft cost + verify cost) / accepted tokens
```

之前我们主要在分子左半边和分母上努力:

- DFlash: draft 很便宜，一次并行提出多个 token。
- MTP: 接受长度高，尤其 122B 这种 verifier 昂贵的场景收益明显。
- 多模态自蒸馏: 让 draft 更贴近 target 的真实输出分布，提升 accepted tokens。

但在高并发系统里还有一个隐藏成本: **verify token 本身占用 target batch capacity**。

例如 spec=7 时，如果后 3 个 token 大概率会被拒绝，仍然把它们送进 target verify，就会浪费 target 的 batch 槽位。轻载时这个浪费不明显，重载时它会直接降低系统可服务的并发数。

所以 DSpark 同时解决两件事:

1. **draft better**: 用 Markov / RNN 轻串行头补 DFlash 的局部依赖，提高后缀接受率。
2. **verify smarter**: 用 confidence + scheduler 控制 verify 长度，不再盲验整块。

## 2. DeepSpec 仓库是什么

DeepSpec 是 DeepSeek 开源的投机解码训练/评测仓库，包含:

- data preparation: 下载 prompt、重新生成 target answer、准备 target cache。
- training: 训练 draft model。
- evaluation: 在 GSM8K、MATH500、AIME25、HumanEval、MBPP、LiveCodeBench、MT-Bench、Alpaca、Arena-Hard-V2 等任务上测 speculative decoding acceptance。
- algorithms: Eagle3、DFlash、DSpark。

公开配置覆盖:

| Algorithm | Qwen3-4B | Qwen3-8B | Qwen3-14B | Gemma4-12B-it |
|---|---|---|---|---|
| Eagle3 | 有 | 有 | 有 | 有 |
| DFlash | 有 | 有 | 有 | 有 |
| DSpark | 有 | 有 | 有 | 有 |

需要注意两个工程现实:

1. **默认训练假设单节点 8 GPU**。`scripts/train/train.sh` 会让 `train.py` 自己按可见 GPU spawn workers。
2. **target cache 很大**。README 里写默认 `Qwen/Qwen3-4B` 设置下 target cache 约 38 TB。也就是说 full training 不是“git clone 后随手跑一下”的量级。

DeepSpec 的 DSpark 配置和 DFlash 配置对比非常直接:

| 配置项 | DFlash | DSpark |
|---|---|---|
| block_size | 7 | 7 |
| draft layers | 5 | 5 |
| target layers | `[1, 9, 17, 25, 33]` | `[1, 9, 17, 25, 33]` |
| Markov head | 关闭，`markov_rank=0` | 开启，`markov_rank=256` |
| confidence head | 关闭，`confidence_head_alpha=0.0` | 开启，`confidence_head_alpha=1.0` |
| loss | CE-only | CE + TV/L1 distribution matching + confidence loss |

所以从实现角度看，DeepSpec 的 DFlash 是 DSpark 的一个退化版本: 关掉 Markov 和 confidence，它就回到纯并行 drafter。

## 3. DSpark 算法拆解

### 3.1 Anchor 是什么

在 speculative decoding 的每轮循环里，target model 上一轮会生成一个 token，论文里把这个 token 叫 **anchor token**，也可以理解成上一轮的 bonus token。

这个 anchor 是下一轮 draft 的起点:

```text
上下文:  A B C
target 先生成: D
D 成为 anchor
draft 基于 D 一次提出: E F G H
target 再并行 verify: D E F G H
```

DFlash 的关键是: anchor + mask tokens 一起进 draft backbone，一次 forward 产出后面多个位置的 logits。它的重计算是并行的，成本基本不随 block size 线性增长。

DSpark 保留这个优势。论文里还做了一个小改动: 不再用 `anchor + gamma 个 mask` 只预测 mask，而是把 anchor 本身也作为第一个 prediction position，让 `anchor + gamma-1 个 mask` 产出 gamma 个 draft logits。这个改动主要是省一点 draft compute，算法本质仍然是并行 backbone。

### 3.2 Markov Head 做了什么

DFlash 的问题是每个 draft 位置“各想各的”。parallel backbone 会一次产出:

```text
U1, U2, U3, ...
```

这里 `Uk` 是第 k 个 draft 位置上的 base logits，也就是一个 vocab 维度的打分向量。纯 DFlash 直接做:

```text
x1 ~ softmax(U1)
x2 ~ softmax(U2)
x3 ~ softmax(U3)
```

问题在于: `U2` 在计算时并不知道 `x1` 最后采成了什么。

例如上下文里同时存在两个高概率模式:

```text
of course
no problem
```

并行预测时，位置 1 可能从自己的边缘分布里采到 `of`，位置 2 可能从自己的边缘分布里采到 `problem`，于是组合成:

```text
of problem
```

这就是论文里说的 multi-modal collision。单看每个位置都像合理 token，但连起来不合理，target verify 很容易在后缀拒绝。

DSpark 的 Markov head 只做一个很轻的修正:

```text
DFlash: x2 ~ softmax(U2)
DSpark: x2 ~ softmax(U2 + B(x1))
```

`B(x1)` 是一个由前一个 token 决定的 transition bias。默认实现里它是低秩分解:

```text
B(prev, .) = W1[prev] @ W2
rank = 256
```

所以当 `x1 = of` 时，Markov head 可以把 `course` 的 logit 拉高，把 `problem` 的 logit 压低。它不是重跑 transformer，也不是让整个 draft 变成重自回归，而是只在 vocab logits 上串行加一个低秩 bias。

这就是它聪明的地方:

- 重的 parallel backbone 仍然一次 forward 出所有 `Uk`。
- 轻的 sampling loop 从左到右执行，给每一步加 `B(prev)`。
- 额外成本远小于重新跑一遍 draft transformer。
- 但它补上了 MTP 最关键的“后一个 token 看前一个”的局部条件。

DeepSpec 代码里还实现了 gated Markov 和 RNN head。论文默认使用 Markov head，因为 RNN 在长 proposal 上只带来边际增益，但部署复杂度更高。

### 3.3 Confidence Head 做了什么

Markov head 解决的是“draft 后缀更连贯”。Confidence head 解决的是“哪些 token 值得送去 verify”。

DSpark 给每个 draft 位置预测一个:

```text
c_k = P(position k accepted | positions < k all accepted)
```

注意这不是普通的 token probability，而是**条件接受概率**。因为 speculative decoding 只接受连续前缀，所以第 j 个 token 真正有价值的概率是 prefix survival:

```text
a_j = c_1 * c_2 * ... * c_j
```

训练标签来自 draft distribution 和 target distribution 的 total variation distance:

```text
c*_k = 1 - 0.5 * ||p_d_k - p_t_k||_1
```

这个量正好对应 rejection sampling 里单步接受概率的期望形式。因此 confidence head 不是拍脑袋打分，而是在学“这个 draft 分布和 target 分布到底有多接近”。

论文还强调了校准。因为 scheduler 需要用 `a_j` 估算 expected accepts，confidence 的绝对值不能太飘。DSpark 用 Sequential Temperature Scaling 在 held-out set 上从左到右校准 cumulative survival。论文里的 Alpaca reliability diagram 显示，原始 confidence 有 3% 到 8% 的 ECE，STS 后平均 ECE 降到约 1%。

### 3.4 Hardware-Aware Prefix Scheduler 做了什么

有了 `a_j` 之后，最简单做法是设一个静态阈值:

```text
如果 a_j 太低，就不验后面
```

但静态阈值不知道系统负载。轻载时多验几个 token 可能几乎免费；重载时一个低置信 token 可能挤掉其他请求的 batch capacity。

DSpark 的 scheduler 把问题写成系统吞吐最大化:

```text
Theta = expected_accepted_tokens * SPS(B)
```

其中:

- `B` 是本轮 target verify 的总 token batch size。
- `SPS(B)` 是引擎在 batch size 为 B 时的 steps per second，由硬件 profiling 得到。
- `expected_accepted_tokens` 由每条请求的 prefix survival `a_j` 累加得到。

直觉上，scheduler 会把所有请求的候选前缀 token 按 `a_j` 从高到低排序，优先把最可能被接受的 token 送进 target verify；一旦继续加 token 让 `Theta` 下降，就停。

这和我们的 D-Cut 的思想完全一致: **只把有正收益的连续前缀交给 verifier**。区别是:

- 静态 D-Cut 用 heuristic，比如 draft prob、entropy、position decay。
- DSpark 用训练出来并校准过的 confidence，再乘硬件吞吐曲线做全局调度。

还有一个非常重要的 lossless 约束: scheduler 不能利用未来 token 决定当前 token 是否被 verify，否则会破坏 speculative decoding 的目标分布。论文把这个叫 non-anticipating property。工程上如果我们做 D-Cut 或 DSpark scheduler，都必须守住这一点。

## 4. 为什么 DSpark 有效

### 4.1 它吃到了 DFlash 的 position-1 优势

论文里有一个很好的解释: DFlash 这种 parallel drafter 不是天然弱，反而在第一个 token 上很强。因为它只需要一次 forward，不像 autoregressive drafter 那样每个 token 都要串行跑，所以可以用更深的 draft backbone。在 position 1，大家都只看上下文，DFlash 的容量优势会很明显。

这也解释了我们之前的观察: trained DFlash 在 9B 上接受长度低于 MTP，但 QPS 可以更高。它的草稿成本更低，position-1 又不差。

### 4.2 它补了 DFlash 的后缀衰减

DFlash 的问题不是第一步，而是后面位置。因为每个位置没有条件于真实采样出的前缀，所以后缀容易 mode collision。

Markov head 的收益来自一个很朴素的事实: 自然语言、代码、标点、模板里，大量局部结构其实是一阶转移就能救的。

```text
of -> course
no -> problem
for -> (
return -> value
```

DSpark 不需要为此付出完整自回归 draft 的代价，只要给 logits 加一个低秩 transition bias。

### 4.3 它把“验多长”从固定参数变成 serving 决策

固定 spec=7 在离线 benchmark 里好理解，但生产系统里不是最优。不同请求、不同领域、不同负载，最优 verify length 都不同。

论文的 confidence threshold sweep 很直观:

- Math / Code 这类结构化任务本来 acceptance 高，剪得比较温和。
- Chat 这种开放任务尾部风险高，剪尾收益大。
- Chat 上随着 threshold 提高，接受率可以从 45.7% 提到 95.7%，代价是平均每步 verify token 变少。

这说明 confidence head 能识别“尾巴里大概率要被拒的 token”。再叠加硬件曲线后，系统就能在轻载时多验、重载时少验。

## 5. 和我们已有工作的关系

### 5.1 DFlash、MTP、DSpark、D-Cut 对照

| 方案 | 是否需要训练 | draft 方式 | 解决什么 | 当前对我们的意义 |
|---|---|---|---|---|
| DFlash | 有现成/可微调 | 并行 block draft | 草稿便宜，吞吐潜力高 | 我们已有 checkpoint，是近期落地底座 |
| MTP | 微调 verifier 原生头 | 带局部依赖的多 token prediction | 接受长度高 | 122B 上目前最强，但部署/训练耦合更重 |
| DSpark Markov | 需要训练 | DFlash backbone + 轻串行 logits bias | 修 DFlash 后缀 | 中期训练目标 |
| DSpark confidence | 需要训练+校准 | per-position survival prediction | 学会少验尾巴 | 中期 scheduler 目标 |
| 静态 D-Cut | 不需要训练 | serving-side verify pruning | 先少验高风险尾巴 | 短期 MVP，最适合先迁到 NPU |

### 5.2 关键判断

**第一，完整 DSpark 不是直接 patch 推理代码就能得到。**  
Markov head 的 `W1/W2` 和 confidence head 都要训练；否则没有可靠的 transition bias，也没有可用的 survival estimate。

**第二，D-Cut 静态版仍然非常有价值。**  
它不需要等训练，可以直接用现有 DFlash 输出的概率、entropy、位置衰减等信息做 verify prefix 裁剪。只要裁剪规则不看未来 token，rejection sampling 的 lossless 性质仍然可以保持。

**第三，我们现在最应该先打通 variable verify runtime。**  
无论未来是 heuristic D-Cut，还是 learned confidence scheduler，底层都需要同一套能力:

- proposer 能输出候选 token、draft probs、可选 confidence。
- metadata 能表达每条请求不同的 draft/verify 长度。
- runner 能把变长 verify prefix 组 batch。
- metrics 能统计 accepted length、effective verify length、wasted verify token、tok/s。

只要这条链路跑通，后续把 heuristic 换成 learned confidence 就是增量升级。

## 6. 我们怎么合入

### Phase A: 先做无训练 D-Cut

目标: 在当前 DFlash checkpoint 上实现“少验高风险尾巴”，先拿 serving 侧收益。

要做的事:

1. 在 DFlash proposer 侧保留每个位置的 draft token probability / entropy / rank / margin。
2. 做一个静态 prefix policy，例如:

```text
keep k while score_k >= threshold(position, load)
score_k 可来自 draft prob、entropy、累积 survival proxy
```

3. verify input 只拼接 `anchor + kept_prefix`。
4. rejection sampler 仍然使用原始 draft probs，并且只对提交给 verifier 的前缀执行标准接受规则。
5. 加 metrics:
   - mean accepted / draft round
   - mean verify tokens / round
   - wasted verify tokens
   - first-position acceptance
   - per-position conditional acceptance
   - tok/s under batch/concurrency sweep

这个阶段的价值是低风险: 不重训、不动主模型、不引入新 head，重点验证 vLLM/NPU 能否支持变长 verify。

### Phase B: 训练 Markov Head

目标: 把 DFlash 的并行草稿升级成 DSpark-style semi-autoregressive draft。

需要关注:

1. 用 target 模型生成训练 cache，至少要有 target logits 或可计算 target distribution 的隐藏态。
2. 在 DFlash backbone 上增加 `markov_rank=256` 的 Markov head。
3. 推理时 sampling 必须用修正后的 logits:

```text
x_k ~ softmax(U_k + B(x_{k-1}))
```

4. rejection sampling 也必须使用修正后的 draft probability:

```text
p_d_k = softmax(U_k + B(x_{k-1}))
```

这点非常关键。如果采样用了 Markov 修正，但 verifier 接受率计算还用原始 `softmax(U_k)`，就会破坏 speculative decoding 的无损性。

### Phase C: 训练 Confidence Head + 校准

目标: 从 heuristic D-Cut 升级到 learned confidence。

训练标签:

```text
c*_k = 1 - 0.5 * ||p_d_k - p_t_k||_1
```

推理信号:

```text
c_k = P(token k accepted | prefix < k accepted)
a_j = product(c_1...c_j)
```

上线前必须做 calibration:

- 看 per-position ECE / AUC / Brier。
- 看 cumulative survival 的 reliability。
- 用 held-out set 做温度校准，避免 confidence 过度自信。

### Phase D: 硬件感知 scheduler

目标: 把“验几个 token”从 per-request heuristic 升级成 batch-aware 调度。

需要准备:

1. profiling 得到 `SPS(B)` 或 NPU/vLLM 上等价的 step throughput curve。
2. 对每轮 active requests 计算所有候选前缀 token 的 `a_j`。
3. 按 expected throughput 选择每条请求的 verify prefix length。
4. 保证 non-anticipating: 当前是否验，不能依赖未来还没被合法观察的 token。
5. 在生产系统里处理异步调度、CUDA/NPU graph、变长 token flatten、attention marker 等工程问题。

## 7. 风险与注意事项

### 7.1 没有现成 Qwen3.5 多模态 DSpark checkpoint

DeepSpec 公开 checkpoint 是 Qwen3 与 Gemma4，不是我们的 Qwen3.5 多模态/122B-A10B 路线。因此它更像“训练和算法参考”，不是可以直接替换上线的权重。

如果要完整复现 DSpark 到我们模型上，需要重新生成 target cache、训练 Markov/confidence、做 calibration 和 serving 评测。

### 7.2 训练成本和数据成本不低

DeepSpec README 提到，默认 Qwen3-4B target cache 约 38 TB。我们自己的多模态目标如果要保留 target distribution / hidden states，也会遇到类似存储和 IO 压力。

这也是为什么短期应先做 D-Cut: 它先把 serving runtime 链路打通，同时不阻塞后面的训练路线。

### 7.3 Lossless 不是自动成立的

投机解码无损依赖两个条件:

1. verifier 使用标准 rejection sampling。
2. draft probability 与实际 draft sampling distribution 一致。

对 DSpark 来说，这意味着:

- Markov 修正后的 logits 用于采样，也必须用于 draft probs。
- confidence / D-Cut 只能决定“提交多长前缀”，不能改变已提交 token 的接受规则。
- scheduler 不能用未来 token 反过来决定当前 token 是否进入 verify。

### 7.4 静态 D-Cut 和 learned confidence 不要混淆

静态 D-Cut 是我们可以马上做的工程版本，但它的 signal 是 heuristic。它适合先验证:

- 平均 verify token 是否下降。
- target batch capacity 是否释放。
- tok/s 是否随 concurrency 改善。
- 输出是否保持 lossless。

DSpark confidence 是训练后的升级版。它需要数据、训练、校准，但能更接近“每个 token 的真实 prefix survival”。

## 8. 建议的下一步

### 8.1 近期: 用 DFlash + D-Cut 打通 NPU variable verify

优先级最高的是 runtime，不是马上训练 DSpark:

1. 固化 DFlash proposer 输出 token/probs。
2. 增加 per-request draft length / verify length metadata。
3. runner 支持变长 verify prefix。
4. 先用静态 D-Cut policy 做 sweep:
   - fixed spec=7
   - D-Cut avg verify ~= 6
   - D-Cut avg verify ~= 5
   - D-Cut avg verify ~= 4
5. 在 ALLaVA、MMStar 和纯文本集上同时看 acceptance 与 tok/s。

### 8.2 中期: 训练小模型 DSpark 头做机制验证

建议先不直接冲 122B。可以先选一个小 target 做 DSpark head 训练，验证:

- Markov head 是否显著抬后缀 conditional acceptance。
- confidence 是否能预测 prefix survival。
- calibration 后 ECE 是否降到可用范围。
- scheduler 是否真的比静态 threshold 稳。

### 8.3 长期: 把 scheduler 做成 hardware-aware

当 variable verify 链路稳定后，再把 D-Cut policy 替换成:

```text
score = calibrated prefix survival a_j
objective = expected_accepts * SPS(B)
```

这时 D-Cut 就自然演进成 DSpark scheduler。最终形态不是“固定少验几个 token”，而是:

- 低并发: 多验，换用户侧速度。
- 高并发: 少验，把 target batch capacity 留给更高价值 token。
- 不同请求: 根据自己的 prefix survival 分配不同 verify budget。

## 9. 一句话收束

DeepSeek 这套 DSpark 给我们的启发非常清楚:

> **投机推理的下一阶段，不只是把 draft 猜准，而是让系统知道哪些 token 值得验。**

我们当前已有 DFlash/MTP 的接受率优化结果，短期最应该把 D-Cut 的 variable verify runtime 跑通；中期再训练 Markov/confidence heads，把 heuristic D-Cut 升级成 DSpark-style confidence scheduling。这样路线最稳，也最符合我们现在“已有 DFlash checkpoint、还没有 DSpark checkpoint”的现实条件。

## 参考

- DeepSpec GitHub: https://github.com/deepseek-ai/DeepSpec
- DSpark paper: `DSpark_paper.pdf`
- DeepSpec implementation: `deepspec/modeling/dspark/markov_head.py`, `deepspec/eval/dspark/draft_ops.py`, `deepspec/modeling/dspark/loss.py`
- 本系列上一篇: `reports/dflash_mtp_internal_community_report.md`
