# MMStar 10k Continue-DFlash Checkpoint Sweep Summary

来源日志：`origin/main:debug_error_from_inside` at commit `014b894`

测试命令实际使用的是 10k continue run 的 checkpoint 目录：

```bash
INFER_NUM_SPEC=7 \
CHECKPOINT_FIND_ROOT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_10k_continue_dflash/dflash_qwen35_9b_allava_10k_continue_dflash_20260605_031136/checkpoints/ \
bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
```

注意：最开始尝试的 100k 路径不存在：

```text
/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_100k_continue_dflash
```

## 一句话结论

这条 10k continue-DFlash run 里面，所有 checkpoint 都能正常加载和推理，但没有任何一个 checkpoint 在 MMStar open-ended、`INFER_NUM_SPEC=7` 下超过 native/raw DFlash baseline。

## 排名表

| checkpoint | trained tok/s | trained/native | mean accept/draft | first-pos accept | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 56.816 | 0.810 | 1.390 | 0.569 | not improved |
| 4 | 59.495 | 0.859 | 1.523 | 0.672 | not improved |
| 9 | 59.285 | **0.869** | 1.556 | 0.694 | ratio 最好 |
| 14 | **60.642** | 0.844 | **1.564** | 0.700 | trained tok/s / acceptance 最好 |
| 19 | 58.939 | 0.830 | 1.548 | **0.701** | first-pos 最好 |
| 24 | 59.154 | 0.819 | 1.541 | 0.695 | not improved |
| 29 | 58.527 | 0.851 | 1.548 | 0.689 | not improved |
| 34 | 57.653 | 0.796 | 1.520 | 0.692 | not improved |

## Native Baseline 水平

Native/raw DFlash baseline 在这组 sweep 中基本稳定：

```text
native tok/s:                68.2 - 72.4
native mean accept/draft:    about 2.09
native token accept rate:    about 0.299
native first-pos accept:     about 0.764 - 0.766
native accepted tokens:      about 11150 - 11170
```

而 trained checkpoint 最好也只有：

```text
best trained tok/s:             60.642   (checkpoint 14)
best trained/native ratio:      0.869    (checkpoint 9)
best trained mean accept/draft: 1.564    (checkpoint 14)
best trained first-pos accept:  0.701    (checkpoint 19)
```

所以当前差距主要不是最终回答质量，而是 DFlash draft 被 verifier 接受得不够多。

## 为什么吞吐追不上

典型对比：

```text
native:
  draft steps 约 5330
  draft tokens 约 37300
  accepted tokens 约 11160
  mean accept/draft 约 2.09

trained:
  draft steps 约 6400-6500
  draft tokens 约 44800-45600
  accepted tokens 约 9900-10016
  mean accept/draft 约 1.52-1.56
```

DFlash 加速的核心是“draft 少一点、accepted 多一点”。当前 trained checkpoint 反而是 draft 更多、accepted 更少，所以速度一定追不上 native。

## 训练趋势

这条 run 的趋势大致是：

```text
checkpoint 0:  明显较差，first-pos accept 只有 0.569
checkpoint 4:  明显恢复
checkpoint 9:  trained/native ratio 最好，0.869
checkpoint 14: trained tok/s 和 mean accept/draft 最好
checkpoint 19+: 平台期或回落
checkpoint 34: ratio 掉到 0.796
```

如果必须从这条 run 里选一个 checkpoint，优先看：

```text
checkpoint 14
```

它的 trained tok/s 和 mean accept/draft 最好。但它仍然没有超过 native baseline。

## 当前判断

1. Continue training 是 work 的：所有 checkpoint 都能推理，且有非零 DFlash acceptance。
2. 但这条 10k ALLaVA continue run 没有提升 native DFlash；它更像是把 native DFlash 的通用 draft 能力往当前 10k ALLaVA 分布上拉偏了。
3. `checkpoint_best` 不应该只按训练/validation loss 选，还需要按 MMStar `INFER_NUM_SPEC=7` 的 acceptance/throughput 选。
4. 当前最重要的指标是：

```text
trained/native > 1.0
trained mean accept/draft > 2.09
```

目前最好 checkpoint 仍只有：

```text
trained/native = 0.869
trained mean accept/draft = 1.564
```

## 下一步建议

建议不要继续沿用当前参数硬训。下一组实验建议降低 warm-start 学习率，并更密集保存 checkpoint：

```bash
LR_FT=1e-5 \
EPOCHS=20 \
CHECKPOINT_FREQ=1 \
MAX_SAMPLES=100000 \
bash examples/train/nohup_dflash_qwen3.5_9b_allava_full.sh
```

然后继续用：

```bash
INFER_NUM_SPEC=7 \
CHECKPOINT_FIND_ROOT=/path/to/new_run/checkpoints \
bash examples/evaluate/sweep_dflash_mmstar_checkpoints.sh
```

重点看每个 checkpoint 是否能把：

```text
mean accept/draft
per-position acceptance
trained/native throughput ratio
```

推到 native baseline 以上。
