# MMStar MTP vs Original DFlash Spec Sweep Summary

来源日志：`origin/main:debug_error_from_inside` at commit `55129fe`

测试命令对应的是 `test_result` 分支里的：

```bash
MTP_SPECS="3 5 7" \
DFLASH_SPECS="3 5 7" \
NUM_PROMPTS=128 \
bash examples/evaluate/sweep_mtp_dflash_original_mmstar_specs.sh
```

结果目录：

```text
/data/wenxuan/speculators/output/mmstar_mtp_dflash_spec_sweeps/single_n128_tok128
```

## 一句话结论

在这组 128 条 MMStar open-ended 单图请求上，`original DFlash @ spec=7`
仍然是最高吞吐配置：`71.629 tok/s`。最好的 MTP 是 `MTP @ spec=7`，
`68.965 tok/s`，比 `original DFlash @ 7` 慢约 `3.7%`。

## 排名表

| rank | method | spec | status | tok/s | vs best | mean accept/draft | token accept | first-pos accept | completed |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | original DFlash | 7 | seeded | **71.629** | 1.000 | 2.097 | 0.300 | 0.767 | 128/128 |
| 2 | MTP | 7 | ok | **68.965** | 0.963 | **2.822** | 0.403 | 0.828 | 128/128 |
| 3 | MTP | 5 | ok | 68.424 | 0.955 | 2.600 | 0.520 | 0.837 | 128/128 |
| 4 | MTP | 3 | seeded | 67.393 | 0.941 | 2.045 | **0.682** | **0.852** | 128/128 |
| 5 | original DFlash | 5 | ok | 66.930 | 0.934 | 1.944 | 0.389 | 0.766 | 128/128 |
| 6 | original DFlash | 3 | ok | 61.084 | 0.853 | 1.638 | 0.546 | 0.779 | 128/128 |

说明：`seeded` 表示这两个点复用了之前
`test_mtp_vs_dflash_original_mmstar.sh` 已经跑完的结果；其余 `ok` 是本次
sweep 新跑的结果。所有配置都完成 `128/128`，没有启动或客户端失败。

## 关键对比

### 最高吞吐

```text
best: original DFlash @ spec=7 = 71.629 tok/s
best MTP: MTP @ spec=7 = 68.965 tok/s
original DFlash@7 / MTP@7 = 1.039
MTP@7 / original DFlash@7 = 0.963
```

也就是说，在当前这组 MMStar 单图设置下，原生 DFlash 最佳点比原生 MTP
最佳点快约 `3.9%`。

### MTP 趋势

| MTP spec | tok/s | mean accept/draft | token accept | first-pos accept |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 67.393 | 2.045 | 0.682 | 0.852 |
| 5 | 68.424 | 2.600 | 0.520 | 0.837 |
| 7 | **68.965** | **2.822** | 0.403 | 0.828 |

MTP 增大 spec 后吞吐小幅上升：`spec=7` 比 `spec=3` 快约 `2.3%`。
token-level accept rate 会下降，这是正常的，因为更深位置更难接受；但
`mean accept/draft` 从 `2.045` 提升到 `2.822`，所以总体吞吐还是变好。

### Original DFlash 趋势

| DFlash spec | tok/s | mean accept/draft | token accept | first-pos accept |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 61.084 | 1.638 | 0.546 | 0.779 |
| 5 | 66.930 | 1.944 | 0.389 | 0.766 |
| 7 | **71.629** | **2.097** | 0.300 | 0.767 |

DFlash 对 spec 深度更敏感：`spec=7` 比 `spec=3` 快约 `17.3%`，比
`spec=5` 快约 `7.0%`。所以当前 DFlash baseline 应继续用
`DFLASH_SPEC=7`，不要降到 3 或 5。

## 为什么 MTP 接受更多但吞吐仍没赢

这组结果里，`MTP@7` 的 `mean accept/draft=2.822`，明显高于
`DFlash@7` 的 `2.097`。但最终吞吐仍然是 DFlash 更高：

```text
MTP@7:    68.965 tok/s, mean accept/draft 2.822
DFlash@7: 71.629 tok/s, mean accept/draft 2.097
```

这说明当前瓶颈不只由接受长度决定，还包括 draft 模块本身的计算开销、
vLLM speculative 调度路径、MTP head 的实现成本、以及多模态场景下 verifier
prefill/decode 的整体开销。简单说：MTP 猜得更长，但每一步可能不够便宜；
DFlash 接受长度短一点，但整体路径在这里更快。

## 当前基线选择

后续所有 continued DFlash 训练结果，建议同时对照两条线：

```text
必须先超过 MTP@7:             68.965 tok/s
最终要超过 original DFlash@7:  71.629 tok/s
```

如果只想用一个主基准，当前应使用：

```text
original DFlash @ spec=7
```

因为它是这轮 sweep 里实际最高吞吐点。

## 下一步建议

1. 把 trained DFlash 的 checkpoint sweep 继续固定在 `INFER_NUM_SPEC=7`。
2. 如果要和 MTP 对齐展示，可以额外报告 `MTP@7`，而不是 `MTP@3`。
3. 对最终论文/汇报表格，建议列三列：
   `MTP@7`、`original DFlash@7`、`trained DFlash@7`。
4. 如果以后测 10-image grouped 场景，重新跑同样的 spec sweep；单图结论不一定能直接外推到 10 图。
