# 中期汇报 — Qwen3.5 多模态投机解码(MTP / DFlash speculator)

内部工作证明。本分支(`zhongqihuibao`,orphan,只含本项目产物)= **代码 diff + 测试用例脚本(按阶段分文件夹)+ 结果报告**。
diff 基准:`upstream/main = vllm-project/speculators @ d1a3ff3`。

## 1. 工作内容
为 **Qwen3.5 多模态 verifier(9B 及 122B-A10B MoE)** 训练**投机解码 draft**(MTP 头微调 + DFlash 块扩散 draft),提升推理吞吐;搭了完整的 **蒸馏 → 训练 → 评测 → 部署** 流水线,并在 ALLaVA(域内)+ MMStar(域外)上量化接受率。

## 2. 关键结果(详见 `reports/`)
| 对比 | in-domain (ALLaVA) | OOD (MMStar) |
|---|---|---|
| 122B MTP `β=0.6` vs 原生 MTP | mean-accept **+11.5%** / tok/s +7.3% ✅ | −2.4%(已优于 `β=1.0`) |
| 122B MTP 权重汤 `α=0.5` vs 原生 MTP | mean-accept **+10.2%** / tok/s +4.5% ✅ | **+2.3%** ✅ |
| 训好的 DFlash vs 原始 DFlash | mean-accept **+30.2%** ✅ | +8.4% ✅ |
| 最强方法 | **MTP(2.92× no-spec baseline)** | MTP |

- 实测 **接受率与吞吐近 1:1**(+10.1% 接受 → +9.9% tok/s),印证 `throughput ∝ 接受长度`。
- 新结果:122B 上 `α=0.5` MTP soup 不用重训,保住 ALLaVA 大部分增益,同时把 MMStar 从轻微回退修到**正收益**。
- 报告:`reports/mtp_dflash_122b_report.md`(122B)、`mtp_finetune_report.md`(9B MTP)、`dflash_ce_finetune_report.md` · `dflash_100k_three_way_report.md`(9B DFlash)。

## 3. 代码贡献(`diff/`,vs upstream d1a3ff3)
| 分支 | 内容 | 规模 |
|---|---|---|
| `mtp-training` | MTP 训练 + 共享设施(MTP 模型/转换、122B 张量并行训练、MoE `intermediate_size` 修复、step-weight) | **46 文件 +5764** |
| `dflash-122b-distill` | 122B 自蒸馏数据生成 | **48 文件 +5970** |
| `test_result122B` | 多模态评测 harness + 报告 | **128 文件 +11878** |

每个 `*_vs_upstream.diff` 是该分支相对 upstream 的完整改动;`SUMMARY.txt` 是改动文件清单。三分支共享核心 `src/speculators` 改动,故 diff 有重叠。

## 4. 测试用例 / 脚本(`scripts/`)
| 文件夹 | 内容 |
|---|---|
| `scripts/distill/` | 122B 自蒸馏 ALLaVA(`distill_allava_122b*.sh` + 跨机分片 + worker `distill_allava_with_qwen.py`) |
| `scripts/train/` | MTP/DFlash 训练 launcher(122B tensor-parallel + 9B online) |
| `scripts/eval/` | **接受率评测**:`mtp_accept/`(native-vs-trained MTP、自动 stitch、双机 A/B `eval_mtp_ab.sh`)、DFlash 三路对比(`test_three_way_mmstar_allava.sh` 等)、多模态客户端(`mmstar_weight_client.py`/`bench_mm_speculative.py`) |
| `scripts/serve/` | GPU 部署(serve + vLLM M-RoPE patch) |
| `RUN_commands.md` | **单一来源**:所有 蒸馏/训练/评测/部署 的 copy-paste 命令 |

## 5. 流水线(端到端)
1. **蒸馏** `scripts/distill/distill_allava_122b_split.sh` —— 122B 自蒸馏 ALLaVA(双机分片、32 并发、TP=8)→ 训练数据。
2. **训练** `scripts/train/nohup_mtp_122b_allava_distilled.sh` —— MTP(verifier TP=4 + trainer;微调 verifier 自带 MTP 头)。
3. **评测** `scripts/eval/mtp_accept/` —— serve native vs 训好的,读 `/metrics` per-position 接受率(ALLaVA 域内 + MMStar 域外)。
4. **部署** `scripts/eval/mtp_accept/stitch_mtp.py` —— 把训好的 MTP 头缝回 verifier → 可直接上线(质量 = base,仅提速)。

> 最新:step-weight A/B 已确认 `β=0.6` 更优;122B MTP soup `α=0.5` 已把 OOD 修到正收益。下一步做更细的 `α` sweep + deploy 深度 spec sweep。
