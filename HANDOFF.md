# HANDOFF (latest) — 122B MTP/DFlash speculator(下个 session 读这个)

一行接上:`git show test_result:HANDOFF.md`。**「一切命令进 RUN.md」是硬规矩**(`mtp-training:RUN.md` 单一来源;`test_result122B:mtp_accept/RUN.md`)。

## 分支布局(别混 9B/122B)
| branch | 用途 |
|---|---|
| `test_result` | **9B / GPU**(canonical handoff 在这) |
| `test_result_npu` | 9B / NPU |
| **`test_result122B`** | **122B / GPU 评测**(`mtp_accept/` + DFlash 三路 + 报告 + `output_log` 回传结果) |
| `mtp-training` | **MTP 训练**(122B launcher + RUN.md「MTP 50k A/B」节, steps=5/seq=4096) |
| `dflash-122b-distill` | 蒸馏数据生成(`distill_allava_122b*.sh`,50k done) |
| **`zhongqihuibao`** | **中期汇报证据包**(diff vs upstream + 脚本分文件夹 + 报告 + 客户报告);也记了 self-forcing 探索结果 |

## 当前最优 & 进展
- **当前最优配置 = β=0.6 + MTP 权重汤(WiSE-FT soup)**。self-forcing 探索过 → 机制有效但**无净收益,已搁置**(记在 `zhongqihuibao:reports/spec_decoding_customer_report_2026-06-22.md`)。
- **MTP 50k A/B(steps=5,β=0.6 vs 1.0)评测已出**(`test_result122B:output_log`,`mtp_accept/results/arm_{a,b}/`):in-domain(ALLaVA)~**+10%**;OOD(MMStar)随配置变(有一臂到 +2.3%、有一臂 −5.7%)。**⚠️ output_log 是多轮累积(A/B + soup + 早期),下个 session 第一件事:把 arm_a/arm_b 的 `*_summary.json` 拉出来做干净对比,定 β。**
- **122B MTP 权重汤 alpha sweep** 已跑(`mtp_accept/results/soup_sweep/.../soup_a<XX>/`;blend = alpha·finetuned+(1-alpha)·native)。9B soup 是赢的,122B 看 sweep 结果选 alpha。

## ⏳ 下个 session 待办
1. **consolidate A/B + soup**:从 summary.json 出干净表,定最终 β + soup alpha。
2. **deploy 深度 = serve-time spec sweep(3/5/7,不重训)**——per-position 条件接受率 ~0.73 没塌,深位还产出,别先入为主定 3。
3. **客户问题(spec=3 接受 2.4)**:先 spec sweep(免费);**增训值不值取决于 draft 有没有对齐 SFT *后* 的分布**——SFT 挪了目标分布,draft 若没在 SFT 后输出上重训(on-policy)就有空间,对齐了则 spec=3 已近顶(~+一成)。

## ✅ 本轮做完的(我这条线)
- **122B 评测 + 报告** `test_result122B:examples/evaluate/mtp_dflash_122b_report.md`:MTP best vs 原生 in-domain **+10.1% 接受/+9.9% tok/s**;训好的 DFlash 超原始 **+30.2%** 但仍 < MTP;**MTP 最强(2.92× baseline)**。
- **评测 harness**:`mtp_accept/run_mtp_accept_compare.sh`(native-vs-trained,自动 stitch+hardlink 省盘+自动切后10% val)+ `eval_mtp_ab.sh`(双机 A/B,ARM=a/b)。
- **50k 蒸馏**(`distill_allava_122b_split.sh`,双机各 20k、32 并发、TP=8)。
- **分支重构**(test_result122B 独立)+ **`zhongqihuibao` 汇报证据包**(diff +5.7k~11.9k 行/分支 + 脚本分类 + 报告)。
- **源码结论**:DFlash 在 vLLM 自 0.20.0、vllm-ascend 自 0.19.1rc1;「掩盖」=async_scheduling+fused materialize(vLLM 自带,不用移植);async 帮 DFlash 不帮 MTP。

## 💡 关键经验(记牢)
- **接受率 metric 不含 free-lunch bonus**:`spec_mean_accepted_tokens_per_draft`(如 3.314)= 接受的 draft 数(含 pos0、不含 bonus);**真实每步吐 L = accepted + 1**(3.314 → L=4.314)。报"接受长度"用 L,报"draft 接受"用前者。`TPOT ∝ 1/L`。
- **step-weight**:默认 FastMTP β=0.6 衰减压中后位;openPangu 说配平好。但 122B A/B 初步看 β=0.6 OOD 更稳(等权更易 in-domain 过拟合)→ 以 A/B 干净对比为准。**当前最优仍 β=0.6 + soup**。
- **增训前先 spec sweep(免费)**;**增训要 on-policy**(在目标 SFT 后的自身 continuation 上训)。

## ⚠️ 坑
- MTP=bf16(fp32 崩 lm_head);DFlash=fp32。122B heads=32 → TP 只能 {4,8}(verifier TP=4 GPU0-3 + trainer 4-7)。
- **steps 大→trainer OOM**(全词表 248320×seq×steps logits)→ 砍 steps 别砍 seq;**别**用 `expandable_segments`(破坏 vLLM hidden-states connector)。
- MTP head 不能裸 serve → `stitch_mtp.py` 缝;stitched=base+训好的头,**可直接上线(质量=base 只提速)**,搬运注意 hardlink 实体化。
- DFlash eval 的 `BASELINE_DRAFT` 要真实路径;`/data` vs `/home` 按机器;跑前 `pkill -f vllm`;`/health` 探活要 `no_proxy=localhost`;内网只 pull,结果走 `output_log`/`output_debug`。

下个 session:`git show test_result:HANDOFF.md` → consolidate A/B+soup → spec sweep → 客户增训决策。
