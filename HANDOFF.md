# HANDOFF 2026-06-17 — 122B MTP/DFlash speculator(下个 session 读这个)

一行接上:`git show test_result:HANDOFF.md`。**「一切命令进 RUN.md」是硬规矩**(`mtp-training:RUN.md` = 449 行单一来源;`test_result122B:mtp_accept/RUN.md`)。

## 分支布局(本 session 重构,别再混 9B/122B)
| branch | tip | 用途 |
|---|---|---|
| `test_result` | 48cad26 | **9B / GPU**(canonical handoff 在这) |
| `test_result_npu` | fe095d5 | 9B / NPU(`npu_mtp_accept/`) |
| **`test_result122B`** | 0364986 | **122B / GPU 评测**(`mtp_accept/` + DFlash 三路 + 报告) |
| `mtp-training` | a50322a | **MTP 训练**(122B launcher + RUN.md「MTP 50k A/B」节) |
| `dflash-122b-distill` | 4738c03 | **蒸馏数据生成**(`distill_allava_122b*.sh`) |

## ⏳ 下个 session 第一件事:MTP 50k 接受率 A/B(进行中)
命令全在 `mtp-training:RUN.md` 的「MTP 50k 接受率 A/B」节。
- **目标**:提升 MTP 接受率(中后位)。两台 A800,**唯一变量 = `STEP_WEIGHT_BETA`**;配置 **steps=5 / seq=4096(全长)**、50k 数据。机器1 A=`beta=0.6`、机器2 B=`beta=1.0`。
- **当前状态**:steps=7 的 smoke **OOM 了**(trainer 端,全词表 248320 × seq × 7 步 logits 撑爆 GPU4-7)→ 决定改 **steps=5 全长**(省 ~8G,不截 seq)。**下一步:跑 steps=5 smoke(RUN.md 有命令)→ 过了起两台正式。**
- 跑完:各取 `checkpoint_best`,切 `test_result122B` 用 `mtp_accept/run_mtp_accept_compare.sh` 评(ALLaVA+MMStar),比 **B(等权)中后位 per-position 是否高于 A**。
- 然后:**纯 serve 的 spec sweep(3/5/7,不重训)** 定 deploy 深度(per-position 条件接受率 ~0.73 没塌,深位还产出,别先入为主定 3)。

## ✅ 本 session 做完的
1. **122B 评测出报告** = `test_result122B:examples/evaluate/mtp_dflash_122b_report.md`:
   - **MTP best vs 原生**:ALLaVA mean-accept 3.066→3.376(**+10.1%**)/ tok/s +9.9% ✅;MMStar(OOD)−5.7%/−4.7%(first-pos 不动)。
   - **训好的 DFlash vs 原始**:ALLaVA **+30.2%**、MMStar +8.4% ✅;但仍落后 MTP。
   - **MTP 最强**(2.92× no-spec baseline)。
2. **50k 蒸馏完成**(10k→50k):`dflash-122b-distill:distill_allava_122b_split.sh`,两台各 20k(SKIP/MAX 自动分片)、32 并发、TP=8、temp=0 → `data/allava/allava_122b_distill_50k.jsonl`。
3. **分支重构** + **stitch 自包含 + hardlink 省盘**(122B 不再 244GB 全拷)+ mtp_accept 自动切后 10% val。
4. **DFlash v2 / SGLang↔vLLM 源码结论**:DFlash 算法在 vLLM 自 **0.20.0**、vllm-ascend 自 **0.19.1rc1**(0.18 都没);「掩盖」= `async_scheduling` + fused materialize,vLLM 自带、**不用从 SGLang 移植**;async 帮 DFlash 不帮 MTP(并行 draft → host-bound)。

## 💡 关键经验(借自 openPangu MTP 报告 + 实测)
- **step-weight 配平**:我们默认 FastMTP `beta=0.6` 衰减 [.51,.31,.18] 压中后位;openPangu:配平最优、压小后位掉 1–1.7%。→ A/B 测 `beta=1.0`(等权);first-pos 已饱和 ~0.87,空间全在中后位。
- **train-what-you-serve 深度**:launcher 默认 train steps=3 但 serve spec=7 → 后位没训过(本想 steps=7,OOM → steps=5 折中)。
- **TPOT ∝ 1/L**(L=每步吐 token=accept+1 bonus):+9% 接受 → TPOT 降 ~6.3%(metric 不含 bonus)~8.3%(含);实测 +10.1% 接受 ≈ +9.9% tok/s 印证。

## ⚠️ 坑(记牢)
- **MTP 必须 bf16**(fp32 崩 lm_head);DFlash 用 fp32。
- **122B heads=32 → TP 只能 {4,8}**;verifier TP=4(GPU0-3)+ trainer(4-7)。
- **steps 大 → trainer OOM**(全词表 248320 × seq × steps logits)→ 砍 steps 别砍 seq(截 VLM 样本丢信号);**别**用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`(破坏 vLLM hidden-states connector)。
- **DFlash eval 的 `BASELINE_DRAFT`** 必须是真实原始 122B DFlash 路径(占位符会 `[fatal]` 退,本 session 踩过)。
- **MTP head 不能裸 serve** → `stitch_mtp.py` 缝进 verifier;stitched = base + 训好的头,**可直接上线(质量=base,只提速)**;搬运注意 hardlink 会实体化(生产建议在目标机现 stitch)。
- 路径 `/data/wenxuan` vs `/home/wenxuan` 按机器调;跑前 `pkill -f vllm`;内网只 pull,结果走 `output_log`/`output_debug` 回传;`/health` 探活要 `no_proxy=localhost`(否则走代理超时)。

下个 session:`git show test_result:HANDOFF.md` → 跑 steps=5 smoke → A/B → eval → spec sweep。
