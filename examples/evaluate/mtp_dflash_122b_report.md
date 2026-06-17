# 122B MTP & DFlash best-checkpoint 接受率评测 — ALLaVA(in-domain)+ MMStar(OOD)

- **日期**:2026-06-17 · **分支**:`test_result122B`
- **Verifier**:Qwen3.5-122B-A10B(MoE,~10B active)
- **数据**:ALLaVA val 尾(in-domain,完整集自动切后 10%)+ MMStar(OOD);各 **128 prompts**,全部 128/128 完成
- **设置**:spec=7,greedy(temp=0),TP=4
- **方法**:MTP 走 `mtp_accept/`(native vs stitch 后的 trained,`qwen3_5_mtp`);DFlash 走 `test_three_way_mmstar_allava.sh`(mtp / trained_dflash / dflash_original 三路)

---

## TL;DR

1. **MTP best 击败原生 MTP(in-domain)**:ALLaVA mean-accept **3.066 → 3.376(+10.1%)**、tok/s **+9.9%**;MMStar(OOD)**轻微回退** −5.7% / tok/s −4.7%,first-pos 几乎不变。= 典型微调专化,和 9B 同一模式。
2. **训好的 DFlash 击败原始 DFlash(两数据集)**:ALLaVA mean-accept **+30.2%**、MMStar +8.4% —— 122B DFlash 蒸馏训练有效。
3. **MTP 仍是最强配置**:两数据集 acceptance 与 tok/s 均最高,对 no-spec baseline **~2.9×**;训好的 DFlash 第二(~2.6×),仍落后 MTP(tok/s ~0.9×、mean-accept ~0.66–0.71×)。
4. **吞吐 ∝ 接受长度得到实测印证**:MTP best 的 +10.1% 接受对应 +9.9% tok/s(近 1:1)。

---

## 1. MTP —— 原生 vs 训好的 best checkpoint

| metric | dataset | native | trained | ratio |
|---|---|---:|---:|---:|
| mean accept/draft | ALLaVA | 3.066 | **3.376** | **1.101** |
| token acceptance | ALLaVA | 0.438 | 0.482 | 1.101 |
| first-pos acceptance | ALLaVA | 0.848 | 0.873 | 1.029 |
| output tok/s | ALLaVA | 33.337 | **36.630** | **1.099** |
| mean accept/draft | MMStar | 2.930 | 2.764 | 0.943 |
| token acceptance | MMStar | 0.419 | 0.395 | 0.943 |
| first-pos acceptance | MMStar | 0.831 | 0.827 | 0.995 |
| output tok/s | MMStar | 33.348 | 31.775 | 0.953 |

**解读**:in-domain 全面提升(+10.1% 接受 / +9.9% tok/s);OOD 轻微回退(−5.7% 接受),first-pos 几乎不动(−0.5%)→ 草稿"下一个 token"质量保住,损失在更深的位置专化。净故事 = **域内提升换域外小代价**,正常微调专化,域定向部署没问题;需要通用服务则混入通用数据。
**注**:first-pos 已接近饱和(0.85–0.87),空间在中/后位 → 见下方 next step 的 step-weight 实验。tok/s 单次有 ±几% 噪声,以 acceptance 为准。

## 2. DFlash —— 三路(trained vs original vs MTP)

| dataset | method | tok/s | mean accept/draft | token accept | first-pos |
|---|---|---:|---:|---:|---:|
| ALLaVA (in-domain) | mtp | 34.375 | 3.209 | 0.458 | 0.865 |
| ALLaVA (in-domain) | **trained_dflash** | 30.918 | 2.283 | 0.326 | 0.745 |
| ALLaVA (in-domain) | dflash_original | 26.645 | 1.753 | 0.250 | 0.691 |
| MMStar (OOD) | mtp | 33.590 | 3.238 | 0.463 | 0.847 |
| MMStar (OOD) | **trained_dflash** | 30.415 | 2.133 | 0.305 | 0.750 |
| MMStar (OOD) | dflash_original | 28.513 | 1.967 | 0.281 | 0.710 |

**训好的 DFlash vs 对照**:
- vs 原始 DFlash:ALLaVA mean-accept **1.302×(+30.2%)** / first-pos 1.078× / tok/s 1.160×;MMStar mean-accept 1.084× / first-pos 1.056× / tok/s 1.067×。→ **训练在两域都涨,域内尤其大。**
- vs MTP:ALLaVA tok/s 0.899× / mean-accept 0.711×;MMStar tok/s 0.906× / mean-accept 0.659×。→ **仍落后 MTP。**

## 3. 对 no-spec baseline 的整体加速(ALLaVA)

| 方法 | tok/s | 对 baseline |
|---|---:|---:|
| MTP | 34.375 | **2.92×** |
| trained_dflash | 30.918 | 2.63× |
| dflash_original | 26.645 | 2.27× |
| baseline (no spec) | 11.756 | 1.00× |

---

## 4. 结论与下一步

- **MTP best 确认可用**:in-domain +10% 接受 / +10% tok/s。域外轻微回退 → 通用场景混通用数据缓解。
- **DFlash 蒸馏方向有效但未超 MTP**:训好的 DFlash 大幅超原始(+30% 域内),但在这个 122B verifier 上 MTP 仍是最强 spec 方法(和 9B 结论一致)。
- **next:50k 重训 + `--step-weight-beta` A/B**。当前 FastMTP 默认 `beta=0.6` → 步权重 [0.51,0.31,0.18] 在压中/后位,而我们的提升空间正是中/后位(first-pos 已饱和)。openPangu MTP 经验:配平系数最优、压小后位系数掉 1–1.7% 接受。→ 跑 `beta=0.6` vs `1.0`(等权)两组,目标抬高后位接受、进一步缩小与理论上限的差距。
- **吞吐模型**:实测 +10.1% 接受 → +9.9% tok/s,近 1:1,印证 `throughput ∝ 接受长度`、`TPOT ∝ 1/L`。

**产物**:`output/three_way_both/20260617_025446/`(DFlash combined_summary)、`mtp_accept/results/20260617_024459/`(MTP native-vs-trained summary)。
