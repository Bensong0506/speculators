# DFlash + MTP 多模态蒸馏 — Session 交接（2026-06-15）

> 读法：`git show test_result:HANDOFF.md`。本文件取代 2026-06-12 旧版。

## 项目目标
提升投机解码 draft 在 VLM 上的接受率→吞吐。verifier 有两个：**Qwen3.5-9B**（已做透）和
**Qwen3.5-122B-A10B**（新目标）。方法：DFlash draft（从头/warm-start 训）+ MTP（微调
verifier 自带的原生 MTP 头）。

## 仓库 / 工作流（重要）
- 内网：`/data/wenxuan/speculators`（注意是 **/data** 不是 /home，旧脚本路径要留意）。
- remote：`origin = Bensong0506/speculators`，`upstream = vllm-project/speculators`。
- **回传约定**：内网**只能 pull、不能 push**。结果通过仓库根的 **`output_log_debug`** 文件回传
  （脚本已改成把结果 tee/cp 进它）；Claude 在 Mac 改→push→内网 pull+跑→看 `output_log_debug`。
- **交付即脚本**：要内网跑的一律做成脚本推上去，别让用户手抄命令。
- **每次跑前 `pkill -f vllm`**（崩溃会留僵尸占端口/显存 → EADDRINUSE）。

## 分支最新 tip
| 分支 | tip | 内容 |
|---|---|---|
| `mtp-training` | `15ed030` | MTP 训练/评测 + 122B 脚本 + base launcher TP |
| `test_result` | `8bf34d4` | 所有评测脚本 + 报告 + 本 HANDOFF |
| `allava-qwen-distill-10k` | `9734833` | DFlash 训练分支（100k 已训完） |
| `dflash-causal-block-mask` | `74b3552` | causal 实验（**已搁置**，见下） |
| `main` | `a9d0a1e` | debug 通道 |

## 本 session 关键结果

### ① MTP 微调（9B）域内涨、域外基本守住 ✅（已完成、已出报告）
微调 Qwen3.5-9B 原生 MTP 头（100k 蒸馏 ALLaVA，bf16/LR3e-5），vs 原生 MTP：
- **域内 ALLaVA**：mean-accept **+7.7%**（2.92→3.14）、tok/s **+6.1%**、first-pos +0.7%
- **域外 MMStar**：first-pos 基本持平（-0.5%）、mean-accept 小幅回退 **-4.6%**（0.954×，非崩溃）
- 原生 MTP 在 MMStar 复现已知基线 **0.828/2.822**（口径正确）
- 报告：`mtp-training:examples/evaluate/mtp_finetune_report.md`（域内+域外合并版，汇报用）
- **怎么评 trained-MTP（已打通）**：vLLM 不能直接挂裸 MTP ckpt → `scripts/stitch_mtp.py` 把微调头
  缝回 verifier 再 serve；评测脚本 `test_mtp_allava_orig_vs_trained.sh`（域内自动 stitch）+
  `test_mtp_mmstar_orig_vs_trained.sh`（域外）。`qwen3_5_mtp` 能直接读缝合后的头。

### ② DFlash causal block-mask 实验 —— 已搁置 ⏸️
- 发现：DFlash block 内注意力默认**非因果（双向）**→ first-pos 随 num_spec 掉（3>5>7）。
  `dflash-causal-block-mask` 分支把训练 mask 改 causal（仅训练侧）。
- **为什么搁置**：用户 GPU 跑的是 **stock pip vLLM 0.22.0，它的 DFlash 写死非因果、无 causal 开关**
  （`dflash.py` `causal=False` + 断言；`get_dflash_causal` 只存在于用户的 `vllm-fork`，而那个 fork 没在用）。
  serve 端 causal 做不了；离线诊断也踩坑（base launcher GPU 变量曾硬编码、DP 撞端口）。用户决定先放下。

### ③ DFlash 100k 评测 —— ⏳ 待跑（**新 session 的即时下一步**）
DFlash 100k 已训完。三路对比脚本就绪但**用户还没跑**。命令：
```bash
pkill -f vllm ; sleep 3
cd /data/wenxuan/speculators && git checkout test_result && git pull
DRAFT=/data/wenxuan/speculators/output/<dflash-100k-run>/checkpoints/checkpoint_best \
ALLAVA_JSONL=/data/wenxuan/speculators/data/allava/allava_qwen35_distill_100k.jsonl \
INFER_NUM_SPEC=7 NUM_PROMPTS=128 GPUS=0 \
bash examples/evaluate/test_three_way_mmstar_allava.sh
```
跑 trained_dflash vs dflash_original vs mtp，在 ALLaVA+MMStar；结果自动写 `output_log_debug`。
若 ALLaVA 全 FAILED → 图片路径坑：`sed -i 's#/home/wenxuan#/data/wenxuan#g' <jsonl>`。

### ④ 122B MTP —— 脚本就绪，待跑
- **模型确认（HF Qwen3.5-122B-A10B）**：MoE 122B/~10B 激活，256 experts，**48 层，头 32 / KV 2**，
  hidden 3072，vocab 248320，**原生 MTP 头**(`mtp_num_hidden_layers=1`)，混合(线性 DeltaNet+全注意力)，多模态。
- **头=32 ⇒ TP 只能 4 或 8（不能 6）**。
- 脚本（`mtp-training`）：
  - `examples/train/distill_allava_122b.sh` — 用 122B 自蒸馏 ALLaVA（MTP 必须用本模型 response），TP=8 占满 8 卡。
  - `examples/train/nohup_mtp_122b_allava_distilled.sh` — MTP 训练，verifier **TP=4**(GPU0-3，KV 极小所以 61GB/卡能塞)+ trainer(GPU4-7)。
  - base launcher `dflash_qwen3.5_9b_multimodal_online.sh` 已加 `VLLM_TP`，并把
    `VLLM_GPUS/VLLM_DP/TRAIN_GPUS/NUM_TRAIN_GPUS/GEN_GPU_MEM_UTIL` 改成 env 可覆盖（原来是硬编码!）。
- 两步：先 `distill_allava_122b.sh` → 再 `nohup_mtp_122b_allava_distilled.sh`（详见 RUN.md 122B 段）。
- **待用户确认**：内网 122B 实际路径（脚本默认 `/data/wenxuan/Qwen3.5-122B-A10B`）。

## 关键事实 / 坑
- **MTP 必须 bf16**（fp32 破坏 lm_head dtype）。DFlash 用 fp32+CE+LR3e-5。
- **MTP 自蒸馏**：训练目标要 verifier 自己生成的 response。
- **path-prefix 坑**：蒸馏 jsonl 图片路径可能写死 `/home/wenxuan`，本机 `/data/wenxuan` → 404 → 请求全 FAILED；sed 修。
- **checkpoint_best 按 val-loss 选**；选最优应按真实接受率（`sweep_dflash_allava_checkpoints.sh`）。
- **eval val 尾巴要对齐训练数据**：`ALLAVA_JSONL` 用哪份训的就用哪份（后 10% 当 val）。
- **基线 @spec7**：MMStar 原生 MTP 0.828/2.822；ALLaVA 原版 DFlash ~0.728/1.94。
- 在线管线：verifier(vLLM) 和 trainer(torchrun) 占**不同 GPU**；大 verifier 必须 **TP**（DP 会每卡复制整模型）。

## 下一步（按优先级）
1. **跑 DFlash 100k 三路评测**（命令见 ③）→ 判读 trained vs original vs MTP。
2. **122B**：确认路径 → 跑 `distill_allava_122b.sh` → MTP 训练 → 评测。
3.（搁置）causal：若要做，需在 vllm-fork 上 serve，或把 causal 支持移植进 stock 0.22。
