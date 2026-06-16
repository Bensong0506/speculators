# NPU MTP 接受率测试 (vllm-ascend, offline)

测之前训好的 9B MTP 在昇腾上的**接受率 / 接受长度**。假设 NPU 上 `vllm` + `vllm_ascend` 已装好,无需重装。

## 1. 准备(每次跑前)
```bash
pkill -f vllm || true                 # 清掉残留 vllm 进程
export ASCEND_RT_VISIBLE_DEVICES=0    # 选卡(多卡 TP 时写 0,1,2,3)
export VLLM_USE_V1=1                  # 接受率指标在 V1 引擎里
```

## 2. 填路径 + 跑
`MTP_MODEL` 填你那份**带 MTP heads 的 9B 权重**。`NUM_SPEC_TOKENS` **必须等于你训练时的 MTP 深度**(填错会报 0 或异常)。
```bash
cd npu_mtp_accept
export MTP_MODEL=/home/wenxuan/path/to/qwen3.5-9b-mtp   # ← 改成你的路径
export NUM_SPEC_TOKENS=3                                 # ← 改成你训的 MTP 深度
export TP=1
python bench_mtp_accept.py
```

若 MTP 是**独立 checkpoint**(没 fuse 进 base):再加
```bash
export MTP_MODEL=/path/to/qwen3.5-9b-base
export MTP_DRAFT=/path/to/your-trained-mtp
```

## 3. 看结果
跑完打印这块:
```
========================================================
  draft tokens    : 12345
  accepted tokens : 8210
  ACCEPTANCE RATE : 0.6650   (accepted / draft)   ← 这就是接受率
  ACCEPT LENGTH   : 2.995 tok/step
  per-position    : [0.78, 0.66, 0.55]            ← 第 1/2/3 个 draft 位的接受率
========================================================
```
若 `get_metrics()` 取不到,翻引擎日志里的 `Spec decode` / acceptance 行(`disable_log_stats=False` 已开)。

## 4. 可调 env(都有默认)
| env | 默认 | 说明 |
|---|---|---|
| `MTP_MODEL` | (必填) | 带 MTP 的 9B 权重 |
| `MTP_DRAFT` | 空 | MTP 是独立 ckpt 时填 |
| `NUM_SPEC_TOKENS` | 3 | **必须匹配训练 MTP 深度** |
| `TP` | 1 | tensor parallel |
| `NUM_PROMPTS` / `MAX_TOKENS` | 64 / 256 | 样本数 / 每条生成长度 |
| `TEMP` | 0 | 贪心,接受率干净可复现;>0 走 rejection sampling |
| `DTYPE` | bfloat16 | MTP 必须 bf16 |
| `DATASET` | 空 | 自带 jsonl(`{"prompt": "..."}` 每行一条),否则用内置 prompt |
| `ENFORCE_EAGER` | 1 | 接受率与图模式无关,默认 eager 最稳;测吞吐再设 0 |

## 注意
- **接受率只反映 draft 质量**(MTP 预测得准不准),跟"掩盖"/吞吐无关 —— 想看 tok/s 收益是另一个实验。
- 若这个 9B 其实是**多模态 VLM**,纯文本 prompt 会低估真实场景;请用真实 VQA prompt(含图)的 DATASET。
- `NUM_SPEC_TOKENS` 填错是最常见的坑:必须 ≤ 且通常 = 你训练的 MTP module 数。
