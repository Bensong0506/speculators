# NPU MTP 接受率:原生 MTP vs 训好的 MTP (vllm-ascend, serve-based)

复用仓库自带的 `eval-guidellm` harness:`vllm serve` 起服务 → `guidellm` 压测 →
`parse_logs.py` 从 server 日志的 `SpecDecoding metrics` 抽 per-position 接受率。
两个 MTP 各跑一遍,串行对比。NPU 上 serve 由 vllm-ascend 接管,无需重装。

## 1. 准备(每次跑前)
```bash
cd /path/to/speculators
git fetch origin test_result_npu && git checkout test_result_npu && git pull
pkill -f vllm || true
export ASCEND_RT_VISIBLE_DEVICES=0        # 多卡 TP 写 0,1,2,3
command -v guidellm >/dev/null || pip install guidellm   # harness 依赖
```

## 2. 填两个模型路径 + 跑
```bash
export NATIVE_MTP_MODEL=/home/wenxuan/<原生/baseline 的 9B-MTP>    # ← 必填
export TRAINED_MTP_MODEL=/home/wenxuan/<你训好的 9B-MTP>          # ← 必填
export NUM_SPEC_TOKENS=3        # 改成你训练的 MTP 深度
export TP=1
# 内网不通 HF → 必须给本地数据集:
export DATASET=/home/wenxuan/<your_eval>.jsonl

bash npu_mtp_accept/run_mtp_accept_compare.sh
```
- MTP 若是**独立 speculator**(非内置头):再设 `NATIVE_MTP_DRAFT=` / `TRAINED_MTP_DRAFT=`。
- "原生 MTP" = 官方/未微调那份;"训好的" = 你的 checkpoint。两者都 `METHOD=mtp`。

## 3. 看结果
脚本末尾直接并排打印两份 `acceptance_analysis.txt`:
```
----- NATIVE  (...) -----
Weighted per-position acceptance rates: [0.79 0.59 0.44]
Conditional acceptance rates:           [0.79 0.74 0.75]
----- TRAINED (...) -----
Weighted per-position acceptance rates: [0.85 0.69 0.55]   ← 训好的应更高
```
原始日志在 `npu_mtp_accept/results/<时间戳>/{native,trained}/`(`vllm_server.log` +
`acceptance_analysis.txt` + `guidellm_*`)。

## 可调 env
| env | 默认 | 说明 |
|---|---|---|
| `NATIVE_MTP_MODEL` / `TRAINED_MTP_MODEL` | (必填) | 两个对比模型 |
| `NATIVE_MTP_DRAFT` / `TRAINED_MTP_DRAFT` | 空 | MTP 是独立 speculator 时填 |
| `NUM_SPEC_TOKENS` | 3 | = 训练的 MTP 深度 |
| `DATASET` | HF math_reasoning | 内网改本地 jsonl |
| `TP` / `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | 1 / 8192 / 0.85 | |
| `TEMP` | 0.6 | harness 默认采样 |

## 注意
- 接受率从 server 日志的 `SpecDecoding metrics: ... Per-position acceptance rate: ...` 抽,
  所以必须有真实流量(guidellm 提供)。两份都没有该行 → 检查 spec config / 模型是否带 MTP。
- 接受率只反映 draft 质量,与吞吐/掩盖无关;tok/s 对比是另一个实验。
- 两次 serve 用同端口、串行跑;脚本每步后 `pkill -f vllm`。
