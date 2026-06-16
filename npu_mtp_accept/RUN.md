# NPU MTP 接受率:原生 MTP vs 训好的 MTP (vllm-ascend, serve-based)

`test_result_npu` 分支专用。一个命令跑完:自动 stitch 训好的 MTP 头 → 起服务 → guidellm 压测
→ 从 server 日志抽 per-position 接受率 → native/trained 并排打印。**直接复制下面的块,不用从聊天里抄。**

原理:训好的 MTP head 嵌在 verifier 权重里、不能直接 serve,所以先 `stitch_mtp.py` 缝进基座;
native arm = 基座自带的 MTP 头,trained arm = 缝了你训好头的副本,两边同 method/spec/数据,只差权重。

---

## 0. 一次性:checkout + 依赖
```bash
cd /path/to/speculators
git fetch origin test_result_npu && git checkout test_result_npu && git pull
export ASCEND_RT_VISIBLE_DEVICES=0                 # 多卡 TP 写 0,1,2,3
pip install -q guidellm typer rich huggingface_hub 2>/dev/null || true   # harness + stitch 依赖
chmod +x npu_mtp_accept/*.sh npu_mtp_accept/stitch_mtp.py
```

## 1. 跑(一条命令)
```bash
export MODEL=/home/model/Qwen3.5-9B                # 基座 = native arm + stitch verifier
export TRAINED_MTP_CKPT=/home/wenxuan/.../checkpoint_best   # 你训练产出的原始 MTP ckpt(自动 stitch)
export NUM_SPEC_TOKENS=3                            # = 你训练的 MTP 深度
export TP=1
export DATASET=/home/wenxuan/<本地eval>.jsonl       # 内网不通 HF,必须给本地集
bash npu_mtp_accept/run_mtp_accept_compare.sh
```
- **已经 stitch 过了?** 不记得就当没 stitch(给 `TRAINED_MTP_CKPT` 让它自动缝);若确定有 stitch 好的目录,改成
  `export TRAINED_MTP_MODEL=/path/to/stitched-dir`(跳过 stitch)。
- 重新 stitch:`FORCE_STITCH=1`。缝出来的目录默认在 `npu_mtp_accept/stitched/`。

## 2. 看结果
脚本末尾并排打印,训好的 per-position 接受率应高于原生:
```
----- NATIVE  (...) -----
Weighted per-position acceptance rates: [0.79 0.59 0.44]
----- TRAINED (...) -----
Weighted per-position acceptance rates: [0.85 0.69 0.55]   ← 训好的更高 = 有效
```
原始日志/中间结果:`npu_mtp_accept/results/<时间戳>/{native,trained}/`
(`vllm_server.log`、`acceptance_analysis.txt`、`guidellm_*`)。

## 3. 只想单独 stitch(可选)
```bash
python npu_mtp_accept/stitch_mtp.py /path/to/finetuned-mtp /home/model/Qwen3.5-9B \
  --output-path npu_mtp_accept/stitched/my-trained-stitched
# 然后:export TRAINED_MTP_MODEL=npu_mtp_accept/stitched/my-trained-stitched && bash ...compare.sh
```

## 可调 env
| env | 默认 | 说明 |
|---|---|---|
| `MODEL` | (必填) | 基座 9B = native arm + stitch verifier |
| `TRAINED_MTP_CKPT` | — | 原始训练 ckpt(自动 stitch) |
| `TRAINED_MTP_MODEL` | — | 已 stitch 的目录(给了就跳过 stitch) |
| `NUM_SPEC_TOKENS` | 3 | = 训练的 MTP 深度 |
| `MTP_METHOD` | mtp | trained≈native 时改 `qwen3_5_mtp` 重试 |
| `DATASET` | HF math_reasoning | 内网改本地 jsonl |
| `TP` / `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | 1 / 8192 / 0.85 | |
| `FORCE_STITCH` | — | =1 重新 stitch |

## 注意
- **sanity**:若 trained ≈ native,多半 stitch 的头没被 vLLM 加载 → `export MTP_METHOD=qwen3_5_mtp` 重跑。
- 接受率从 `vllm_server.log` 的 `SpecDecoding metrics: ... Per-position acceptance rate:` 抽,必须有真实流量(guidellm 提供)。
- 接受率只反映 draft 质量,与吞吐/掩盖无关;tok/s 是另一个实验。
- `stitch_mtp.py` 是自包含版(remap 表已内联),不依赖 `speculators.convert.mtp`;只需 torch/safetensors/typer/rich。
- 每个 serve 前后都 `pkill -f vllm`;两 arm 串行、同端口。
