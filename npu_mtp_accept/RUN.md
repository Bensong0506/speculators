# NPU MTP 接受率:原生 MTP vs 训好的 MTP (多模态 ALLaVA + MMStar)

`test_result_npu` 专用。**用的就是你 test_result 里跑 ALLaVA/MMStar 的方式**:多模态 serve
(`qwen3_5_mtp` + `--allowed-local-media-path` + `--limit-mm-per-prompt '{"image":1}'`)
→ `mmstar_weight_client.py` 发图文请求、读 `/metrics` → `{arm}_summary.json`。
不是 guidellm 文本集(那个已删)。一条命令跑完,复制即可,不用从聊天抄。

native 臂 = serve 基座自带 MTP 头;trained 臂 = serve stitch 后的目录(脚本自动 stitch)。

---

## 0. checkout + 依赖
```bash
cd /path/to/speculators
git fetch origin test_result_npu && git checkout test_result_npu && git pull
export ASCEND_RT_VISIBLE_DEVICES=0
pip install -q typer rich huggingface_hub 2>/dev/null || true   # stitch 用(vllm/torch 已有)
chmod +x npu_mtp_accept/*.sh npu_mtp_accept/stitch_mtp.py
```

## 1. 备数据(若还没有 jsonl)
```bash
# ALLaVA:用你训练同款的 val 尾巴,或从 ALLaVA json 转
python3 scripts/llava_to_jsonl.py --in /home/wenxuan/ALLaVA-4V/...json \
  --image-root /home/wenxuan/ALLaVA-4V --out-jsonl data/allava/allava_val.jsonl
# MMStar(可选):从 mmstar_answers.json 转
python3 scripts/mmstar_to_jsonl.py   # 产 data/mmstar/mmstar.jsonl(详见脚本 -h)
```

## 2. 跑(一条命令,自动 stitch + 两数据集 native/trained 各一臂)
```bash
export MODEL=/home/model/Qwen3.5-9B                       # 基座 = native 臂
export TRAINED_MTP_CKPT=/home/wenxuan/.../checkpoint_best # 原始训练 ckpt(自动 stitch)
export ALLAVA_JSONL=$PWD/data/allava/allava_qwen35_distill_100k.jsonl  # 完整集,自动切后 10% 当 val(无泄漏)
export ALLAVA_IMAGE_ROOT=/home/wenxuan/ALLaVA-4V          # = --allowed-local-media-path
# 已有现成 val 则改给 ALLAVA_VAL_JSONL 跳过切分;比例 VAL_RATIO=0.1(默认)
# 可选 OOD:
export MMSTAR_JSONL=$PWD/data/mmstar/mmstar.jsonl
export MMSTAR_IMAGE_ROOT=/home/wenxuan/mmstar/images
export NUM_SPEC_TOKENS=7 TP=1 GPUS=0
bash npu_mtp_accept/run_mtp_accept_compare.sh
```
- 已 stitch 过的目录 → 改设 `TRAINED_MTP_MODEL=/path/to/stitched`,跳过 stitch。
- 路径在 `/data/wenxuan` 还是 `/home/wenxuan` 按你那台机器改(memory 里那个老坑)。

## 3. 看结果
脚本末尾打印对比表(训好的应更高):
```
=== allava ===
metric                                       native   trained    ratio
spec_mean_accepted_tokens_per_draft           2.916     3.142    1.077
spec_token_acceptance_rate                    0.417     0.449    1.077
spec_first_position_acceptance_rate           0.838     0.844    1.007
output_tok_per_sec                            66.4      70.4     1.060
=== mmstar ===   (OOD,若开了)
...
```
每臂的 `*_summary.json` / `*_responses.jsonl` / `*_vllm.log` 在 `npu_mtp_accept/results/<时间戳>/`;
汇总也复制进了 `output_log_debug/` 方便回传。

## 可调 env
| env | 默认 | 说明 |
|---|---|---|
| `MODEL` | (必填) | 基座 9B = native 臂 + stitch verifier |
| `TRAINED_MTP_CKPT` / `TRAINED_MTP_MODEL` | (二选一) | 原始 ckpt(自动 stitch)/ 已 stitch 目录 |
| `ALLAVA_VAL_JSONL` / `ALLAVA_IMAGE_ROOT` | (必填) | ALLaVA 图文 jsonl / 图片根 |
| `MMSTAR_JSONL` / `MMSTAR_IMAGE_ROOT` | 空 | 设了才跑 MMStar(OOD) |
| `MTP_METHOD` | qwen3_5_mtp | trained≈native 时换 `mtp` |
| `NUM_SPEC_TOKENS` | 7 | = 训练 MTP 深度 |
| `NUM_PROMPTS` / `MAX_TOKENS` | 128 / 256 | |
| `TP` / `GPUS` / `PORT` | 1 / 0 / 8100 | |

## 注意
- **代理坑已修**:脚本里 `export no_proxy/NO_PROXY=localhost,127.0.0.1,::1`,所以 `/health` 和客户端不会再走代理超时。
- **sanity**:trained ≈ native → stitch 的头没被 vLLM 加载 → `export MTP_METHOD=mtp` 重跑。
- 接受率来自 server `/metrics` 的 spec_decode delta(`mmstar_weight_client.py` 抓);要有真实流量。
- 图片必须在 `--allowed-local-media-path`(=`ALLAVA_IMAGE_ROOT`/`MMSTAR_IMAGE_ROOT`)下,否则 404、接受率全 0。
- 每臂前后 `pkill -f vllm`,串行同端口;`--enforce-eager`(接受率与图模式无关)。
