#!/usr/bin/env bash
# 原生 MTP vs 训好的 MTP —— 接受率对比 (NPU), 用 test_result 的多模态方式:
#   多模态 serve (qwen3_5_mtp + --allowed-local-media-path + --limit-mm-per-prompt '{"image":1}')
#   + mmstar_weight_client.py 客户端 -> {arm}_summary.json
#     (spec_mean_accepted_tokens_per_draft / spec_token_acceptance_rate /
#      spec_first_position_acceptance_rate / output_tok_per_sec)
# native 臂 = serve 基座(自带 MTP 头);trained 臂 = serve stitch 后的目录。
# 数据集:ALLaVA(in-domain,必)+ MMStar(OOD,设了 MMSTAR_JSONL 才跑)。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"; cd "$REPO_ROOT"
CLIENT="examples/evaluate/mmstar_weight_client.py"

# 本机绕过代理(否则 /health 和客户端命中 localhost 会走代理 -> 超时,见上次的坑)
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

: "${MODEL:?set MODEL=/abs/path/to/Qwen3.5-9B (基座 = native 臂 + stitch verifier)}"
: "${ALLAVA_VAL_JSONL:?set ALLAVA_VAL_JSONL=/abs/path/to/allava_val.jsonl (图文对话 jsonl)}"
: "${ALLAVA_IMAGE_ROOT:?set ALLAVA_IMAGE_ROOT=/abs/path (= --allowed-local-media-path,图片根目录)}"
command -v vllm >/dev/null || { echo "[ERROR] 环境里没有 vllm"; exit 1; }
[ -f "$CLIENT" ] || { echo "[ERROR] 缺 $CLIENT (checkout test_result_npu?)"; exit 1; }

# --- trained 臂模型:已 stitch 的目录优先;否则自动 stitch ---
if [ -z "${TRAINED_MTP_MODEL:-}" ]; then
  : "${TRAINED_MTP_CKPT:?set TRAINED_MTP_CKPT=/abs/path/to/finetuned-mtp (原始 ckpt;已 stitch 则改设 TRAINED_MTP_MODEL)}"
  STITCH_OUT="${STITCH_OUT:-$HERE/stitched/$(basename "$TRAINED_MTP_CKPT")-stitched}"
  if [ -d "$STITCH_OUT" ] && [ -z "${FORCE_STITCH:-}" ]; then
    echo "[stitch] 复用 $STITCH_OUT (FORCE_STITCH=1 重建)"
  else
    echo "[stitch] $TRAINED_MTP_CKPT + $MODEL -> $STITCH_OUT"
    python3 "$HERE/stitch_mtp.py" "$TRAINED_MTP_CKPT" "$MODEL" --output-path "$STITCH_OUT"
  fi
  TRAINED_MTP_MODEL="$STITCH_OUT"
fi

MTP_METHOD="${MTP_METHOD:-qwen3_5_mtp}"      # trained≈native 时换 mtp 重试
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"; MAX_TOKENS="${MAX_TOKENS:-256}"
TP="${TP:-1}"; GPUS="${GPUS:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"; PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"; MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"; GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"; DTYPE="${DTYPE:-bfloat16}"
MMSTAR_JSONL="${MMSTAR_JSONL:-}"; MMSTAR_IMAGE_ROOT="${MMSTAR_IMAGE_ROOT:-}"
SPEC="{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"enforce_eager\":true}"

STAMP="$(date +%Y%m%d_%H%M%S)"; RUN="${RUN_DIR:-$HERE/results/$STAMP}"; mkdir -p "$RUN"
echo "[INFO] native=$MODEL"; echo "[INFO] trained=$TRAINED_MTP_MODEL"
echo "[INFO] method=$MTP_METHOD spec=$NUM_SPEC_TOKENS prompts=$NUM_PROMPTS run=$RUN"

cleanup(){ pkill -f vllm 2>/dev/null || true; sleep 5; }
serve(){ # $1 model  $2 media_root  $3 logfile
  cleanup
  env ASCEND_RT_VISIBLE_DEVICES="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS" \
    vllm serve "$1" --seed 42 --trust-remote-code \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" --dtype "$DTYPE" --enforce-eager \
    --allowed-local-media-path "$2" --limit-mm-per-prompt '{"image":1}' \
    --generation-config vllm --no-enable-chunked-prefill --host 0.0.0.0 --port "$PORT" \
    --speculative-config "$SPEC" > "$3" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "  ready"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  [server 挂了] tail:"; tail -n 60 "$3"; return 1; }
    sleep 5
  done
  echo "  [timeout] tail:"; tail -n 80 "$3"; return 1
}
run_arm(){ # $1 arm  $2 model  $3 media  $4 dataset
  echo "==== $1 (model=$2) ===="
  serve "$2" "$3" "$RUN/$1_vllm.log" || { echo "[skip] $1 serve 失败"; return 1; }
  python3 "$CLIENT" --endpoint "http://localhost:$PORT/v1" --data-jsonl "$4" \
    --out-jsonl "$RUN/$1_responses.jsonl" --summary-json "$RUN/$1_summary.json" \
    --num "$NUM_PROMPTS" --max-tokens "$MAX_TOKENS" || echo "[warn] $1 bench 失败 (看 $RUN/$1_vllm.log)"
  cleanup
}

run_arm allava_native  "$MODEL"             "$ALLAVA_IMAGE_ROOT" "$ALLAVA_VAL_JSONL"
run_arm allava_trained "$TRAINED_MTP_MODEL" "$ALLAVA_IMAGE_ROOT" "$ALLAVA_VAL_JSONL"
if [ -n "$MMSTAR_JSONL" ] && [ -n "$MMSTAR_IMAGE_ROOT" ]; then
  run_arm mmstar_native  "$MODEL"             "$MMSTAR_IMAGE_ROOT" "$MMSTAR_JSONL"
  run_arm mmstar_trained "$TRAINED_MTP_MODEL" "$MMSTAR_IMAGE_ROOT" "$MMSTAR_JSONL"
fi

python3 - "$RUN" <<'PY'
import json, os, sys
run = sys.argv[1]
keys = ["spec_mean_accepted_tokens_per_draft","spec_token_acceptance_rate",
        "spec_first_position_acceptance_rate","output_tok_per_sec","completed","num_requested"]
def load(a):
    p = os.path.join(run, f"{a}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
def f(x): return f"{x:.3f}" if isinstance(x, float) else ("n/a" if x is None else str(x))
print("\n################ 原生 MTP vs 训好的 MTP 接受率 ################")
for ds in ["allava", "mmstar"]:
    n, t = load(f"{ds}_native"), load(f"{ds}_trained")
    if not n and not t: continue
    print(f"\n=== {ds} ===\n{'metric':42s}{'native':>10s}{'trained':>10s}{'ratio':>9s}")
    for k in keys:
        nv, tv = (n or {}).get(k), (t or {}).get(k)
        r = (tv/nv) if (isinstance(nv,(int,float)) and isinstance(tv,(int,float)) and nv) else None
        print(f"{k:42s}{f(nv):>10s}{f(tv):>10s}{(f'{r:.3f}' if r else 'n/a'):>9s}")
print("\n[sanity] trained ≈ native => stitch 的头没被加载,换 MTP_METHOD=mtp 重跑")
PY
# 回传(内网只能 pull):汇总丢进 output_log_debug
cp -f "$RUN"/*_summary.json output_log_debug/ 2>/dev/null || true
echo "[INFO] 结果/日志: $RUN/"
