#!/bin/bash
# 3-way MTP acceptance compare: native vs PREV-trained vs NEW-trained.
#
# 用途:对比「原生 MTP 头」/「之前训的 MTP(teacher-forcing)」/「这次训的 MTP
# (self-forcing)」三者的接受率,看 self-forcing 有没有抬中后位。三臂跑 *同一个*
# spec 方法(qwen3_5_mtp)、*同一* spec 步数、*同一* val(ALLaVA 后 10% 尾巴
# [+ 可选 MMStar OOD]),唯一差别是 MTP 头权重。
#
# ⚠️ confound 提醒:若你的 PREV 是 beta=0.6/sf=0、NEW 是 beta=1.0/sf=0.5,则
#    new-prev 同时混了 beta 与 self-forcing 两个变量。要干净隔离 self-forcing,
#    PREV 应是同 beta、sf=0 的对照(beta=1.0/sf=0)。没有也能看"新配方整体好不好"。
#
# USAGE
#   MODEL=/home/wenxuan/Qwen3.5-9B \
#   PREV_MTP_CKPT=$PWD/output/mtp_qwen3.5_9b_mm_distilled/<prev_run>/checkpoints/checkpoint_best \
#   NEW_MTP_CKPT=$PWD/output/mtp_qwen3.5_9b_mm_distilled/<new_sf_run>/checkpoints/checkpoint_best \
#   ALLAVA_JSONL=$PWD/data/allava/allava_qwen35_distill_100k.jsonl \
#   ALLAVA_IMAGE_ROOT=/home/wenxuan/ALLaVA-4V \
#   MMSTAR_JSONL=$PWD/data/mmstar/mmstar.jsonl MMSTAR_IMAGE_ROOT=/home/wenxuan/mmstar/images \
#   INFER_NUM_SPEC=7 GPUS=0 \
#   bash examples/evaluate/compare_mtp_3way.sh
#
#   - ALLAVA_JSONL 必须是 NEW/PREV 训练所用的同一份(后 10% val 才对得上)。
#   - 训的是 steps=5 -> 想看"匹配深度"再补跑一遍 INFER_NUM_SPEC=5。
#   - MMStar 不设就只跑 ALLaVA(in-domain)。
# OUTPUT: output/mtp_3way/<stamp>/mtp_3way_summary.md (+ 每 arm/数据集 *_summary.json)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

# ---- paths (mirror test_mtp_allava_orig_vs_trained.sh defaults) ----
DEFAULT_ROOT="${DEFAULT_ROOT:-/home/wenxuan}"
[ ! -d "$DEFAULT_ROOT/Qwen3.5-9B" ] && [ -d /data/wenxuan/Qwen3.5-9B ] && DEFAULT_ROOT="/data/wenxuan"
MODEL="${MODEL:-$DEFAULT_ROOT/Qwen3.5-9B}"
ALLAVA_IMAGE_ROOT="${ALLAVA_IMAGE_ROOT:-$DEFAULT_ROOT/ALLaVA-4V}"
PREV_MTP_CKPT="${PREV_MTP_CKPT:?set PREV_MTP_CKPT=/abs/.../checkpoint_best (之前 teacher-forcing 的 MTP)}"
NEW_MTP_CKPT="${NEW_MTP_CKPT:?set NEW_MTP_CKPT=/abs/.../checkpoint_best (这次 self-forcing 的 MTP)}"
ALLAVA_JSONL="${ALLAVA_JSONL:-$REPO_ROOT/data/allava/allava_qwen35_distill_100k.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
ALLAVA_VAL_JSONL="${ALLAVA_VAL_JSONL:-$REPO_ROOT/data/allava/$(basename "${ALLAVA_JSONL%.jsonl}")_val_tail10pct.jsonl}"
MMSTAR_JSONL="${MMSTAR_JSONL:-}"; MMSTAR_IMAGE_ROOT="${MMSTAR_IMAGE_ROOT:-}"

# ---- serve knobs (mirror the 9B 2-arm script) ----
GPUS="${GPUS:-0}"; TP="${TP:-1}"; PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"; DTYPE="${DTYPE:-bfloat16}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"; MAX_TOKENS="${MAX_TOKENS:-128}"
INFER_NUM_SPEC="${INFER_NUM_SPEC:-7}"; MTP_METHOD="${MTP_METHOD:-qwen3_5_mtp}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"; DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"; FORCE_STITCH="${FORCE_STITCH:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_MODEL_LEN + MAX_NUM_SEQS * INFER_NUM_SPEC))}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/output/mtp_3way/$STAMP}"; mkdir -p "$RUN_DIR"

CLIENT="examples/evaluate/mmstar_weight_client.py"
[ -f "$CLIENT" ] || { echo "[fatal] 缺 $CLIENT"; exit 1; }
for p in "$MODEL" "$PREV_MTP_CKPT" "$NEW_MTP_CKPT" "$ALLAVA_IMAGE_ROOT"; do
  [ -e "$p" ] || { echo "[fatal] 路径不存在: $p"; exit 1; }
done
[ -s "$ALLAVA_JSONL" ] || { echo "[fatal] ALLAVA_JSONL 空/缺: $ALLAVA_JSONL"; exit 1; }

# ---- stitch prev + new (两个 ckpt 路径不同 -> stitch 目录不同,不会撞名) ----
stitch_one() { # $1 ckpt  $2 out_dir
  local ckpt="$1" out="$2"
  if [ "$FORCE_STITCH" != "1" ] && [ -f "$out/config.json" ] && \
     { [ -f "$out/model.safetensors" ] || [ -f "$out/model.safetensors.index.json" ]; }; then
    echo "[stitch] 复用 $out (FORCE_STITCH=1 重建)"; return 0
  fi
  echo "[stitch] $ckpt + $MODEL -> $out"; rm -rf "$out"
  python3 scripts/stitch_mtp.py "$ckpt" "$MODEL" --output-path "$out" || { echo "[fatal] stitch 失败: $ckpt"; exit 1; }
}
STITCHED_PREV="${STITCHED_PREV:-$REPO_ROOT/output/mtp_stitched/3way_prev_$(basename "$(dirname "$(dirname "$PREV_MTP_CKPT")")")}"
STITCHED_NEW="${STITCHED_NEW:-$REPO_ROOT/output/mtp_stitched/3way_new_$(basename "$(dirname "$(dirname "$NEW_MTP_CKPT")")")}"
stitch_one "$PREV_MTP_CKPT" "$STITCHED_PREV"
stitch_one "$NEW_MTP_CKPT"  "$STITCHED_NEW"

# ---- ALLaVA val tail (后 VAL_RATIO,无泄漏) ----
python3 - "$ALLAVA_JSONL" "$ALLAVA_VAL_JSONL" "$VAL_RATIO" <<'PY'
import sys; from pathlib import Path
src, dst, ratio = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
if not lines: raise SystemExit(f"[fatal] empty: {src}")
val = lines[int(len(lines)*(1-ratio)):]
if not val: raise SystemExit("[fatal] val tail empty")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(val)+"\n", encoding="utf-8")
print(f"[info] ALLaVA val tail -> {dst} rows={len(val)} (of {len(lines)})")
PY

# ---- serve / client ----
SERVER_PID=""
cleanup(){ [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }; pkill -f vllm 2>/dev/null || true; sleep 3; }
trap cleanup EXIT
serve(){ # $1 model  $2 media  $3 log
  cleanup
  local args=( vllm serve "$1" --served-model-name mtp-3way --seed 42 --trust-remote-code
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" --dtype "$DTYPE"
    --allowed-local-media-path "$2" --limit-mm-per-prompt '{"image":1}'
    --generation-config vllm --host 0.0.0.0 --port "$PORT"
    --speculative-config "{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$INFER_NUM_SPEC,\"enforce_eager\":true}" )
  [ "$ENFORCE_EAGER" = "1" ] && args+=(--enforce-eager)
  [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
  [ "$DISABLE_CHUNKED_PREFILL" = "1" ] && args+=(--no-enable-chunked-prefill)
  env CUDA_VISIBLE_DEVICES="$GPUS" "${args[@]}" > "$3" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "  ready"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  [server 挂] tail:"; tail -n 60 "$3"; return 1; }
    sleep 5
  done; echo "  [timeout]"; tail -n 80 "$3"; return 1
}
run_arm(){ # $1 tag(ds_arm)  $2 model  $3 media  $4 jsonl
  echo "==== $1 (model=$2) ===="
  serve "$2" "$3" "$RUN_DIR/$1_vllm.log" || { echo '{}' > "$RUN_DIR/$1_summary.json"; return 0; }
  python3 "$CLIENT" --endpoint "http://localhost:$PORT/v1" --data-jsonl "$4" \
    --out-jsonl "$RUN_DIR/$1_responses.jsonl" --summary-json "$RUN_DIR/$1_summary.json" \
    --num "$NUM_PROMPTS" --max-tokens "$MAX_TOKENS" || echo "[warn] $1 bench 失败"
  cleanup
}

# ---- run matrix: dataset outer (media 不同需重起), arm inner ----
declare -a DS_NAMES DS_JSONL DS_MEDIA
DS_NAMES+=("allava"); DS_JSONL+=("$ALLAVA_VAL_JSONL"); DS_MEDIA+=("$ALLAVA_IMAGE_ROOT")
if [ -n "$MMSTAR_JSONL" ] && [ -n "$MMSTAR_IMAGE_ROOT" ]; then
  DS_NAMES+=("mmstar"); DS_JSONL+=("$MMSTAR_JSONL"); DS_MEDIA+=("$MMSTAR_IMAGE_ROOT")
fi
for i in "${!DS_NAMES[@]}"; do
  ds="${DS_NAMES[$i]}"; jsonl="${DS_JSONL[$i]}"; media="${DS_MEDIA[$i]}"
  run_arm "${ds}_native" "$MODEL"          "$media" "$jsonl"
  run_arm "${ds}_prev"   "$STITCHED_PREV"  "$media" "$jsonl"
  run_arm "${ds}_new"    "$STITCHED_NEW"   "$media" "$jsonl"
done

# ---- summary ----
python3 - "$RUN_DIR" "$PREV_MTP_CKPT" "$NEW_MTP_CKPT" "$INFER_NUM_SPEC" <<'PY' | tee "$RUN_DIR/mtp_3way_summary.stdout.txt"
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); prev_ck, new_ck, spec = sys.argv[2], sys.argv[3], sys.argv[4]
def load(tag):
    try: return json.loads((run / f"{tag}_summary.json").read_text())
    except Exception: return {}
def f(x): return f"{x:.4f}" if isinstance(x, float) else ("n/a" if x is None else str(x))
def d(a,b): return (a-b) if (isinstance(a,(int,float)) and isinstance(b,(int,float))) else None
def r(a,b): return (a/b) if (isinstance(a,(int,float)) and isinstance(b,(int,float)) and b) else None
def bypos(x):
    v = x.get("spec_accepted_tokens_by_position")
    if isinstance(v, list): return v
    if isinstance(v, dict): return [v[k] for k in sorted(v, key=lambda z:int(z))]
    return None
KEYS = [("first-pos","spec_first_position_acceptance_rate"),
        ("mean-accept/draft","spec_mean_accepted_tokens_per_draft"),
        ("token-accept","spec_token_acceptance_rate"),
        ("tok/s","output_tok_per_sec")]
md = [f"# MTP 3-way: native vs prev vs new (self-forcing)  ·  spec={spec}", "",
      f"prev ckpt: `{prev_ck}`  ", f"new  ckpt: `{new_ck}`  ",
      "> 关键看 **new vs prev** 的 mean-accept 和中后位 per-position;both vs native 看相对原生。", ""]
present = [ds for ds in ("allava","mmstar") if load(f"{ds}_new") or load(f"{ds}_native")]
for ds in present:
    nat, prv, new = load(f"{ds}_native"), load(f"{ds}_prev"), load(f"{ds}_new")
    md += [f"## {ds}",
           f"requests: native {f(nat.get('completed'))}/{f(nat.get('num_requested'))}, "
           f"prev {f(prv.get('completed'))}/{f(prv.get('num_requested'))}, "
           f"new {f(new.get('completed'))}/{f(new.get('num_requested'))}", "",
           "| metric | native | prev | new | new−prev | new/prev | new/native |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for label,k in KEYS:
        a,b,c = nat.get(k), prv.get(k), new.get(k)
        md.append(f"| {label} | {f(a)} | {f(b)} | {f(c)} | {f(d(c,b))} | {f(r(c,b))} | {f(r(c,a))} |")
    pn,pp,pc = bypos(nat), bypos(prv), bypos(new)
    if pp or pc:
        md += ["", "### accepted tokens by position (counts; 看 new−prev 哪几位涨)",
               "| pos | native | prev | new | new−prev |", "|---:|---:|---:|---:|---:|"]
        n = max(len(pn or []), len(pp or []), len(pc or []))
        for j in range(n):
            av = pn[j] if pn and j<len(pn) else None
            bv = pp[j] if pp and j<len(pp) else None
            cv = pc[j] if pc and j<len(pc) else None
            md.append(f"| {j} | {f(av)} | {f(bv)} | {f(cv)} | {f(d(cv,bv))} |")
    # verdict per ds
    ma, mb, mc = nat.get("spec_mean_accepted_tokens_per_draft"), prv.get("spec_mean_accepted_tokens_per_draft"), new.get("spec_mean_accepted_tokens_per_draft")
    md += ["", "**判断**: "]
    if isinstance(mc,float) and isinstance(mb,float):
        if abs(mc-mb) < 1e-3:
            md[-1] += f"new ≈ prev(mean {f(mc)} vs {f(mb)}) → self-forcing 没动接受率(或头没加载,查 trained≈native)。"
        elif mc > mb:
            md[-1] += f"new > prev(mean {f(mc)} vs {f(mb)},+{f(d(mc,mb))})→ self-forcing 抬了接受率。"
        else:
            md[-1] += f"new < prev(mean {f(mc)} vs {f(mb)})→ self-forcing 没帮上(看是否 p 太大/val 抖)。"
    else:
        md[-1] += "指标不全,看 *_summary.json / *_vllm.log。"
    md += [""]
md += ["---", "⚠️ confound:若 prev 是 beta=0.6/sf=0、new 是 beta=1.0/sf=0.5,则 new−prev 混了 beta 与 self-forcing;"
       "要干净隔离请用同 beta、sf=0 的对照。"]
out = run / "mtp_3way_summary.md"; out.write_text("\n".join(md)+"\n", encoding="utf-8")
print("\n".join(md)); print(f"\n[written] {out}")
PY
echo; echo "Artifacts: $RUN_DIR/mtp_3way_summary.md  (+ {allava,mmstar}_{native,prev,new}_summary.json)"
