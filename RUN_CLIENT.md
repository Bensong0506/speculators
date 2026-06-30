# RUN_CLIENT — MTP 增训 on the client's post-SFT 122B

Goal: re-train the MTP draft head so it aligns to the client's **post-SFT** target
distribution. The whole chain is driven by one variable — point everything at the
client's SFT'd weights (`CLIENT_MODEL`) and at client-domain prompts (`CLIENT_DATA`).

- Backbone: **Qwen3.5-122B-A10B** (MoE, 48 layers, heads=32 ⇒ TP ∈ {4,8}, native MTP head).
- Box: 8× A800 80GB. bf16 (fp32 breaks the MTP lm_head).
- Run from the `mtp-training` env (`pip install -e . --no-deps` or `PYTHONPATH=$PWD/src`).

Box: PAI node `lshb-reservedpool-byzo-14`, 8× H800 80GB (reported as L20Y, disguised),
conda. Use the FULL-PRECISION SFT weights (`qwen3.5-vl-122B`), NOT the
msModelSlim-quantized `Qwen3.5-122B-A10B` (GPU vLLM can't load that). Install vLLM
0.22.0 first — see `install/`.

Set these once and export them in your shell:

```bash
export CLIENT_MODEL=/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/qwen3.5-vl-122B
export CLIENT_TRAIN_JSONL=/mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/train.jsonl
```

Client data (`train.jsonl`, 8137 rows) is the 小红书 "问一问" RAG search SFT set —
OpenAI chat shape, one record per line:
```json
{ "messages": [ {"role":"system","content":"<big instruction>"},
                {"role":"user","content":"...<image>...<image>..."},
                {"role":"assistant","content":"<GT answer — DROPPED & regenerated>"} ],
  "images":   ["/abs/path/a.jpg", ...] }   // ordered, one per <image> token; [] if text-only
```

---

## STEP 1 — Data distillation (on-policy, client domain)

Serves the **post-SFT** 122B (TP=8) and regenerates the answers with it. Default is
**TEXT-ONLY** (the task is text-dominated RAG; bootstraps the whole pipeline without
the 20-images/sample + long-context cost). `scripts/distill_client_messages.py` does
the work; the shell script serves + drives it.

```bash
MAX_SAMPLES=8137 bash examples/train/distill_client_122b.sh
# -> /mnt/tidal-alsh01/dataset/pai/zhaofei4/huawei/client_122b_distill_text_8137.jsonl
```

> **Safety (client machine):** the original `train.jsonl` is never modified. STEP 1
> writes into the SAME `…/huawei/` folder as train.jsonl: a read-only (`0444`) copy
> `train_source_copy.jsonl` (we read only the copy) + the distilled
> `client_122b_distill_*.jsonl` + a `logs/` subdir. Distinct names, so they never
> collide with `train.jsonl`; the distiller also hard-refuses if `--out-jsonl`
> resolves to any input path. Override with `WORK_DIR=...`.

Multimodal (phase 2 — needs the image root visible + per-prompt image cap):
```bash
MODE=multimodal IMAGE_MEDIA_ROOT=/mnt/tidal-alsh01 LIMIT_IMAGES=20 \
  MAX_SAMPLES=8137 bash examples/train/distill_client_122b.sh
```

Two unconnected machines (split, then concat):
```bash
SKIP_SAMPLES=0    MAX_SAMPLES=4000 OUT_JSONL=data/client/distill_partA.jsonl bash examples/train/distill_client_122b.sh
SKIP_SAMPLES=4000 MAX_SAMPLES=4137 OUT_JSONL=data/client/distill_partB.jsonl bash examples/train/distill_client_122b.sh
cat data/client/distill_partA.jsonl data/client/distill_partB.jsonl > data/client/client_122b_distill_text_8137.jsonl
```

## STEP 2 — MTP training

Extracts + fine-tunes `CLIENT_MODEL`'s native mtp.* head (verifier vLLM TP=4 on GPUs
0–3, trainer on GPUs 4–7). Smoke-test first:

```bash
# smoke (validates TP layout + MTP extraction on the SFT'd model)
MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=0 \
  CLIENT_DISTILL_JSONL=data/client/client_122b_distill_text_8137.jsonl \
  bash examples/train/nohup_mtp_client_122b.sh
tail -f run_logs/mtp_client_122b_*.nohup.log

# full run
CLIENT_DISTILL_JSONL=data/client/client_122b_distill_text_8137.jsonl \
  EPOCHS=10 LR=3e-5 NUM_SPECULATIVE_STEPS=3 STEP_WEIGHT_BETA=0.6 \
  bash examples/train/nohup_mtp_client_122b.sh
# -> output/mtp_client_122b/<run>/checkpoints/checkpoint_best
```

> Long prompts: the launcher defaults `SEQ_LENGTH=16384` (client RAG prompts run
> ~9k–15k tokens). If STEP 2 OOMs, lower `SEQ_LENGTH` or `NUM_SPECULATIVE_STEPS`
> first (full-vocab logits scale with seq × steps).

## STEP 3 — Test the trained head

Stitches the trained head back into a servable 122B copy (sharded-aware) and compares
the client's **stock SFT** MTP head vs **our trained** head on the client-domain val
tail. Same method/spec/val — only the head weights differ.

```bash
CLIENT_DISTILL_JSONL=data/client/client_122b_distill_text_8137.jsonl \
  MTP_CKPT=output/mtp_client_122b/<run>/checkpoints/checkpoint_best \
  INFER_NUM_SPEC=7 NUM_PROMPTS=128 \
  bash examples/evaluate/test_mtp_client_122b.sh
# -> output/mtp_orig_vs_trained/<stamp>/mtp_orig_vs_trained_summary.md
```

Read **first-pos** and **mean-accept/draft** (tok/s is noisy ±%). Real tokens/step
`L = mean-accept + 1`, `TPOT ∝ 1/L`. If trained ≈ original, the head didn't load —
retry with `TRAINED_MTP_METHOD=mtp`. A small OOD dip on general data is normal
finetune specialization; optionally WiSE-FT-soup it back (see `examples/evaluate/sweep_mtp_soup_alpha.sh`).

---

### What's reused vs new

| Stage | Client script | Notes |
|---|---|---|
| 1 distill | `examples/train/distill_client_122b.sh` + `scripts/distill_client_messages.py` | serves full-precision 122B, regenerates answers (text-only default / multimodal flag) |
| 2 train | `examples/train/nohup_mtp_client_122b.sh` → `nohup_mtp_122b_allava_distilled.sh` | verifier TP=4 GPUs0-3 + trainer GPUs4-7, bf16, SEQ_LENGTH=16384 |
| 3 test | `examples/evaluate/test_mtp_client_122b.sh` → `test_mtp_allava_orig_vs_trained.sh` + `scripts/stitch_mtp.py` | native SFT MTP head vs trained head on client val tail |

The distiller is client-specific (the data is `messages`+`images`, multi-image); the
train/test wrappers thread `CLIENT_MODEL`/paths + 122B TP layout through the already
validated 122B MTP pipeline. Spec-decode caveat: this is a heavy-prefill / light-decode
RAG load, so end-to-end speedup from the draft is diluted by the long retrieval prompts.
