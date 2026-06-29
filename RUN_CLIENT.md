# RUN_CLIENT — MTP 增训 on the client's post-SFT 122B

Goal: re-train the MTP draft head so it aligns to the client's **post-SFT** target
distribution. The whole chain is driven by one variable — point everything at the
client's SFT'd weights (`CLIENT_MODEL`) and at client-domain prompts (`CLIENT_DATA`).

- Backbone: **Qwen3.5-122B-A10B** (MoE, 48 layers, heads=32 ⇒ TP ∈ {4,8}, native MTP head).
- Box: 8× A800 80GB. bf16 (fp32 breaks the MTP lm_head).
- Run from the `mtp-training` env (`pip install -e . --no-deps` or `PYTHONPATH=$PWD/src`).

Set these once and export them in your shell:

```bash
export CLIENT_MODEL=/data/wenxuan/Qwen3.5-122B-A10B-sft     # post-SFT 122B weights
export CLIENT_DATA="/data/client/domain_train.json"          # LLaVA/ALLaVA-style prompts
export CLIENT_IMAGE_ROOT=/data/client/images                 # root for record `image` paths
```

`CLIENT_DATA` schema (a JSON list, or `{"data": [...]}`):
```json
{ "image": "rel/or/abs/path.jpg",
  "conversations": [ {"from": "human", "value": "<image>\nQ..."},
                     {"from": "gpt",   "value": "orig answer — DROPPED & regenerated"} ] }
```

---

## STEP 1 — Data distillation (on-policy, client domain)

Serves the **post-SFT** 122B (TP=8, all GPUs, greedy) and regenerates the answers so
the MTP head learns *this* model's own continuations.

```bash
MAX_SAMPLES=10000 bash examples/train/distill_client_122b.sh
# -> data/client/client_122b_distill_10k.jsonl
```

Two unconnected machines (50k each, then concat):
```bash
# machine A
SKIP_SAMPLES=0     MAX_SAMPLES=50000 OUT_JSONL=data/client/client_122b_distill_partA.jsonl \
  bash examples/train/distill_client_122b.sh
# machine B
SKIP_SAMPLES=50000 MAX_SAMPLES=50000 OUT_JSONL=data/client/client_122b_distill_partB.jsonl \
  bash examples/train/distill_client_122b.sh
# then
cat data/client/client_122b_distill_partA.jsonl data/client/client_122b_distill_partB.jsonl \
  > data/client/client_122b_distill_100k.jsonl
```

## STEP 2 — MTP training

Extracts + fine-tunes `CLIENT_MODEL`'s native mtp.* head (verifier vLLM TP=4 on GPUs
0–3, trainer on GPUs 4–7). Smoke-test first:

```bash
# smoke (5 min): validates TP layout + MTP extraction on the SFT'd model
MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=0 \
  CLIENT_DISTILL_JSONL=data/client/client_122b_distill_10k.jsonl \
  bash examples/train/nohup_mtp_client_122b.sh
tail -f run_logs/mtp_client_122b_*.nohup.log

# full run
CLIENT_DISTILL_JSONL=data/client/client_122b_distill_10k.jsonl \
  EPOCHS=10 LR=3e-5 NUM_SPECULATIVE_STEPS=3 STEP_WEIGHT_BETA=0.6 \
  bash examples/train/nohup_mtp_client_122b.sh
# -> output/mtp_client_122b/<run>/checkpoints/checkpoint_best
```

## STEP 3 — Test the trained head

Stitches the trained head back into a servable 122B copy (sharded-aware) and compares
the client's **stock SFT** MTP head vs **our trained** head on the client-domain val
tail. Same method/spec/val — only the head weights differ.

```bash
CLIENT_DISTILL_JSONL=data/client/client_122b_distill_10k.jsonl \
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

| Stage | New (client wrapper) | Wraps (validated) |
|---|---|---|
| 1 distill | `examples/train/distill_client_122b.sh` | `distill_allava_122b.sh` → `scripts/distill_allava_with_qwen.py` |
| 2 train | `examples/train/nohup_mtp_client_122b.sh` | `nohup_mtp_122b_allava_distilled.sh` → `dflash_qwen3.5_9b_multimodal_online.sh` |
| 3 test | `examples/evaluate/test_mtp_client_122b.sh` | `test_mtp_allava_orig_vs_trained.sh` + `scripts/stitch_mtp.py` |

The wrappers only swap in `CLIENT_MODEL` / `CLIENT_DATA` / `CLIENT_IMAGE_ROOT` and the
122B TP layout — all underlying logic is the already-validated 122B MTP pipeline.
