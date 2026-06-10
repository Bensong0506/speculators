# DFlash CE Fine-tune — Results Report (2026-06-10)

## TL;DR

Switching the continue-training loss from `kl_div` to **`ce`** (cross-entropy on
the verifier's argmax) fixed the long-standing failure where the trained DFlash
draft *lost* to the original open-source draft. The CE `checkpoint_best`:

- **In-domain (ALLaVA val): modest WIN over original DFlash** — mean-accept 1.99
  vs 1.94 (+2.6%), tok/s 64.2 vs 62.8 (+2.2%), first-pos tied (0.727 vs 0.729).
- **OOD (MMStar): ≈ TIE with original** — mean-accept 2.085 vs 2.088, first-pos
  0.752 vs 0.764. No catastrophic forgetting (kl_div used to regress to ~0.87x).
- **vs MTP: MTP still has clearly higher acceptance** (mean-accept ~2.8 on both
  sets, +35–40% over DFlash; first-pos ~+10%), **but trained DFlash matches MTP on
  throughput** because the DFlash draft is far cheaper per step (tok/s within
  ±3–5%, noisy across runs).

Project bar — beat the open-source DFlash draft in-domain — is **met (modestly)**.
The remaining gap to MTP is an acceptance-quality gap, concentrated at **first
position**, which did not move.

## Setup

- Verifier: Qwen3.5-9B. Draft: DFlash, warm-started from `z-lab/Qwen3.5-9B-DFlash`,
  continue-trained on 10k Qwen-distilled ALLaVA (`allava_qwen35_distill_10k.jsonl`).
- Decisive change: `LOSS_FN=ce` (was `kl_div`). Checkpoint: `checkpoint_best`
  (selected by min val loss; under CE, val loss tracks top-1).
- Eval: vLLM speculative decoding, `num_speculative_tokens=7`, 128 prompts,
  temp 0; acceptance read from vLLM `/metrics` (`spec_decode_*`).
- ALLaVA val = in-domain (distilled val tail, same split as training);
  MMStar = OOD generalization / forgetting check.

## Results

### ALLaVA val (in-domain), @spec7

| method | tok/s | mean accept/draft | token accept | first-pos |
|---|---:|---:|---:|---:|
| MTP@7 | 65.997 | **2.807** | 0.401 | **0.818** |
| **trained DFlash@7** | 64.197 | 1.993 | 0.285 | 0.727 |
| original DFlash@7 | 62.821 | 1.942 | 0.277 | 0.729 |

- trained vs original: mean-accept **1.026x**, tok/s **1.022x**, first-pos 0.997x (tie).
- trained vs MTP: mean-accept 0.710x, first-pos 0.889x, tok/s 0.973x.

### MMStar (OOD), @spec7 — single run, three-way

| method | tok/s | mean accept/draft | token accept | first-pos |
|---|---:|---:|---:|---:|
| trained DFlash@7 | 70.316 | 2.085 | 0.298 | 0.752 |
| MTP@7 | 68.074 | **2.822** | 0.403 | **0.828** |
| original DFlash@7 | 66.388 | 2.088 | 0.298 | 0.764 |

- trained vs original: mean-accept **0.999x** (tie), first-pos 0.985x, tok/s 1.059x.
- trained vs MTP: mean-accept 0.739x, first-pos 0.909x, tok/s 1.033x.

**tok/s is noisy.** A separate MMStar 2-way run gave original 69.3 / trained 67.8
(trained slower); this 3-way gave 66.4 / 70.3 (trained faster). Run-to-run tok/s
variance is several %, so small tok/s gaps are not reliable. The stable signals
are first-pos and mean-accept — both say **trained ≈ original on MMStar**.

## Trajectory: kl_div → CE (ALLaVA val)

| metric | kl_div (ep6) | CE (checkpoint_best) | original |
|---|---:|---:|---:|
| first-pos | 0.672 | 0.727 | ~0.729 |
| mean-accept | 1.615 | 1.993 | ~1.942 |
| mean vs original | 0.826x (lose) | **1.026x (win)** | 1.0 |

`kl_div` lowered the soft-distribution loss without moving top-1; acceptance@temp0
only counts top-1, so kl_div drafts always lost. `ce` targets the verifier's
argmax directly → top-1 / acceptance rose, in-domain crossed original, and OOD
stopped regressing.

## Conclusions

1. **CE was the right objective.** It turned a consistent loss into an in-domain
   win + OOD parity — the headline result of this round.
2. **The in-domain win is real but small** (+2–3% mean-accept), and **first-pos
   did not move** (0.727 vs 0.729). The per-block acceptance gate is unchanged;
   the mean-accept gain came from deeper positions only.
3. **MTP is still the stronger drafter on acceptance** (+35–40% mean-accept on
   both sets), **but DFlash matches it on throughput** because DFlash drafting is
   much cheaper per step. Implication: closing DFlash's acceptance gap is
   high-value — a higher-accept DFlash would beat MTP on tok/s outright.
4. **Bottleneck = first-position acceptance.** It is the gate for the whole draft
   block and the main axis where DFlash trails both its own potential and MTP.

## Open levers for the next round (to discuss, not yet decided)

- **LR / schedule:** warm-start LR 1e-5 → 3e-5 + warmup — is first-pos LR-starved?
- **Checkpoint selection:** confirm `checkpoint_best` (val-loss) is also the best
  epoch by real acceptance (the per-epoch ALLaVA sweep would settle this).
- **Target first-pos directly:** position-1-weighted CE, more/larger draft layers,
  or different aux hidden-state layers.
- **Data:** 10k may be too little / too narrow — replay or data mixing to lift
  acceptance without re-introducing OOD regression.
- **More training / capacity** if the above plateau.

## Reproduce

```bash
# ALLaVA four-way (in-domain)
DRAFT=.../checkpoints/checkpoint_best ALLAVA_JSONL="$(pwd)/data/allava/allava_qwen35_distill_10k.jsonl" \
INFER_NUM_SPEC=7 MTP_SPEC=7 NUM_PROMPTS=128 bash examples/evaluate/test_dflash_allava_val_weights.sh

# MMStar three-way (OOD), one run, same spec
DRAFT=.../checkpoints/checkpoint_best INFER_NUM_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_dflash_mmstar_three_way.sh
```

Source logs: `origin/main:debug_error_from_inside`. Artifacts:
`output/allava_val_weight_tests/20260610_073126`, `output/mmstar_weight_tests/20260610_081248`,
`output/mmstar_three_way_tests/20260610_083756`.
