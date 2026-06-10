# ALLaVA Val Four-Way Benchmark Summary

## TL;DR (current, 2026-06-10)

The **CE-trained `checkpoint_best`** is the first continued-training DFlash that
**beats the original DFlash draft in-domain** on the ALLaVA distilled val tail:
+2.2% tok/s, +2.6% mean accepted tokens/draft, +2.9% token acceptance.
First-position acceptance is a tie (0.727 vs 0.729). MTP is still the strongest
single method, but the project goal -- beat the open-source DFlash draft within
the ALLaVA domain -- is met (modestly).

This flips the previous (kl_div) result, which clearly lost (see history below).
Three things changed: `LOSS_FN=ce` (target the verifier's top-1, not the soft
KL), the correct distilled val jsonl (`allava_qwen35_distill_10k.jsonl`, not
`allava_10000.jsonl`), and selecting `checkpoint_best` (min val loss; under CE
val loss tracks top-1) instead of the MMStar-throughput-selected epoch 6.

## Source log

- `origin/main:debug_error_from_inside` commit `e5ee89d`
- Artifact dir: `/home/wenxuan/speculators/output/allava_val_weight_tests/20260610_073126`

Command:

```bash
DRAFT=.../dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/<CE_RUN>/checkpoints/checkpoint_best \
ALLAVA_JSONL="$(pwd)/data/allava/allava_qwen35_distill_10k.jsonl" \
INFER_NUM_SPEC=7 MTP_SPEC=7 NUM_PROMPTS=128 \
bash examples/evaluate/test_dflash_allava_val_weights.sh
```

## Results (CE checkpoint_best, @spec7, n=128)

| rank | method | tok/s | vs baseline | mean accept/draft | token accept | first-pos accept | completed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | MTP@7 | **65.997** | 2.149x | **2.807** | **0.401** | **0.818** | 128/128 |
| 2 | trained DFlash@7 | 64.197 | 2.091x | 1.993 | 0.285 | 0.727 | 128/128 |
| 3 | original DFlash@7 | 62.821 | 2.046x | 1.942 | 0.277 | 0.729 | 128/128 |
| 4 | baseline, no spec | 30.704 | 1.000x | n/a | n/a | n/a | 128/128 |

Trained DFlash vs original DFlash (the win):

```text
tok/s:              64.197 / 62.821 = 1.022x  (+2.2%)
mean accept/draft:  1.993  / 1.942  = 1.026x  (+2.6%)
token accept:       0.285  / 0.277  = 1.029x  (+2.9%)
first-pos accept:   0.727  / 0.729  = 0.997x  (tie, within noise)
```

Trained DFlash vs MTP (still behind):

```text
tok/s:              64.197 / 65.997 = 0.973x  (~2.7% slower)
mean accept/draft:  1.993  / 2.807  = 0.710x  (~29% lower)
```

Ordering: `MTP@7 > trained DFlash@7 > original DFlash@7 > baseline`.

## Interpretation

- The gain is real but small, and it is concentrated at deeper draft positions,
  not position 1. First-pos is flat (0.727 vs 0.729), while mean accept/draft
  rises (1.993 vs 1.942) -- CE pushed positions 2-6 acceptance up a bit, which is
  what lifts tok/s above original. Trained accepted-by-position:
  `{0:3988, 1:2552, 2:1610, 3:1091, 4:809, 5:538, 6:341}` over 5483 draft steps.
- Sanity: original DFlash first-pos here (0.729) matches history (~0.731), so the
  harness/data split is consistent; the win is not a measurement artifact.
- MTP remains the strongest method on ALLaVA (mean 2.807). Closing the
  DFlash-vs-MTP gap is a separate, harder goal than beating original DFlash.

## History — previous kl_div result (the baseline this beat)

`origin/main:debug_error_from_inside` commit `cdab42c`, checkpoint `6` (selected
by MMStar throughput), `ALLAVA_JSONL=allava_10000.jsonl`:

| method | tok/s | mean accept/draft | token accept | first-pos accept |
|---|---:|---:|---:|---:|
| MTP@7 | 67.073 | 2.821 | 0.403 | 0.817 |
| original DFlash@7 | 64.950 | 1.954 | 0.279 | 0.731 |
| trained DFlash@7 (kl_div) | 59.099 | 1.615 | 0.231 | 0.672 |
| baseline | 31.592 | n/a | n/a | n/a |

Then trained LOST on every metric (e.g. mean accept 1.615 vs 1.954, 0.826x).
CE + checkpoint_best + the correct distilled jsonl turned that into a win.

## Next

1. OOD check: run `test_dflash_mmstar_weights.sh` (original vs trained @7) on the
   same `checkpoint_best`. In-domain FT historically regressed OOD MMStar (best
   ever ~0.87x); the question is whether CE forgets less. Not expected to beat the
   high native bar (~0.764 / 2.09).
2. Push the in-domain gain further: higher warm-start LR (`LR_FT=3e-5`) and/or
   more epochs while watching the ALLaVA sweep curve; consider replay/data mixing
   to lift first-pos (the metric that did not move).
