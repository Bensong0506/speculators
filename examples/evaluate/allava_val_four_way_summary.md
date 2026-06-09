# ALLaVA Val Four-Way Benchmark Summary

Source log:

- `origin/test_result` commit `9e822ea`
- `origin/main:debug_error_from_inside` commit `cdab42c`
- Artifact dir: `/data/wenxuan/speculators/output/allava_val_weight_tests/20260609_072029`

Command family:

```bash
DRAFT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609_012511/checkpoints/6 \
ALLAVA_JSONL="$(pwd)/data/allava/allava_10000.jsonl" \
INFER_NUM_SPEC=7 \
MTP_SPEC=7 \
NUM_PROMPTS=128 \
bash examples/evaluate/test_dflash_allava_val_weights.sh
```

## One-Line Conclusion

The distilled continued DFlash checkpoint does not improve even on the ALLaVA validation tail. It is faster than no-spec baseline, but slower than both native MTP and original DFlash, and its acceptance metrics are lower than original DFlash.

## Results

| rank | method | tok/s | vs baseline | mean accept/draft | token accept | first-pos accept | completed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | MTP@7 | **67.073** | **2.123x** | **2.821** | **0.403** | **0.817** | 128/128 |
| 2 | original DFlash@7 | 64.950 | 2.056x | 1.954 | 0.279 | 0.731 | 128/128 |
| 3 | trained DFlash@7 | 59.099 | 1.871x | 1.615 | 0.231 | 0.672 | 128/128 |
| 4 | baseline, no spec | 31.592 | 1.000x | n/a | n/a | n/a | 128/128 |

## Key Comparisons

Trained DFlash vs original DFlash:

```text
tok/s:              59.099 / 64.950 = 0.910x  (~9.0% slower)
mean accept/draft:  1.615 / 1.954   = 0.826x  (~17.4% lower)
token accept:       0.231 / 0.279   = 0.827x  (~17.3% lower)
first-pos accept:   0.672 / 0.731   = 0.919x  (~8.1% lower)
```

Trained DFlash vs MTP:

```text
tok/s:              59.099 / 67.073 = 0.881x  (~11.9% slower)
mean accept/draft:  1.615 / 2.821   = 0.572x  (~42.8% lower)
```

All speculative configs still help over no-spec baseline, but the ordering is:

```text
MTP@7 > original DFlash@7 > trained DFlash@7 > baseline
```

## Interpretation

This result weakens the "MMStar is just out-of-domain" hypothesis. On ALLaVA val, which is much closer to the distilled training data, trained DFlash still has worse first-position acceptance and lower mean accepted tokens than original DFlash.

The training validation loss can still improve while inference acceptance drops because the two measurements are not identical:

- Training validation is teacher-forced on fixed target responses.
- Speculative decoding needs the draft argmax to match the verifier under autoregressive prefixes.
- A lower loss can come from better probability mass on teacher tokens without improving exact top-1 agreement at decode time.
- Continued training on only 10k samples can overfit the ALLaVA response style and degrade the original DFlash draft's broader alignment with the verifier.

The ALLaVA drop is smaller than the MMStar drop:

| dataset | original DFlash@7 | trained DFlash@7 | trained/original |
|---|---:|---:|---:|
| MMStar | 71.629 | 62.615 | 0.874x |
| ALLaVA val | 64.950 | 59.099 | 0.910x |

So the distilled training may be slightly less harmful in-domain, but it is still not a win.

## Caveat

This run evaluated checkpoint `6`, which was selected by MMStar throughput sweep. If the training `checkpoint_best` symlink points to a different epoch by validation loss, it is worth running this exact ALLaVA four-way script with:

```bash
DRAFT=/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609_012511/checkpoints/checkpoint_best
```

That will tell us whether validation-best and throughput-best disagree.

## Recommended Next Experiments

1. Sweep all checkpoints on ALLaVA val, not only MMStar, and select by `first-pos accept` or `tok/s`.
2. Run the same ALLaVA four-way test on `checkpoint_best` if it differs from checkpoint `6`.
3. Try lower LR for continue training, e.g. `1e-5` or `3e-5`; current behavior looks like over-updating the original DFlash draft.
4. Add replay/mixing from original DFlash-style or broader distilled data instead of only 10k ALLaVA.
5. Treat `reference_contains_rate=0.0` as meaningless here; ALLaVA answers are long open-ended strings, so exact containment is not a useful quality metric.
