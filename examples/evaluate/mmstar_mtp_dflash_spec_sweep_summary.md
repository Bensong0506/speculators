=== Aggregating trained-vs-baseline results ===
# MMStar Trained DFlash Best Checkpoint vs Baselines

Best checkpoint selected by `trained_tok_s` at `spec=7`:

`/data/wenxuan/speculators/output/dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash/dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609_012511/checkpoints/6`

Selection value: `63.913`

## Ranked Results

| rank | group | method | spec | status | tok/s | mean accept/draft | token accept | first-pos accept | completed |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | dflash_original | 7 | seeded | 71.629 | 2.097 | 0.300 | 0.767 | 128/128 |
| 2 | baseline | mtp | 7 | ok | 68.965 | 2.822 | 0.403 | 0.828 | 128/128 |
| 3 | baseline | mtp | 5 | ok | 68.424 | 2.600 | 0.520 | 0.837 | 128/128 |
| 4 | baseline | mtp | 3 | seeded | 67.393 | 2.045 | 0.682 | 0.852 | 128/128 |
| 5 | baseline | dflash_original | 5 | ok | 66.930 | 1.944 | 0.389 | 0.766 | 128/128 |
| 6 | trained | trained_dflash | 7 | ok | 62.615 | 1.704 | 0.243 | 0.690 | 128/128 |
| 7 | baseline | dflash_original | 3 | ok | 61.084 | 1.638 | 0.546 | 0.779 | 128/128 |
| 8 | trained | trained_dflash | 5 | ok | 59.932 | 1.611 | 0.322 | 0.700 | 128/128 |
| 9 | trained | trained_dflash | 3 | ok | 54.642 | 1.402 | 0.467 | 0.707 | 128/128 |

## Key Comparisons

- Best trained config: `trained_dflash@7` = `62.615` tok/s.
- vs `MTP@7`: trained/MTP@7 = `0.908`.
- vs `original DFlash@7`: trained/original = `0.874`.
- vs best baseline `dflash_original@7`: trained/best-baseline = `0.874`.

## Verdict

Trained DFlash does not beat the main baselines on this MMStar slice.

final_jsonl=/data/wenxuan/speculators/output/mmstar_trained_dflash_best_vs_baselines/single_n128_tok128/output_dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash_dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609/final_results.jsonl
final_csv=/data/wenxuan/speculators/output/mmstar_trained_dflash_best_vs_baselines/single_n128_tok128/output_dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash_dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609/final_results.csv
final_md=/data/wenxuan/speculators/output/mmstar_trained_dflash_best_vs_baselines/single_n128_tok128/output_dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash_dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609/final_results.md

Artifacts:
  /data/wenxuan/speculators/output/mmstar_trained_dflash_best_vs_baselines/single_n128_tok128/output_dflash_qwen3.5_9b_mm_distilled_10k_continue_dflash_dflash_qwen35_9b_allava_distilled_10k_continue_dflash_20260609
