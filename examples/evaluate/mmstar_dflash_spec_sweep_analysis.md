# MMStar DFlash Spec-Length Sweep

Source log: `origin/main:debug_error_from_inside` at commit `33f4a5d`.

This note compares the native/raw DFlash baseline against the continued-training
DFlash checkpoint on the same 128 MMStar open-ended prompts. Both runs use
Qwen3.5-9B as the verifier model. The only intended sweep variable is
`num_speculative_tokens`.

## Summary

The continued-training checkpoint loads and runs, but it does not improve over
the native DFlash baseline for either tested inference length.

| infer spec | native tok/s | trained tok/s | trained/native | native mean accept/draft | trained mean accept/draft | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 68.679 | 59.762 | 0.870 | 2.092 | 1.556 | not improved |
| 3 | 59.322 | 52.696 | 0.888 | 1.631 | 1.269 | not improved |

## Detailed Metrics

### `INFER_NUM_SPEC=7`

| metric | native DFlash | trained DFlash | trained/native |
| --- | ---: | ---: | ---: |
| completed | 128/128 | 128/128 | n/a |
| output tok/s | 68.679 | 59.762 | 0.870 |
| reference hit | 0.234 | 0.234 | 1.000 |
| draft steps | 5,333 | 6,430 | 1.206 |
| draft tokens | 37,331 | 45,010 | 1.206 |
| accepted tokens | 11,159 | 10,003 | 0.896 |
| token acceptance rate | 0.299 | 0.222 | 0.743 |
| first-position acceptance | 0.764 | 0.690 | 0.903 |
| mean accepted tokens / draft | 2.092 | 1.556 | 0.743 |

Per-position acceptance from the vLLM log:

| position | native | trained |
| ---: | ---: | ---: |
| 0 | 0.763 | 0.690 |
| 1 | 0.483 | 0.416 |
| 2 | 0.298 | 0.219 |
| 3 | 0.200 | 0.106 |
| 4 | 0.148 | 0.061 |
| 5 | 0.119 | 0.040 |
| 6 | 0.082 | 0.024 |

### `INFER_NUM_SPEC=3`

| metric | native DFlash | trained DFlash | trained/native |
| --- | ---: | ---: | ---: |
| completed | 128/128 | 128/128 | n/a |
| output tok/s | 59.322 | 52.696 | 0.888 |
| reference hit | 0.227 | 0.227 | 1.000 |
| draft steps | 6,228 | 7,218 | 1.159 |
| draft tokens | 18,684 | 21,654 | 1.159 |
| accepted tokens | 10,158 | 9,160 | 0.902 |
| token acceptance rate | 0.544 | 0.423 | 0.778 |
| first-position acceptance | 0.777 | 0.682 | 0.878 |
| mean accepted tokens / draft | 1.631 | 1.269 | 0.778 |

Per-position acceptance from the vLLM log:

| position | native | trained |
| ---: | ---: | ---: |
| 0 | 0.777 | 0.682 |
| 1 | 0.519 | 0.381 |
| 2 | 0.335 | 0.206 |

## Interpretation

1. The trained checkpoint is functional: both runs complete all 128 requests and
   emit DFlash speculative metrics.
2. The native DFlash baseline is better on the speed-driving metrics in both
   settings. It accepts more draft tokens while drafting fewer total steps.
3. `INFER_NUM_SPEC=7` is faster than `INFER_NUM_SPEC=3` for both models in this
   test. Although `spec=3` has a higher token acceptance rate, it accepts fewer
   tokens per draft step, so the scheduling/draft overhead is paid more often.
4. The reference-hit scores are essentially tied. This smoke metric is not a
   strong quality metric, and speculative decoding should preserve the verifier
   distribution; the useful signal here is throughput and acceptance.
5. The current result supports "continued-training DFlash works" but not
   "continued-training improves over native DFlash" on this MMStar slice.

## Recommended Next Checks

1. Keep `INFER_NUM_SPEC=7` as the primary comparison point unless a later sweep
   finds a better setting.
2. Compare more checkpoints along the continued-training trajectory, especially
   later checkpoints after more ALLaVA samples.
3. Verify that the auto-selected `DRAFT` is the intended continued-training
   checkpoint by checking the `Checkpoint sanity OK (trained)` path printed at
   the start of each run.
4. If the goal is to beat native DFlash, track `mean accepted tokens / draft` and
   per-position acceptance first; these metrics explain the current throughput
   gap more directly than the reference-hit score.
