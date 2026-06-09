
Summary
{
  "endpoint": "http://localhost:8100/v1",
  "model": "qwen3.5-9b-allava-val-weight-test",
  "num_requested": 128,
  "completed": 128,
  "failed": 0,
  "completion_tokens": 16384,
  "wall_sec": 277.2314842239721,
  "output_tok_per_sec": 59.09862671572886,
  "mean_latency_sec": 2.165717885937738,
  "reference_contains_rate": 0.0,
  "reference_count": 128,
  "spec_metrics_delta": {
    "vllm:spec_decode_num_drafts_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 6274.0,
    "vllm:spec_decode_num_drafts_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 0.0,
    "vllm:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 43918.0,
    "vllm:spec_decode_num_draft_tokens_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 10130.0,
    "vllm:spec_decode_num_accepted_tokens_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"0\"}": 4214.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"1\"}": 2532.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"2\"}": 1350.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"3\"}": 888.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"4\"}": 566.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"5\"}": 370.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"6\"}": 210.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"0\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"1\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"2\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"3\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"4\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"5\"}": 0.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{engine=\"0\",model_name=\"qwen3.5-9b-allava-val-weight-test\",position=\"6\"}": 0.0
  },
  "spec_draft_steps_total": 6274.0,
  "spec_draft_tokens_total": 43918.0,
  "spec_accepted_tokens_total": 10130.0,
  "spec_accepted_tokens_by_position": {
    "0": 4214.0,
    "1": 2532.0,
    "2": 1350.0,
    "3": 888.0,
    "4": 566.0,
    "5": 370.0,
    "6": 210.0
  },
  "spec_token_acceptance_rate": 0.23065713374925997,
  "spec_first_position_acceptance_rate": 0.6716608224418233,
  "spec_mean_accepted_tokens_per_draft": 1.6145999362448198
}
Stopping vLLM server pid=87390
# ALLaVA Val Four-Way Benchmark

| rank | method | tok/s | mean accept/draft | token accept | first-pos accept | completed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | mtp | 67.073 | 2.821 | 0.403 | 0.817 | 128/128 |
| 2 | dflash_original | 64.950 | 1.954 | 0.279 | 0.731 | 128/128 |
| 3 | trained_dflash | 59.099 | 1.615 | 0.231 | 0.672 | 128/128 |
| 4 | baseline | 31.592 | n/a | n/a | n/a | 128/128 |

## Key Ratios

- trained/original DFlash tok/s: `0.910`
- trained/MTP tok/s: `0.881`
- trained/baseline tok/s: `1.871`
- original DFlash/baseline tok/s: `2.056`
- MTP/baseline tok/s: `2.123`

## Verdict

Trained DFlash does not beat original DFlash on acceptance or tok/s here.

summary_jsonl=/data/wenxuan/speculators/output/allava_val_weight_tests/20260609_072029/allava_val_four_way_summary.jsonl
summary_csv=/data/wenxuan/speculators/output/allava_val_weight_tests/20260609_072029/allava_val_four_way_summary.csv
summary_md=/data/wenxuan/speculators/output/allava_val_weight_tests/20260609_072029/allava_val_summary.md

Artifacts:
  /data/wenxuan/speculators/output/allava_val_weight_tests/20260609_072029
