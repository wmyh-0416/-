# Evaluation Report

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Model source used: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`
- Adapter path: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_cot_qwen25_3b/checkpoint-80`
- Test set: `data/dataset_release/final_clean_527/test_eval.jsonl`
- Output directory: `/home/ym3447/学霸学渣SFT+DPO/outputs/evals_sft_cot_smoke`

## Metric

- Accuracy is computed from `extracted_final_answer` against `gold_answer` after normalization.
- Open-ended items additionally allow normalized containment matching.

## Summary

| Mode | Simple Accuracy | Complex Accuracy | Overall Accuracy | Simple Avg Tokens | Complex Avg Tokens | Overall Avg Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0.0000 | 0.0000 | 0.0000 | 0.00 | 128.00 | 128.00 |
| cot | 0.0000 | 0.0000 | 0.0000 | 0.00 | 160.00 | 160.00 |

## Files

- Direct predictions: `sft_cot_smoke_direct_predictions.jsonl`
- CoT predictions: `sft_cot_smoke_cot_predictions.jsonl`
- Stats JSON: `sft_cot_smoke_eval_stats.json`
