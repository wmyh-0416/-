# Base Evaluation Report

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Model source used: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`
- Test set: `data/dataset_release/final_clean_527/test_eval.jsonl`
- Output directory: `/home/ym3447/学霸学渣SFT+DPO/outputs/baselines`

## Metric

- Accuracy is computed from `extracted_final_answer` against `gold_answer` after normalization.
- Open-ended items additionally allow normalized containment matching.

## Summary

| Mode | Simple Accuracy | Complex Accuracy | Overall Accuracy | Simple Avg Tokens | Complex Avg Tokens | Overall Avg Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0.2464 | 0.1026 | 0.1944 | 21.48 | 32.41 | 25.43 |
| cot | 0.1884 | 0.0513 | 0.1389 | 87.58 | 117.49 | 98.38 |

## Files

- Direct predictions: `base_direct_predictions.jsonl`
- CoT predictions: `base_cot_predictions.jsonl`
- Stats JSON: `base_eval_stats.json`
