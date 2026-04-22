# SFT Dataset Length Comparison

- Source splits: `sft_cot_train.jsonl` vs `sft_nocot_train.jsonl`
- Lengths are measured on assistant message content only, without chat template wrapper.

| Dataset | Count | Avg Chars | Avg Words | Avg Tokens | Median Tokens | Min Tokens | Max Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-CoT Train | 367 | 384.10 | 67.54 | 118.08 | 77.00 | 24 | 707 |
| SFT-NoCoT Train | 367 | 35.53 | 5.68 | 10.33 | 8.00 | 1 | 41 |

## Delta

- Avg assistant chars delta (`NoCoT - CoT`): `-348.57`
- Avg assistant tokens delta (`NoCoT - CoT`): `-107.75`
