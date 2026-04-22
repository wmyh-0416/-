# Final Results Report

## Prompt Modes

- `Base direct`: explicit concise-answer prompt.
- `Base cot`: explicit think-then-answer prompt.
- `SFT-CoT`, `SFT-NoCoT`, `DPO-Adaptive`: default user-only chat prompt so the model chooses its own response length.

## Summary Table

| Model | Prompt Mode | Simple Accuracy | Complex Accuracy | Overall Accuracy | Simple Avg Tokens | Complex Avg Tokens | Overall Avg Tokens | Length Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base direct | direct | 0.2464 | 0.1026 | 0.1944 | 21.48 | 32.41 | 25.43 | 10.93 |
| Base cot | cot | 0.1884 | 0.0513 | 0.1389 | 87.58 | 117.49 | 98.38 | 29.91 |
| SFT-CoT | default | 0.2029 | 0.1282 | 0.1759 | 59.48 | 142.97 | 89.63 | 83.50 |
| SFT-NoCoT | default | 0.2609 | 0.1282 | 0.2130 | 9.09 | 29.33 | 16.40 | 20.25 |
| DPO-Adaptive | default | 0.1014 | 0.1026 | 0.1019 | 82.61 | 145.97 | 105.49 | 63.37 |

## Interpretation

- Most adaptive by length gap under the default prompt: `SFT-CoT`.
- It shows a length gap of `83.50` tokens with overall accuracy `0.1759`.
- A larger positive length gap means the model spends more tokens on complex items than on simple items.
