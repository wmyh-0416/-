# Split Report

切分是在题目级 master dataset 上完成的，随后才从各 split 派生 SFT 和 DPO 数据，以避免信息泄漏。

## Source Counts

- total: 527
- simple: 335
- complex: 192
- objective: 446
- open_ended: 81

## Expected vs Actual

- expected total/simple/complex: 527 / 335 / 192
- actual total/simple/complex: 527 / 335 / 192

## Split Counts

### train

- total: 367
- simple: 233
- complex: 134
- objective: 311
- open_ended: 56

### val

- total: 52
- simple: 33
- complex: 19
- objective: 44
- open_ended: 8

### test

- total: 108
- simple: 69
- complex: 39
- objective: 91
- open_ended: 17

## Export Counts

- sft_cot_train: 367
- sft_nocot_train: 367
- dpo_adaptive_train: 367
- sft_cot_val: 52
- sft_nocot_val: 52
- dpo_adaptive_val: 52
- test_eval: 108

## Consistency Checks

- train 三份文件是否逐行对齐: True
- val 三份文件是否逐行对齐: True
- train/val/test 是否互斥: True
- 三个 split 合并后是否正好恢复到 527 条: True

