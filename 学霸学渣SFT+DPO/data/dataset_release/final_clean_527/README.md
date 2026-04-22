# Final Clean Dataset (527)

这个目录是当前可直接用于后训练的最终版本数据集。

## 内容

- `master_clean_dataset_final.jsonl`
  当前最终主数据集，共 `527` 题。每题同时包含：
  - `question`
  - `gold_answer`
  - `cot_response`
  - `nocot_response`
  - `difficulty`
  - `question_type`
  - `subset`
  - `metadata`

- `master_train.jsonl`
- `master_val.jsonl`
- `master_test.jsonl`
  在题目级主表上完成的切分结果，互斥且可复现，切分参数：
  - `seed = 42`
  - `train = 70%`
  - `val = 10%`
  - `test = 20%`

- `sft_cot_train.jsonl`
- `sft_nocot_train.jsonl`
- `dpo_adaptive_train.jsonl`
  从 `master_train.jsonl` 派生的训练文件。

- `sft_cot_val.jsonl`
- `sft_nocot_val.jsonl`
- `dpo_adaptive_val.jsonl`
  从 `master_val.jsonl` 派生的验证文件。

- `test_eval.jsonl`
  从 `master_test.jsonl` 派生的轻量评测文件，只用于测试评估，不参与训练。

- `cleaning_report_final.md`
- `cleaning_stats_final.json`
  最终清洗报告。

- `split_report.md`
- `split_stats.json`
  切分与一致性检查报告。

## 推荐用法

### 1. 训练 SFT-CoT

使用：

- `sft_cot_train.jsonl`
- `sft_cot_val.jsonl`

### 2. 训练 SFT-NoCoT

使用：

- `sft_nocot_train.jsonl`
- `sft_nocot_val.jsonl`

### 3. 训练 DPO-Adaptive

使用：

- `dpo_adaptive_train.jsonl`
- `dpo_adaptive_val.jsonl`

DPO 构造规则：

- `simple`：`chosen = nocot_response`，`rejected = cot_response`
- `complex`：`chosen = cot_response`，`rejected = nocot_response`

### 4. 最终评测

使用：

- `test_eval.jsonl`

不要把 `test` 用于训练或调参。

## 设计原则

这版数据是从对齐后的 `928` 题主索引出发构建的，最终保留 `527` 题。

保留原则：

- 答案正确优先
- `CoT` 需要有基本推理痕迹且明显长于 `NoCoT`
- `NoCoT` 允许一句简短说明，但不允许明显多步推理
- 删除空输出、拒答、伪 CoT、明显错答、明显跑题样本

## 一致性说明

- `master_train / master_val / master_test` 三者互斥
- `train` 内三份文件逐行对齐
- `val` 内三份文件逐行对齐
- `test` 不导出训练用 SFT / DPO 文件

