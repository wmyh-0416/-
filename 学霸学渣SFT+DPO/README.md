# 学霸学渣SFT+DPO

一个用于“学霸/学渣”风格数据构建、SFT 训练与 DPO 对齐实验的新项目骨架。

## 目录结构

```text
学霸学渣SFT+DPO/
├── README.md
├── .gitignore
├── configs/
├── data/
├── scripts/
└── src/
```

## 计划用途

- `data/`: 放原始数据、清洗数据和偏好对数据
- `configs/`: 放 SFT / DPO 训练配置
- `scripts/`: 放数据处理、训练和评估脚本
- `src/`: 放项目核心代码
- `models/`: 放本地模型权重

## 当前已下载模型

- `Qwen/Qwen2.5-3B-Instruct`
- 本地路径: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`

## 下一步建议

1. 明确“学霸”和“学渣”两种风格的定义与标注规则。
2. 先准备 SFT 数据，再构造 chosen/rejected 偏好对用于 DPO。
3. 补充训练脚本、配置文件和评估流程。
